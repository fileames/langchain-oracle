# Bug fix: `OracleSemanticCache` collapses agent state when wired via `set_llm_cache`

**Package:** `langchain-oracledb`
**Files changed:**
- `libs/oracledb/langchain_oracledb/cache.py` — add `_has_tool_calls` + `_reset_generation_ids` helpers; guard the write path in `OracleSemanticCache.update`; reset message ids in `OracleSemanticCache.lookup`.

**Scope:** two small helpers, one guard in `update`, one call in `lookup`. No schema migration, no serde change, no public API change.

---

## Symptom

Any LangChain agent built with `langchain.agents.create_agent` (or the deprecated `langgraph.prebuilt.create_react_agent`) and `set_llm_cache(OracleSemanticCache(...))` fails on its first tool-calling turn with:

```
KeyError: 'model'
File ".../langgraph/graph/_branch.py", line 202, in BranchSpec._finish
    destinations: Sequence[Send | str] = [
        r if isinstance(r, Send) else self.ends[r] for r in result
    ]
KeyError: 'model'
During task with name 'model' and id '50e8238e-bb58-e196-1741-d78284637217'
```

Without the cache installed, the same agent runs to completion cleanly. Minimal reproduction (requires Oracle 23ai + an embedding model + an OpenAI-compatible LLM):

```python
import oracledb
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_oracledb import OracleSemanticCache

@tool
def check_inventory(sku: str) -> str:
    """stub"""
    return f"{sku}: stock=87"

conn = oracledb.connect(user=..., password=..., dsn="...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                             encode_kwargs={"normalize_embeddings": True})
cache = OracleSemanticCache(client=conn, embedding=emb,
                             table_name="DEMO", score_threshold=0.35)
set_llm_cache(cache)

llm = ChatOpenAI(model="gpt-5.2", max_completion_tokens=600)
agent = create_agent(model=llm, tools=[check_inventory],
                     system_prompt="Check inventory when asked.")
agent.invoke({"messages": [HumanMessage(content="Check SKU-1002 stock")]})
# -> KeyError: 'model'
```

A second, quieter symptom surfaces once you work around the first: every turn after turn 1 silently returns turn 1's final answer (or, before the id reset, echoes the user's question back verbatim) even when the user asks a different question.

## Root cause (two linked issues)

### 1. Tool-call `AIMessage`s must not be cached

The post-model router in `langchain/agents/factory.py` (`_make_model_to_tools_edge`) has six cases. Case 6 fires when `last_ai_message.tool_calls` is non-empty **but** every tool call has a matching `ToolMessage.tool_call_id` already in state — i.e., "this AI message's tool calls are already satisfied." In that case the router returns `model_destination` (the string `'model'`) to jump back to the model node. But the conditional edge emitted from the `model` node has `ends = ['tools', '__end__']` — `'model'` is not in it, so `BranchSpec._finish` raises `KeyError`.

How does an `AIMessage` with already-satisfied tool_calls end up in state, causing this branch to re-fire? Sequence:

1. First model call emits `AIMessage(id=A, tool_calls=[X])`. Router returns `Send("tools", ...)` → tool runs → `ToolMessage(tool_call_id=X)` appended.
2. Control returns to the model node for the summarization step.
3. With `OracleSemanticCache` installed, `BaseChatModel._generate_with_cache` serializes the new message list and calls `cache.lookup(prompt, llm_string)`. Vector search finds turn 1's cached entry (the content is largely shared), returns the **same** `AIMessage(id=A, tool_calls=[X])` — the cached procedural step.
4. LangGraph's `add_messages` reducer dedupes by `id`. Since `id=A` is already in state, the cached message **replaces** the existing one instead of being appended. State is unchanged.
5. The router runs on the same state: `last_ai = AI(id=A, tool_calls=[X])`, `tool_messages = [Tool(call_id=X)]`, `pending_tool_calls = []` → case 6 fires → returns `'model'` → `KeyError`.

The underlying principle: **tool-call `AIMessage`s are procedural intermediates, not final answers**. They are meaningful only inside a specific agent step; replaying one is never correct. A semantic cache that persists and re-emits them necessarily breaks agent loops.

### 2. Cached chat messages must not keep their original `id`

Even after tool-call caching is blocked, the same dedup mechanism bites the *final* summarization message. On a subsequent agent turn, the cache can hit on a semantically close prompt and return the cached final `AIMessage(id=A, content="...")`. If `id=A` already lives in the current state — for example, because an upstream helper (like `OracleChatMessageHistory`) re-hydrated that same message from a prior turn — the reducer again dedupes, no new message is appended, the graph terminates without making progress, and `result["messages"][-1]` ends up being the user's own `HumanMessage`. The user sees their question echoed back as the agent's reply.

The fix mirrors what `BaseChatModel._convert_cached_generations` already does for `usage_metadata.total_cost` (it mutates the cached `AIMessage` via `model_copy` so the caller sees a non-misleading value): on load, drop the stale `id` so downstream reducers treat the cached message like a fresh generation.

## Fix

### `libs/oracledb/langchain_oracledb/cache.py`

Two module-level helpers — one guards the write path, one sanitizes the read path.

```python
def _reset_generation_ids(generations: RETURN_VAL_TYPE) -> None:
    """Clear the ``.id`` on cached chat messages before returning them.

    Cached ``AIMessage``s keep whatever ``id`` they had at write time.
    When LangChain caching is wired into a LangGraph agent, the graph's
    ``add_messages`` reducer dedupes by ``id``: a cached message with an
    id already present in state replaces the existing one instead of
    being appended, and state stops progressing. Dropping the id on load
    matches how a fresh LLM call would behave (the runtime mints a new
    ``lc_run--<uuid>`` id when the message is emitted).
    """
    for gen in generations:
        message = getattr(gen, "message", None)
        if message is not None and getattr(message, "id", None) is not None:
            try:
                message.id = None
            except (AttributeError, TypeError):
                pass


def _has_tool_calls(generations: RETURN_VAL_TYPE) -> bool:
    """Return True if any generation carries a tool-call message."""
    for gen in generations:
        message = getattr(gen, "message", None)
        if message is None:
            continue
        if getattr(message, "tool_calls", None):
            return True
        if getattr(message, "invalid_tool_calls", None):
            return True
    return False
```

`OracleSemanticCache.lookup` — strip ids from any chat messages on their way out:

```diff
         return_val = document.metadata.get(self.RETURN_VAL)
         if not isinstance(return_val, str):
             return None
-        return _loads_generations(return_val)
+        generations = _loads_generations(return_val)
+        if generations is not None:
+            _reset_generation_ids(generations)
+        return generations
```

`OracleSemanticCache.update` — refuse to persist tool-call generations:

```diff
-    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
-        """Insert or update a semantic cache entry."""
+    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
+        """Insert or update a semantic cache entry.
+
+        Generations whose message carries ``tool_calls`` are intentionally
+        **not** cached. Tool-call responses are procedural (one step inside
+        an agent loop, not a final answer), and replaying them collapses
+        agent state: a cached tool-call ``AIMessage`` keeps its original
+        ``id``, LangGraph's ``add_messages`` reducer dedupes by ``id``, and
+        the graph fails to advance past the tool node.
+        """
+        if _has_tool_calls(return_val):
+            return
+
         metadata = {
             self.PROMPT_HASH: _hash_value(prompt),
             self.LLM_HASH: _hash_value(llm_string),
             self.RETURN_VAL: _dumps_generations(return_val),
         }
         self._vector_store.add_texts(
             [prompt],
             [metadata],
             ids=[_cache_entry_id(prompt, llm_string)],
         )
```

## Alternatives considered

1. **Lower `score_threshold` until false-positive hits stop.** Works in the narrow "agent + cache" scenario but is brittle: the right threshold depends on prompt length and embedding model, and the same value that stops agent collisions also kills paraphrase recall for the single-shot use case the cache was designed for. Rejected as a library fix — the threshold is a legitimate tuning knob, not a bug workaround. The notebook now uses a *tight* threshold (0.02) for the agent cache and a looser one (0.35) for the directed paraphrase-recall tests, and the two can coexist because this patch makes them non-destructive.
2. **Stop using `set_llm_cache` for agents and cache at the user-question layer.** A perfectly valid application-level pattern (extract the user's latest question, probe `cache.lookup(user_text, key)` yourself, short-circuit on hit). But it would mean the library can't be used with the standard LangChain cache contract against chat agents — a significant regression. Rejected in favor of making `set_llm_cache` work correctly.
3. **Never cache any `ChatGeneration`.** Too broad — correctly-cached final answers are the main value of `OracleSemanticCache` for chat models. Rejected.
4. **`_has_tool_calls` on write + `_reset_generation_ids` on read (chosen).** Minimal, local, and each change addresses a distinct failure mode (procedural-step caching; dedup-by-id state collapse). The write guard prevents tool-call `AIMessage`s from ever entering the cache in the first place; the read sanitizer covers the case where any stored message — including legacy entries from earlier versions — could collide with an id already in the caller's state.

## Verification

Live 23ai instance, `gpt-5.2` via `ChatOpenAI`, the supply-chain agent from `samples/11-oracle-cache-and-chat-history/supply_chain_assistant.ipynb`.

Before the fix:

```
--- Turn 1 ---
USER : Hi, I need to check on SKU-1002. Whats our current stock?
[ERROR: KeyError: 'model']
```

With `_has_tool_calls` guard only (id reset not yet applied):

```
--- Turn 1 --- AGENT: SKU-1002 is in stock: 87 units.        ← real summary
--- Turn 2 --- AGENT: Is that below the reorder point?        ← user's question echoed
--- Turn 3 --- AGENT: Who supplies that SKU and how long...   ← user's question echoed
```

With both fixes applied:

```
--- Turn 1 --- AGENT: SKU-1002 current on-hand stock is 87 units in Dallas-C (reorder point: 200).
--- Turn 2 --- AGENT: Yes. SKU-1002 is below the reorder point: 87 on hand vs 200 reorder point (short by 113 units).
--- Turn 3 --- AGENT: SKU-1002 is supplied by PrecisionForge Ltd. (SUP-302) with a 21-day lead time.
```

The agent completes the full tool-call → tool-run → summarize loop, carries context across turns, and the cache table grows with only final answers (never tool-call intermediates).

The directed `§3` tests in the sample notebook — which exercise `OracleSemanticCache` through its public API with plain `Generation(text=...)` values, not `ChatGeneration`s — are unaffected: `_has_tool_calls` returns `False` for plain generations, and `_reset_generation_ids` no-ops for generations without a `.message` attribute.

## Suggested follow-ups (out of scope for this fix)

- Regression test in `tests/integration_tests/test_cache.py` that exercises the full `langchain.agents.create_agent` + `set_llm_cache(OracleSemanticCache(...))` loop across at least two turns and asserts (a) no exception and (b) distinct replies per turn.
- Unit test that writes a tool-call `ChatGeneration` to `OracleSemanticCache.update` and asserts the table row count does **not** increase — i.e., the write was skipped.
- Unit test that round-trips a `ChatGeneration(AIMessage(id="X", content="hi"))` through `update` → `lookup` and asserts the returned message's `id` is `None`.
- Consider exposing the tool-call skip behavior as a constructor flag (`skip_tool_call_generations: bool = True`) so users with non-LangGraph agent frameworks can opt out if they have a different caching contract.
