# VecDB branch sources

This private package carries VecDB code until the corresponding SDK changes ship.
Keep the upstream module paths, class names, request dictionaries and validation
rules so updates can be compared against their source. It is not a public SDK.

Repository: https://orahub.oci.oraclecorp.com/database-appliedai-dev/vecdb-python-sdk

| Local file | Upstream file | Branch / commit |
| --- | --- | --- |
| `parameter_validation.py` | `src/oracle_vecdb/parameter_validation.py` | `feature/anmathad_vec-821`, `b343f231f1c5df23526031ed236fb78e41e45ff4` |
| `services/sqlnet/query_mapper.py` | `src/oracle_vecdb/services/sqlnet/query_mapper.py` | `feature/anmathad_vec-821`, `b343f231f1c5df23526031ed236fb78e41e45ff4` |
| `services/ords/hybrid_search.py` | `src/oracle_vecdb/services/ords/hybrid_search.py` | `feature/tshirish_hybrid_json`, `cee35a4ee240dcf764d1258f15ac56a6d93c4129` |

## Local changes to copied files

- Preserve the Oracle copyright and UPL notices; apply repository formatting.
- Use `Optional[...]` in Pydantic schemas for the host package's Python 3.9 support.
- Keep only the parameter schemas and SQL templates used by supported native
  operations; omit unused rebuild, rerank and model/job lifecycle definitions.
- Keep the hybrid request builder and its reachable validation helpers. Omit
  unused option whitelists, standalone index planning, conflicting-index drop,
  ORDS execution/hydration, preview facade and scheduler/job lifecycle code.
  Two lookup methods are adapter extension points instead of ORDS queries.
- Extract the hybrid DDL renderer from `_hybrid_index_plan` into
  `_create_hybrid_index_statement`. Both SQL-layout and VecDB index creation use
  this single renderer with their existing caller-managed vectorizer preferences.

## LangChain boundary

`vectorstores/_vecdb_adapter.py` owns output-bind adaptation, JSON response
normalization, document mapping, and database-generation differences. The
transport retains connection/pool ownership and native async execution.

The adapter uses copied SQL templates for search, listing, deletion, description,
and real index creation. It wraps SELECT results/local CLOBs in a `:result`
output bind. Table creation, table dropping, and upserts retain the previously
supported package function signatures, which differ from the branch's legacy
SQL*Net mappings. These differences are isolated in `_function_statement`.

Native requests use the copied parameter validation. Existing LangChain HNSW
helpers explicitly select `distribute_method=AUTO` when none was supplied;
callers can override it. HNSW requests also materialize OracleVS's existing
`neighbors=32` and `efConstruction=200` defaults: the live VecDB database rejected
omitted construction settings with ORA-51915. Explicit SDK advanced parameters
retain precedence over these defaults. Filters pass through without copying the SQL*Net
service's limited Python filter evaluator or its synthetic result fallbacks.

Hybrid retrieval uses `_hybrid_search_request` and the SDK's serialized vector,
JSON return defaults and metadata text path. A single adapter mapping owns
VecDB retriever options, mode-to-fusion defaults and literal text handling; it no
longer builds a SQL-layout request first. Constructor and runtime option checks
are shared. Query vectors come from LangChain,
so the adapter requires the stored text path but not a database embedding model
for request construction. New SEARCH options pass through, matching the source
request builder rather than its older, unused option whitelist. Hydration uses
one bound SQL join instead of the branch's per-hit SQL requests. Catalog index
names retain their exact case and punctuation (including `$`) in bound JSON;
they are not interpreted as SQL text. Document text
comes from metadata, not a potentially truncated search snippet.

Hybrid DDL retains LangChain's existing preference management, metadata-index
name, callback, scheduling settings, transaction guard and best-effort recovery.
The copied implementation contains no alternative index-drop or lifecycle path.
The branch's synthetic SQL*Net indexes/jobs and ORDS-only scheduler are not used.
REST clients continue to use public methods from the optional installed SDK.

## Updating or replacing the copy

Compare the three files against the recorded commits. Refresh the source helpers
and run the adapter, VecDB, index recovery, hybrid/text retriever and compatibility
tests. Once an SDK release provides equivalent commands, validation and native
async behavior, replace the copied implementation behind the adapter, verifying
connection ownership, supported database generations, scores and index recovery.
Do not expose the private copied modules as public LangChain APIs.
