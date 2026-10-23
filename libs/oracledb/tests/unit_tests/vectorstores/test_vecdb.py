# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Backend selection, SDK contracts, and bounded VecDB layout repair."""

import json
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.retrievers.hybrid_search import (
    OracleHybridSearchRetriever,
)
from langchain_oracledb.retrievers.text_search import OracleTextSearchRetriever
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
    _layout,
    _migration_actions,
    layout_error,
    native_selection,
    parse_version,
    table_identifier,
    validate_filter,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy


class Embedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def sdk():
    oracle_vecdb = pytest.importorskip("oracle_vecdb")
    client = oracle_vecdb.OracleVecDB(
        oracle_vecdb.Configuration(
            rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
            access_token="test",
        )
    )
    client.describe_vector_table = MagicMock(return_value={"table_name": "DOCS"})
    client.create_vector_table = MagicMock(return_value={"table_name": "DOCS"})
    client.upsert_vectors = MagicMock(return_value={"upserted_count": 1})
    client.delete_vectors = MagicMock(return_value={"message": "deleted"})
    client.list_vectors = MagicMock(return_value={"items": []})
    client.query = MagicMock(
        return_value={
            "items": [
                {
                    "id": "a",
                    "metadata": {"text": "hello", "tenant": "one"},
                    "distance": 0.25,
                    "vector": [1.0, 0.0],
                }
            ]
        }
    )
    return client


@pytest.fixture
def store(sdk):
    return OracleVS(sdk, Embedding(), "DOCS", mutate_on_duplicate=True)


@pytest.mark.parametrize(
    "version, expected",
    [("23.26.1.0.0", (23, 26, 1)), ("Release 23.0.0 Version 23.26.3.0.0", (23, 26, 3))],
)
def test_version(version, expected):
    assert parse_version(version) == expected


@pytest.mark.parametrize(
    "name", ["DOCS; DROP TABLE x", 'x"y', "schema.docs", " ", "DOCS) --"]
)
def test_identifiers_reject_sql(name):
    with pytest.raises(ValueError):
        table_identifier(name)


def test_optional_sdk_not_imported_by_sql_package():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import langchain_oracledb; "
            "assert 'oracle_vecdb' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_factory_reports_missing_sdk():
    with patch.dict(sys.modules, {"oracle_vecdb": None}):
        with pytest.raises(ImportError, match=r"langchain-oracledb\[vecdb\]"):
            OracleVS.from_vecdb_config(object(), "DOCS", Embedding())


def selection_connection(version, columns, supported=True):
    connection = MagicMock(version=version)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(column,) for column in columns]
    cursor.fetchone.return_value = (int(supported),)
    return connection, cursor


@pytest.mark.parametrize(
    "version, expected",
    [
        ("23.4.0.0.0", "sql"),
        ("23.26.1.0.0", "vecdb"),
        ("23.26.2.0.0", "vecdb"),
        ("23.26.3.0.0", "vecdb"),
    ],
)
def test_auto_new_table(version, expected):
    conn, _ = selection_connection(version, [])
    assert native_selection(conn, "DOCS", "auto")[0] == expected


def test_auto_preserves_existing_sql_even_on_current_database():
    conn, cursor = selection_connection(
        "23.26.3", {"ID", "EMBEDDING", "TEXT", "METADATA"}
    )
    assert native_selection(conn, "DOCS", "auto")[0] == "sql"
    assert cursor.execute.call_count == 1
    with pytest.raises(ValueError, match="ordinary OracleVS"):
        native_selection(conn, "DOCS", "vecdb")


def test_auto_missing_package_uses_sql_but_explicit_vecdb_fails():
    conn, _ = selection_connection("23.26.3", [], False)
    assert native_selection(conn, "DOCS", "auto")[0] == "sql"
    with pytest.raises(ValueError, match="unavailable"):
        native_selection(conn, "DOCS", "vecdb")


def test_explicit_sql_does_not_probe_capabilities():
    conn, cursor = selection_connection("23.26.3", [])
    assert native_selection(conn, "DOCS", "sql")[0] == "sql"
    cursor.execute.assert_not_called()


def test_current_table_rejected_on_legacy_database():
    conn, _ = selection_connection("23.26.1", CURRENT_COLUMNS)
    with pytest.raises(ValueError, match="legacy database"):
        native_selection(conn, "DOCS", "auto")


def test_rest_store_roundtrip_mapping_without_metadata_mutation(store, sdk):
    metadata = {"tenant": "one"}
    assert store.backend == "vecdb"
    assert store.add_texts(["hello"], [metadata], ids=["a"]) == ["a"]
    assert metadata == {"tenant": "one"}
    records = sdk.upsert_vectors.call_args.kwargs["vectors"]
    assert records == [
        {
            "id": "a",
            "dense_vector": [1.0, 0.0],
            "metadata": {"tenant": "one", "text": "hello"},
        }
    ]
    hits = store.similarity_search_with_score(
        "query", k=3, filter={"tenant": {"$eq": "one"}}
    )
    assert hits[0][0] == Document(
        id="a", page_content="hello", metadata={"tenant": "one"}
    )
    assert hits[0][1] == 0.25  # lower-is-better distance, same as SQL OracleVS
    assert sdk.query.call_args.kwargs["filters"] == {"tenant": {"$eq": "one"}}
    assert sdk.query.call_args.kwargs["advanced_options"] == {
        "distance_metric": "EUCLIDEAN"
    }


def test_reserved_text_key_and_bad_batch_do_not_write(store, sdk):
    with pytest.raises(ValueError, match="reserved"):
        store.add_texts(["hello"], [{"text": "user value"}])
    with pytest.raises(ValueError, match="batch sizes"):
        store.add_texts(["hello"], ids=[])
    sdk.upsert_vectors.assert_not_called()


def test_empty_batch_does_not_write(store, sdk):
    assert store.add_texts([]) == []
    sdk.upsert_vectors.assert_not_called()


def test_insert_only_rest_requires_native_instead_of_racy_upsert(sdk):
    store = OracleVS(sdk, Embedding(), "DOCS")
    with pytest.raises(ValueError, match="REST SDK does not support"):
        store.add_texts(["hello"], ids=["a"])
    sdk.upsert_vectors.assert_not_called()


def test_get_pages_id_batches_and_delete(store, sdk):
    sdk.list_vectors.return_value = {
        "items": [{"id": "a", "metadata": {"text": "hello"}}]
    }
    assert store.get_by_ids([str(i) for i in range(205)])[0].page_content == "hello"
    assert [call.kwargs["limit"] for call in sdk.list_vectors.call_args_list] == [
        100,
        100,
        5,
    ]
    store.delete(["a"])
    sdk.delete_vectors.assert_called_once_with(table_name="DOCS", ids=["a"])
    with pytest.raises(ValueError):
        store.delete([])


def test_return_embeddings(store):
    hit = store.similarity_search_by_vector_returning_embeddings([1.0, 0.0])[0]
    assert hit[0].id == "a"
    assert hit[2].tolist() == [1.0, 0.0]


@pytest.mark.asyncio
async def test_rest_async_paths(store, sdk):
    assert await store.aadd_texts(["hello"], ids=["a"]) == ["a"]
    assert (await store.asimilarity_search("q"))[0].id == "a"
    assert (await store.asimilarity_search_by_vector_returning_embeddings([1.0, 0.0]))[
        0
    ][2].tolist() == [1.0, 0.0]
    await store.adelete(["a"])
    sdk.delete_vectors.assert_called_once()


def test_only_404_causes_rest_creation(sdk):
    for status in [401, 403, 500]:
        error = RuntimeError("request failed")
        error.status = status
        sdk.describe_vector_table.side_effect = error
        with pytest.raises(RuntimeError):
            OracleVS(sdk, Embedding(), "DOCS")
    sdk.create_vector_table.assert_not_called()
    error = RuntimeError("not found")
    error.status = 404
    sdk.describe_vector_table.side_effect = error
    OracleVS(sdk, Embedding(), "DOCS")
    sdk.create_vector_table.assert_called_once_with(
        name="DOCS", table_params={"auto_generate_id": False}
    )


def test_migration_is_error_triggered_and_retry_is_bounded():
    transport = MagicMock()
    mismatch = RuntimeError('ORA-00904: "CONTENT_METADATA": invalid identifier')
    transport.call.side_effect = [mismatch, {"results": []}]
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert backend.search([1.0], 2, None) == []
    transport.migrate.assert_called_once_with("DOCS")
    assert transport.call.call_count == 2
    transport.reset_mock()
    transport.call.side_effect = [mismatch, mismatch]
    with pytest.raises(RuntimeError):
        backend.search([1.0], 2, None)
    assert transport.call.call_count == 2
    transport.migrate.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("HTTP 401"),
        RuntimeError("ORA-00904: tenant"),
        RuntimeError("ORA-01031: insufficient privileges"),
    ],
)
def test_unrelated_errors_never_migrate(error):
    transport = MagicMock()
    transport.call.side_effect = error
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(RuntimeError):
        backend.search([1.0], 2, None)
    transport.migrate.assert_not_called()


def test_migration_steps_are_resumable_and_reject_ambiguous_layout():
    assert len(_migration_actions({"ID", "DENSE_VECTOR", "METADATA"}, '"DOCS"')) == 4
    assert (
        len(
            _migration_actions(
                {"ID", "DENSE_VECTOR", "CONTENT_METADATA", "CONTENT"}, '"DOCS"'
            )
        )
        == 2
    )
    assert _migration_actions(CURRENT_COLUMNS, '"DOCS"') == []
    with pytest.raises(ValueError):
        _migration_actions(CURRENT_COLUMNS | {"METADATA"}, '"DOCS"')
    with pytest.raises(ValueError):
        _layout({"ID", "TEXT"})


def test_pending_transaction_prevents_migration():
    connection = SimpleNamespace(transaction_in_progress=True, version="23.26.3")

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(RuntimeError, match="pending work"):
            NativeTransport(object(), (23, 26, 3)).migrate("DOCS")


def test_native_package_signatures():
    legacy = NativeTransport(object(), (23, 26, 1))
    current = NativeTransport(object(), (23, 26, 3))
    old_sql, old_binds = legacy._statement(
        "create_vector_table",
        {"name": "DOCS", "table_params": {"auto_generate_id": False}},
    )
    new_sql, new_binds = current._statement(
        "create_vector_table",
        {"name": "DOCS", "table_params": {"auto_generate_id": False}},
    )
    assert "table_name => :table_name" in old_sql
    assert old_binds["auto_generate_id"] is False
    assert "name => :name" in new_sql
    assert "table_params => JSON(:table_params)" in new_sql
    assert json.loads(new_binds["table_params"]) == {"auto_generate_id": False}
    assert "table_params" not in old_binds


def test_generation_specific_index_json():
    old = VecDBBackend(
        NativeTransport(None, (23, 26, 1)), "DOCS", DistanceStrategy.COSINE
    )
    new = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    assert old.index_params({})["indexing"] == "auto"
    assert new.index_params({})["vector_index_params"]["auto_index"] is True
    assert (
        new.index_params({"efConstruction": 20})["vector_index_params"][
            "advanced_params"
        ]["efConstruction"]
        == 20
    )
    with pytest.raises(ValueError):
        new.index_params({"accuracy": 101})


def test_filter_rejects_operators_paths_and_nonfinite_values():
    for value in [
        {"$sql": "x"},
        {"a.b": 3},
        {"text": "x"},
        {"a": {"$in": []}},
        {"a": float("nan")},
    ]:
        with pytest.raises(ValueError):
            validate_filter(value)
    assert validate_filter({"$and": [{"tenant": "one"}, {"size": {"$gt": 2}}]})


def test_text_retriever_uses_metadata_backend(store):
    store._vecdb.text_search = MagicMock(
        return_value=[Document(id="a", page_content="hello")]
    )
    retriever = OracleTextSearchRetriever(vector_store=store)
    assert retriever.invoke("hello")[0].id == "a"
    store._vecdb.text_search.assert_called_once_with('"hello"', 4)


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_hybrid_retriever_calls_native_search_without_application_fusion(store, mode):
    store._vecdb.transport = NativeTransport(object(), (23, 26, 4))
    expected = [
        Document(id="native", page_content="native result", metadata={"score": 7})
    ]
    store._vecdb.hybrid_search = MagicMock(return_value=expected)
    store._vecdb.text_search = MagicMock(
        side_effect=AssertionError("separate text query")
    )
    store.similarity_search = MagicMock(
        side_effect=AssertionError("separate vector query")
    )
    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name="HVI", search_mode=mode, return_scores=True
    )
    assert retriever.invoke("hello") == expected
    params, scores = store._vecdb.hybrid_search.call_args.args
    assert scores is True
    assert params["hybrid_index_name"] == '"HVI"'
    assert ("text" in params) == (mode != "semantic")
    assert ("vector" in params) == (mode != "keyword")
    if mode != "semantic":
        assert params["text"] == {"json_textcontains": ["$.text", '"hello"']}
    if mode != "keyword":
        assert params["vector"] == {"search_vector": [1.0, 0.0]}


@pytest.mark.asyncio
async def test_async_hybrid_uses_native_search_and_async_query_embedding(store):
    store._vecdb.transport = NativeTransport(object(), (23, 26, 4))
    store.embedding_function.aembed_query = AsyncMock(return_value=[0.0, 1.0])
    store.embedding_function.embed_query = MagicMock(
        side_effect=AssertionError("sync embed")
    )
    expected = [Document(id="native", page_content="hello")]
    store._vecdb.run_async = AsyncMock(return_value=expected)
    retriever = OracleHybridSearchRetriever(vector_store=store, idx_name="HVI")
    assert await retriever.ainvoke("hello") == expected
    method, params, scores = store._vecdb.run_async.call_args.args
    assert method == "hybrid_search"
    assert params["vector"] == {"search_vector": [0.0, 1.0]}
    store.embedding_function.aembed_query.assert_awaited_once_with("hello")


def test_native_hybrid_transport_binds_query_and_maps_database_scores():
    transport = NativeTransport(object(), (23, 26, 4))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    original = {"text": "hello", "tenant": "one"}
    cursor.fetchall.return_value = [("a", original, 9, 2, 7)]
    cursor.fetchone.return_value = ("HVI", "SIMPLIFIED_JSON_HYBRID")
    params = {
        "hybrid_index_name": "HVI",
        "text": {"json_textcontains": ["$.text", "x' OR 1=1"]},
    }

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        docs = transport.hybrid_search("DOCS", params, True)
    assert docs[0].id == "a"
    assert docs[0].page_content == "hello"
    assert docs[0].metadata == {
        "tenant": "one",
        "score": 9,
        "text_score": 2,
        "vector_score": 7,
    }
    assert original == {"text": "hello", "tenant": "one"}
    statement = cursor.execute.call_args.args[0]
    assert "DBMS_HYBRID_VECTOR.SEARCH(JSON(:search_params))" in statement
    assert "ORDER BY h.position" in statement
    assert "x' OR 1=1" not in statement
    assert json.loads(cursor.execute.call_args.kwargs["search_params"]) == params


def test_rest_hybrid_reports_unsupported_operation(store):
    with pytest.raises(ValueError, match="REST SDK does not support"):
        OracleHybridSearchRetriever(vector_store=store, idx_name="HVI").invoke("hello")


@pytest.mark.parametrize("params", [{"rrf_k": 60}, {"fetch_k": 20}])
def test_hybrid_rejects_previous_application_fusion_parameters(store, params):
    store._vecdb.transport = NativeTransport(object(), (23, 26, 4))
    with pytest.raises(ValueError, match="DBMS_HYBRID_VECTOR"):
        OracleHybridSearchRetriever(
            vector_store=store, idx_name="HVI", params=params
        ).invoke("hello")


@pytest.mark.asyncio
async def test_async_native_query_retries_once():
    transport = AsyncNativeTransport(None, (23, 26, 3))
    transport.acall = AsyncMock(
        side_effect=[RuntimeError('ORA-00904: "CONTENT_METADATA"'), {"results": []}]
    )
    transport.amigrate = AsyncMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert await backend.run_async("search", [1.0], 3, None) == []
    transport.amigrate.assert_awaited_once_with("DOCS")
    assert transport.acall.await_count == 2


def test_native_text_index_replaces_guide_preserving_settings():
    transport = NativeTransport(object(), (23, 26, 3))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    connection.transaction_in_progress = False
    cursor = connection.cursor.return_value.__enter__.return_value
    original = (
        "SIMPLIFIED_JSON dataguide on change "
        "sys.dbms_vector_database.handle_mvi_index search_on none "
        'sync (every "freq=secondly; interval=30" )'
    )
    cursor.fetchall.return_value = [("VEC_METADATA_IDX", original)]

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        transport.create_text_index("DOCS", "REQUESTED_NAME")
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert statements[1] == 'DROP INDEX "VEC_METADATA_IDX"'
    assert 'CREATE SEARCH INDEX "VEC_METADATA_IDX"' in statements[2]
    assert "SEARCH_ON TEXT" in statements[2]
    assert "search_on none" not in statements[2]
    assert 'sync (every "freq=secondly; interval=30" )' in statements[2]
    assert "sys.dbms_vector_database.handle_mvi_index" in statements[2]
    assert "MEMORY" not in statements[2] and "ON COMMIT" not in statements[2]


def test_native_text_index_reuses_already_enabled_index():
    transport = NativeTransport(object(), (23, 26, 3))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [("VEC_METADATA_IDX", "search_on text_value")]

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        transport.create_text_index("DOCS", "REQUESTED_NAME")
    assert cursor.execute.call_count == 1


def test_native_insert_only_skips_duplicates_atomically():
    transport = NativeTransport(object(), (23, 26, 3))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.getbatcherrors.return_value = [SimpleNamespace(code=1, offset=0)]
    records = [
        {"id": id_, "dense_vector": [1.0], "metadata": {"text": "x"}}
        for id_ in ["a", "b"]
    ]

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        assert transport.insert("DOCS", records) == ["b"]
    assert "INSERT INTO" in cursor.executemany.call_args.args[0]
    assert "CONTENT_METADATA" in cursor.executemany.call_args.args[0]
    assert cursor.executemany.call_args.kwargs["batcherrors"] is True
    connection.commit.assert_called_once()


@pytest.mark.asyncio
async def test_native_async_crud_uses_async_driver_not_thread_wrapping():
    transport = AsyncNativeTransport(None, (23, 26, 3))
    transport.acall = AsyncMock(return_value={"items": []})
    transport.ainsert = AsyncMock(return_value=["a"])
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert await backend.run_async("add", ["x"], [[1.0]], [{}], ["a"], False) == ["a"]
    transport.ainsert.assert_awaited_once()
    assert await backend.run_async("get", ["a"]) == []
    await backend.run_async("delete", ["a"])
    assert transport.acall.await_count == 2


def test_migration_applies_missing_actions_without_table_drop():
    import re

    from langchain_oracledb.vectorstores import _vecdb

    columns = {"ID", "DENSE_VECTOR", "CONTENT_METADATA", "CONTENT"}
    connection = MagicMock(transaction_in_progress=False, version="23.26.3")
    cursor = connection.cursor.return_value.__enter__.return_value

    def execute(statement, **binds):
        if " ADD " in statement:
            columns.add(re.search(r" ADD (\w+)", statement).group(1))

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = lambda: [(column,) for column in columns]

    @contextmanager
    def get(_):
        yield connection

    with patch.object(_vecdb, "_get_connection", get):
        NativeTransport(None, (23, 26, 3)).migrate("DOCS")
    assert columns == CURRENT_COLUMNS
    actions = [
        call.args[0]
        for call in cursor.execute.call_args_list
        if "ALTER TABLE" in call.args[0]
    ]
    assert len(actions) == 2
    assert not any("RENAME" in action or "DROP" in action for action in actions)


def test_from_texts_propagates_backend_ids_and_mutation_mode(sdk):
    store = OracleVS.from_texts(
        ["hello"],
        Embedding(),
        client=sdk,
        table_name="DOCS",
        backend="vecdb",
        mutate_on_duplicate=True,
        ids=["custom"],
    )
    assert store.backend == "vecdb"
    assert sdk.upsert_vectors.call_args.kwargs["vectors"][0]["id"] == "custom"


@pytest.mark.parametrize(
    "value", [{1: "x"}, {"a": {"$eq": {"$sql": "x"}}}, {"a": {"$in": [{"$eq": 1}]}}]
)
def test_filter_rejects_structured_comparison_values(value):
    with pytest.raises(ValueError):
        validate_filter(value)


def test_layout_error_checks_actual_missing_column_not_request_context():
    assert layout_error(
        RuntimeError('ORA-00904: "T"."CONTENT_METADATA": invalid identifier')
    )
    assert not layout_error(
        RuntimeError(
            "Request table_name CONTENT_METADATA; "
            'ORA-00904: "TENANT": invalid identifier'
        )
    )
    assert not layout_error(
        RuntimeError('ORA-00904: "CONTENT_METADATA_OTHER": invalid identifier')
    )


def test_native_pool_initialization_does_not_acquire_a_second_connection():
    from langchain_oracledb.vectorstores import _vecdb, oraclevs

    pool = object()
    connection = MagicMock(version="23.26.3", thin=True)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (1,)
    cursor.var.return_value.getvalue.return_value = '{"table_name":"DOCS"}'
    acquired = False

    @contextmanager
    def get(client):
        nonlocal acquired
        if client is pool:
            assert not acquired, "Nested pool acquisition would deadlock max=1 pools"
            acquired = True
            try:
                yield connection
            finally:
                acquired = False
        else:
            assert client is connection
            yield connection

    with (
        patch.object(oraclevs, "_get_connection", get),
        patch.object(_vecdb, "_get_connection", get),
    ):
        store = OracleVS(pool, Embedding(), "DOCS")
    assert store.backend == "vecdb"
    assert store._vecdb.transport.client is pool


def test_migration_refreshes_version_after_pool_database_upgrade():
    from langchain_oracledb.vectorstores import _vecdb

    connection = MagicMock(version="23.26.4", transaction_in_progress=False)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(column,) for column in CURRENT_COLUMNS]

    @contextmanager
    def get(_):
        yield connection

    transport = NativeTransport(None, (23, 26, 1))
    with patch.object(_vecdb, "_get_connection", get):
        transport.migrate("DOCS")
    assert transport.version == (23, 26, 4)


@pytest.mark.asyncio
async def test_async_native_hybrid_transport_returns_database_scores():
    from contextlib import asynccontextmanager

    transport = AsyncNativeTransport(object(), (23, 26, 4))
    transport.acolumns = AsyncMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=("HVI", "SIMPLIFIED_JSON_HYBRID"))
    cursor.fetchall = AsyncMock(
        return_value=[("a", {"text": "hello", "tenant": "one"}, 9, 2, 7)]
    )

    @asynccontextmanager
    async def get(_):
        yield connection

    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    params = {"hybrid_index_name": "HVI", "vector": {"search_vector": [1.0, 0.0]}}
    with patch("langchain_oracledb.vectorstores._vecdb._aget_connection", get):
        docs = await backend.run_async("hybrid_search", params, False)
    assert docs == [Document(id="a", page_content="hello", metadata={"tenant": "one"})]
    assert json.loads(cursor.execute.call_args.kwargs["search_params"]) == params
    assert "DBMS_HYBRID_VECTOR.SEARCH" in cursor.execute.call_args.args[0]


@pytest.mark.parametrize(
    "version, column", [((23, 26, 1), "METADATA"), ((23, 26, 4), "CONTENT_METADATA")]
)
def test_vecdb_hybrid_ddl_uses_metadata_column(store, version, column):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        _get_hybrid_index_ddl,
    )

    store._vecdb.transport = NativeTransport(object(), version)
    pref = OracleVectorizerPreference.__new__(OracleVectorizerPreference)
    pref.vs = store
    pref.preference_name = "MY_PREF"
    ddl = _get_hybrid_index_ddl(pref, '"HVI"', {})
    assert f'"DOCS"({column})' in ddl
    assert "CREATE HYBRID VECTOR INDEX" in ddl


@pytest.mark.parametrize(
    "operation", ["text", "hybrid", "text_index", "hybrid_index", "preference"]
)
def test_rest_unsupported_operations_fail_without_native_fallback(store, operation):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        create_hybrid_index,
    )
    from langchain_oracledb.retrievers.text_search import create_text_index

    store._embed_query = MagicMock(side_effect=AssertionError("must not embed"))
    operations = {
        "text": lambda: OracleTextSearchRetriever(vector_store=store).invoke("hello"),
        "hybrid": lambda: OracleHybridSearchRetriever(
            vector_store=store, idx_name="IDX"
        ).invoke("hello"),
        "text_index": lambda: create_text_index(
            store.client, "IDX", vector_store=store
        ),
        "hybrid_index": lambda: create_hybrid_index(
            store.client, "IDX", vector_store=store
        ),
        "preference": lambda: OracleVectorizerPreference.create_preference(store),
    }
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection") as connect:
        with pytest.raises(
            ValueError, match="oracle-vecdb REST SDK does not support"
        ) as error:
            operations[operation]()
    assert "native_client" not in str(error.value)
    connect.assert_not_called()
    store._embed_query.assert_not_called()
    store.client.query.assert_not_called()
    store.client.upsert_vectors.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["text", "hybrid", "text_index", "hybrid_index", "preference"]
)
async def test_async_rest_unsupported_operations_fail_without_native_fallback(
    store, operation
):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        acreate_hybrid_index,
    )
    from langchain_oracledb.retrievers.text_search import acreate_text_index

    store._aembed_query = AsyncMock(side_effect=AssertionError("must not embed"))
    operations = {
        "text": lambda: OracleTextSearchRetriever(vector_store=store).ainvoke("hello"),
        "hybrid": lambda: OracleHybridSearchRetriever(
            vector_store=store, idx_name="IDX"
        ).ainvoke("hello"),
        "text_index": lambda: acreate_text_index(
            store.client, "IDX", vector_store=store
        ),
        "hybrid_index": lambda: acreate_hybrid_index(
            store.client, "IDX", vector_store=store
        ),
        "preference": lambda: OracleVectorizerPreference.acreate_preference(store),
    }
    with patch("langchain_oracledb.vectorstores._vecdb._aget_connection") as connect:
        with pytest.raises(
            ValueError, match="oracle-vecdb REST SDK does not support"
        ) as error:
            await operations[operation]()
    assert "native_client" not in str(error.value)
    connect.assert_not_called()
    store._aembed_query.assert_not_called()
    store.client.query.assert_not_called()
    store.client.upsert_vectors.assert_not_called()


def test_rest_layout_error_reports_unsupported_migration_without_retry(store):
    store.client.query.side_effect = RuntimeError(
        'ORA-00904: "CONTENT_METADATA": invalid identifier'
    )
    with pytest.raises(
        RuntimeError, match="REST SDK does not support schema migration"
    ):
        store.similarity_search("hello")
    assert store.client.query.call_count == 1


@pytest.mark.asyncio
async def test_async_rest_layout_error_reports_unsupported_migration_without_retry(
    store,
):
    store.client.query.side_effect = RuntimeError(
        'ORA-00904: "CONTENT_METADATA": invalid identifier'
    )
    with pytest.raises(
        RuntimeError, match="REST SDK does not support schema migration"
    ):
        await store.asimilarity_search("hello")
    assert store.client.query.call_count == 1


def test_public_store_factories_have_no_native_companion_parameter():
    import inspect

    assert "native_client" not in inspect.signature(OracleVS).parameters
    assert "native_client" not in inspect.signature(OracleVS.acreate).parameters
