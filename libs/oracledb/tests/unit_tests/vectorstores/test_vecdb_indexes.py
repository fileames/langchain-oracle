# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Metadata index replacement, recovery, and sync/async public entry points."""

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_oracledb.retrievers.hybrid_search import (
    OracleVectorizerPreference,
    _vecdb_hybrid_builder,
    acreate_hybrid_index,
    create_hybrid_index,
)
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
    _metadata_index_plan,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy

SCHEDULE = 'sync (every "freq=secondly; interval=30" )'
GUIDE = (
    "SIMPLIFIED_JSON dataguide on change "
    "sys.dbms_vector_database.handle_mvi_index search_on none " + SCHEDULE
)
TEXT = GUIDE.replace("search_on none", "search_on text")
HVI = (
    "SIMPLIFIED_JSON_HYBRID vectorizer PREF dataguide on change "
    "sys.dbms_vector_database.handle_mvi_index " + SCHEDULE
)


def preference(transport):
    pref = OracleVectorizerPreference.__new__(OracleVectorizerPreference)
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    pref.vs = SimpleNamespace(_vecdb=backend, table_name='"DOCS"')
    pref.preference_name = "PREF"
    return pref


@pytest.mark.parametrize(
    "columns", [CURRENT_COLUMNS, {"ID", "DENSE_VECTOR", "METADATA"}]
)
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("original", [GUIDE, TEXT])
def test_plan_preserves_physical_name_callback_and_schedule(columns, hybrid, original):
    pref = preference(NativeTransport(None, (23, 26, 4)))
    physical, ddl, restore = _metadata_index_plan(
        "DOCS",
        "REQUESTED_NAME",
        columns,
        ("VECIDX$_1_SEARCH_IDX", original),
        hybrid,
        _vecdb_hybrid_builder(pref, None),
    )
    assert physical == '"VECIDX$_1_SEARCH_IDX"'
    if not hybrid and original == TEXT:
        assert ddl is restore is None
        return
    assert physical in ddl
    assert "REQUESTED_NAME" not in ddl
    assert SCHEDULE in ddl
    assert "sys.dbms_vector_database.handle_mvi_index" in ddl
    assert "MEMORY" not in ddl and "ON COMMIT" not in ddl
    assert "SIMPLIFIED_JSON" not in ddl
    assert ("CREATE HYBRID VECTOR INDEX" in ddl) is hybrid
    assert ("SEARCH_ON TEXT" in ddl) is (not hybrid)
    assert "search_on none" not in ddl
    assert original.removeprefix("SIMPLIFIED_JSON ") in restore
    meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
    assert meta in ddl


@pytest.mark.parametrize("hybrid", [False, True])
def test_existing_hybrid_is_never_downgraded_or_rebuilt(hybrid):
    assert _metadata_index_plan(
        "DOCS", "REQUESTED", CURRENT_COLUMNS, ("ACTUAL", HVI), hybrid, None
    ) == ('"ACTUAL"', None, None)


@pytest.mark.parametrize("hybrid", [False, True])
def test_missing_index_uses_requested_name_and_vecdb_callback(hybrid):
    pref = preference(NativeTransport(None, (23, 26, 4)))
    physical, ddl, restore = _metadata_index_plan(
        "DOCS",
        "REQUESTED",
        CURRENT_COLUMNS,
        None,
        hybrid,
        _vecdb_hybrid_builder(pref, None),
    )
    assert physical == '"REQUESTED"'
    assert "SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX" in ddl
    assert "interval=30" in ddl
    assert restore is None


@pytest.mark.parametrize(
    "parameters",
    [
        "custom index configuration",
        GUIDE.replace("sys.dbms_vector_database.handle_mvi_index", "MY_CALLBACK"),
        GUIDE.replace(
            "dataguide on change sys.dbms_vector_database.handle_mvi_index",
            "dataguide off",
        ),
    ],
)
def test_unknown_index_configuration_is_rejected_before_replacement(parameters):
    with pytest.raises(ValueError):
        _metadata_index_plan(
            "DOCS", "IDX", CURRENT_COLUMNS, ("OLD", parameters), True, lambda *_: "DDL"
        )


def test_sync_override_is_rejected_before_replacement():
    pref = preference(NativeTransport(None, (23, 26, 4)))
    build = _vecdb_hybrid_builder(pref, {"parameters": {"sync": "(ON COMMIT)"}})
    with pytest.raises(ValueError, match="do not override sync"):
        _metadata_index_plan(
            "DOCS", "IDX", CURRENT_COLUMNS, ("OLD", GUIDE), True, build
        )


@pytest.mark.parametrize(
    "options",
    [
        {"parameters": {"dataguide": "OFF"}},
        {"parameters": {"sync": "(ON COMMIT)"}},
        {"parameters": {"search_on": "NONE"}},
        {"parallel": -1},
        {"order_by": ["x; DROP TABLE DOCS"]},
    ],
)
def test_invalid_hybrid_options_fail_before_drop(options):
    transport, connection, cursor, get = sync_transport()
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(ValueError):
            create_hybrid_index(
                connection,
                "REQUESTED",
                vectorizer_preference=preference(transport),
                params=options,
            )
    assert not any("DROP" in c.args[0] for c in cursor.execute.call_args_list)


def sync_transport(parameters=GUIDE):
    transport = NativeTransport(object(), (23, 26, 4))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    connection.transaction_in_progress = False
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [("OLD", parameters)]
    cursor.fetchone.return_value = None

    @contextmanager
    def get(_):
        yield connection

    return transport, connection, cursor, get


def test_public_hybrid_helper_upgrades_same_name_text_index():
    transport, connection, cursor, get = sync_transport(TEXT)
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        create_hybrid_index(
            connection, "OLD", vectorizer_preference=preference(transport)
        )
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert statements[1] == 'DROP INDEX "OLD"'
    assert 'CREATE HYBRID VECTOR INDEX "OLD"' in statements[2]
    assert SCHEDULE in statements[2]


@pytest.mark.parametrize("partial_index", [False, True])
def test_failed_replacement_restores_original_and_raises_original_error(partial_index):
    transport, _, cursor, get = sync_transport()
    cursor.fetchone.return_value = ("OLD",) if partial_index else None
    original_error = RuntimeError("invalid hybrid preference")

    def execute(sql, **kwargs):
        if sql.startswith("CREATE HYBRID"):
            raise original_error

    cursor.execute.side_effect = execute
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(RuntimeError) as caught:
            transport.create_metadata_index(
                "DOCS", "NEW", hybrid=True, build=lambda *_: "CREATE HYBRID INVALID"
            )
    assert caught.value is original_error
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert statements[-1].startswith('CREATE SEARCH INDEX "OLD"')
    assert "search_on none" in statements[-1] and SCHEDULE in statements[-1]
    assert ('DROP INDEX "OLD" FORCE' in statements) is partial_index


def test_recovery_failure_explicitly_requires_manual_recovery():
    transport, _, cursor, get = sync_transport()

    def execute(sql, **kwargs):
        if sql.startswith("CREATE"):
            raise RuntimeError("database creation failed")

    cursor.execute.side_effect = execute
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(RuntimeError, match="manual recovery"):
            transport.create_text_index("DOCS", "NEW")


def test_pending_transaction_is_not_implicitly_committed():
    transport, connection, cursor, get = sync_transport()
    connection.transaction_in_progress = True
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(RuntimeError, match="pending work"):
            transport.create_text_index("DOCS", "NEW")
    assert len(cursor.execute.call_args_list) == 1


def test_requested_name_collision_is_checked_without_dropping():
    transport, _, cursor, get = sync_transport()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = ("OTHER_TABLE",)
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(ValueError, match="incompatible"):
            transport.create_text_index("DOCS", "TAKEN")
    assert not any("DROP" in c.args[0] for c in cursor.execute.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_async_native_replacement_and_recovery(failure):
    transport = AsyncNativeTransport(object(), (23, 26, 4))
    transport.acolumns = AsyncMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    connection.transaction_in_progress = False
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall = AsyncMock(return_value=[("OLD", TEXT)])
    cursor.fetchone = AsyncMock(return_value=None)

    async def execute(sql, **kwargs):
        if failure and sql.strip().startswith("CREATE HYBRID"):
            raise RuntimeError("failed creation")

    cursor.execute = AsyncMock(side_effect=execute)

    @asynccontextmanager
    async def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._aget_connection", get):
        if failure:
            with pytest.raises(RuntimeError, match="failed creation"):
                await acreate_hybrid_index(
                    connection, "NEW", vectorizer_preference=preference(transport)
                )
        else:
            await acreate_hybrid_index(
                connection, "NEW", vectorizer_preference=preference(transport)
            )
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert 'DROP INDEX "OLD"' in statements
    assert any('CREATE HYBRID VECTOR INDEX "OLD"' in q for q in statements)
    if failure:
        assert statements[-1].startswith('CREATE SEARCH INDEX "OLD"')
        assert "search_on text" in statements[-1]


@pytest.mark.parametrize("columns", [set(), {"ID", "EMBEDDING", "METADATA", "TEXT"}])
def test_non_vecdb_layout_is_rejected_before_index_ddl(columns):
    with pytest.raises(ValueError, match="recognized VecDB table"):
        _metadata_index_plan("DOCS", "IDX", columns, ("OLD", GUIDE), False, None)


def test_inherited_on_commit_schedule_is_rejected_before_drop():
    transport, connection, cursor, get = sync_transport(
        GUIDE.replace(SCHEDULE, "SYNC (ON COMMIT)")
    )
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(ValueError, match="cannot inherit SYNC"):
            create_hybrid_index(
                connection, "REQUESTED", vectorizer_preference=preference(transport)
            )
    assert not any("DROP" in c.args[0] for c in cursor.execute.call_args_list)
