# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Private VecDB transport, layout compatibility, and document mapping.

REST uses the optional public SDK; native calls use DBMS_VECTOR_DATABASE.
There are no SDK imports on the SQL-only import path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any, Callable, Optional

import numpy as np
import oracledb
from langchain_core.documents import Document

from .utils import (
    DistanceStrategy,
    _aget_connection,
    _get_connection,
    _quote_indentifier,
)

logger = logging.getLogger(__name__)

TEXT_KEY = "text"
LEGACY_VERSION = (23, 26, 1)
CURRENT_VERSION = (23, 26, 3)
DENSE_COMPATIBLE = (23, 4, 0)
SPARSE_COMPATIBLE = (23, 6, 0)
COMPATIBILITY_SQL = "BEGIN DBMS_UTILITY.DB_VERSION(:db_version, :compatibility); END;"
BASE_COLUMNS = {"ID", "DENSE_VECTOR"}
CURRENT_COLUMNS = BASE_COLUMNS | {
    "CONTENT_METADATA",
    "SPARSE_VECTOR",
    "CONTENT",
    "CONTENT_TYPE",
}


def parse_version(version: str) -> tuple[int, ...]:
    matches = re.findall(r"(?<!\d)(\d+)\.(\d+)\.(\d+)", version)
    if not matches:
        raise ValueError("Cannot determine the Oracle Database version")
    return tuple(map(int, matches[-1]))


def table_identifier(name: str) -> tuple[str, str]:
    """Use the shared identifier handling; return SQL and API forms."""
    quoted = _quote_indentifier(name)
    return quoted, quoted.replace('"', "")


def as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "read"):
        response = response.read()
    if isinstance(response, str):
        response = json.loads(response)
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    elif hasattr(response, "model_dump"):
        response = response.model_dump()
    if not isinstance(response, dict):
        raise ValueError("Malformed VecDB response: expected a JSON object")
    return {key.lower(): value for key, value in response.items()}


def rows(response: Any, key: str) -> list[dict[str, Any]]:
    obj = as_dict(response)
    result = obj.get(key, obj.get("items"))
    if not isinstance(result, list):
        raise ValueError(f"Malformed VecDB response: missing {key} array")
    return [as_dict(item) for item in result]


def document(row: dict[str, Any]) -> Document:
    from .oraclevs import INTERNAL_ID_KEY

    metadata = dict(row.get("metadata") or {})
    text = metadata.pop(TEXT_KEY, None)
    if not isinstance(text, str) or not isinstance(row.get("id"), str):
        raise ValueError("VecDB document is missing its ID or metadata text")
    # The SQL backend's private ID field is not part of public metadata.
    metadata.pop(INTERNAL_ID_KEY, None)
    return Document(id=row["id"], page_content=text, metadata=metadata)


def _columns_sql() -> str:
    return "SELECT column_name FROM user_tab_columns WHERE table_name = :name"


def _metadata_index_sql() -> str:
    return (
        "SELECT i.index_name, i.parameters FROM user_indexes i "
        "JOIN user_ind_columns c ON c.index_name = i.index_name "
        "WHERE i.table_name = :name AND c.column_name = :column "
        "AND i.ityp_owner = 'CTXSYS' AND i.ityp_name IN ('CONTEXT', 'CONTEXT_V2')"
    )


def _text_enabled(parameters: Optional[str]) -> bool:
    if "SIMPLIFIED_JSON_HYBRID" in (parameters or "").upper():
        return True
    return bool(
        re.search(
            r"\bsearch_on\s+text(?:_value(?:_string)?)?\b",
            parameters or "",
            re.IGNORECASE,
        )
    )


DATAGUIDE_CALLBACK = "ON CHANGE SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX"
DEFAULT_METADATA_PARAMETERS = (
    f'DATAGUIDE {DATAGUIDE_CALLBACK} SYNC (EVERY "freq=secondly; interval=30")'
)


def _metadata_parameters(parameters: str, hybrid: bool) -> str:
    """Retain catalog settings, excluding the index-kind marker/search mode."""
    if not re.match(r"^SIMPLIFIED_JSON\b", parameters.strip(), re.IGNORECASE):
        raise ValueError("Only a recognized VecDB JSON metadata index can be replaced")
    parameters = re.sub(r"^\s*SIMPLIFIED_JSON\s*", "", parameters, flags=re.IGNORECASE)
    callback = re.search(
        r"\bDATAGUIDE\s+ON\s+CHANGE\s+([\w.$#]+)", parameters, re.IGNORECASE
    )
    if callback and callback[1].upper() != "SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX":
        raise ValueError("Cannot replace a metadata index with a different callback")
    if not callback:
        if re.search(r"\bDATAGUIDE\b", parameters, re.IGNORECASE):
            raise ValueError("Cannot replace an unrecognized data-guide configuration")
        parameters += f" DATAGUIDE {DATAGUIDE_CALLBACK}"
    parameters = re.sub(
        r"\bSEARCH_ON\s+(?:TEXT_VALUE_STRING|TEXT_VALUE|TEXT|VALUE|NONE)\b",
        "",
        parameters,
        flags=re.IGNORECASE,
    ).strip()
    if not hybrid:
        parameters += " SEARCH_ON TEXT"
    return parameters


def _search_index_ddl(index: str, table: str, meta: str, parameters: str) -> str:
    escaped = parameters.replace("'", "''")
    return (
        f"CREATE SEARCH INDEX {index} ON {table} ({meta}) FOR JSON "
        f"PARAMETERS ('{escaped}')"
    )


def _metadata_index_plan(
    name: str,
    index: str,
    columns: set[str],
    existing: Optional[tuple],
    hybrid: bool,
    build: Optional[Callable[[str, str, str], str]],
) -> tuple[str, Optional[str], Optional[str]]:
    """Return physical name, replacement DDL, and original-index recovery DDL."""
    quoted, _ = table_identifier(name)
    requested, _ = table_identifier(index)
    if _layout(columns) != "vecdb":
        raise ValueError("Metadata indexing requires a recognized VecDB table")
    meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
    physical = table_identifier(existing[0])[0] if existing else requested
    if existing and (
        "SIMPLIFIED_JSON_HYBRID" in (existing[1] or "").upper()
        or (not hybrid and _text_enabled(existing[1]))
    ):
        return physical, None, None
    parameters = (
        _metadata_parameters(existing[1] or "", hybrid)
        if existing
        else DEFAULT_METADATA_PARAMETERS + ("" if hybrid else " SEARCH_ON TEXT")
    )
    restore = None
    if existing:
        original = re.sub(
            r"^\s*SIMPLIFIED_JSON\s*", "", existing[1], flags=re.IGNORECASE
        )
        restore = _search_index_ddl(physical, quoted, meta, original)
    if hybrid:
        if build is None:
            raise ValueError("Hybrid indexing requires a vectorizer preference")
        ddl = build(physical, parameters, meta)
    else:
        ddl = _search_index_ddl(physical, quoted, meta, parameters)
    return physical, ddl, restore


def _layout(columns: set[str]) -> str:
    if not columns:
        return "missing"
    if {"ID", "EMBEDDING", "METADATA", "TEXT"} <= columns:
        return "sql"
    if not BASE_COLUMNS <= columns or {"METADATA", "CONTENT_METADATA"} <= columns:
        raise ValueError("Incompatible VecDB table layout")
    if "METADATA" not in columns and "CONTENT_METADATA" not in columns:
        raise ValueError("VecDB table has no recognized metadata column")
    return "vecdb"


def _migration_actions(columns: set[str], quoted: str) -> list[str]:
    if _layout(columns) != "vecdb":
        raise ValueError("Only recognized VecDB layouts can be migrated")
    actions = []
    if "METADATA" in columns:
        actions.append(
            f"ALTER TABLE {quoted} RENAME COLUMN METADATA TO CONTENT_METADATA"
        )
    for name, sql_type in (
        ("SPARSE_VECTOR", "VECTOR(*, *, SPARSE)"),
        ("CONTENT", "BLOB"),
        ("CONTENT_TYPE", "VARCHAR2(256)"),
    ):
        if name not in columns:
            actions.append(f"ALTER TABLE {quoted} ADD {name} {sql_type}")
    return actions


def layout_error(error: Exception) -> bool:
    """Only a missing current-layout column is a candidate for repair.

    Physical inspection and database generation are checked before any DDL.
    ORA-00904 alone (e.g. a bad user identifier) never triggers migration.
    """
    message = str(error).upper().replace('\\"', '"')
    return bool(
        re.search(
            r'ORA-00904:\s*(?:(?:"[^"]+"|\w+)\.)*"?'
            r'(?:CONTENT_METADATA|SPARSE_VECTOR|CONTENT_TYPE|CONTENT)"?'
            r"(?=\s*(?::|$))",
            message,
        )
    )


def _hybrid_sql(name: str, columns: set[str]) -> str:
    quoted, _ = table_identifier(name)
    _layout(columns)
    meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
    return (
        f"SELECT t.ID, t.{meta}, h.score, h.text_score, h.vector_score "
        "FROM JSON_TABLE(DBMS_HYBRID_VECTOR.SEARCH(JSON(:search_params)), "
        "'$[*]' COLUMNS (position FOR ORDINALITY, "
        "rid VARCHAR2(128) PATH '$.rowid', score NUMBER PATH '$.score', "
        "text_score NUMBER PATH '$.text_score', "
        "vector_score NUMBER PATH '$.vector_score')) h "
        f"JOIN {quoted} t ON t.ROWID = CHARTOROWID(h.rid) ORDER BY h.position"
    )


def _hybrid_documents(result: list, scores: bool) -> list[Document]:
    docs = []
    for id_, metadata, score, text_score, vector_score in result:
        doc = document({"id": str(id_), "metadata": metadata})
        if scores:
            doc.metadata.update(
                score=score, text_score=text_score, vector_score=vector_score
            )
        docs.append(doc)
    return docs


class NativeTransport:
    """Execute a bounded set of package calls using caller-owned connections."""

    def __init__(self, client: Any, version: tuple[int, ...]):
        self.client = client
        self.version = version

    def columns(self, name: str) -> set[str]:
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_columns_sql(), name=name)
                return {row[0] for row in cursor.fetchall()}

    def _statement(self, operation: str, params: dict) -> tuple[str, dict]:
        # Function and parameter names are internal constants, not caller SQL.
        allowed = {
            "create_vector_table",
            "describe_vector_table",
            "drop_vector_table",
            "upsert_vectors",
            "list_vectors",
            "search",
            "delete_vectors",
            "create_index",
        }
        if operation not in allowed:
            raise ValueError("Unsupported VecDB operation")
        params = dict(params)
        if operation in {
            "describe_vector_table",
            "drop_vector_table",
            "create_vector_table",
        }:
            if self.version < CURRENT_VERSION:
                params["table_name"] = params.pop("name")
                if operation == "create_vector_table":
                    params.pop("table_params", None)
                    params.pop("embed_params", None)
                    params.update(
                        auto_generate_id=False,
                        vector_type="dense",
                        debug_flags=None,
                        request_id=None,
                    )
        arguments = []
        for key, value in params.items():
            if not re.fullmatch(r"[a-z_]+", key):
                raise ValueError("Invalid internal VecDB parameter")
            if key in {
                "vectors",
                "ids",
                "query_by",
                "filters",
                "advanced_options",
                "index_params",
                "table_params",
                "embed_params",
                "annotations",
                "debug_flags",
            }:
                params[key] = (
                    json.dumps(value, allow_nan=False) if value is not None else None
                )
                arguments.append(f"{key} => JSON(:{key})")
            else:
                arguments.append(f"{key} => :{key}")
        return (
            f"BEGIN :result := DBMS_VECTOR_DATABASE.{operation}("
            f"{', '.join(arguments)}); END;",
            params,
        )

    def call(self, operation: str, **params: Any) -> dict:
        statement, binds = self._statement(operation, params)
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                out = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.setinputsizes(
                    **{
                        key: oracledb.DB_TYPE_CLOB
                        for key in binds
                        if f"JSON(:{key})" in statement
                    }
                )
                cursor.execute(statement, result=out, **binds)
                return as_dict(out.getvalue())

    def migrate(self, name: str) -> None:
        quoted, name = table_identifier(name)
        with _get_connection(self.client) as connection:
            self.version = parse_version(connection.version)
            if self.version < CURRENT_VERSION:
                raise ValueError("Layout migration requires Oracle Database 23.26.3+")
            if connection.transaction_in_progress:
                raise RuntimeError(
                    "VecDB migration requires a connection without pending work"
                )
            with connection.cursor() as cursor:

                def inspect() -> set[str]:
                    cursor.execute(_columns_sql(), name=name)
                    return {row[0] for row in cursor.fetchall()}

                for action in _migration_actions(inspect(), quoted):
                    try:
                        cursor.execute(action)
                    except oracledb.DatabaseError as exc:
                        # Another process may have completed this exact DDL.
                        code = exc.args[0].code
                        if code not in {904, 1430} or action in _migration_actions(
                            inspect(), quoted
                        ):
                            raise
                if not CURRENT_COLUMNS <= inspect():
                    raise RuntimeError(
                        "VecDB migration did not produce the required layout"
                    )

    def create_metadata_index(
        self,
        name: str,
        index: str,
        *,
        hybrid: bool = False,
        build: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        _, name = table_identifier(name)
        _, requested = table_identifier(index)
        columns = self.columns(name)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_metadata_index_sql(), name=name, column=meta)
                found = cursor.fetchall()
                if len(found) > 1:
                    raise ValueError("Multiple metadata domain indexes found")
                existing = found[0] if found else None
                physical, ddl, restore = _metadata_index_plan(
                    name, index, columns, existing, hybrid, build
                )
                if ddl is None:
                    return
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                if not existing:
                    cursor.execute(
                        "SELECT table_name FROM user_indexes WHERE index_name = :name",
                        name=requested,
                    )
                    if cursor.fetchone():
                        raise ValueError(
                            "Index name is already used by an incompatible index"
                        )
                if restore:
                    cursor.execute(f"DROP INDEX {physical}")
                try:
                    cursor.execute(ddl)
                except Exception:
                    if restore:
                        try:
                            cursor.execute(
                                "SELECT index_name FROM user_indexes "
                                "WHERE index_name = :name",
                                name=table_identifier(physical)[1],
                            )
                            if cursor.fetchone():
                                cursor.execute(f"DROP INDEX {physical} FORCE")
                            cursor.execute(restore)
                            logger.warning(
                                "Restored metadata index %s after replacement failed",
                                physical,
                            )
                        except Exception as recovery:
                            raise RuntimeError(
                                f"Could not restore metadata index {physical} after "
                                "replacement failed; manual recovery is required"
                            ) from recovery
                    raise
                logger.info(
                    "Created %s metadata index %s",
                    "hybrid" if hybrid else "text",
                    physical,
                )

    def create_text_index(self, name: str, index: str) -> None:
        self.create_metadata_index(name, index)

    def insert(self, name: str, records: list[dict]) -> list[str]:
        import array

        quoted, name = table_identifier(name)
        columns = self.columns(name)
        _layout(columns)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        values = [
            (r["id"], array.array("f", r["dense_vector"]), r["metadata"])
            for r in records
        ]
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                cursor.executemany(
                    f"INSERT INTO {quoted} (ID, DENSE_VECTOR, {meta}) "
                    "VALUES (:1, :2, :3)",
                    values,
                    batcherrors=True,
                )
                errors = cursor.getbatcherrors() or []
                fatal = [error for error in errors if error.code != 1]
                if fatal:
                    raise RuntimeError(
                        f"VecDB insert failed with ORA-{fatal[0].code:05d}"
                    )
                connection.commit()
                failed = {error.offset for error in errors}
                return [
                    record["id"] for i, record in enumerate(records) if i not in failed
                ]

    def hybrid_search(self, name: str, params: dict, scores: bool) -> list:
        columns = self.columns(name)
        sql = _hybrid_sql(name, columns)
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    _metadata_index_sql(),
                    name=table_identifier(name)[1],
                    column="CONTENT_METADATA"
                    if "CONTENT_METADATA" in columns
                    else "METADATA",
                )
                existing = cursor.fetchone()
                if (
                    not existing
                    or "SIMPLIFIED_JSON_HYBRID" not in (existing[1] or "").upper()
                ):
                    raise ValueError(
                        "Create a hybrid vector index on the VecDB "
                        "metadata column first"
                    )
                params = {**params, "hybrid_index_name": existing[0]}
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_CLOB)
                cursor.execute(sql, search_params=json.dumps(params, allow_nan=False))
                return _hybrid_documents(cursor.fetchall(), scores)

    def text_search(self, name: str, query: str, k: int) -> list:
        quoted, name = table_identifier(name)
        columns = self.columns(name)
        _layout(columns)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        # JSON_TEXTCONTAINS limits matches to document text, not other metadata.
        sql = (
            f"SELECT id, {meta} FROM {quoted} "
            f"WHERE JSON_TEXTCONTAINS({meta}, '$.text', :query) "
            "ORDER BY id FETCH FIRST :k ROWS ONLY"
        )
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, query=query, k=k)
                return [
                    document({"id": str(row[0]), "metadata": row[1]})
                    for row in cursor.fetchall()
                ]


class RestTransport:
    """Only public SDK methods; no direct HTTP or private SDK attributes."""

    def __init__(self, client: Any):
        self.client = client

    def call(self, operation: str, **params: Any) -> dict:
        method = "query" if operation == "search" else operation
        return as_dict(getattr(self.client, method)(**params))

    def migrate(self, name: str) -> None:
        raise RuntimeError("oracle-vecdb REST SDK does not support schema migration")

    def text_search(self, name: str, query: str, k: int) -> list:
        raise ValueError("oracle-vecdb REST SDK does not support metadata text search")


def is_sdk_client(client: Any) -> bool:
    # Avoid importing the optional SDK for a native/SQL store.
    return any(
        cls.__module__.startswith("oracle_vecdb.") and cls.__name__ == "OracleVecDB"
        for cls in type(client).__mro__
    )


def _compatible_backend(
    version: tuple[int, ...], compatibility: str, kind: str, backend: str
) -> str:
    minimum = SPARSE_COMPATIBLE if version >= CURRENT_VERSION else DENSE_COMPATIBLE
    if parse_version(compatibility) < minimum:
        required = ".".join(map(str, minimum))
        if backend == "vecdb" or kind == "vecdb":
            message = (
                f"VecDB requires COMPATIBLE >= {required}; "
                f"configured value is {compatibility}"
            )
            if kind == "vecdb":
                message += ". An existing VecDB table cannot use the SQL backend."
            raise ValueError(message)
        logger.info(
            "Using SQL backend: COMPATIBLE %s is below VecDB requirement %s",
            compatibility,
            required,
        )
        return "sql"
    return "vecdb"


def native_selection(
    connection: Any, name: str, backend: str
) -> tuple[str, tuple[int, ...]]:
    version = parse_version(connection.version)
    if backend == "sql":
        return "sql", version
    _, name = table_identifier(name)
    with connection.cursor() as cursor:
        cursor.execute(_columns_sql(), name=name)
        columns = {row[0] for row in cursor.fetchall()}
        kind = _layout(columns)
        if kind == "sql":
            if backend == "vecdb":
                raise ValueError("An ordinary OracleVS table cannot be used as VecDB")
            return "sql", version
        if version < LEGACY_VERSION:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("VecDB requires Oracle Database 23.26.1+")
            return "sql", version
        cursor.execute(
            "SELECT COUNT(*) FROM all_procedures WHERE owner = 'SYS' "
            "AND object_name = 'DBMS_VECTOR_DATABASE' AND procedure_name = 'SEARCH'"
        )
        supported = cursor.fetchone()[0] > 0
        if not supported:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("DBMS_VECTOR_DATABASE.SEARCH is unavailable")
            return "sql", version
        if (
            kind == "vecdb"
            and version < CURRENT_VERSION
            and "CONTENT_METADATA" in columns
        ):
            raise ValueError(
                "Current VecDB table layout cannot be used on a legacy database"
            )
        db_version = cursor.var(str, size=128)
        compatibility = cursor.var(str, size=128)
        cursor.execute(
            COMPATIBILITY_SQL, db_version=db_version, compatibility=compatibility
        )
        return _compatible_backend(
            version, compatibility.getvalue(), kind, backend
        ), version


class VecDBBackend:
    def __init__(self, transport: Any, name: str, metric: DistanceStrategy):
        self.transport = transport
        self.quoted, self.name = table_identifier(name)
        from .oraclevs import _get_distance_function

        self.metric = _get_distance_function(
            DistanceStrategy.DOT_PRODUCT
            if metric == DistanceStrategy.MAX_INNER_PRODUCT
            else metric
        )

    def prepare(self) -> None:
        if isinstance(self.transport, NativeTransport):
            if not self.transport.columns(self.name):
                self.transport.call(
                    "create_vector_table",
                    name=self.name,
                    table_params={"auto_generate_id": False},
                )
            return
        try:
            self.transport.call("describe_vector_table", name=self.name)
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise
            self.transport.call(
                "create_vector_table",
                name=self.name,
                table_params={"auto_generate_id": False},
            )

    def query(self, operation: str, **params: Any) -> dict:
        try:
            return self.transport.call(operation, **params)
        except Exception as exc:
            if not layout_error(exc):
                raise
            self.transport.migrate(self.name)
            return self.transport.call(operation, **params)

    def records(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> list[dict]:
        if not texts:
            return []
        if len(texts) != len(vectors):
            raise ValueError("Text and embedding batch sizes must match")
        records = []
        for text, vector, metadata, id_ in zip(texts, vectors, metadatas, ids):
            if TEXT_KEY in metadata:
                raise ValueError("Document metadata contains a reserved key")
            records.append(
                {
                    "id": id_,
                    "dense_vector": vector,
                    "metadata": {**metadata, TEXT_KEY: text},
                }
            )
        return records

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
        mutate: bool,
    ) -> list[str]:
        records = self.records(texts, vectors, metadatas, ids)
        if not records:
            return []
        if not mutate:
            if isinstance(self.transport, RestTransport):
                raise ValueError(
                    "oracle-vecdb REST SDK does not support insert-only writes; "
                    "it exposes upserts"
                )
            return self.transport.insert(self.name, records)
        self.transport.call("upsert_vectors", table_name=self.name, vectors=records)
        return ids

    def search(
        self,
        embedding: list[float],
        k: int,
        filter: Optional[dict],
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> list:
        response = self.query(
            "search",
            table_name=self.name,
            query_by={"vector": embedding},
            top_k=k,
            filters=filter,
            include_vectors=include_vectors,
            advanced_options={"distance_metric": self.metric},
        )
        return self.search_results(response, include_vectors)

    def search_results(self, response: dict, include_vectors: bool) -> list:
        result = []
        for row in rows(response, "results"):
            distance = row.get("distance")
            if not isinstance(distance, (int, float)) or not math.isfinite(distance):
                raise ValueError("VecDB returned an invalid distance")
            hit: tuple[Any, ...] = (document(row), float(distance))
            if include_vectors:
                if not isinstance(row.get("vector"), list):
                    raise ValueError("VecDB did not return requested embeddings")
                hit += (np.asarray(row["vector"], dtype=np.float32),)
            result.append(hit)
        return result

    def get(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        result: list[Document] = []
        for offset in range(0, len(ids), 100):
            batch = ids[offset : offset + 100]
            response = self.query(
                "list_vectors",
                table_name=self.name,
                ids=batch,
                limit=len(batch),
                offset=0,
            )
            result.extend(document(row) for row in rows(response, "vectors"))
        return result

    def delete(self, ids: Optional[list[str]]) -> None:
        self.transport.call("delete_vectors", table_name=self.name, ids=ids)

    def text_search(self, query: str, k: int) -> list[Document]:
        return self.transport.text_search(self.name, query, k)

    def native_transport(
        self, operation: str = "SQL-based metadata index operations"
    ) -> NativeTransport:
        if isinstance(self.transport, RestTransport):
            raise ValueError(f"oracle-vecdb REST SDK does not support {operation}")
        return self.transport

    def hybrid_search(self, params: dict, scores: bool) -> list[Document]:
        return self.native_transport("hybrid search").hybrid_search(
            self.name, params, scores
        )

    def create_text_index(self, index: str) -> None:
        self.native_transport("text index creation").create_text_index(self.name, index)

    def index_params(self, params: Optional[dict]) -> dict:
        from .oraclevs import (
            _validate_index_type,
            _validate_int_param,
            _validate_vector_index_common,
        )

        params = dict(params or {})
        kind = params.get("idx_type", "HNSW")
        kind = "IVF" if isinstance(kind, str) and kind.upper() == "IVF" else "HNSW"
        _validate_index_type(params, kind)
        params.setdefault("idx_name", kind)
        _validate_vector_index_common(params)
        params.pop("idx_type", None)
        params.pop("idx_name", None)  # VecDB owns the physical index name.
        accuracy = params.pop("accuracy", 90)
        parallel = params.pop("parallel", None)
        if kind == "IVF" and "partitions" in params:
            params.setdefault("neighbor_part", params.pop("partitions"))
        mapped: dict[str, Any] = {
            "organization": "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS",
            "distance_metric": self.metric,
            "accuracy": accuracy,
        }
        advanced = {}
        ranges = (
            {"neighbors": (2, 2048), "efConstruction": (1, 65535)}
            if kind == "HNSW"
            else {"neighbor_part": (1, 10000000)}
        )
        for key, (low, high) in ranges.items():
            _validate_int_param(params, key, low, high)
            if key in params:
                advanced["partitions" if key == "neighbor_part" else key] = params.pop(
                    key
                )
        mapped.update(params)  # The SDK/database validates additional options.
        if advanced:
            mapped["advanced_params"] = advanced
        if getattr(self.transport, "version", CURRENT_VERSION) < CURRENT_VERSION:
            mapped["indexing"] = "auto"
            if parallel is not None:
                mapped["parallel_creation"] = parallel
            return mapped
        mapped["auto_index"] = True
        mapped["organization"] = "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS"
        result: dict[str, Any] = {"vector_index_params": mapped}
        if parallel is not None:
            result["parallel_creation"] = parallel
        return result

    def create_index(self, params: Optional[dict]) -> None:
        self.transport.call(
            "create_index", table_name=self.name, index_params=self.index_params(params)
        )

    async def aquery(self, operation: str, **params: Any) -> dict:
        try:
            return await self.transport.acall(operation, **params)
        except Exception as exc:
            if not layout_error(exc):
                raise
            await self.transport.amigrate(self.name)
            return await self.transport.acall(operation, **params)

    async def run_async(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(getattr(self, method), *args, **kwargs)
        if method == "create_index":
            await self.transport.acall(
                "create_index",
                table_name=self.name,
                index_params=self.index_params(args[0]),
            )
            return
        if method == "create_text_index":
            await self.transport.acreate_text_index(self.name, args[0])
            return
        if method == "prepare":
            if not await self.transport.acolumns(self.name):
                await self.transport.acall(
                    "create_vector_table",
                    name=self.name,
                    table_params={"auto_generate_id": False},
                )
            return
        if method == "add":
            texts, vectors, metadatas, ids, mutate = args
            records = self.records(texts, vectors, metadatas, ids)
            if not records:
                return []
            if not mutate:
                return await self.transport.ainsert(self.name, records)
            await self.transport.acall(
                "upsert_vectors", table_name=self.name, vectors=records
            )
            return ids
        if method == "delete":
            await self.transport.acall(
                "delete_vectors", table_name=self.name, ids=args[0]
            )
            return
        if method == "get":
            ids = args[0]
            result: list[Document] = []
            for offset in range(0, len(ids), 100):
                batch = ids[offset : offset + 100]
                response = await self.aquery(
                    "list_vectors",
                    table_name=self.name,
                    ids=batch,
                    limit=len(batch),
                    offset=0,
                )
                result.extend(document(row) for row in rows(response, "vectors"))
            return result
        if method == "search":
            embedding, k, filter = args[:3]
            include = args[3] if len(args) > 3 else False
            response = await self.aquery(
                "search",
                table_name=self.name,
                query_by={"vector": embedding},
                top_k=k,
                filters=filter,
                include_vectors=include,
                advanced_options={"distance_metric": self.metric},
            )
            return self.search_results(response, include)
        if method == "hybrid_search":
            return await self.transport.ahybrid_search(self.name, *args)
        if method == "text_search":
            query, k = args
            return await self.transport.atext_search(self.name, query, k)
        raise ValueError("Unsupported async VecDB operation")


class AsyncNativeTransport(NativeTransport):
    """Same package contract using python-oracledb's async connections."""

    async def acolumns(self, name: str) -> set[str]:
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(_columns_sql(), name=name)
                return {row[0] for row in await cursor.fetchall()}

    async def acall(self, operation: str, **params: Any) -> dict:
        statement, binds = self._statement(operation, params)
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                out = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.setinputsizes(
                    **{
                        key: oracledb.DB_TYPE_CLOB
                        for key in binds
                        if f"JSON(:{key})" in statement
                    }
                )
                await cursor.execute(statement, result=out, **binds)
                value = out.getvalue()
                if hasattr(value, "read"):
                    value = await value.read()
                return as_dict(value)

    async def amigrate(self, name: str) -> None:
        quoted, name = table_identifier(name)
        async with _aget_connection(self.client) as connection:
            self.version = parse_version(connection.version)
            if self.version < CURRENT_VERSION:
                raise ValueError("Layout migration requires Oracle Database 23.26.3+")
            if connection.transaction_in_progress:
                raise RuntimeError(
                    "VecDB migration requires a connection without pending work"
                )
            with connection.cursor() as cursor:

                async def inspect() -> set[str]:
                    await cursor.execute(_columns_sql(), name=name)
                    return {row[0] for row in await cursor.fetchall()}

                for action in _migration_actions(await inspect(), quoted):
                    try:
                        await cursor.execute(action)
                    except oracledb.DatabaseError as exc:
                        if exc.args[0].code not in {
                            904,
                            1430,
                        } or action in _migration_actions(await inspect(), quoted):
                            raise
                if not CURRENT_COLUMNS <= await inspect():
                    raise RuntimeError(
                        "VecDB migration did not produce the required layout"
                    )

    async def acreate_metadata_index(
        self,
        name: str,
        index: str,
        *,
        hybrid: bool = False,
        build: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        _, name = table_identifier(name)
        _, requested = table_identifier(index)
        columns = await self.acolumns(name)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(_metadata_index_sql(), name=name, column=meta)
                found = await cursor.fetchall()
                if len(found) > 1:
                    raise ValueError("Multiple metadata domain indexes found")
                existing = found[0] if found else None
                physical, ddl, restore = _metadata_index_plan(
                    name, index, columns, existing, hybrid, build
                )
                if ddl is None:
                    return
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                if not existing:
                    await cursor.execute(
                        "SELECT table_name FROM user_indexes WHERE index_name = :name",
                        name=requested,
                    )
                    if await cursor.fetchone():
                        raise ValueError(
                            "Index name is already used by an incompatible index"
                        )
                if restore:
                    await cursor.execute(f"DROP INDEX {physical}")
                try:
                    await cursor.execute(ddl)
                except Exception:
                    if restore:
                        try:
                            await cursor.execute(
                                "SELECT index_name FROM user_indexes "
                                "WHERE index_name = :name",
                                name=table_identifier(physical)[1],
                            )
                            if await cursor.fetchone():
                                await cursor.execute(f"DROP INDEX {physical} FORCE")
                            await cursor.execute(restore)
                            logger.warning(
                                "Restored metadata index %s after replacement failed",
                                physical,
                            )
                        except Exception as recovery:
                            raise RuntimeError(
                                f"Could not restore metadata index {physical} after "
                                "replacement failed; manual recovery is required"
                            ) from recovery
                    raise
                logger.info(
                    "Created %s metadata index %s",
                    "hybrid" if hybrid else "text",
                    physical,
                )

    async def acreate_text_index(self, name: str, index: str) -> None:
        await self.acreate_metadata_index(name, index)

    async def ainsert(self, name: str, records: list[dict]) -> list[str]:
        import array

        quoted, name = table_identifier(name)
        columns = await self.acolumns(name)
        _layout(columns)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        values = [
            (r["id"], array.array("f", r["dense_vector"]), r["metadata"])
            for r in records
        ]
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                await cursor.executemany(
                    f"INSERT INTO {quoted} (ID, DENSE_VECTOR, {meta}) "
                    "VALUES (:1, :2, :3)",
                    values,
                    batcherrors=True,
                )
                errors = cursor.getbatcherrors() or []
                fatal = [error for error in errors if error.code != 1]
                if fatal:
                    raise RuntimeError(
                        f"VecDB insert failed with ORA-{fatal[0].code:05d}"
                    )
                await connection.commit()
                failed = {error.offset for error in errors}
                return [
                    record["id"] for i, record in enumerate(records) if i not in failed
                ]

    async def ahybrid_search(self, name: str, params: dict, scores: bool) -> list:
        columns = await self.acolumns(name)
        sql = _hybrid_sql(name, columns)
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(
                    _metadata_index_sql(),
                    name=table_identifier(name)[1],
                    column="CONTENT_METADATA"
                    if "CONTENT_METADATA" in columns
                    else "METADATA",
                )
                existing = await cursor.fetchone()
                if (
                    not existing
                    or "SIMPLIFIED_JSON_HYBRID" not in (existing[1] or "").upper()
                ):
                    raise ValueError(
                        "Create a hybrid vector index on the VecDB "
                        "metadata column first"
                    )
                params = {**params, "hybrid_index_name": existing[0]}
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_CLOB)
                await cursor.execute(
                    sql, search_params=json.dumps(params, allow_nan=False)
                )
                return _hybrid_documents(await cursor.fetchall(), scores)

    async def atext_search(self, name: str, query: str, k: int) -> list:
        quoted, name = table_identifier(name)
        columns = await self.acolumns(name)
        _layout(columns)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        sql = (
            f"SELECT id, {meta} FROM {quoted} "
            f"WHERE JSON_TEXTCONTAINS({meta}, '$.text', :query) "
            "ORDER BY id FETCH FIRST :k ROWS ONLY"
        )
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(sql, query=query, k=k)
                return [
                    document({"id": str(row[0]), "metadata": row[1]})
                    for row in await cursor.fetchall()
                ]


async def anative_selection(
    connection: Any, name: str, backend: str
) -> tuple[str, tuple[int, ...]]:
    version = parse_version(connection.version)
    if backend == "sql":
        return "sql", version
    _, name = table_identifier(name)
    with connection.cursor() as cursor:
        await cursor.execute(_columns_sql(), name=name)
        columns = {row[0] for row in await cursor.fetchall()}
        kind = _layout(columns)
        if kind == "sql":
            if backend == "vecdb":
                raise ValueError("An ordinary OracleVS table cannot be used as VecDB")
            return "sql", version
        if version < LEGACY_VERSION:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("VecDB requires Oracle Database 23.26.1+")
            return "sql", version
        await cursor.execute(
            "SELECT COUNT(*) FROM all_procedures WHERE owner = 'SYS' "
            "AND object_name = 'DBMS_VECTOR_DATABASE' AND procedure_name = 'SEARCH'"
        )
        if not (await cursor.fetchone())[0]:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("DBMS_VECTOR_DATABASE.SEARCH is unavailable")
            return "sql", version
        if (
            kind == "vecdb"
            and version < CURRENT_VERSION
            and "CONTENT_METADATA" in columns
        ):
            raise ValueError(
                "Current VecDB table layout cannot be used on a legacy database"
            )
        db_version = cursor.var(str, size=128)
        compatibility = cursor.var(str, size=128)
        await cursor.execute(
            COMPATIBILITY_SQL, db_version=db_version, compatibility=compatibility
        )
        return _compatible_backend(
            version, compatibility.getvalue(), kind, backend
        ), version
