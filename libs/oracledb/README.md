# langchain-oracledb

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-oracledb?label=%20&logo=pypi&logoColor=white)](https://pypi.org/project/langchain-oracledb/)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-oracledb)](https://pypistats.org/packages/langchain-oracledb)
[![License](https://img.shields.io/badge/license-UPL--1.0-blue)](https://oss.oracle.com/licenses/upl/)

This package contains the LangChain integrations with [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/).

> ⭐ If this package is useful to you, consider [starring the repository](https://github.com/oracle/langchain-oracle) — it helps other developers discover these integrations.

## Installation

```bash
python -m pip install -U langchain-oracledb
```

## Documentation

- [Oracle AI Vector Search: Vector Store](https://python.langchain.com/docs/integrations/vectorstores/oracle/)
- [Oracle AI Vector Search: Generate Summary](https://python.langchain.com/docs/integrations/tools/oracleai/)
- [Oracle AI Vector Search: Document Processing](https://python.langchain.com/docs/integrations/document_loaders/oracleai/)
- [Oracle AI Vector Search: Generate Embeddings](https://python.langchain.com/docs/integrations/text_embedding/oracleai/)

## Examples

The following examples showcase basic usage of the components provided by `langchain-oracledb`.

Please refer to our complete demo guide [Oracle AI Vector Search End-to-End Demo Guide](https://github.com/langchain-ai/langchain/blob/v0.3/cookbook/oracleai_demo.ipynb) to build an end to end RAG pipeline with the help of Oracle AI Vector Search.

### Connect to Oracle Database

Some examples below require a connection with Oracle Database through [`python-oracledb`](https://pypi.org/project/oracledb/). The following sample code will show how to connect to Oracle Database. By default, `python-oracledb` runs in a ‘Thin’ mode which connects directly to Oracle Database. This mode does not need Oracle Client libraries. However, some additional functionality is available when python-oracledb uses them. Python-oracledb is said to be in ‘Thick’ mode when Oracle Client libraries are used. Both modes have comprehensive functionality supporting the Python Database API v2.0 Specification. See the following [guide](https://python-oracledb.readthedocs.io/en/latest/user_guide/appendix_a.html#featuresummary) that talks about features supported in each mode. You might want to switch to Thick mode if you are unable to use Thin mode. For python-oracledb installation help, see [Installing python-oracledb](https://python-oracledb.readthedocs.io/en/latest/user_guide/installation.html).

Check your database connectivity:

```python
import os

import oracledb

# Please update with your username, password, hostname, port and service_name
username = os.environ["ORACLE_DB_USERNAME"]
password = os.environ["ORACLE_DB_PASSWORD"]
dsn = os.environ["ORACLE_DB_DSN"]

connection = oracledb.connect(user=username, password=password, dsn=dsn)
print("Connection successful!")
```

### Vector Stores

#### OracleVS backends and VecDB

`OracleVS` defaults to `backend="auto"`. With a native connection, it checks the
Oracle Database version, the availability of `DBMS_VECTOR_DATABASE.SEARCH`,
and the database `COMPATIBLE` setting (read through `DBMS_UTILITY.DB_VERSION`).
New tables use VecDB on supported databases (23.26.1+), and SQL otherwise.
The legacy VecDB generation requires `COMPATIBLE >= 23.4.0`; the current schema
(23.26.3+) requires `COMPATIBLE >= 23.6.0` for its sparse-vector column. A lower
setting selects SQL for new tables in `auto` mode. Explicit VecDB requests and
existing VecDB tables instead report the unmet prerequisite before DDL; an
existing VecDB layout is never treated as an ordinary SQL table. No extra
constructor parameter is needed.
Existing ordinary OracleVS tables continue using SQL. Explicit `backend="sql"`
and `backend="vecdb"` overrides are available. Both backends reuse the existing
OracleVS identifier handling, batch preparation, and index-parameter validators.
VecDB filters are passed to the SDK or database without a separate operator or
metadata-path whitelist.

REST uses the optional Oracle VecDB SDK:

```bash
pip install "langchain-oracledb[vecdb]"
```

The REST extra requires Python 3.10–3.15. SQL-only installation continues to
support Python 3.9. The SDK is imported only when needed.

```python
import os
from oracle_vecdb import Configuration
from langchain_oracledb import OracleVS

# embedding_model implements LangChain's Embeddings interface.
config = Configuration(
    rest_url=os.environ["ORACLE_VECDB_REST_URL"],
    username=os.environ["ORACLE_VECDB_USERNAME"],
    password=os.environ["ORACLE_VECDB_PASSWORD"],
    # Alternatively use access_token=... without username/password.
)
store = OracleVS.from_vecdb_config(
    config=config,
    table_name="DOCUMENTS",
    embedding=embedding_model,
    mutate_on_duplicate=True,
)
store.add_texts(["Oracle vector search"], ids=["doc-1"])
documents = store.similarity_search("vector search", k=4)
```

You can also pass an existing `OracleVecDB` as `client`; it remains caller-owned.
The SDK supports REST only. Native connections use a private PL/SQL executor.
For REST clients, API availability establishes VecDB support. The adapter uses
only operations exposed by the public `oracle-vecdb` SDK. Unsupported operations
raise clear errors; there is no secondary connection parameter or automatic
fallback to another transport.

REST ingestion uses the SDK's upsert operation and therefore requires the existing
`mutate_on_duplicate=True` option. Insert-only writes are not supported by this
SDK. SDK documents and metadata are mapped to the same LangChain IDs. Search
scores retain OracleVS's lower-is-better distance convention, and requested
distance metrics are sent explicitly to VecDB.

VecDB stores document text under the reserved metadata key `text`, in `METADATA`
on 23.26.1/23.26.2 or `CONTENT_METADATA` on 23.26.3+. User metadata must not contain
`text` or the existing internal document-ID key. Text is returned as
`Document.page_content`, including long documents, without the reserved field in
`Document.metadata`. No additional document table is created.

`OracleTextSearchRetriever` searches only `$.text` using `JSON_TEXTCONTAINS`.
`create_text_index(..., vector_store=store)` replaces VecDB's metadata guide index
with a text-enabled JSON search index when necessary. It preserves the physical
index name, data-guide callback, synchronization schedule, and other catalog
parameters; only the `SEARCH_ON` mode changes to `TEXT`. An existing text or
hybrid index is reused. Matches are ordered by ID; `return_scores=True` returns
a Boolean match score of 1.0.

`create_hybrid_index` replaces the guide or text-only index with an Oracle hybrid
vector index on the metadata column. It retains the existing physical name and
compatible parameters, including
`DATAGUIDE ON CHANGE SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX`. Model-backed
VecDB preferences default to the `$.text` string path. Explicit hybrid parameters
can override memory and maintenance settings, but cannot change the inherited
synchronization schedule or remove the VecDB data-guide callback. A live check
with `SYNC (ON COMMIT)` and newly introduced metadata paths failed during ingestion;
sync overrides and conversion of an existing `ON COMMIT` index to HVI are
rejected before replacement. The separate `DENSE_VECTOR` index is retained.

VecDB owns the physical metadata-index name. When an index already exists, the
helpers preserve that name even if a different `idx_name` is requested. The hybrid
retriever discovers the actual HVI on the store's metadata column, so existing
retriever calls continue working. Repeated requests reuse an existing HVI;
requesting text indexing afterward does not downgrade it. No index is replaced
merely by constructing a store.

`OracleHybridSearchRetriever` executes `DBMS_HYBRID_VECTOR.SEARCH`; Oracle Database
performs ranking and returns overall/text/vector scores. There is no application
fusion. The REST SDK does not expose metadata text search, native hybrid search,
metadata text/hybrid index creation, vectorizer preferences, or schema migration;
those operations raise unsupported-operation errors for SDK clients. They remain
available for stores constructed with an oracledb connection/pool as `client`.
Model-backed HVI creates its own embeddings rather than automatically reusing `DENSE_VECTOR`.

Index replacement uses DDL and requires a connection without pending application
work. Queries may be unavailable between drop and recreation; concurrent index
DDL is not supported. If creation fails, the adapter attempts to recreate the
original JSON search index with its previous parameters and raises the error.
Recovery is best effort, not transactional rollback; a recovery failure explicitly
requires manual intervention. Unknown index configurations and foreign data-guide
callbacks are rejected before dropping anything.

The inherited synchronization schedule governs text, hybrid, and data-guide
freshness. VecDB's default is every 30 seconds, so committed changes need not
appear immediately in those indexes. SQL-only stores keep their existing behavior.
Replacement and new-path MVI generation were validated on DB 23.26.4.1 / SDK
1.0.6; official VecDB lifecycle compatibility and older DB versions remain
unconfirmed.

After an upgrade to 23.26.3+, queries against recognized legacy VecDB layouts
may trigger migration: rename `METADATA`, add the three missing content/sparse
columns, re-inspect, and retry once. Only a missing current-layout column error
qualifies, and only the missing actions run. This preserves data and supports
interrupted migrations. Migration is supported only for native `client`
connections without pending application work because Oracle DDL implicitly commits. A recognized layout
error through the REST SDK reports that schema migration is unsupported.
Authentication, network, and unrelated query errors never trigger migration or backend fallback.

Opt-in REST integration tests use `ORACLE_VECDB_REST_URL` with either
`ORACLE_VECDB_ACCESS_TOKEN` or `ORACLE_VECDB_USERNAME`/`ORACLE_VECDB_PASSWORD`.
`ORACLE_VECDB_CA_CERT` optionally supplies a trusted CA bundle. Tests use unique
temporary tables and drop them afterward. Index lifecycle tests additionally use
`ORACLE_VECDB_NATIVE_DSN` and username/password for a native store. The hybrid
lifecycle test also needs `ORACLE_VECDB_HYBRID_MODEL`, naming an installed database
embedding model.

#### OracleVS

Use Oracle Vector Database with `OracleVS`. More information can be found in [Oracle AI Vector Search: Vector Store](https://python.langchain.com/docs/integrations/vectorstores/oracle/) documentation.


```python
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.oraclevs import create_index
from langchain_oracledb.document_loaders.oracleai import OracleTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_oracledb.vectorstores import DistanceStrategy

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)
vector_store = OracleVS(conn, embedding_model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)

# add texts to the vector database
texts = ["A tablespace can be online (accessible) or offline (not accessible) whenever the database is open.\nA tablespace is usually online so that its data is available to users. The SYSTEM tablespace and temporary tablespaces cannot be taken offline.", "The database stores LOBs differently from other data types. Creating a LOB column implicitly creates a LOB segment and a LOB index. "]
metadata = [
    {"id": "100", "link": "Document Example Test 1"},
    {"id": "101", "link": "Document Example Test 2"},
]

vector_store.add_texts(texts, metadata)

# for large documents, chunk before ingesting so each chunk is stored in its own row
splitter = OracleTextSplitter(conn=conn, params={"split": "sentence", "max": 20})
documents = [
    Document(
        page_content=(
            "A tablespace can be online (accessible) or offline (not accessible). "
            "The SYSTEM tablespace cannot be taken offline."
        ),
        metadata={"id": "200", "link": "Large Document Example"},
    )
]
vector_store.add_documents(
    documents,
    text_splitter=splitter,
    ids=["200"],  # becomes 200#chunk-0, 200#chunk-1, ...
)

create_index(
    conn, vector_store, params={"idx_name": "hnsw_oravs", "idx_type": "HNSW"}
)

# perform siliarity search
vector_store.similarity_search("How does a database stores LOBs?", 1)

```

### Document Loaders

#### OracleDocLoader

Load your documents using `OracleDocLoader`. More information can be found in [Oracle AI Vector Search: Document Processing](https://python.langchain.com/docs/integrations/document_loaders/oracleai/) documentation.

```python
from langchain_oracledb.document_loaders.oracleai import OracleDocLoader

"""
# loading a local file
loader_params = {}
loader_params["file"] = "<file>"

# loading from a local directory
loader_params = {}
loader_params["dir"] = "<directory>"
"""

# loading from Oracle Database table
loader_params = {
    "owner": "<owner>",
    "tablename": "demo_tab",
    "colname": "data",
}

# load the docs
loader = OracleDocLoader(conn=conn, params=loader_params)
docs = loader.load()

# verify
print(f"Number of docs loaded: {len(docs)}")
```

#### OracleTextSplitter

Chunk your documents using `OracleTextSplitter`. More information can be found in [Oracle AI Vector Search: Document Processing](https://python.langchain.com/docs/integrations/document_loaders/oracleai/) documentation.

```python
from langchain_oracledb.document_loaders.oracleai import OracleTextSplitter
from langchain_oracledb.document_loaders.oracleai import OracleDocLoader

# loading from Oracle Database table
loader_params = {
    "owner": "<owner>",
    "tablename": "demo_tab",
    "colname": "data",
}

# load the docs
loader = OracleDocLoader(conn=conn, params=loader_params)
docs = loader.load()

"""
# some examples
# split by chars, max 500 chars
splitter_params = {"split": "chars", "max": 500, "normalize": "all"}

# split by words, max 100 words
splitter_params = {"split": "words", "max": 100, "normalize": "all"}

# split by sentence, max 20 sentences
splitter_params = {"split": "sentence", "max": 20, "normalize": "all"}
"""

# split by default parameters
splitter_params = {"normalize": "all"}

# get the splitter instance
splitter = OracleTextSplitter(conn=conn, params=splitter_params)

list_chunks = []
for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    list_chunks.extend(chunks)

# verify
print(f"Number of Chunks: {len(list_chunks)}")
# print(f"Chunk-0: {list_chunks[0]}") # content
```

#### OracleAutonomousDatabaseLoader

Load documents from Oracle Autonomous Database using `OracleAutonomousDatabaseLoader`. More information can be found in [Oracle Autonomous Database](https://python.langchain.com/docs/integrations/document_loaders/oracleadb_loader/) documentation.

```python
from langchain_oracledb.document_loaders import OracleAutonomousDatabaseLoader
from settings import s

SQL_QUERY = "select channel_id, channel_desc from sh.channels where channel_desc = :1 fetch first 5 rows only"

doc_loader = OracleAutonomousDatabaseLoader(
    query=SQL_QUERY,
    user=s.USERNAME,
    password=s.PASSWORD,
    schema=s.SCHEMA,
    dsn=s.DSN,
    parameters=["Direct Sales"],
)
doc = doc_loader.load()
```

With mutual TLS authentication (mTLS), wallet_location and wallet_password are required to create the connection, user can create connection by providing either connection string or tns configuration details. With TLS authentication, wallet_location and wallet_password are not required. Bind variable option is provided by argument "parameters".

### Embeddings

#### OracleEmbeddings

Generate embeddings for your documents using `OracleEmbeddings`. More information can be found in [Oracle AI Vector Search: Generate Embeddings](https://python.langchain.com/docs/integrations/text_embedding/oracleai/) documentation.

```python
from langchain_oracledb.embeddings.oracleai import OracleEmbeddings

"""
# using ocigenai
embedder_params = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-light-v3.0",
}

# using huggingface
embedder_params = {
    "provider": "huggingface",
    "credential_name": "HF_CRED",
    "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "wait_for_model": "true"
}
"""

# using ONNX model loaded to Oracle Database
embedder_params = {"provider": "database", "model": "demo_model"}

# if a proxy is not required for your environment, you can omit the 'proxy' parameter below
embedder = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)
embed = embedder.embed_query("Hello World!")

# verify
print(f"Embedding generated by OracleEmbeddings: {embed}")
```

### Utilities

#### OracleSummary

Generate summary for your documents using `OracleSummary`. More information can be found in [Oracle AI Vector Search: Generate Summary](https://python.langchain.com/docs/integrations/tools/oracleai/) documentation.

```python
from langchain_oracledb.utilities.oracleai import OracleSummary
from langchain_core.documents import Document

"""
# using 'ocigenai' provider
summary_params = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/summarizeText",
    "model": "cohere.command",
}

# using 'huggingface' provider
summary_params = {
    "provider": "huggingface",
    "credential_name": "HF_CRED",
    "url": "https://api-inference.huggingface.co/models/",
    "model": "facebook/bart-large-cnn",
    "wait_for_model": "true"
}
"""

# using 'database' provider
summary_params = {
    "provider": "database",
    "glevel": "S",
    "numParagraphs": 1,
    "language": "english",
}

# get the summary instance
# remove proxy if not required
summ = OracleSummary(conn=conn, params=summary_params, proxy=proxy)
summary = summ.get_summary(
    "In the heart of the forest, "
    + "a lone fox ventured out at dusk, seeking a lost treasure. "
    + "With each step, memories flooded back, guiding its path. "
    + "As the moon rose high, illuminating the night, the fox unearthed "
    + "not gold, but a forgotten friendship, worth more than any riches."
)

print(f"Summary generated by OracleSummary: {summary}")
```
