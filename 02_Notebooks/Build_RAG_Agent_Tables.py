# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Build RAG Agent Tables
# MAGIC
# MAGIC Builds the Unity Catalog assets needed for the Databricks RAG agent:
# MAGIC 1. `product_details` — parsed text of every product PDF in the volume
# MAGIC 2. `product_master` — `products` joined with `product_details`, plus an HTML-tag "combined" column
# MAGIC 3. A Vector Search index on `product_master.product_combined`
# MAGIC 4. Two Unity Catalog functions (agent tools) over `policies` and `cust_service_data`
# MAGIC
# MAGIC Verified against the live workspace before writing this notebook:
# MAGIC - Catalog/schema: `uc_agentic_ai.agentic_ai_schema`
# MAGIC - Volume: `uc_agentic_ai.agentic_ai_schema.data_files`, PDFs at `/Volumes/uc_agentic_ai/agentic_ai_schema/data_files/01_Data_Files/product_docs/` (509 files)
# MAGIC - `products` schema: `product_id, product_name, product_category, product_sub_category, _rescued_data` (553 rows, 509 distinct `product_name`)
# MAGIC - `policies` schema: `policy, policy_details, last_updated, _rescued_data`
# MAGIC - `cust_service_data` schema: `customer_id, name, email, phone_number, address, interaction_id, date_time, issue_category, issue_description, agent_id, _rescued_data`
# MAGIC - PDF filenames replace `:` with `_` (e.g. product `Advanced Algebra: Concepts and Applications` -> file `Advanced Algebra_ Concepts and Applications.pdf`). Normalizing `:` -> `_` on `product_name` gives a 100% join match against the PDFs — confirmed with a live query, 0 unmatched rows.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

catalog = "uc_agentic_ai"
schema = "agentic_ai_schema"
volume_path = f"/Volumes/{catalog}/{schema}/data_files/01_Data_Files/product_docs/"

vs_endpoint_name = "agentic_ai_vs_endpoint"          # change if you already have an endpoint
vs_index_name = f"{catalog}.{schema}.product_master_index"
embedding_model_endpoint = "databricks-gte-large-en"  # standard Databricks-hosted embedding model; change if your workspace uses a different one

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — `product_details` from the PDFs
# MAGIC
# MAGIC Uses the built-in `ai_parse_document` SQL function (confirmed available in this workspace) to parse each PDF, then concatenates all parsed elements into one text field. `product_name` is the filename without the `.pdf` extension.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.product_details AS
SELECT
  regexp_replace(element_at(split(path, '/'), -1), '\\\\.pdf$', '') AS product_name,
  array_join(
    transform(
      CAST(parsed:document:elements AS ARRAY<VARIANT>),
      x -> CAST(x:content AS STRING)
    ),
    '\\n\\n'
  ) AS product_desc
FROM (
  SELECT path, ai_parse_document(content) AS parsed
  FROM READ_FILES('{volume_path}', format => 'binaryFile')
)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Sanity check — should be 509 rows, matching the number of PDFs in the volume.

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) AS row_count FROM {catalog}.{schema}.product_details"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — `product_master`
# MAGIC
# MAGIC Joins `products` with `product_details` on a normalized `product_name` (colon -> underscore, to match PDF filenames). Adds `product_id` as the primary key (required by Vector Search) and a `product_combined` column formatted as HTML-style tags for embedding.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.product_master AS
SELECT
  p.product_id,
  p.product_name,
  p.product_category,
  p.product_sub_category,
  d.product_desc,
  concat(
    '<product_name>', coalesce(p.product_name, ''), '</product_name>\\n',
    '<product_category>', coalesce(p.product_category, ''), '</product_category>\\n',
    '<product_sub_category>', coalesce(p.product_sub_category, ''), '</product_sub_category>\\n',
    '<product_desc>', coalesce(d.product_desc, ''), '</product_desc>'
  ) AS product_combined
FROM {catalog}.{schema}.products p
INNER JOIN {catalog}.{schema}.product_details d
  ON regexp_replace(p.product_name, ':', '_') = d.product_name
""")

# Vector Search Delta Sync Index requires Change Data Feed on the source table
spark.sql(f"""
ALTER TABLE {catalog}.{schema}.product_master
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Sanity check — should be 553 rows (one per product, including duplicate product names), all with a non-null `product_desc`.

# COMMAND ----------

display(spark.sql(f"""
SELECT COUNT(*) AS total_rows, COUNT(product_desc) AS rows_with_desc
FROM {catalog}.{schema}.product_master
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Vector Search index on `product_combined`
# MAGIC
# MAGIC Requires the `databricks-vectorsearch` Python package (`%pip install databricks-vectorsearch` if not already on the cluster) and runs on cluster/serverless Python compute — not on a SQL warehouse, so this cell must be run from this notebook rather than the SQL-only path used above.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "uc_agentic_ai"
schema = "agentic_ai_schema"
vs_endpoint_name = "agentic_ai_vs_endpoint"
vs_index_name = f"{catalog}.{schema}.product_master_index"
embedding_model_endpoint = "databricks-gte-large-en"
source_table = f"{catalog}.{schema}.product_master"

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if vs_endpoint_name not in existing_endpoints:
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

existing_indexes = [i["name"] for i in vsc.list_indexes(vs_endpoint_name).get("vector_indexes", [])]
if vs_index_name not in existing_indexes:
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        source_table_name=source_table,
        index_name=vs_index_name,
        pipeline_type="TRIGGERED",
        primary_key="product_id",
        embedding_source_column="product_combined",
        embedding_model_endpoint_name=embedding_model_endpoint,
    )
else:
    vsc.get_index(vs_endpoint_name, vs_index_name).sync()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Unity Catalog functions (agent tools)
# MAGIC
# MAGIC Structured lookup tools to pair with the vector index: one over `policies`, one over `cust_service_data`.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_policy_details(
  policy_name STRING COMMENT 'Name or keyword of the policy to search for, e.g. "return policy" or "warranty"'
)
RETURNS TABLE (policy STRING, policy_details STRING, last_updated DATE)
COMMENT 'Looks up company policy details by policy name or keyword.'
RETURN
  SELECT policy, policy_details, last_updated
  FROM {catalog}.{schema}.policies
  WHERE lower(policy) LIKE lower(concat('%', policy_name, '%'))
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_customer_service_history(
  customer_identifier STRING COMMENT 'Customer id, email, or name to search for'
)
RETURNS TABLE (
  customer_id STRING,
  name STRING,
  email STRING,
  interaction_id STRING,
  date_time TIMESTAMP,
  issue_category STRING,
  issue_description STRING,
  agent_id BIGINT
)
COMMENT 'Looks up a customer''s past service interactions by customer id, email, or name.'
RETURN
  SELECT customer_id, name, email, interaction_id, date_time, issue_category, issue_description, agent_id
  FROM {catalog}.{schema}.cust_service_data
  WHERE customer_id = customer_identifier
     OR lower(email) = lower(customer_identifier)
     OR lower(name) LIKE lower(concat('%', customer_identifier, '%'))
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick tests

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.get_policy_details('return')"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.get_customer_service_history('Robert Butler')"))