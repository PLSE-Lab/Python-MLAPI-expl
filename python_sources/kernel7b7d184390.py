# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
patentsview = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="patentsview")
                                   
# View table names under the patentsview data table
bq_assistant = BigQueryHelper("patents-public-data", "patentsview")
bq_assistant.list_tables()

bq_assistant.head("patent", num_rows=10)

bq_assistant.table_schema("patent")

query1 = """
SELECT DISTINCT
  type
FROM
  `patents-public-data.patentsview.patent`
LIMIT
  8;
        """
response1 = patentsview.query_to_pandas_safe(query1)
response1.head(20)

query2 = """
SELECT DISTINCT
  title
FROM
  `patents-public-data.patentsview.patent`
LIMIT
  20;
        """
response2 = patentsview.query_to_pandas_safe(query2)
response2.head(20)

query3 = """
SELECT DISTINCT
  patent_id
FROM
  `patents-public-data.patentsview.application_201708`
WHERE
    series_code = '03'
LIMIT
  20;
        """
response3 = patentsview.query_to_pandas_safe(query3)
response3.head(20)

query4 = """
SELECT DISTINCT
  number
FROM
  `patents-public-data.patentsview.application_201708`
WHERE
   date <= '1988-01-01' AND date >= '1987-12-31' 
LIMIT
  20;
        """
response4 = patentsview.query_to_pandas_safe(query4)
response4.head(20)




