#### This class makes quick overview-ing a table easier 
#### i.e. you don't need to write lines of code to get the table

from google.cloud import bigquery

################ If runing on COLAB:
#from google.colab import drive
#drive.mount('/content/drive')

#### Credentials need to run on COLAB
#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/.json"


'''Get an overview of a dataset in BigQuery public data:
1. See the list of tables:
   e.g. Overview("crypto_bitcoin").list_tables()

2. Get a table from the dataset:
   e.g. transaction_table = Overview("crypto_bitcoin").get_table("transactions")
   
3. Check table schema:
   e.g. transactin_table.schema

4. Check first a few lines (5 lines) of a table:

   e.g. Overview("crypto_bitcoin").head_lines("transactions")
   
3. Get the dry-run estimate and get the dataframe from a query
   e.g. query = """
                 WITH time AS
                 (
                     SELECT DATE(block_timestamp) AS trans_date
                     FROM `bigquery-public-data.crypto_bitcoin.transactions`
                 )
                 SELECT COUNT(1) AS transactions,
                        trans_date
                 FROM time
                 GROUP BY trans_date
                 ORDER BY trans_date
                 """

            bitcoin = Overview("crypto_bitcoin")

            bitcoin.get_df_dryrun(query)
            df = bitcoin.get_df_safe(query)

   '''

class Overview():

    def __init__(self,dataset):

        self.dataset = dataset
        self.project = "bigquery-public-data"
        self.client = bigquery.Client()
        self.dataset_ref = self.client.dataset(dataset, project=self.project)
        self.dataset = self.client.get_dataset(self.dataset_ref)
        self.tables = list(self.client.list_tables(self.dataset))

    def list_tables(self):
        tables = list(self.client.list_tables(self.dataset))

        for table in tables:
            print(table.table_id)
        

    
    def get_table(self,table_name):

        table_ref=self.dataset_ref.table(table_name)
        table = self.client.get_table(table_ref)
        return table
    
    
    def head_lines(self,table_name):
        
        table = self.get_table(table_name)
        return self.client.list_rows(table,max_results = 5).to_dataframe()
        
        
    def get_df_safe(self,query):

        safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
        query_job = self.client.query(query, job_config=safe_config)
        df = query_job.to_dataframe()

        return df

    def get_df_dryrun(self,query):

        dry_run_config = bigquery.QueryJobConfig(dry_run=True)

        # API request - dry run query to estimate costs
        dry_run_query_job = self.client.query(query, job_config=dry_run_config)

        print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))

