# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
		
	SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS Day
            FROM time           
            GROUP BY Day 
            ORDER BY Day
        """

tran_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
print (tran_per_day)


query3 = """ SELECT COUNT(transaction_id) AS transactions, merkle_root AS markleroot
	     FROM `bigquery-public-data.bitcoin_blockchain.transactions`
	     GROUP BY markleroot	
	 """  

tran_per_merkel = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=40)
print (tran_per_merkel) 



# Any results you write to the current directory are saved as output.