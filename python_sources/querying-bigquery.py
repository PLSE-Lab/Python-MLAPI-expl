#!/usr/bin/env python
# coding: utf-8

# Kaggle can access data in BigQuery. If you do not set a Google Cloud project ID, you will be using the Kaggle BigQuery client with 5TB query quota per month free. This client can access any public BigQuery dataset.

# In[ ]:


from google.cloud import bigquery

client = bigquery.Client()


# In[ ]:


query = """
SELECT country_code, COUNT(*) AS cnt
FROM `patents-public-data.patents.publications`
GROUP BY country_code;
"""

df = client.query(query).to_dataframe()
df.head(20)


# # Analyze data

# In[ ]:


df.sort_values('cnt', ascending=False).head(20).plot.bar(x='country_code', y='cnt')


# # Download data

# In[ ]:


df.to_csv("data.csv")


# ![www.kaggle.com_wetherbeei_kernel6a41bdf0ef_edit%20%281%29.png](attachment:www.kaggle.com_wetherbeei_kernel6a41bdf0ef_edit%20%281%29.png)

# **Copyright 2019 Google LLC**
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
