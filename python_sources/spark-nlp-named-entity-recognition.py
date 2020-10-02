#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os

# Install java
get_ipython().system(' apt-get install -y openjdk-8-jdk-headless -qq > /dev/null')
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
get_ipython().system(' java -version')

# Install pyspark
get_ipython().system(' pip install --ignore-installed pyspark==2.4.5 spark-nlp==2.5.1')


# In[ ]:


import sparknlp

spark = sparknlp.start()

print("Apache Spark version")
spark.version


# In[ ]:


print("Spark NLP version")
sparknlp.version()


# **Download the pretrained Pipeline**
# `recognize_entities_dl`
# 

# In[ ]:


from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('recognize_entities_dl', 'en')


# In[ ]:


result = pipeline.annotate('Donald John Trump (born June 14, 1946) is the 45th and current president of the United States.')


# In[ ]:


result


# In[ ]:


print(result['ner'])


# In[ ]:


ner = [result['ner'] for content in result]
token = [result['token'] for content in result]
# let's put token and tag together
list(zip(token[0], ner[0]))


# In[ ]:


print(result['entities'])


# - **Home repository: ** https://github.com/JohnSnowLabs/spark-nlp
# - **Full list of pretrained models/pipelines:** https://github.com/JohnSnowLabs/spark-nlp-models
# - **All examples:** https://github.com/JohnSnowLabs/spark-nlp-workshop

# In[ ]:




