#!/usr/bin/env python
# coding: utf-8

# # Tutorial 2
# 
# **CS3481 Fundamentals of Data Science**
# 
# *Semester B 2019/20*
# ___
# **Instructions:**
# - same as [Tutorial 1](http://bit.ly/CS3481T1).
# ___

# ## Exercise 1 (submit via uReply)

# Complete the tutorial exercises of [[Witten11]](https://ebookcentral.proquest.com/lib/cityuhk/reader.action?docID=634862&ppg=595) from **Exercise 17.1.3** to **17.1.7**, and read up to and including the subsection  [**The Visualize Panel**](https://ebookcentral.proquest.com/lib/cityuhk/reader.action?docID=634862&ppg=597). Submit your answers of 17.1.3-5 through [uReply](https://cityu.ed2.mobi/student/mobile_index.php) section number **LM715**.
# 
# [*Hint: See the [documentation](https://ebookcentral.proquest.com/lib/cityuhk/reader.action?ppg=438&docID=634862&tm=1547446912037) of WEKA for more details.*]

# 
# ___
# **Answers to 17.1.3:** 
# 
# **Answers to 17.1.4:** 
# 
# **Answers to 17.1.5:**
# ___
# 

# ## Exercise 2 (no submission required)

# Use a text editor of your choice to create an ARFF file for the AND gate $Y=X_1\cdot X_2$, and then load the file into WEKA to ensure it is correct.
# 
# [*See the [documentation](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/) for the ARFF file format, or take a look at some of the ARFF files in the WEKA data folder as examples.*]

# ___
# **Answer**: Modify the following to create the desired ARFF.

# In[ ]:


text = '''@RELATION AND
@ATTRIBUTE X1 {0, _}
@ATTRIBUTE X2 {0, _}
@ATTRIBUTE Y {0, _}
@DATA
_, _, _
_, _, _
_, _, _
_, _, _
'''

with open('AND.arff','w') as file:
  file.write(text)


# Load the ARFF file into a dataframe to check if the ARFF file is correct. 

# In[ ]:


from scipy.io import arff
import pandas as pd

data = arff.loadarff('AND.arff')
df = pd.DataFrame(data[0])

df.head()


# ## Exercise 3 (no submission required)
# 
# [[Han11]](https://www.sciencedirect.com/science/article/pii/B9780123814791000022#s0185) **Question 2.5**: Briefly outline how to compute the dissimilarity between objects described by the following:

# (a) Nominal attributes
# 

# ___
# Answer:
# 
# ___

# (b) Asymmetric binary attributes

# ___
# Answer:
# 
# ___

# (c) Numeric attributes

# ___
# Answer:
# 
# ___

# (d) Term-frequency vectors

# ___
# Answer:
# 
# ___

# ## Exercise 4 (Optional)

# The following illustrates some methods of loading datasets into CoLab. You can execute the code by `shift+enter`.

# ### (a) Load sample datasets from scikit-learn.

# Import the `sklearn.datasets` package, which contains the desired iris dataset. Then, load the iris data sets and print its content.

# In[ ]:


from sklearn import datasets # see https://scikit-learn.org/stable/datasets/index.html

iris = datasets.load_iris()
iris # to print out the content


# The field `DESCR` (description) contains some background information of the dataset. We can pretty-print only the description as follows.

# In[ ]:


print(iris.DESCR)


# Convert the dataset to a Pandas dataframe.

# In[ ]:


import pandas as pd
import numpy as np # 

iris_pd = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns = iris['feature_names']+['target'])
iris_pd


# The function `np.c_` above concatenates the columns `iris['data']` of input features and the column `iris['target']` of class values together. For more details of the function, use the `help` function.

# ### (b) Download and load datasets from openML

# Download the `weather.nominal` dataset from [openml.org](https://www.openml.org/d/41521). 
# 
# [*See the [documentation](https://scikit-learn.org/stable/datasets/index.html#downloading-datasets-from-the-openml-org-repository) for more details.*]

# In[ ]:


from sklearn.datasets import fetch_openml
weather = fetch_openml(data_id=41521)
weather


# Pretty-print with text and wrap it to 100 characters per line.

# In[ ]:


import textwrap
print(textwrap.fill(weather.DESCR,100))


# Conversion to dataframe.

# In[ ]:


import pandas as pd
import numpy as np

weather_pd = pd.DataFrame(data=np.c_[weather.data,weather.target],columns=weather.feature_names+['target'])
weather_pd


# Modify the dataframe so that the columns for `outlook` and `windy` use their respective category labels instead of indexes.

# ### (c) Download a CSV file from UCI Machine Learning repository and read it into a dataframe directly

# In[ ]:


import urllib.request
import io
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
ftpstream = urllib.request.urlopen(url)
iris = pd.read_csv(io.StringIO(ftpstream.read().decode('utf-8')))
iris


# There is something wrong with the above dataframe. Use the additional options `names` and `index_col` of `read_csv` to read the CSV file correctly.
