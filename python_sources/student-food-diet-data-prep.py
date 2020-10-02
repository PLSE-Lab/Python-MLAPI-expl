#!/usr/bin/env python
# coding: utf-8

# # Pending Actions
# 
#   1. Classify features as categorical, numerical or count
#   2. Identify missing data 
#   3. Figure out how to extract features from text

# In[ ]:


from pandas import read_csv
data = read_csv("../input/food_coded.csv")


# In[ ]:


features = data.columns


# In[ ]:


data.shape


# In[ ]:


data.isnull().any().value_counts()


# # Feature Classification
# 
# 
# ----------
# 

# In[ ]:


features_by_dtype = {}
for f in features:
    dtype = str(data[f].dtype)
    
    if dtype not in features_by_dtype.keys():
        features_by_dtype[dtype] = [f]
    else:
        features_by_dtype[dtype] += [f]


# In[ ]:


features_by_dtype.keys()


# In[ ]:


keys = iter(features_by_dtype.keys())


# In[ ]:


key = next(keys)
dtype_list = features_by_dtype[key]
for f in dtype_list:
    string = "{}: {}".format(f,len(data[f].unique()))
    print(string)


# In[ ]:


textual_data = dtype_list 


# In[ ]:


key = next(keys)
dtype_list = features_by_dtype[key]
for f in features_by_dtype[key]:
    string = "{}: {}".format(f,data[f].unique())
    print(string)


# In[ ]:


binary_features = [f for f in dtype_list if len(data[f].unique()) == 2]
categorical_features = binary_features
numerical_features = [f for f in dtype_list if f not in categorical_features]
count_features = numerical_features


# In[ ]:


key = next(keys)
features_by_dtype[key]
for f in features_by_dtype[key]:
    string = "{}: {}".format(f,data[f].unique())
    print(string)


# In[ ]:


numerical_features


# In[ ]:


data[numerical_features].head()


# In[ ]:


categorical_features


# In[ ]:


data[categorical_features].head()


# # Missing Data
# 
# 
# ----------
# 

# In[ ]:


dictionary = {}

for feature in features:
    
    column = data[feature]
    
    has_null = any(column.isnull())
    
    if(has_null):
        
        null_count = column.isnull().value_counts()[True]
        not_null_count = column.notnull().value_counts()[True]
        total_rows = len(column)
        
        row = {}
        row["Null Count"] = null_count
        row["Not Null Count"] = not_null_count
        row["Null Count / Total Rows"] = "%s / %s" %  (null_count, total_rows)
        row["Percentage of Nulls"] = "%.2f" % ((null_count / total_rows) * 100) + "%"
        row["Ratio (Not Null : Null)"] = "%.2f : 1" %  ((null_count / not_null_count))
        
        dictionary[feature] = row

ordered_columns = ["Null Count", "Not Null Count", "Ratio (Not Null : Null)", "Null Count / Total Rows", "Percentage of Nulls"]

from pandas import DataFrame

new_dataframe = DataFrame.from_dict(data = dictionary, orient="index")
new_dataframe[ordered_columns].sort_values("Null Count", ascending=False)


# # Textual Features for NLP
# 
# 
# ----------
# 

# In[ ]:


data[textual_data].head()


# # Feature Extraction Brainstorm
# 
#  - Word Count
#  - Character Count
#  -  Noun Count
#  - Noun Phrase Count
#  - Verb Phrase Count
#  - Adjective Count
#  - Food as noun phrase one hot encoded
#  - Vocabulary Complexity (unique words / total count of words)
#  - Categorize parent careers as working class, blue collar, etc
#  - Detect comparative adjectives
#  - Detect financial and timing themes
#  - Literacy score e.g. spelling mistakes / grammar errors
#  - Categorize sports by contact / intensity / team vs solo

# In[ ]:


data["comfort_food_reasons"].head(20)


# In[ ]:


data["eating_changes"].head(20)


# In[ ]:


data["ideal_diet"].head(20)


# In[ ]:


data["diet_current"].head(20)

