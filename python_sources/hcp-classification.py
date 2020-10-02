#!/usr/bin/env python
# coding: utf-8

# # UCSanDiegoX: DSE200x Python for Data Science
# 
# ## Week 10 - Final Project
# 
# **Note:** This notebook runs on Kaggle, there's no need to download data.
# 
# ### Step 1:  Find a dataset or datasets
#  Prescription-based prediction dataset can be found at Kaggle.com
#  https://www.kaggle.com/roamresearch/prescriptionbasedprediction
#  
#  This dataset is a compilation from medicare Part D and other datasets from the US department of health and human services.
# 

# In[ ]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2:  Explore the dataset(s)
# 
# **Read in dataset**
# 
# The file is in JSONL format (one JSON record per line):

# In[ ]:


data = pd.read_json("../input/roam_prescription_based_prediction.jsonl", lines=True)
data.shape


# In[ ]:


data.head()


# **Data Preparation**

# In[ ]:


Rx = pd.DataFrame([v for v in data["cms_prescription_counts"]])
Rx.shape


# In[ ]:


Rx.head()


# In[ ]:


Rx.sum().sort_values(ascending=False).head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Presctiption Counts')


# In[ ]:


hcp = pd.DataFrame([v for v in data["provider_variables"]])
hcp.shape


# In[ ]:


hcp.head()


# In[ ]:


hcp['gender'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Gender')


# In[ ]:


hcp['region'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Region')


# In[ ]:


hcp['settlement_type'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Settlement Type')


# In[ ]:


hcp['specialty'].value_counts().sort_values(ascending=False).head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Provider Specialties')


# In[ ]:


hcp['years_practicing'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Years in Practice')


# ### Step 3:  Identify 1-3 research questions
# 
# Can we use providers attributes such as gender, specialty and years of practice to predict their likelihood of prescribing a given medication?

# ### Step 5:  Identify your research methods
# 
# Humira is one of the best selling medication, I'll use it to set up a lebel for the classification algorithm. I will use providers variables as features.

# In[ ]:


Rx_Humira = Rx[['HUMIRA']]
Rx_Humira.shape


# In[ ]:


Rx_Humira.head()


# In[ ]:


clean_data = Rx_Humira.copy()
clean_data['prescribe_label'] = (clean_data['HUMIRA'] > 1)*1
print(clean_data[['HUMIRA', 'prescribe_label']])


# In[ ]:


y = clean_data[['prescribe_label']].copy()


# In[ ]:


clean_data['HUMIRA'].head()


# In[ ]:


y.head()


# In[ ]:


hcp = pd.DataFrame([v for v in data["provider_variables"]])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

lower = hcp['brand_name_rx_count'].mean() - hcp['brand_name_rx_count'].std()
upper = hcp['brand_name_rx_count'].mean() + hcp['brand_name_rx_count'].std()
hist_data = [x for x in hcp[:10000]['brand_name_rx_count'] if x>lower and x<upper ]


# hist_data = hcp['brand_name_rx_count']
print(len(hist_data))


# In[ ]:


plt.hist(hist_data, 10, normed=False, facecolor='green')

plt.xlabel('Brand Prescriptions')
plt.ylabel('Number of Prescribers')
plt.title('Prescribers Distribution')

plt.grid(True)

plt.show()


# To remove noise from data I will limit the providers universe to the following specialties and providers with 50 scripts or more.

# In[ ]:


specFilter = ['Rheumatology', 'Family', 'Medical', 'Adult Health', 'Procedural Dermatology', 
             'Geriatric Medicine', 'Acute Care', 'MOHS-Micrographic Surgery', 'Allergy & Immunology',
             'Cardiovascular Diseas', 'Clinical & Laboratory', 'Dermatological Immunology', 
             'Surgical Technologist', 'Dermatopathology', 'Hematology & Oncology']

filterMesh = (hcp['specialty'].isin(specFilter)) & (hcp['brand_name_rx_count'] >= 50)
hcp_features = hcp.loc[filterMesh]


# In[ ]:


hcp_features.shape


# In[ ]:


hcp_features.columns


# In[ ]:


X = pd.get_dummies(hcp_features)
X.columns


# In[ ]:


X.shape


# In[ ]:


y = pd.merge(y, X, left_index = True, right_index = True)[['prescribe_label']]
y.head()


# In[ ]:


y.columns


# In[ ]:


y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# In[ ]:


hcp_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
hcp_classifier.fit(X_train, y_train)


# In[ ]:


type(hcp_classifier)


# In[ ]:


predictions = hcp_classifier.predict(X_test)


# In[ ]:


predictions[:30]


# In[ ]:


y_test['prescribe_label'][:20]


# In[ ]:


accuracy_score(y_true = y_test, y_pred = predictions)


# In[ ]:


print (hcp_classifier)


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))


# In[ ]:


print("Precision: %0.4f" % metrics.precision_score(y_test, predictions))


# In[ ]:


print("Recall: %0.4f" % metrics.recall_score(y_test, predictions))


# In[ ]:


## Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)


# In[ ]:


## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




