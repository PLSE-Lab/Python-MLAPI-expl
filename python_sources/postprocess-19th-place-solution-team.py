#!/usr/bin/env python
# coding: utf-8

# # 19th place solution (postprocess)
# boost our score +0.0007 in Public and +0.0004 in Private.<br>
# This notebook use this great notebook's submittion. <br>
# (I study a lot from this notebook and other @Konstantin Yakovlev's notebooks. thanks a lot!<br>
# and congratuation!!) <br> 
# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again <br>
# boost this notebook's score +0.0042 in Public and +0.0032 in Private <br>

# In[ ]:





# In[ ]:


import pandas as pd
import datetime


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')


# In[ ]:


submit = pd.read_csv("../input/ieee-gb-2-make-amount-useful-again/submission.csv")


# In[ ]:


def make_day(df):
    def fillna(x):
        if "nan" in x:
            return np.nan
        else:
            return x
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    df['Date'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
    df['Day'] = (df["Date"].dt.year - 2017) * 365 + df["Date"].dt.dayofyear
    df["ID1"] = df["card1"].astype(str)+df["card2"].astype(str)+df["card3"].astype(str)+df["card4"].astype(str)+                df["card5"].astype(str)+df["card6"].astype(str)
    df["ID_D1"] = fillna(df["ID1"].astype(str) + (df["Day"] - df["D1"]).astype(str))
    df["ID_D1D10"] = fillna(df["ID_D1"].astype(str) + (df["Day"] - df["D10"]).astype(str))
    df["ID_D1D12"] = fillna(df["ID_D1"].astype(str) + (df["Day"] - df["D12"]).astype(str))
    return df


# In[ ]:


train_transaction = make_day(train_transaction)
test_transaction = make_day(test_transaction)


# In[ ]:


q_id1 = train_transaction[["ID_D1D10", "isFraud"]].groupby("ID_D1D10").agg({"isFraud": ["count", "mean"]}).reset_index()
q_id1.columns = ["ID_D1D10", "isFraud_countD1D10", "isFraud_meanD1D10"]
q_id2 = train_transaction[["ID_D1D10", "isFraud"]].groupby("ID_D1D10").agg({"isFraud": ["count", "mean"]}).reset_index() # I found this bug after competition finished...
q_id2.columns = ["ID_D1D12", "isFraud_countD1D12", "isFraud_meanD1D12"]


# In[ ]:


test_transaction = pd.merge(test_transaction, q_id1, how="left", on="ID_D1D10")
test_transaction = pd.merge(test_transaction, q_id2, how="left", on="ID_D1D12")


# In[ ]:


submit = pd.merge(submit, test_transaction, how="left", on="TransactionID")


# # fix submittion file to isFraud=1

# In[ ]:


q = "isFraud_countD1D10 > 1 and isFraud_meanD1D10 > 0.7 and ProductCD != 'C'"


# In[ ]:


submit.query(q)["isFraud"].describe()


# In[ ]:


submit["isFraud"][submit.query(q).index] = 1


# In[ ]:


submit.query(q)["isFraud"].describe()


# In[ ]:


q = "isFraud_countD1D12 > 1 and isFraud_meanD1D12 == 1 and ProductCD == 'C'"


# In[ ]:


submit.query(q)["isFraud"].describe()


# In[ ]:


submit["isFraud"][submit.query(q).index] = 1
submit.query(q)["isFraud"].describe()


# In[ ]:





# # fix submittion file to isFraud=0

# In[ ]:


q = "isFraud_countD1D10 > 5 and isFraud_meanD1D10 == 0"


# In[ ]:


submit.query(q)["isFraud"].describe()


# In[ ]:


submit["isFraud"][submit.query(q).index] = 0


# In[ ]:


submit.query(q)["isFraud"].describe()


# In[ ]:


submit[["TransactionID", "isFraud"]].to_csv("postprocessed.csv", index=False)

