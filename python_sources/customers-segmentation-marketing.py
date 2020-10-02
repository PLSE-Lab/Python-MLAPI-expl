#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import re

py.init_notebook_mode(connected=False)


# In[ ]:


def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


# # Load Dataset

# In[ ]:


df_customers = pd.read_csv("../input/olist_customers_dataset.csv")
df_orders = pd.read_csv("../input/olist_orders_dataset.csv")
df_order_payments = pd.read_csv("../input/olist_order_payments_dataset.csv")


# In[ ]:


df_order_payments = df_order_payments.groupby("order_id").agg({"payment_value": "sum"}).reset_index()

df_tmp = pd.merge(df_orders, df_order_payments, on=["order_id"], how="inner")
df = pd.merge(df_tmp, df_customers, on=["customer_id"], how="inner")

cond = df["order_status"] == "delivered"
df = df.loc[cond]

df["order_purchase_date"] = df["order_purchase_timestamp"].str.slice(0, 10)

df["order_purchase_date"] = pd.to_datetime(df["order_purchase_date"], format="%Y-%m-%d")


# # Clustering 
# 
# The function performs the creation of a hierarchical clustering defined by the following rules:
# 
# **Segment**
# 
# >  **incative**: Customer who did not make any purchases or that their last purchase was more than three times the period defined as a parameter.
# 
# > **cold**: Customer that your last purchase was greater than twice period and less than three times the period.
# 
# > **hot**: Customer that your last purchase was greater than the period and less than twice the period.
# 
# > **active**: Customer that your last purchase is less than the period.
# 
# 
# **Sub segment**
# 
# >  **incative**: Customer who did not make any purchases or that their last purchase was more than three times the period defined as a parameter.
# 
# > **cold_high_payment_value**: Customer that segment is equal to cold and have purchases greater than or equal to the median of all purchases.
# 
# > **cold_low_payment_value**: Customer that segment is equal to cold and have purchases less than the median of all purchases.
# 
# > **hot_high_payment_value**: Customer that segment is equal to hot and have purchases greater than or equal to the median of all purchases.
# 
# > **hot_low_payment_value**: Customer that the segment is equal to hot and have purchases less than the median of all purchases.
# 
# > **active_high_payment_value**: Customer segment is equal to active and have purchases greater than or equal to the median of all purchases.
# 
# > **active_low_payment_value**: Customer segment is equal to active and have less than the median purchases of all purchases.
# 
# 
# **New customer**
# 
# > **new_customer**: Clientes que a primeira compra for menor que duas vezes o periodo.
# 
# 
# 
# *period*: number of days
# 
# 
# 

# In[ ]:


def clustering_customers(df, date_max, date_min=False, group_range_days=False):
    df = df.copy()

    if(date_min == False):
        cond_f = df["order_purchase_date"] <= pd.to_datetime(date_max)
    else:
        cond_1 = df["order_purchase_date"] <= pd.to_datetime(date_max)
        cond_2 = df["order_purchase_date"] >= pd.to_datetime(date_min)
        cond_f = cond_1 & cond_2

    df = df.loc[cond_f]

    df["today"] = df["order_purchase_date"].max()

    df["today"] = df["today"].dt.date
    df["today"] = pd.to_datetime(df["today"], format="%Y-%m-%d")

    df["order_purchase_days_since"] = df["today"]  - df["order_purchase_date"]
    df["order_purchase_days_since"] = df["order_purchase_days_since"].astype(str)
    df["order_purchase_days_since"] = df["order_purchase_days_since"].str.replace(r'\s+days.*', '', regex=True)
    df["order_purchase_days_since"] = df["order_purchase_days_since"].astype(int)
    df["order_purchase_year"] = df["order_purchase_date"].dt.year

    agg_group = {
        "order_purchase_days_since": ["min", "max", "count"],
        "payment_value": ["sum","mean"]
    }

    df_group = df.groupby(["customer_unique_id"]).agg(agg_group).reset_index()

    df_group.columns = [' '.join(col).strip() for col in df_group.columns.values]

    columns_rename = {
        "order_purchase_days_since min": "first_order_purchase",
        "order_purchase_days_since max": "last_order_purchase",
        "order_purchase_days_since count": "order_purchase_qty",
        "payment_value mean": "payment_value_mean",
        "payment_value sum": "payment_value_sum"
    }

    df_group.rename(columns_rename, axis=1, inplace=True)

    median_payment = df_group["payment_value_mean"].median()

    if(group_range_days == False):
        major_group = 4

        range_days = str(df["order_purchase_date"].max() - df["order_purchase_date"].min()) 
        group_range_days = int(re.sub(r'\s+days.*', '', range_days))/major_group

    cond_payment_zero = df_group['payment_value_mean'] == 0.0

    cond_inactive_1 = df_group['last_order_purchase'] > group_range_days*3
    cond_inactive = cond_inactive_1 | cond_payment_zero

    cond_cold_1 = df_group['last_order_purchase'] > group_range_days*2
    cond_cold_2 = df_group['last_order_purchase'] <= group_range_days*3
    cond_cold = cond_cold_1 & cond_cold_2 & ~(cond_payment_zero)

    cond_hot_1 = df_group['last_order_purchase'] > group_range_days
    cond_hot_2 = df_group['last_order_purchase'] <= group_range_days*2
    cond_hot = cond_hot_1 & cond_hot_2 & ~(cond_payment_zero)

    cond_active_1 = df_group['last_order_purchase'] <= group_range_days
    cond_active = cond_active_1 & ~(cond_payment_zero)

    df_group.loc[cond_inactive, "segment"] = "inactive"
    df_group.loc[cond_cold, "segment"] = "cold"
    df_group.loc[cond_hot, "segment"] = "hot"
    df_group.loc[cond_active, "segment"] = "active"

    cond_hot_high_payment_1 = df_group["segment"] == "hot"
    cond_hot_high_payment_2 = df_group["payment_value_mean"] >= median_payment
    cond_hot_high_payment = cond_hot_high_payment_1 & cond_hot_high_payment_2

    cond_hot_low_payment_1 = df_group["segment"] == "hot"
    cond_hot_low_payment_2 = df_group["payment_value_mean"] < median_payment
    cond_hot_low_payment = cond_hot_low_payment_1 & cond_hot_low_payment_2

    cond_active_high_payment_1 = df_group["segment"] == "active"
    cond_active_high_payment_2 = df_group["payment_value_mean"] >= median_payment
    cond_active_high_payment = cond_active_high_payment_1 & cond_active_high_payment_2

    cond_active_low_payment_1 = df_group["segment"] == "active"
    cond_active_low_payment_2 = df_group["payment_value_mean"] < median_payment
    cond_active_low_payment = cond_active_low_payment_1 & cond_active_low_payment_2

    cond_cold_high_payment_1 = df_group["segment"] == "cold"
    cond_cold_high_payment_2 = df_group["payment_value_mean"] >= median_payment
    cond_cold_high_payment = cond_cold_high_payment_1 & cond_cold_high_payment_2

    cond_cold_low_payment_1 = df_group["segment"] == "cold"
    cond_cold_low_payment_2 = df_group["payment_value_mean"] < median_payment
    cond_cold_low_payment = cond_cold_low_payment_1 & cond_cold_low_payment_2

    df_group["sub_segment"] = "inactive"
    df_group.loc[cond_hot_high_payment, "sub_segment"] = "hot_high_payment_value"
    df_group.loc[cond_hot_low_payment, "sub_segment"] = "hot_low_payment_value"
    df_group.loc[cond_active_high_payment, "sub_segment"] = "active_high_payment_value"
    df_group.loc[cond_active_low_payment, "sub_segment"] = "active_low_payment_value"
    df_group.loc[cond_cold_high_payment, "sub_segment"] = "cold_high_payment_value"
    df_group.loc[cond_cold_low_payment, "sub_segment"] = "cold_low_payment_value"

    cond_new_customer = df_group["first_order_purchase"] <= group_range_days*2
    df_group["new_customer"] = 0
    df_group.loc[cond_new_customer, "new_customer"] = 1
    
    return group_range_days, df_group


# # Descriptive Analysis

# In[ ]:


period_2018, df_clustering_2018 = clustering_customers(df, "2018-12-31")

df_clustering_2018.head()


# **`Faturamento por segmento 2018`**
# 
# 

# In[ ]:


df_revenue_subsegment_2018 = df_clustering_2018.groupby(["sub_segment"]).agg({"payment_value_sum": "sum"}).reset_index()


# In[ ]:


configure_plotly_browser_state()
trace0 = go.Bar(
    x=df_revenue_subsegment_2018["sub_segment"].values,
    y=df_revenue_subsegment_2018["payment_value_sum"].values,
    marker=dict(
        color=['rgba(36,123,160,1)', 
               'rgba(75,147,177,1)',
               'rgba(112,193,179,1)', 
               'rgba(138,204,192,1)',
               'rgba(243,255,189,1)',
               'rgba(247,255,213,1)',
               'rgba(255,22,84,1)']),
)

data = [trace0]

layout = go.Layout(
    title='Revenue 2018',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# **`Repurchase Amount by Segment 2018`**

# In[ ]:


df_clustering_2018_qty = df_clustering_2018.loc[df_clustering_2018["order_purchase_qty"] > 1]

df_qty_subsegment_2018 = df_clustering_2018_qty.groupby(["sub_segment"]).agg({"order_purchase_qty": "count"}).reset_index()


# In[ ]:


configure_plotly_browser_state()
trace0 = go.Bar(
    x=df_qty_subsegment_2018["sub_segment"].values,
    y=df_qty_subsegment_2018["order_purchase_qty"].values,
    marker=dict(
        color=['rgba(36,123,160,1)', 
               'rgba(75,147,177,1)',
               'rgba(112,193,179,1)', 
               'rgba(138,204,192,1)',
               'rgba(243,255,189,1)',
               'rgba(247,255,213,1)',
               'rgba(255,22,84,1)']),
)

data = [trace0]

layout = go.Layout(
    title='Repurchase Amount 2018',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# **`Average days between first and last purchase by segment 2018`**

# In[ ]:


df_days_repurchase_subsegment_2018 = df_clustering_2018_qty.groupby(["sub_segment"]).agg({"first_order_purchase": "mean", "last_order_purchase": "mean"}).reset_index()

df_days_repurchase_subsegment_2018["diff_order_purchase"] = df_days_repurchase_subsegment_2018["last_order_purchase"].values - df_days_repurchase_subsegment_2018["first_order_purchase"].values

df_days_repurchase_subsegment_2018["diff_order_purchase"] = df_days_repurchase_subsegment_2018["diff_order_purchase"].round(0)


# In[ ]:


configure_plotly_browser_state()
trace0 = go.Bar(
    x=df_days_repurchase_subsegment_2018["sub_segment"].values,
    y=df_days_repurchase_subsegment_2018["diff_order_purchase"].values,
    marker=dict(
        color=['rgba(36,123,160,1)', 
               'rgba(75,147,177,1)',
               'rgba(112,193,179,1)', 
               'rgba(138,204,192,1)',
               'rgba(243,255,189,1)',
               'rgba(247,255,213,1)',
               'rgba(255,22,84,1)']),
)

data = [trace0]

layout = go.Layout(
    title='Avg days between first and last purchase',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# 
# 
# ---
# 
# 

# In[ ]:


period_2017, df_clustering_2017 = clustering_customers(df, "2017-12-31")

df_clustering_2017.head()


# # Life Time Value

# In[ ]:


df_clutering_2017_2018 =  pd.merge(df_clustering_2017, df_clustering_2018, left_on=["customer_unique_id"], right_on=["customer_unique_id"], how="inner")

categories = ["inactive",
              "hot_high_payment_value",
              "hot_low_payment_value",
              "active_high_payment_value",
              "active_low_payment_value",
              "cold_high_payment_value",
              "cold_low_payment_value"
             ]

x = pd.Categorical(df_clutering_2017_2018["sub_segment_x"].tolist(), categories=categories)
y = pd.Categorical(df_clutering_2017_2018["sub_segment_y"].tolist(), categories=categories)

df_cross_2017_2018 =  pd.crosstab(x,y)
df_cross_2017_2018 = df_cross_2017_2018.reindex(index=categories, columns=categories, fill_value=0.0)

df_cross_2017_2018


# **`Transition Matrix`**

# In[ ]:


df_transition = df_cross_2017_2018.div(df_cross_2017_2018.sum(axis=1), axis=0)

df_transition


# **`Predicting Customer Transition`**

# In[ ]:


years = ["2019","2020","2021","2022","2023"]

df_seg =  pd.DataFrame(index=categories, columns=years)

df_seg["seg_tmp"] = df_seg.index

df_clustering_2018_count = df_clustering_2018.groupby(["sub_segment"]).count().reset_index()

df_seg = pd.merge(df_seg, df_clustering_2018_count[["sub_segment", "customer_unique_id"]], left_on=["seg_tmp"], right_on=["sub_segment"], how="left")

df_seg.drop(["sub_segment", "seg_tmp"], axis=1, inplace=True)

df_seg.rename({"customer_unique_id": "2018"}, axis=1, inplace=True)

df_seg["2018"].fillna(0.0, inplace=True)

df_seg.index = categories

df_seg["2019"] = np.dot(df_seg["2018"].values, df_transition.values)
df_seg["2020"] = np.dot(df_seg["2019"].values, df_transition.values)
df_seg["2021"] = np.dot(df_seg["2020"].values, df_transition.values)
df_seg["2022"] = np.dot(df_seg["2021"].values, df_transition.values)
df_seg["2023"] = np.dot(df_seg["2022"].values, df_transition.values)

df_seg = df_seg[["2018"] + years].round(0)

df_seg


# **`Predicting Revenue`**

# In[ ]:


_, df_clustering_only_2018 = clustering_customers(df, "2018-12-31", "2018-01-01", period_2018)

df_revenue_only_2018 = df_clustering_only_2018.groupby(["sub_segment"]).agg({"payment_value_sum": "mean"}).reset_index()

df_revenue_only_2018


# In[ ]:


df_seg["seg_tmp"] = df_seg.index

df_seg_revenue = pd.merge(df_seg, df_revenue_only_2018, left_on=["seg_tmp"], right_on=["sub_segment"], how="left")

df_seg_revenue["payment_value_sum"].fillna(0.0, inplace=True)

df_seg_revenue.index = df_seg_revenue["seg_tmp"].values

df_seg_revenue["2018"] = df_seg_revenue["2018"].values * df_seg_revenue["payment_value_sum"].values
df_seg_revenue["2019"] = df_seg_revenue["2019"].values * df_seg_revenue["payment_value_sum"].values
df_seg_revenue["2020"] = df_seg_revenue["2020"].values * df_seg_revenue["payment_value_sum"].values
df_seg_revenue["2021"] = df_seg_revenue["2021"].values * df_seg_revenue["payment_value_sum"].values
df_seg_revenue["2022"] = df_seg_revenue["2022"].values * df_seg_revenue["payment_value_sum"].values
df_seg_revenue["2023"] = df_seg_revenue["2023"].values * df_seg_revenue["payment_value_sum"].values

df_seg_revenue = df_seg_revenue.round(2)

df_seg_revenue.drop(["sub_segment", "seg_tmp", "payment_value_sum"], axis=1, inplace=True)

df_seg_revenue


# **`How much is the customer base in the year 2023?`**
# 
# 
# 
# > To perform this calculation were considered a discount factor of 10%. Of course, there are several variables that were not considered.
# 
# 

# In[ ]:


df_seg_revenue_sum = df_seg_revenue.sum(axis=0)

discount = []
discount_rate = 0.10
for i in range(0,len(years)+1):
    discount.append(1 / ((1 + discount_rate)**i))

dis_revenue = df_seg_revenue_sum.values * discount

print("2023 - R$",round(dis_revenue.cumsum()[5] - df_seg_revenue_sum.iloc[0], 2))


# # Conclusion
# 
# 
# 
# > I believe that the analysis can be useful at a certain point, but there are several issues that can not be clearly defined because the database is relatively young to perform certain analysis.
# 

# # Work in Progress
# 
# 
# 
# 1.   Perform a clusterization using machine learning with more information than the database has.
# 2.   Extract profiles from the clustering performed
# 
# 
