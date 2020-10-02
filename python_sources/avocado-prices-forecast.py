#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install nb_black -q')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# # Import libraries and dataset

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/avocado-prices/avocado.csv")
data = data.sort_values("Date")
data.reset_index(inplace=True)
data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


data.groupby(["type", "year"]).describe()["AveragePrice"]


# ## Sample counts per year

# In[ ]:


import plotly.express as px

aux = pd.DataFrame(data.year.value_counts())
aux.columns = ["sample counts"]
aux["year"] = aux.index
aux

fig = px.bar(
    aux, y="sample counts", x="year", text="sample counts", color="sample counts"
)
fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
fig.show()


# ## Cleaning dataset 

# In[ ]:


data.drop(
    [
        "index",
        "Unnamed: 0",
        "Total Volume",
        "4046",
        "4225",
        "4770",
        "Total Bags",
        "Small Bags",
        "Large Bags",
        "XLarge Bags",
    ],
    inplace=True,
    axis=1,
)

data.Date = pd.to_datetime(data.Date)


# ## Transforming - grouping and separe by type (organic and conventional)

# In[ ]:


data_org = data[data.type.isin(["organic"])]
data_org_final = data_org.groupby(["Date"]).mean()
data_org_final["Date"] = data_org_final.index
data_org_final["Type"] = "organic"

data_con = data[data.type.isin(["conventional"])]
data_con_final = data_con.groupby(["Date"]).mean()
data_con_final["Date"] = data_con_final.index
data_con_final["Type"] = "conventional"

data_final = pd.concat([data_con_final, data_org_final])


# In[ ]:


import plotly.express as px

fig = px.line(
    data_final, x="Date", y="AveragePrice", color="Type", title="Avocado Price"
)
fig.show()


# In[ ]:


import plotly.express as px


fig = px.box(
    data,
    x="year",
    y="AveragePrice",
    color="type",
    title="Avocado Price per year and type",
)
fig.update_traces(quartilemethod="exclusive")
fig.show()


# # Model

# ## Preparing the data for the model

# In[ ]:


data_model_avocado_org = data_org_final[["Date", "AveragePrice"]]
data_model_avocado_org.reset_index(inplace=True, drop=True)
data_model_avocado_org.columns = ["ds", "y"]

data_model_avocado_con = data_con_final[["Date", "AveragePrice"]]
data_model_avocado_con.reset_index(inplace=True, drop=True)
data_model_avocado_con.columns = ["ds", "y"]


# ## Model function and print

# In[ ]:


from fbprophet import Prophet
import plotly.graph_objects as go


def train_predict(data, periods, kind):

    model = Prophet(yearly_seasonality=True)
    model.fit(data[:-periods])

    future = model.make_future_dataframe(
        periods=periods, freq="W", include_history=True
    )
    forecast = model.predict(future)
    print(forecast.shape, data.shape)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predict Values"
        )
    )
    fig.add_trace(
        go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines", name="Trend")
    )
    fig.add_trace(
        go.Scatter(x=data["ds"], y=data["y"], mode="lines", name="Real Values",)
    )

    fig.update_layout(
        title_text=f"Comperating the real x predicted for {kind}",
        yaxis_title="Avocado Price",
        xaxis_title="Date",
    )

    fig.show()


# In[ ]:


train_predict(data_model_avocado_org, 4, "Organic")


# In[ ]:


train_predict(data_model_avocado_con, 4, "Conventional")

