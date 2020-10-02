#!/usr/bin/env python
# coding: utf-8

# # Avocado Price Prediction With Some Cool Packages
# 
# I figured this would be a good dataset to show off the functionality of a few functions/packages that I have been playing around with recently. A little background on each before we continue...
# 
# + Prophet
#   + Allows for some pretty easy time series prediction right out of the box!
# + sklearn TimeSeriesSplit
#    + Quickly split your data into chunks suitable for time series cross validation
# + Plotnine
#   + Plotting using syntax identical to R's ggplot
#   
# Let me know what you found helpful or confusing! I appreciate any feedback!!
# 
# ***
# 
# 
# 

# ## Loading Libraries

# In[ ]:


from plotnine import * #ggplot like library for python!!!!
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit #Splitting for time series CV!
from fbprophet import Prophet 


# ## Loading and Tidying Data

# In[ ]:


#Read in Data, Create New Frame With Relevant Columns
df=pd.read_csv("../input/avocado.csv")
#Filter To TotalUS and Conventional Avocados for simplicity
df=df.loc[np.logical_and(df.region=="TotalUS",df.type=="conventional"),:]
#Clean up data for analysis
df.Date=pd.to_datetime(df.Date)
date_price_df=df.loc[:,["Date","AveragePrice"]]
date_price_df.columns=["ds","y"]
date_price_df=date_price_df.sort_values("ds").reset_index(drop=True)


# ## Use TimeSeriesSplit and Prophet To Cross Validate
# 
# 

# In[ ]:


#Initialize Split Class, we'll split our data 5 times for cv
tscv = TimeSeriesSplit(n_splits=5)


# ### Lets Create A Function That Will Return A Tidy DataFrame With Predictions

# In[ ]:


def pro_ds_data_gen(df,tscv,yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False):
    out_df=pd.DataFrame()
    for i,(train_i,test_i) in enumerate(tscv.split(df)): #For Time Series Split
        #Use indexes to grab the correct data for this split
        train_df=df.copy().iloc[train_i,:]
        test_df=df.copy().iloc[test_i,:]
        #Build our model using prophet and make predictions on the test set
        model=Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        model.fit(train_df)
        predictions=model.predict(test_df)

        #Combine predictions and training into one df for plotting
        pred_df=predictions.loc[:,["ds","yhat"]]
        pred_df["y"]=test_df.y.tolist()
        train_df["train"]="Train"
        pred_df["train"]="Test"
        sub_df=train_df.append(pred_df).reset_index(drop=True)
        sub_df["split"]="Split "+str(i+1)
        sub_df["rmse"]=(np.mean((sub_df.yhat-sub_df.y)**2))**.5 #calculating rmse for the split
        out_df=out_df.append(sub_df).reset_index(drop=True)
    return out_df


# ## Visualizing TS Splits Easily With Plotnine

# In[ ]:


year_weak_seas_df=pro_ds_data_gen(
    date_price_df,tscv,yearly_seasonality=True,weekly_seasonality=True
)

(ggplot(year_weak_seas_df,aes("ds","y",color="factor(train)"))+ geom_point()+facet_grid('split~.'))+labs(title="Train/Test Splits",x="Date",y="Price")+scale_x_date(date_breaks="6 months",date_labels =  "%b %Y")


# #### Lets walk through this code step by step...
# 
# For those of you Python users unfamiliar with ggplot, I highly suggest you check it out. [This cheatsheet](https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf) is an awesome place to start learning the syntax. One of the major bennefits of ggplot is the ability to chain together commands seemlessly. Lets breakdown the previous line step by step.
# 
# + ggplot(total_plot_df,aes("ds","y",color="factor(train)"))
#   + We are using the total_plot_df as our data, the ds column as our x variable, the y column as our y variable, and coloring based on the train column
#   + factor(train) is converting the train column to a factor (R's version of a categorical variable)
# + geom_point()
#   + Signals we want a scatterplot
# + facet_grid('split~.'))
#   + This splits our data based on the split column and stacks the plots on top of each other
# + labs(title="Train/Test Splits",x="Date",y="Price")
#   + Updates Lables
# + scale_x_date(date_breaks="6 months",date_labels =  "%b %Y")
#   + Cleans up the x-axis so its readable!
# 
# 

# ## Visualizing Model Performance

# #### Lets create multiple models by adjusting which seasonality corrections we make. We can use the function we created above to get a dataframe including predictions for each model.

# In[ ]:


no_seas_df=pro_ds_data_gen(date_price_df,tscv,yearly_seasonality=False,weekly_seasonality=False)
year_seas_df=pro_ds_data_gen(date_price_df,tscv,weekly_seasonality=False)
week_seas_df=pro_ds_data_gen(date_price_df,tscv,yearly_seasonality=False)
df_dict={"year_weak":year_weak_seas_df,"none":no_seas_df,"year":year_seas_df,"week":week_seas_df}


# #### Now we can combine the relevent columns of each dataframe into one final frame for plotting!

# In[ ]:


cv_frame=pd.DataFrame()
for name,frame in df_dict.items():
    #grab the one unique rmse for each split
    values_lol=frame.groupby("split").agg({"rmse":"mean"}).values
    values=[item for sublist in values_lol for item in sublist] #returns 2D array with sub-length 1, so we cpllapse
    sub_df=pd.DataFrame({"rmse":values})
    sub_df["model"]=name
    cv_frame=cv_frame.append(sub_df)


# #### Finally, we can create a box plot to see what model performed the best!

# In[ ]:


(ggplot(cv_frame,aes(x="model",y="rmse",fill="model"))+geom_boxplot())


# #### It looks like only taking into account yearly seasonality yields us the most accurate model!

# 

# 

# 
