#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hello there! In this notebook I am going to explains simple linera regression with some awful facts. Some of the beginners should follow this notebook to get insight of simple linear regression. (Dont forget to upvote)
# 
# ![LinearRegression](https://i.ytimg.com/vi/nk2CQITm_eo/maxresdefault.jpg)
# 
# 
# So first we will start with the breif introduction of linear regression.
# 
# So Linear Regression is the way to find out relationship between two variables where 1 is dependent and the other 1 is 
# independent. SO the question is how can we find the realtion?
# 
# Lets take and example:
# Assume that A hotel manger came to you and ask you to make a model for tip prediction model. Now you went to the hotel for data collection and collect data (amount of tip) of 6 random orders. 
# 
# | Order_id  | TIP($) |
# | ------------- | ------------- |
# | 1  | 5  |
# | 2  | 7  |
# | 3  | 11  |
# | 4  | 12  |
# | 5  | 1 |
# | 6  | 5  |
# | 7  | 11  |
# 
# So if is say you to predict next tip with only just 1 variable. How will you predict? (take some time and think)
# 
# Lets Visualize this data

# In[ ]:


import pandas as pd

df = pd.DataFrame({
    "Order_id" : [i for i in range(1,8)],
    "TIP($)"   : [5, 7, 11, 12, 1, 5, 11]
})

import plotly.express as px

fig = px.line(df, x="Order_id", y="TIP($)", title='Tip Calulation')
fig.show()


# So here is the visualisation Now go ahead and try to give me a value for the 7th Tip.
# with only one variable the best way to predict next point is by taking its mean so the next point (amount of tip) will be $6.8.
# 
#  [5 + 7 + 11 + 12 + 1 + 5 + 11] / 7 = 7.42
#  
# with only one variable best possible answer is 6.88.
# 
# 
# 

# In[ ]:




import plotly.graph_objects as go

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=list(df["TIP($)"].values),
                    mode='lines+markers',
                    name='TIP'))
fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=[(sum(df["TIP($)"].values)/ len(df["TIP($)"].values)) for i in range(0,len(df["TIP($)"].values))],
                    mode='lines',
                    name='Mean'))

fig.update_layout(title='Relation in Tips and Mean')
                   
fig.show()




# Now we have to calculate error of this mean line which is our best fit line for now. To find out the error of the line
# we have to calcualte distance of every datapoint from the mean point and add it. so lets see our error.

# In[ ]:


df["distance"]= df["TIP($)"].apply(lambda x: x - 6.8)
print(f"Error of Best Fit Line :{sum(df['distance'].values)}")


# Now lets find second variable for which we can get more information and to be more accurate. 
# 
# Now you again go to manager and manager is happy with your work but after some time when data get increased manager see some anomolies and invite you again to build more accurate model.Now you go again and started to view more orders you see that order bill is relating to the amount of tip. When the order is of 5 dollar amount of tip paid is 1 dollar and when order bill is 50 dollar the amount of tip is 5 dollar you got the variable on which amount is tip is depending on Now you collect the data of the order bills with tip amount now you have 2 variables and when we have more data we have chance to build more accurate model. You come back to your office and start working on the data again.
# 
# 
# 
# | Order_id  | TIP($) | Bill |
# | ------------- | ------------- | ------------- |
# | 1  | 5  |  50  |
# | 2  | 7  |  56.5  |
# | 3  | 11  |  100  |
# | 4  | 12  |  110  |
# | 5  | 1 |  5  |
# | 6  | 5  |  52  |
# | 7  | 11  |  101  |
# 
# 

# In[ ]:


df = pd.DataFrame({
    "Order_id" : [i for i in range(1,8)],
    "TIP($)"   : [5, 7, 11, 12, 1, 5, 11],
    "Bill"   : [50, 56.5 ,100, 110, 5, 52, 101]
})


import plotly.graph_objects as go

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=list(df["TIP($)"].values),
                    mode='lines+markers',
                    name='TIP'))
fig.add_trace(go.Scatter(x=list(df.Order_id.values), y=list(df.Bill.values),
                    mode='lines',
                    name='Mean'))

fig.update_layout(title='Relation in Tips and Mean')
                   
fig.show()


# # Linear Regression Forluma
# 
# Linear Regression formulat =  h(x) = beta _0 + (beta_1 * x)  
# 
# # Error Calculation
# 
# Here, e_i is residual error in ith observation.
# So, our aim is to minimize the total residual error.
# 
# ![Screenshot%202020-06-28%20at%207.39.00%20PM.png](attachment:Screenshot%202020-06-28%20at%207.39.00%20PM.png)

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x     # ss  stands for sum squared
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
    
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show()
    

    
def Linear_regression(x,y):
        # estimating coefficients 
        
        b = estimate_coef(x, y) 
        # Here b[0] is error and b[1] is the value which we got when we multiply it with x (amount bill) to get the amount of tip
        
        print("Estimated coefficients:\nb_0 = {} nb_1 = {}".format(b[0], b[1])) 
        print(b)

        # plotting regression line 
        plot_regression_line(x, y, b) 
        
Linear_regression(df["Bill"],df["TIP($)"])


# Now Linear regression is used to find out relation between two variabe from these linear regression values we can conclude that Bills have some connection with tip because our error is minimized on the other side if our error is not minimized it means that these two variables are not related to each other

# # Hope you find some intresting things here I will try to make next notebook on Multiple linear Regression Dont forget to upvote

# In[ ]:




