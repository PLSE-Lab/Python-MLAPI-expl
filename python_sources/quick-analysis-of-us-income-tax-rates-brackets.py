#!/usr/bin/env python
# coding: utf-8

# # Quick Analysis of US Income Tax Rates & Brackets [Tableau]

# [Firat Gonen](https://www.kaggle.com/frtgnn) recently uploaded the [Historical Income Tax Rates & Brackets](https://www.kaggle.com/frtgnn/historical-income-tax-rates-brackets) dataset.
# 
# Here is a quick analysis using Tableau. I've added the viz screenshots in this notebook. 
# 
# [Link to the visualizations on Tableau Public](https://public.tableau.com/shared/Y3QWDZJSQ?:display_count=y&:origin=viz_share_link)

# ### This is the dataset.

# In[ ]:


import pandas as pd
df = pd.read_csv("/kaggle/input/historical-income-tax-rates-brackets/tax_over_years.csv",sep=";")
df.drop(['Unnamed: 5','Unnamed: 6'],axis=1,inplace=True)
df.head()


# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt

img1 = mpimg.imread('/kaggle/input/viz-screenshots/1.jpg')
img2 = mpimg.imread('/kaggle/input/viz-screenshots/Annotation 2020-05-30 062108.jpg')
img3 = mpimg.imread('/kaggle/input/viz-screenshots/3.jpg')
img4 = mpimg.imread('/kaggle/input/viz-screenshots/4.jpg')
img5 = mpimg.imread('/kaggle/input/viz-screenshots/5.jpg')
img6 = mpimg.imread('/kaggle/input/viz-screenshots/6.jpg')


# ### Let's first create a basic line plot visualizing the Top and Bottom Taxable Bracket Income Change over the years.

# In[ ]:


plt.figure(figsize=(15,10))
plt.imshow(img1)
plt.axis('off')


# * Top Bracket is in 'Millions' whereas Bottom Bracket is in 'Thousands'.
# 
# 
# * Top Bracket seems to have stayed below 1,000,000 for most of the years. Bottom Bracket mostly stayed below 10,000.
# 
# 
# * The **Top Bracket went up to 10,000,000 in 1937!** I wonder why. The Great Depression and World War II? 
# 
# 
# * Bottom Bracket went up to 45,000, and then fell sharply. Here's why:
# "During the 1990s, the top rate jumped to 39.6 percent. However, the Economic Growth and Tax Relief and Reconciliation Act of 2001 dropped the highest income tax rate to 35 percent from 2003 to 2010. The Tax Relief, Unemployment Insurance Reauthorization, and Job Creation Act of 2010 maintained the 35 percent tax rate through 2012." [Link](http://https://bradfordtaxinstitute.com/Free_Resources/Federal-Income-Tax-Rates.aspx)

# ### In 1990, the Top Bracket seems very low. Let's have a closer look.

# In[ ]:


plt.figure(figsize=(15,10))
plt.imshow(img2)
plt.axis('off')


# * Top and Bottom Brackets had same values from 1988-1990. 
# 
# "The Economic Recovery Tax Act of 1981 slashed the highest rate from 70 to 50 percent, and indexed the brackets for inflation.
#  
# Then, the Tax Reform Act of 1986, claiming that it was a two-tiered flat tax, expanded the tax base and dropped the top rate to 28 percent for tax years beginning in 1988.4 The hype here was that the broader base contained fewer deductions, but brought in the same revenue. Further, lawmakers claimed that they would never have to raise the 28 percent top rate.
# 
# The 28 percent top rate promise lasted three years before it was broken."
# 
# [Link](http://https://bradfordtaxinstitute.com/Free_Resources/Federal-Income-Tax-Rates.aspx)
# 
# * The forecast shows Top Bracket will most likely keep on increasing while the Bottom Bracket will remain steady for the next few years.

# ### Now let's see how much the Taxable Income changes with a change in the Bracket Rate%

# In[ ]:


plt.figure(figsize=(15,10))
plt.imshow(img3)
plt.axis('off')


# * Changes in the Top Bracket Rate% between 77% and 81.1% had the most dramatic change.
# 
# * Changing the Rate from 77% to 79% increased the Taxable Income by 18.4 Million! 
# 
# * On the other hand, switching from 79% to 81% lowered the Taxable Income by 15 Million.
# 
# * Switching from 81% to 81.1% had no change.

# In[ ]:


plt.figure(figsize=(15,10))
plt.imshow(img4)
plt.axis('off')


# * First of all, notice that Bottom Bracket Rate% has much lower values (and no. of values) than Top Bracket Rate%.
# 
# * Changing the Bottom Bracket Rate% from 6% to 10% and from 14% to 15% noted a very high increase in Bottom Bracket Taxable Income.
# 
# * Changing the Bottom Bracket Rate% from 10% to 11% and from 15% to 16% resulted in a significant decrease in the Taxable Income.

# ### Now, let's see if there's any significant relationship between Top and Bottom Bracket Taxable Income values.

# In[ ]:


plt.figure(figsize=(15,10))
plt.imshow(img5)
plt.axis('off')


# * Apparently not. But, notice that most of the values for the years 2000-2020 lie in a small cluster. Top Bracket is lower than 1M and Bottom Bracket lies between 10K-20K.
# 
# * This is different than the earlier years (before the 1980s) where the Bottom Bracket was much lower (below 5K).

# In[ ]:


plt.figure(figsize=(15,10))
plt.imshow(img6)
plt.axis('off')


# Earlier we noticed that the two values had no significant relationship. But when you cluster them, things change.
# 
# * The red and blue cluster show positive relationship, i.e., for an increase in Bottom Bracket, there is an increase in the Top Bracket. 
# 
# * The red and blue clusters lie in the lower left of the scatter plot. Top Bracket - Lower than 2,000,000 | Bottom Bracket - lower than 20,000.
# 
# * The other values must be outliers. The ones in the top of the plot definitely are. 
# 
# * If you are going for regression, considering only the red and blue clusters might help.
