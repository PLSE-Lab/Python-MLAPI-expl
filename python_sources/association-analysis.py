#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd
cars93 = pd.read_csv("../input/Cars93.csv")


# In[ ]:


cars93.head()


# In[ ]:


cars93reduced = cars93[["Type","MPG.city","AirBags","DriveTrain","Cylinders",
                        "EngineSize","Horsepower","RPM","Rev.per.mile",
                        "Fuel.tank.capacity","Length","Wheelbase","Width",
                        "Turn.circle","Weight"]]


# In[ ]:


columns = ["MPG.city","EngineSize","Horsepower","RPM","Length","Wheelbase","Turn.circle","DriveTrain","Type","AirBags"]
cars93reduced[columns].head(10)


# In[ ]:


sns.pairplot(cars93reduced, vars=["Horsepower","RPM","Wheelbase","MPG.city"])


# In[ ]:


sns.pairplot(cars93reduced, vars=["Horsepower","RPM","EngineSize","MPG.city"])


# In[ ]:


sns.pairplot(cars93reduced,diag_kind="kde", vars=["Length","Wheelbase","Turn.circle"])


# In[ ]:


cars93["MPG.city"].mean()


# In[ ]:


cars93["EngineSize"].mean()


# In[ ]:


sum((cars93["MPG.city"] - (cars93["MPG.city"].mean())) * (cars93["EngineSize"] - (cars93["EngineSize"].mean()))/(cars93.shape[0]-1))


# In[ ]:


cars93["EngineSize"].cov(cars93["MPG.city"])


# In[ ]:


cars93["Horsepower"].cov(cars93["MPG.city"])


# In[ ]:


cars93["RPM"].cov(cars93["MPG.city"])


# In[ ]:


cars93["EngineSize"].cov(cars93["Horsepower"])


# In[ ]:


cars93["Length"].cov(cars93["Wheelbase"])


# In[ ]:


cars93["Length"].cov(cars93["Turn.circle"])


# In[ ]:


cars93["Wheelbase"].cov(cars93["Turn.circle"])


# In[ ]:


cars93[["MPG.city","EngineSize","Horsepower","RPM"]].cov()


# In[ ]:


from scipy.stats import shapiro


# In[ ]:


shapiro(cars93["Turn.circle"])


# In[ ]:


shapiro(cars93["Length"])


# In[ ]:


shapiro(cars93["Wheelbase"])


# In[ ]:


cars93["Length"].corr(cars93["Turn.circle"],method="pearson")


# In[ ]:


cars93["RPM"].corr(cars93["MPG.city"],method="pearson")


# In[ ]:


cars93["Wheelbase"].corr(cars93["Turn.circle"],method="pearson")


# In[ ]:


cars93["Length"].corr(cars93["Turn.circle"],method="pearson")


# In[ ]:


cars93["Horsepower"].corr(cars93["MPG.city"],method="spearman")


# In[ ]:


cars93["RPM"].corr(cars93["MPG.city"],method="spearman")


# In[ ]:


cars93["EngineSize"].corr(cars93["MPG.city"],method="spearman")


# In[ ]:


shapiro(cars93["Horsepower"])


# In[ ]:


shapiro(cars93["RPM"])


# In[ ]:


shapiro(cars93["EngineSize"])


# In[ ]:


shapiro(cars93["MPG.city"])


# In[ ]:


first10hp = cars93["Horsepower"].head(10)
first10hp = first10hp.sort_values(ascending=False)
rankfirst10hp = first10hp.rank()
print(list(rankfirst10hp))


# In[ ]:


# cars93 = cars93["RPM"].sort_values(ascending=False)
first25rpm = np.array(cars93["RPM"].head(25).rank())
first25mpg = np.array(cars93["MPG.city"].head(25).rank())
pd.DataFrame(first25rpm,first25mpg)


# **Two Way Tables**

# In[ ]:


two_way_table = pd.crosstab(index=cars93["DriveTrain"], 
                           columns=cars93["Type"])
two_way_table.index = ["4WD","Front","Rear"]
two_way_table


# In[ ]:


from scipy.stats import chi2_contingency,chi2
chi2.ppf(1-0.05,10)
chi2_contingency(two_way_table)


# In[ ]:


cars93[["Man.trans.avail","Origin"]].head(10)


# In[ ]:


origin_trans = cars93[["Man.trans.avail","Origin"]]


# In[ ]:


origin_trans = pd.crosstab(index=cars93["Man.trans.avail"], 
                           columns=cars93["Origin"])
origin_trans.index = ["No","Yes"]
origin_trans["Sum"] = origin_trans.sum(axis=1)
other = origin_trans.sum(axis=0)
other = pd.DataFrame({"Sum":list(other)})
other.index=["USA","non-USA","Sum"]
transposed_other = other.T
origin_trans.append(transposed_other,sort=False)


# In[ ]:


origin_trans
origin_trans.sum(axis=0)


# In[ ]:


man_vec = cars93["Man.trans.avail"].apply(lambda x: 1 if x=="Yes" else 0)
origin_vec = cars93["Origin"].apply(lambda x: 1 if x=="USA" else 0)


# In[ ]:


man_vec.corr(origin_vec, method="pearson")


# In[ ]:


# from mlxtend.frequent_patterns import apriori
# import pandas as pd
# products_train = pd.read_csv("../input/instacart-market-basket-analysis/orders.csv")


# In[ ]:


# frequent_itemsets = apriori(products_train, min_support=0.07, use_colnames=True)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

percs = np.linspace(0,100,21)
qn_a = np.percentile(a, percs)
qn_b = np.percentile(b, percs)

plt.plot(qn_a,qn_b, ls="", marker="o")

x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
plt.plot(x,x, color="k", ls="--")

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
sm.qqplot(cars93["Turn.circle"])
plt.legend(["Turn.circle"])
plt.show()
sm.qqplot(cars93["Length"])
plt.legend(["Length"])
plt.show()
sm.qqplot(cars93["Wheelbase"])
plt.legend(["Wheelbase"])
plt.show()


# In[ ]:



    


# In[ ]:



    

