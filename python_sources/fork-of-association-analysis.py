#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import pandas as pd
cars93 = pd.read_csv("../input/cars93/Cars93.csv")


# In[ ]:


cars93.head()


# In[ ]:


print(list(cars93.columns))


# In[ ]:


cars93reduced = cars93[["Type","AirBags","DriveTrain","Cylinders",
                        "EngineSize","MPG.city","Horsepower","RPM","Rev.per.mile",
                        "Fuel.tank.capacity","Length","Wheelbase","Width",
                        "Turn.circle","Weight"]]
selcols = ["Type","AirBags","DriveTrain","Cylinders",
                        "EngineSize","Horsepower","RPM","Rev.per.mile",
                        "Fuel.tank.capacity","Length","Wheelbase","Width",
                        "Turn.circle","Weight"]

# cars93reduced[selcols]


# In[ ]:


# x=3
# for i in cars93reduced:
# #     plt.subplots(111)
#     plt.figure()
#     plt.scatter(cars93["MPG.city"],cars93reduced[i])
fig, ax = plt.subplots(4, 3,squeeze=False,figsize=(15,15))
# fig.figure()
count=0
for i in range(4):
    for j in range(3):
        ax[i, j].scatter(cars93reduced[selcols[count]],cars93["MPG.city"],label=selcols[count])
        ax[i, j].legend()
        count+=1
# fig


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


cars93[["MPG.city","EngineSize","Horsepower","RPM"]].corr()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
sm.qqplot(cars93["Turn.circle"])
plt.legend(["Turn.circle"])
plt.figure(figsize=(7,7))
plt.show()
sm.qqplot(cars93["Length"])
plt.legend(["Length"])
plt.show()
sm.qqplot(cars93["Wheelbase"])
plt.legend(["Wheelbase"])
plt.show()


# **Apriori**

# In[ ]:


from mlxtend.frequent_patterns import apriori 
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


# In[ ]:


file = open("../input/groceries-sparse-matrix/Groceries.csv",'r')


# In[ ]:


strings = []
for i in file.readlines():
    strings.append(i.strip().split("\n"))
strings


# In[ ]:


dataset = []
for i in strings:
    dataset.append(i[0].split(","))
dataset


# In[ ]:


# df = pd.read_csv("../input/groceries/Groceries.csv")


# In[ ]:


# df.head()


# In[ ]:


# df = df.drop("Unnamed: 0",axis=1)


# In[ ]:


# df.replace(to_replace=False, value = 0)


# In[ ]:


# df


# In[ ]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
# print(te_ary)
df = pd.DataFrame(te_ary, columns=te.columns_)
# df.head()
# df.boxplot()
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets


# In[ ]:


from mlxtend.frequent_patterns import association_rules

association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1).head


# In[ ]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
rules.head(15)


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel("../input/car-accessories/market basket - car accessories.xlsx")
df = df.replace(float("nan"), 0)
df = df.replace("y", 1)
df = pd.DataFrame({"Items":list(df.columns),
                   "Count":list(df.sum(axis=0))
                  })
bar = sns.barplot(data = df,y="Items",x="Count")

