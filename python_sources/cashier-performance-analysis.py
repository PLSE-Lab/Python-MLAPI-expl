#!/usr/bin/env python
# coding: utf-8

# # Supermarket cashiers' performance analysis
# 1. Determine the tasks that are likely to be set by the user (shop owner) that have provided the dataset.
# 2. Solve the tasks.
# 
# *Solution by Ilya Hinzburh, a student of "Data Science Basics" course at IT Academy (Minsk Belarus). Oct 2019. Dataset proposed by our teacher, Roman N. Sidorenko*
# ## Reading and analysing data

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('/kaggle/input/retail-sales/Sales.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data['Cajero'].value_counts()


# In[ ]:


data.dtypes


# There are five fields in the dataset:
# * First column (**Folio**) contains just transaction ID, holding the duplicate information about transaction date and transaction number. We'll have to delete this column.
# * Second column (**Hora**) contains crucial information on the precise transaction time. It should be converted to pd.Timestamp in order to further process it by Pandas and visualize.
# * Third column (**Total**) contains total sum of one check. As long as most purchases are likely to be non-expensive foods, we can assume that **number of purchased items** is proportional to their total cost (idea belongs to our teacher, Roman Sidorenko).
# * Fourth column (**Pago**) contains total sum of money paid by client. We're actually interested not in the sum, but in the difference between the paid money and total check sum, since it demonstrates whether the cashier had to give change, and amount of that change.
# * Fifth column (**Cajero**) contains cashier ID. It's a categorical variable that is currently stored as string, but should be converted to categorical.
# ## Data transformation

# In[ ]:


shop = pd.DataFrame(data['Cajero']).astype('category')
shop.rename(columns={'Cajero': 'Cashier'}, inplace=True)
shop['Time'] = pd.to_datetime(data['Hora'])
shop['Total'] = data['Total']
shop['Change'] = data['Pago'] - data['Total']
shop.head()


# In[ ]:


shop.dtypes


# In[ ]:


shop.shape


# ## Cashier worktime analysis
# Total time a cashier spends on a single check ($\mathtt{t}_{total}$) can be written as $\mathtt{t}_{total} = \mathtt{t}_{min} + \mathtt{t}_{proc} + \mathtt{t}_{change} + \mathtt{t}_{client}$, where the componenents are:
# 1. Minimal time for a check ($\mathtt{t}_{min}$) is time needed to ask client for the cards, give him a plastic bag and so on. A cashier has to perform these operations for every check, even the minimal one.
# 2. Items processing time ($\mathtt{t}_{proc}$) in ideal case is limited to moving scanner over each purchased item. As long as we don't know the number of purchased items in each check, we can roughly estimate the processing time by calculating average processing time of goods costing 1 peso ($\mathtt{t}_{perf}$) for each cashier and multiplicating it by Total: $\mathtt{t}_{proc} = \mathtt{t}_{perf} * Total$.
# 3. $\mathtt{t}_{change}$ is time needed to give change to the client. It can be zero if there is no change, but there are extreme cases when cashier doesn't have enough coins and has to ask for help.
# 4. $\mathtt{t}_{client}$ is time spent by client while cashier just has to wait. A client may take a lot of time on any operation, and we don't have any data about it. So we just have to assume that this time is minimal and ignore it, since everybody wants to pass the check register as quickly as possible.

# ## Determining tasks and exploring possible solutions
# **1. Rating the cashiers' performance**.
# We need to rate the relative performance of each cashier compared to performance of the other cashiers. For instance, if four cashiers of equal qualification were working at the same time, we can assume that each of them will "earn" 25% of total supermarket revenue. Of course, we couldn't rely on this on the small data blocks, since all clients with expensive goods may prefer one cashier to another, but we have a relatively large dataset here.
# 
# **2. Calculating $\mathtt{t}_{min}$, $\mathtt{t}_{proc}$ and $\mathtt{t}_{change}$ for every cashier**.
# These values can be used in further models. They may be used to justify special ways to reduce one or more of them. Unfortunately, we don't have crucial information - a number of items in every check.
# 
# **3. Does the supermarket need more cash registers or cashiers?**
# We don't have information about the time that clients spend in the queue, but we can determine the crisis situations by the increased number of simultaneously working cashiers. If the periods of time when a lot of cashiers were working are regular and don't end quickly, then we need to increase the number of cashiers or at least re-organize their work.
# 
# **4. Does the supermarket need to re-organize some cash registers?**
# Supermarket can designate some cash registers to work "only without change", "only up to five items" and so on. Will that help the supermarket to get rid of the crisis situations or not? To answer this questions we need to determine the ratios of such checks, especially in times of crisis.
# 
# **5. Determining importance of cashiers' fatigue**.
# Cashiers obviously aren't automatons, and they get tired after several hours of work. We need to determine the level of dependence between cashiers' performance and their fatigue. If that dependence would appear to be very strong, then we could suggest to reduce the shift time or organize the breaks.

# ## Weekends and holidays
# Supermarkets obviously work differently on weekends and holidays. People tend to go to supermarkets on weekends, and to spend more money, which creates a lot of work for the cashiers. Holidays are even worse: people tend to buy **a lot**, and even the most poor ones save some money for them.
# 
# [Mexican national holidays](www.tripsavvy.com/mexican-national-holidays-1588997) are New Year (January 1), first Monday in February (Constitution Day), third Monday in March (Birthday of Benito Juarez), May 1 (Labor Day), September 16 (Mexican Independence Day), third Monday in November (Revolution Day) and December 25 (Christmas). Since we have only data from May 1 to August 31, only one national holiday was in that time period - Labor Day.
# 
# Let's add three new columns to our table: **Duration** (time of processing this check), **Holiday** (0 for common days, 1 for a weekend and 2 for a holiday) and **NoBreakTime** (time the cashiers have worked without breaks).

# In[ ]:


def holiday_code(date):
    if date.month == 5 and date.day == 1:
        return 2
    if date.weekday() >= 5:
        return 1
    return 0

def add_duration_holidays(threshold_mins):
    cashier_last = {} # Time of the last check for every cashier 
    cashier_first = {} # Time of the first check for every cashier without breaks 
    duration_col = [] # Duration
    holiday_col = []  # Holiday
    nobreaktime_col = [] # NoBreakTime

    for index, row in shop.iterrows():
        cashier = row.Cashier
        time = row.Time
        holiday_col.append(holiday_code(time))
        time_last = cashier_last.get(cashier, None)
        cashier_last[cashier] = time  
        if time_last is None:  # Nothing to compare with
            duration_col.append(np.NaN)
            cashier_first[cashier] = None  # The non-break work sequence have broken 
        else:
            time_diff = time - time_last
            if time_diff.seconds > threshold_mins*60:  # Processing can't take that much time - it was a break or idle time
                duration_col.append(np.NaN)
                cashier_first[cashier] = None  # Sequence was broken
            else:
                duration_col.append(time_diff.seconds)
        time_first = cashier_first.get(cashier, None)
        if time_first is None:  # Nothing to compare with
            nobreaktime_col.append(np.NaN)
            cashier_first[cashier] = time
        else:
            time_diff = time - time_first
            if time_first.day != time.day:
                nobreaktime_col.append(np.NaN)
                cashier_first[cashier] = None  # The non-break work sequence have broken
            else:
                nobreaktime_col.append(time_diff.seconds/60)
    shop['Duration'] = duration_col
    shop['Holiday'] = holiday_col
    shop['NoBreakTime'] = nobreaktime_col

add_duration_holidays(30)


# In[ ]:


shop[shop['Duration'].between(1750, 1801)].head()


# We can see some garbage values, since processing a check of 9.60 peso can't take almost half an hour. A cashier was likely to had a break at that time. Let's recalculate the column, setting 20 minutes as a threshold:

# In[ ]:


add_duration_holidays(20)
shop[shop['Duration'].between(1150, 1201)].head()


# Here we can see two problems in our data:
# * Some rows contain negative Total (probably, money refunds). We have to exclude such rows from our model since we can't do anything useful with them.
# * Some records still have invalid Duration. Nobody can't believe that processing a check for 1 peso required almost 20 minutes: the cashier either was idle or had a break.
# 
# Still, we can't lower the "trusted" duration threshold even further, as the second row's Duration is real! Processing a huge check of 4341 peso (more than $220) with a change really can take 20 minutes. Let's return to a 30-minute threshold and analyse the largest checks:

# In[ ]:


add_duration_holidays(30)
shop.nlargest(20, 'Total')


# We can see six checks with duration more than half an hour (Duration=NaN). So we have to use another approach in order to avoid losing the data. We will increase the maximum threshold to 2 hours (120 minutes) and then invent a way to remove the unreal durations.

# In[ ]:


add_duration_holidays(120)
shop.nlargest(20, 'Total')


# Maximum time of real check processing was almost 65 minutes. Large purchase with ID=66804 is likely to be the first Jaqueline's check on that day (the check is stamped at 8.03, while the supermarket opens at 7.30), so we don't know the real processing time of that purchase. Let's replace Duration by 30 minutes here.
# 
# The listing shows that, unfortunately, we can't always assume that processing time is proportional to the price: there is a purchase of 4522.36 peso processed in 25 seconds! Half of this listing are purchases that were processed in less than 1000 seconds. But still, we will hold to the hypothesis that processing time is proportional to its total price for the smaller purchases. And the small purchases take the majority of our data:

# In[ ]:


shop['Total'][shop['Total'].between(0, 2000)].hist(bins=30);


# In[ ]:


shop[shop['Total'] >= 1000.0].count()


# As we see, the percentage of large purchases is quite small - lesser than 2%, so we can keep their Duration. Let's invent a condition of "garbage Duration" like this:

# In[ ]:


shop[(shop['Total'] < 1000.0) & (shop['Duration'] > 900) & (shop['Total']*7 < shop['Duration'])]


# In[ ]:


shop = shop[shop['Total']>0]  # Remove negative Totals
shop['Duration'].mask(shop['Total'] == 4980.15, 1800, inplace=True)  # Fix the Jaqueline's check
shop.loc[66804]


# In[ ]:


# Clean the garbage
garbage = (shop['Total'] < 1000.0) & (shop['Duration'] > 900) & (shop['Total']*7 < shop['Duration'])
shop['Duration'].mask(garbage, inplace=True)


# ## Timeline processing
# We need to convert our data to a timeline to solve tasks 1, 3 and 4. This is done in 4 steps:
# 1. Index by Time column.

# In[ ]:


shopt = shop.set_index('Time')
shopt.head()


# 2. Group by Cashier and resample with 10-minute intervals to compare cashiers' performance within each interval and draw graphs.

# In[ ]:


shopt=shopt.groupby(['Cashier']).resample('10Min').sum()
shopt.head()


# 3. Transform the table with unstack() to get Total for every cashier for every 10-minute time interval.

# In[ ]:


shopt=shopt.unstack(level='Cashier')['Total']
shopt.head()


# 4. Add "Grand Total" and "Cashiers Num" columns to store total amount of money the supermarket had gained during that 10-minute interval and total number of worked cashiers.

# In[ ]:


shopt.mask(lambda x: x==0.0, inplace=True)
grand=shopt.sum(axis=1)
cash_num = shopt.agg('notnull').sum(axis=1)
shopt.columns = shopt.columns.add_categories(['Grand Total', 'Cashiers Num'])
shopt['Grand Total'] = grand
shopt['Cashiers Num'] = cash_num
shopt.head()


# In[ ]:


import plotly.graph_objects as go


# In[ ]:


# I don't know why this doesn't show anything in Kaggle, it works in Jupyter notebook
fig = go.Figure()
for name in shopt.columns:
    if name != 'Cashiers Num' and name != 'Grand Total':
        fig.add_trace(go.Scatter(x=shopt.index, y=shopt[name], name=name))

fig.update_layout(title_text="Supermarket sales with a Rangeslider", xaxis_rangeslider_visible=True)
fig.update_yaxes(range=[0, 7500])
fig.show()


# In[ ]:


sns.set(rc={'figure.figsize':(18, 10)})
shopt.iloc[:,:-2]['2018-05-01'].plot();   # Substitute 2018-05-01 with your date here


# ## Tasks 3 and 4
# Let's determine the maximal number of simultaneously worked cashiers:

# In[ ]:


shopt['Cashiers Num'].max()


# In[ ]:


shopt[shopt['Cashiers Num'] > 4].count()


# As long as there never worked more than six cashiers at a time, and 5 or 6 cashiers were working only for 28 hours during four months (1670 minutes), we can safely say that the supermarket **doesn't need more cash registers or cashiers, and it doesn't need any re-organizations**.
# ## Task 1 solution (estimating cashiers' performance)
# To estimate performance of a cashier, as we've already noted, we will just subtract average share of supermarket income from "his" (or "her") share, for every 10-minute interval when that cashier isn't working alone. This method doesn't work when we don't have enough data, but as long as we do, it should really deliver useful information.

# In[ ]:


def get_ratings(data, cashier):
    res = []
    cash_ratings = data[(data['Cashiers Num'] > 1) & data[cashier].notnull()]
    for _, row in cash_ratings.iterrows():
        percent = row[cashier] / row['Grand Total']
        res.append(percent - (1 / row['Cashiers Num']))
    return res

def get_plot_table(data):
    cashiers = data.columns[:-2]  # Ignore last two columns
    res = {}
    max_len = 0
    for cashier in cashiers:
        ratings = get_ratings(data, cashier)
        res[cashier] = ratings
        if len(ratings) > max_len:
            max_len = len(ratings)
    for cashier in cashiers:     # Add NaN's to the short columns
        rating = res[cashier]
        while len(rating) < max_len:
            rating.append(np.nan)
    return pd.DataFrame(res)


# In[ ]:


plot_table = get_plot_table(shopt)
plot_table.head()


# In[ ]:


plt.figure(figsize=(16, 7))
p = sns.boxplot(data=plot_table,
                order = sorted(plot_table.columns),
                orient="h")


# We've got very interesting results here. Six cashiers have bad average performance, and five of them also have less processed checks than the top 10 cashiers, so we can suggest that they were the training cashiers or part-time ones. The sixth underachiever, Monse L., has a lot of positive outliers, so our comparison may simply provide the false result for him or her.
# 
# Most cashier have approximately average performance. The only difference is Eduardo: his performance is far better than performance of the rest cashiers. As long as he has the least number of checks (only 123), I've figured that he may be a VIP cashier only for the elite clients.
# ## Task 2 - calculate $\mathtt{t}_{min}$ and $\mathtt{t}_{proc}$
# Let's start with $\mathtt{t}_{min}$. If we take only the rows without change and select the rows with the minimal Total from them, we will get $\mathtt{t}_{min}$ for a given cashier. It will be logical to take several "minimal" rows to calculate mean and deviation.

# In[ ]:


shop_tmin = shop[shop['Duration'].notnull() & (shop['Change'] == 0.0)]
shop_tmin.shape


# In[ ]:


shop_tmin['Cashier'].value_counts()


# We see that even the most rare cashiers have at least 20 rows without change. Let's take 20 minimal rows for every cashier to calculate $\mathtt{t}_{min}$.

# In[ ]:


min_table = shop_tmin.sort_values("Duration").groupby("Cashier").head(20)
min_table.head()


# In[ ]:


gb = min_table[['Cashier', 'Duration']].groupby('Cashier')
gb.mean()


# In[ ]:


gb.boxplot(subplots=False, figsize=(16,7), rot=90);


# We see a very interesting picture here: $\mathtt{t}_{min}$ is very different for the different cashiers (for instance, ALE HUERTA has approximately 4 times smaller $\mathtt{t}_{min}$ than MAGO), and variance for most cashiers is minimal! Thus, some cashiers can process minimal purchases quickly while other can't. If that supermarket would ever organise a cash register "up to five items", it should assign only the cashiers with low $\mathtt{t}_{min}$ there.
# 
# Now we'll calculate $\mathtt{t}_{perf}$ (average time per 1 peso of purchase) for every cashier.

# In[ ]:


tb=shop_tmin.groupby('Cashier')['Total', 'Duration'].mean()
tb


# In[ ]:


tb['t_obr'] = tb['Duration']-gb['Duration'].mean()
tb['t_perf'] = tb['Duration'] / tb['t_obr']
tb


# In[ ]:


plt_data=tb.reset_index()
fig, ax = plt.subplots(figsize=(16,10))
sns.barplot(ax=ax, x='t_perf', y='Cashier', data=plt_data);


# Now we see that $\mathtt{t}_{perf}$ has real sense, and its value is almost equal for all the cashiers besides two! Now we can easily use formula $\mathtt{t}_{proc} = \mathtt{t}_{perf}*Total$ to estimate $\mathtt{t}_{proc}$. $\mathtt{t}_{perf}$ varies between 1.06 and 1.17 for most cashiers.
# 
# If we need $\mathtt{t}_{change}$, we can calculate it like that: $\mathtt{t}_{change} = Duration - \mathtt{t}_{min} - \mathtt{t}_{proc}$
# 
# Why do we need all these values? To use them in the business modelling software which is specifically designed to model such processes. For instance, AnyLogic has a [ready supermarket example](https://cloud.anylogic.com/model/b4ea28be-2455-4cd4-a4a3-91d0bd068a16?mode=SETTINGS) which I strongly recommend to take a look at. Every task has the best instruments to solve it.
# ## Task 5 - determining importance of fatigue
# Let's build a Machine Learning model to predict Duration from Total, Holiday, Change and NoBreakTime. We don't really need that prediction, but the model will demonstrate the importance of NoBreakTime (time that the cashier had worked without breaks). There is no point in building a general model for all cashiers, as they may have very different tolerance to fatigue. We will build a sample model for the most frequently worked cashier, Maricruz.

# In[ ]:


shop.dropna(subset=['Duration'], inplace=True)
shop['NoBreakTime'].fillna(0.0, inplace=True)


# In[ ]:


mari = shop[shop['Cashier']=='MARICRUZ']
data=mari[['Total', 'Change', 'Holiday', 'NoBreakTime']]
y = mari['Duration']
data.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(data)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
gb_parms= {
    'loss': ['ls', 'lad', 'quantile'],
    'learning_rate': [0.05, 0.1, 0.5],
    'max_depth': [2, 3, 5]
}
gbr = GradientBoostingRegressor()
clf = GridSearchCV(gbr, gb_parms, cv=5)
clf.fit(X_train, y_train)


# In[ ]:


best_xgr = clf.best_estimator_
print(clf.best_params_)


# In[ ]:


print(clf.best_score_)


# In[ ]:


best_xgr.feature_importances_


# Now we see that importance of NoBreakTime is 0.133, which gives it a second place, but Duration is still depends on Total much more than on Mary's fatigue. So we can say that fatigue is important, but it isn't so much important to invent some special measures against it.
