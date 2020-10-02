#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class pairwise_association_mining:
    def __init__(self, list_of_sets, threshold, min_count):
        """
        * Accepts a list of sets of unique items as input
        * Uses Bayes Rule to calculate the probability of
          co-occurence between items
        * Only works on unique items with user-specified
          minimum number of co-occurrences
        * Prints item pairs with probability
          greater than user-specified threshold
        
        Args:
            list_of_sets: list of unique items
            threshold: minimum probability of co-occurrence
            min_count: minimum number of co-occurrences
        Raises:
            Assertion errors for incorrect input types
        Returns:
            Prints co-occurence probability for each item pair
        """
        assert isinstance(list_of_sets, list), "list_of_sets must be a list of sets"
        assert isinstance(list_of_sets[0], set), "list_of_sets must be a list of sets"
        assert isinstance(threshold, float) and threshold > 0  and threshold < 1, "threshold must be between 0 and 1"
        assert isinstance(min_count, int), "min_count must be an int"
        
        self.list_of_sets = list_of_sets
        self.threshold = threshold
        self.min_count = min_count
        
        self.pair_counts = defaultdict(int)
        self.item_counts = defaultdict(int)
        
        self.rules = dict()
        self.find_assoc_rules()
        
        self.pairwise_confidence = {pair:self.rules[pair] for pair in self.rules.keys()                              if self.item_counts[pair[0]] >= self.min_count}
        
    def update_pair_counts(self, itemset):
        """
        Updates a dictionary of pair counts for
        all pairs of items in a given itemset.
        """
        for a,b in combinations(itemset,2):
            self.pair_counts[(a,b)] += 1
            self.pair_counts[(b,a)] += 1
            
    def update_item_counts(self, itemset):
        """
        Updates a dictionary of item counts for
        all pairs of items in a given itemset.
        """
        for item in itemset:
            self.item_counts[item] += 1
            
    def filter_rules_by_conf(self):
        """
        Filters out pairs whose confidence is
        below the user defined threshold.
        """
        for (a,b) in self.pair_counts:
            confidence = self.pair_counts[(a,b)] / self.item_counts[a]
            if confidence >= self.threshold:
                self.rules[(a,b)] = confidence

    def find_assoc_rules(self):
        """
        Set final rules dictionary using
        pairs that appear together with
        confidence greater than or equal to
        the user defined threshold.
        """
        for itemset in self.list_of_sets:
            self.update_pair_counts(itemset)
            self.update_item_counts(itemset)
        rules = self.filter_rules_by_conf()
        return rules
    
    @staticmethod
    def gen_rule_str(a, b, val=None, val_fmt='{:.3f}', sep=" = "):
        text = "{} => {}".format(a, b)
        if val:
            text = "conf(" + text + ")"
            text += sep + val_fmt.format(val)
        return text

    def print_rules(self):
        """
        Pretty print pairwise associations
        """
        from operator import itemgetter
        ordered_rules = sorted(self.pairwise_confidence.items(), key=itemgetter(1), reverse=True)
        for (a, b), conf_ab in ordered_rules:
            print(self.gen_rule_str(a, b, conf_ab))


# # Load Data

# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head()


# In[ ]:


df.info()


# # Preprocess `Date` and `Time`
# 
# Let's convert these to datetime and time types.

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
df['Day'] = df['Date'].dt.day_name()
df['Hour'] = df['Time'].dt.hour
df['Time'] = df['Time'].dt.time


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Exploratory Data Analysis
# 
# Let's explore the data! First lets look at the items sold at this bakery.

# In[ ]:


df.Item.unique()


# Pretty standard fare. Let's examine the best sellers.

# In[ ]:


df.groupby('Item', as_index=False)['Transaction'].count().sort_values(by='Transaction', ascending=False)[:20].plot(x='Item', kind='bar')
plt.title('Number of Transactions by Item')
plt.ylabel('Transactions')
plt.tight_layout()


# Top 5 make the most sense. I wonder what the NONE values represent? Maybe customers just getting cash back? Regardless, we should probably exclude these transactions from our analysis.

# In[ ]:


df = df[df['Item']!='NONE']


# Next let's look at the trend of number of transactions.

# In[ ]:


df.groupby('Date')['Transaction'].count().reset_index().plot(x='Date', y='Transaction')
plt.title('Number of Transactions Vs. Date')
plt.ylabel('Transactions')
plt.tight_layout()


# The dip right at the new year is most likely due to the New Years Day holiday.

# In[ ]:


df.groupby('Date', as_index=False)['Transaction'].count().sort_values(by='Transaction').reset_index().drop('index',axis=1).iloc[0]


# Sure enough. What day were sales the highest?

# In[ ]:


df.groupby('Date', as_index=False)['Transaction'].count().sort_values(by='Transaction', ascending=False).reset_index().drop('index',axis=1).iloc[0]


# Lovebirds getting ready for Valentines day? Maybe...

# # Day of Week Sales Analysis
# 
# Let's explore the relationship between sales and the day of the week.

# In[ ]:


df.groupby('Day')['Transaction'].count().reset_index().plot(x='Day', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Number of Transactions by Day')
plt.ylabel('Transactions')
plt.tight_layout()


# Saturday looks like the prime day to go to the bakery! What about coffee and bread sales by day?

# In[ ]:


coffee = df[df['Item']=='Coffee']
coffee.groupby('Day')['Transaction'].count().reset_index().plot(x='Day', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Number of Coffee Transactions by Day')
plt.ylabel('Transactions')
plt.tight_layout()


# In[ ]:


bread = df[df['Item']=='Bread']
bread.groupby('Day')['Transaction'].count().reset_index().plot(x='Day', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Number of Bread Transactions by Day')
plt.ylabel('Transactions')
plt.tight_layout()


# Sums are fine and dandy but what about the expected number of transactions by day of the week?

# In[ ]:


transactions_per_day = df.groupby(['Date', 'Day'])['Transaction'].count().reset_index()

transactions_per_day.groupby('Day')['Transaction'].mean().reset_index().plot(x='Day', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Average Number of Transactions by Day')
plt.ylabel('Transactions')
plt.tight_layout()


# Looks like weekends are the most popular. What about the worst case scenario?

# In[ ]:


transactions_per_day.groupby('Day')['Transaction'].min().reset_index().plot(x='Day', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Minimum Number of Transactions by Day')
plt.ylabel('Transactions')
plt.tight_layout()


# New Years Day obviously fell on a Sunday in 2017...

# In[ ]:


df[df['Date']=='2017-01-01']['Day']


# What about the best case scenario?

# In[ ]:


transactions_per_day.groupby('Day')['Transaction'].max().reset_index().plot(x='Day', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Maximum Number of Transactions by Day')
plt.ylabel('Transactions')
plt.tight_layout()


# # Time of Day
# 
# It's a bakery so my hypothesis is that they are busier in the mornings. Let's see if this is reflected in the data.

# In[ ]:


df.groupby('Hour')['Transaction'].count().reset_index().plot(x='Hour', y='Transaction', kind='bar', legend=None, color='#1f77b4')
plt.title('Number of Transactions by Hour')
plt.ylabel('Transactions')
plt.tight_layout()


# Looks like this bakery is actually busier during the lunch hours. What about the expected number of transactions per hour?

# In[ ]:


transactions_per_time = df.groupby(['Date', 'Hour']).count().reset_index()

transactions_per_time.groupby('Hour')['Transaction'].mean().reset_index().plot(x='Hour', y='Transaction', kind='bar', legend=None, color='#1f77b4')

plt.title('Average Number of Transactions by Hour')
plt.ylabel('Transactions')
plt.tight_layout()


# That's interesting! Our data appears to be somewhat bimodal in nature with the peak during lunchtime and early-afternoon hours and another mini-rush during the dinner hours.

# Enough with the visualizations! Let's figure out which items are most likely to be sold together. We'll use our good old friend Bayes Rule:
# 
# $$
# P\left(A | B\right) = \frac{P\left(B | A\right)P\left(A\right)}{P\left(B\right)}
# $$
# 
# And the class we wrote out above. Ignore my lack of adherence to PEP 8.

# In[ ]:


checkout_list = defaultdict(list)
trans = dict()
for row in df.groupby(by='Transaction').filter(lambda x: len(x['Item']) >= 1)[['Transaction','Item']].itertuples():
    checkout_list[row.Transaction].append(row.Item)


# Now that we have our dictionary of lists containing items purchased during the same transaction, let's apply Baye's Rule! Feel free to play around with the `THRESHOLD` and `MIN_COUNT` parameters.

# In[ ]:


# Confidence threshold
THRESHOLD = 0.7

# Only consider rules for items appearing at least `MIN_COUNT` times.
MIN_COUNT = 5


# In[ ]:


bakery_itemset = [set(lst) for lst in checkout_list.values()]
pam = pairwise_association_mining(bakery_itemset, THRESHOLD, MIN_COUNT)
pam.print_rules()


# Extra Salami or Feta and Coffee? You do you! 

# In[ ]:


youdoyou1 = df[df['Item'] == 'Extra Salami or Feta']
youdoyou2 = df.loc[df['Item'] == 'Coffee', ['Transaction','Item']]
youdoyou2.head()


# In[ ]:


youdoyou2['Item2'] = youdoyou2.Transaction.map(pd.Series(youdoyou1.Item.values,index=youdoyou1.Transaction).to_dict())
youdoyou2.dropna().drop_duplicates(['Transaction']).reset_index().drop('index',axis=1).iloc[-2:]


# Sure enough... 30 unique transactions of Extra Salami or Feta and Coffee... what a time to be alive!
