#!/usr/bin/env python
# coding: utf-8

# # Objective:
# 1. Predict the number of purchases a customer will do in 2018 H2.
# 2. Predict the total cash credited by the customer in 2018 H2.

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')

import seaborn as sns
import pandas as pd
import lifetimes


# In[ ]:


np.random.seed(42)

import random
random.seed(42)

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8) 
plt.rcParams["figure.dpi"] = 60 

# sns.set(style="ticks")
# sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 5})


# # ModifiedBetaGeoFitter
# - With this technique, we are abstracting:
#   - the nature of the items in the order
#   - the ratings of the users in the order
#   - the credit payment history of the items in the user
#   - the freight rate

# In[ ]:


orders = pd.read_csv('../input/olist_orders_dataset.csv')
customers = pd.read_csv('../input/olist_customers_dataset.csv')
payments = pd.read_csv('../input/olist_order_payments_dataset.csv')
order_items = pd.read_csv("../input/olist_order_items_dataset.csv")

cols = ['customer_id', 'order_id', 'order_purchase_timestamp']
orders = orders[cols]
orders = orders.set_index('customer_id')
orders.drop_duplicates(inplace=True)
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

# aggregate cost of items
costs = order_items.groupby("order_id")["price"].sum()

cols = ['customer_id', 'customer_unique_id']
customers = customers[cols]
customers = customers.set_index('customer_id')

transactions = pd.concat([orders,customers], axis=1, join='inner')
transactions.reset_index(inplace=True)

cols = ['customer_unique_id', 'order_id','order_purchase_timestamp']
transactions = transactions[cols]

transactions['order_purchase_timestamp'] = pd.to_datetime(transactions['order_purchase_timestamp'])
transactions['order_date'] = transactions.order_purchase_timestamp.dt.date
transactions['order_date'] = pd.to_datetime(transactions['order_date'])

cols = ['customer_unique_id', 'order_id', 'order_date']
transactions = transactions[cols]


# In[ ]:


products = pd.read_csv("../input/olist_products_dataset.csv")
sellers = pd.read_csv("../input/olist_sellers_dataset.csv")


# In[ ]:


transactions = transactions.merge(costs.to_frame("total_cost").reset_index(), on='order_id')
transactions.columns = ["customer_id", "order_id", "order_date", "total_cost"]


# In[ ]:


transactions.order_date.value_counts().plot()


# In[ ]:


vc_num_purchases_by_customer = transactions.groupby("customer_id")["order_id"].nunique().value_counts()
vc_num_purchases_by_customer = vc_num_purchases_by_customer.sort_index()
nunique_items_per_customer = transactions.merge(order_items).groupby("customer_id")["product_id"].nunique()
vc_num_items_per_customer = nunique_items_per_customer.value_counts()
vc_num_items_per_customer = vc_num_items_per_customer.sort_index()

print("Average number of purchases: {:.2f}".format(transactions.groupby("customer_id")["order_id"].nunique().mean()))
print("Average number of items bought per customer: {:.2f}".format(
    nunique_items_per_customer.mean()))


# plot of number of orders per customer
ax = plt.figure().add_subplot(211)
vc_num_purchases_by_customer.plot.bar(color='dodgerblue', ax=ax)

pct_cumsum_vc = vc_num_purchases_by_customer.cumsum() / vc_num_purchases_by_customer.sum()

ax2=ax.twinx()
ax.set_ylabel("Number of customers")
ax.set_xlabel("Number of purchases")
ax2.set_ylabel("Cumulative percent")
ax2.plot(range(len(vc_num_purchases_by_customer)), pct_cumsum_vc, linestyle='--', color='salmon')

# plot of number of items bought per customer
ax = plt.figure().add_subplot(212)
vc_num_items_per_customer.plot.bar(color='dodgerblue', ax=ax)

pct_cumsum_vc = vc_num_items_per_customer.cumsum() / vc_num_items_per_customer.sum()

ax2=ax.twinx()
ax.set_ylabel("Number of customers")
ax.set_xlabel("Number of items bought")
ax2.set_ylabel("Cumulative percent")
ax2.plot(range(len(vc_num_items_per_customer)), pct_cumsum_vc, linestyle='--', color='salmon')


# In[ ]:


np.max(transactions.order_date) - np.min(transactions.order_date)


# In[ ]:


from lifetimes.utils import summary_data_from_transaction_data



# summary_data_from_transaction_data(transactions, 
#                                                    customer_id_col = 'customer_id', 
#                                                    datetime_col = 'order_date', 
#                                                    freq = 'D',
#                                         observation_period_end='2018-09-28', monetary_value_col='total_cost' )


# In[ ]:


timestamps = [d.strftime('%Y-%m-%d') for d in pd.date_range(start='2018-04-01', end='2018-10-01', freq='M')]


# In[ ]:


summary_cal_holdout = None

for timestamp in timestamps:
    summary_cal_holdout_partial = summary_data_from_transaction_data(transactions, 
                                                   customer_id_col = 'customer_id', 
                                                   datetime_col = 'order_date', 
                                                   freq = 'D',
                                                observation_period_end=timestamp, monetary_value_col='total_cost' )

    summary_cal_holdout_partial['date'] = timestamp
    
    if summary_cal_holdout is None:
        summary_cal_holdout = summary_cal_holdout_partial
    else:
        summary_cal_holdout = summary_cal_holdout.append([summary_cal_holdout_partial])


# In[ ]:


summary_cal_holdout['date'].unique()


# In[ ]:


from sklearn.model_selection import train_test_split

x = summary_cal_holdout[summary_cal_holdout['date'] != '2018-09-30']
y = summary_cal_holdout[summary_cal_holdout['date'] == '2018-09-30']


# In[ ]:


x = x.reset_index(drop=False)
y = y.reset_index(drop=False)


# In[ ]:


x = x[x['customer_id'].isin(y['customer_id'])]


# In[ ]:


sorted_x = x.sort_values(['customer_id', 'date'])


# In[ ]:


customers = x['customer_id'].unique()
dates = x['date'].unique()


# In[ ]:


index = pd.MultiIndex.from_product([customers, dates], names = ["customer_id", "date"])
all_customers_and_dates = pd.DataFrame(index = index).reset_index()


# In[ ]:


merged = pd.merge(all_customers_and_dates, x,  how='left', left_on=['customer_id', 'date'], right_on = ['customer_id', 'date'])


# In[ ]:


merged.loc[(merged['date'] == '2018-04-30') & (pd.isna(merged['frequency'])), 'frequency'] = 0
merged.loc[(merged['date'] == '2018-04-30') & (pd.isna(merged['recency'])), 'recency'] = 0
merged.loc[(merged['date'] == '2018-04-30') & (pd.isna(merged['T'])), 'T'] = 0
merged.loc[(merged['date'] == '2018-04-30') & (pd.isna(merged['monetary_value'])), 'monetary_value'] = 0


# In[ ]:


merged = merged.fillna(axis = 0, method = 'ffill')


# In[ ]:


dates = merged['date'].values.reshape(95420, 5)
frequencies = merged['frequency'].values.reshape(95420, 5)
recencies = merged['recency'].values.reshape(95420, 5)
Ts = merged['T'].values.reshape(95420, 5)
monetary_values = merged['monetary_value'].values.reshape(95420, 5)


# In[ ]:


reshaped = numpy.hstack((frequencies, recencies, Ts, monetary_values)).reshape(95420, 4, 5)


# In[ ]:


sorted_y = y.sort_values(['customer_id'])


# In[ ]:


output_y = sorted_y[['frequency', 'recency', 'T', 'monetary_value']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(reshaped, output_y, test_size=0.2)


# In[ ]:


y_train


# In[ ]:


from keras import layers
from keras.layers import recurrent
from keras.models import Sequential


# In[ ]:


model = Sequential([
    layers.Dense(60, input_shape=(4,5)),
    layers.Activation('relu'),
    layers.SimpleRNN(60),
    layers.Activation('relu'),
    layers.Dense(4),
    layers.Activation('linear')
])


# In[ ]:


model.compile(optimizer='rmsprop',
              loss='mse')


# In[ ]:


for layer in model.layers:
    print(layer.output_shape)


# In[ ]:


#TODO normalize input
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)


# # NBD Model

# In[ ]:


from lifetimes import BetaGeoFitter, ModifiedBetaGeoFitter
mbgf = ModifiedBetaGeoFitter()
mbgf.fit(summary_cal_holdout["frequency_cal"], summary_cal_holdout["recency_cal"], summary_cal_holdout["T_cal"], 
         iterative_fitting=3, verbose=True)


# In[ ]:


from lifetimes.plotting import plot_period_transactions
ax = plot_period_transactions(mbgf, max_frequency=7)
ax.set_yscale('log')
sns.despine();


# In[ ]:


from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_expected_repeat_purchases


plt.figure(figsize=(10, 10))
plot_frequency_recency_matrix(mbgf, T=120, );

plt.figure(figsize=(10, 10))
plot_probability_alive_matrix(mbgf);


# In[ ]:


n = 7

summary = summary_cal_holdout.copy()
duration_holdout = summary.iloc[0]['duration_holdout']

summary['model_predictions'] = summary.apply(lambda r: mbgf.conditional_expected_number_of_purchases_up_to_time(
    duration_holdout, r['frequency_cal'], r['recency_cal'], r['T_cal']), axis=1)
agg_data = summary.groupby("frequency_cal")[['frequency_holdout', 'model_predictions']].mean()
ax = agg_data.iloc[:n].plot(title="Pearson correlation (calibration & holdout): {:.4f}".format(agg_data.corr().min().min()))
ax.set_xticks(agg_data.iloc[:n].index);


# In[ ]:


t = 120
predicted_purchases = mbgf.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                               summary_cal_holdout['frequency_cal'], 
                                                                               summary_cal_holdout['recency_cal'],
                                                                               summary_cal_holdout['T_cal'])
predicted_purchases.sort_values().tail(4)


# In[ ]:


from lifetimes.plotting import plot_history_alive

fig = plt.figure(figsize=(15, 10))

for idx, customer_id in enumerate(predicted_purchases.sort_values().tail(4).index, 1):
    # all days
    days_since_birth = (max(transactions.order_date - min(transactions.order_date))).days
    sp_trans = transactions.loc[transactions['customer_id'] == customer_id]
    
    plot_history_alive(mbgf, days_since_birth, sp_trans, 'order_date', ax=fig.add_subplot(2, 2, idx))


# # Adding Monetary, GammaGamma Filter

# In[ ]:


# weak correlation between monetary and frequency
returning_customers_summary = summary_cal_holdout[summary_cal_holdout["frequency_cal"] > 0]
print(returning_customers_summary[["frequency_cal", "monetary_value_cal"]].corr())

fig, axes = plt.subplots(1,2,figsize=(12, 5))
sns.distplot(returning_customers_summary["monetary_value_cal"], ax=axes[0], )
sns.distplot(np.log(returning_customers_summary["monetary_value_cal"] + 1), ax=axes[1], axlabel='$log(monetary\_value)$')


# In[ ]:


from lifetimes import GammaGammaFitter

gg = GammaGammaFitter()
gg.fit(returning_customers_summary["frequency_cal"], 
       returning_customers_summary["monetary_value_cal"], verbose=True)


# In[ ]:


expected_average_profit_validation = gg.conditional_expected_average_profit(
    summary_cal_holdout['frequency_holdout'], summary_cal_holdout['monetary_value_holdout'])

expected_average_profit = gg.conditional_expected_average_profit(
    summary_cal_holdout['frequency_cal'], summary_cal_holdout['monetary_value_cal'])

print("With non-repeat buyers")
print("Train correlation")
print(pd.Series.corr(summary_cal_holdout["monetary_value_cal"], expected_average_profit).round(4))

print("Validation correlation")
print(pd.Series.corr(summary_cal_holdout["monetary_value_holdout"], expected_average_profit_validation).round(4))


# In[ ]:


expected_average_profit_validation = gg.conditional_expected_average_profit(
    returning_customers_summary['frequency_holdout'], returning_customers_summary['monetary_value_holdout'])

expected_average_profit = gg.conditional_expected_average_profit(
    returning_customers_summary['frequency_cal'], returning_customers_summary['monetary_value_cal'])

print("Repeat buyers only")
print("Train correlation")
print(pd.Series.corr(returning_customers_summary["monetary_value_cal"], expected_average_profit).round(4))

print("Validation correlation")
print(pd.Series.corr(returning_customers_summary["monetary_value_holdout"], expected_average_profit_validation).round(4))


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_absolute_error(summary_cal_holdout["monetary_value_holdout"], expected_average_profit_validation)


# In[ ]:


print("Expected average profit validation: {:.2f}, Average profit validation: {:.2f}".format(
    gg.conditional_expected_average_profit(
        summary_cal_holdout['frequency_cal'],
        summary_cal_holdout['monetary_value_cal']
    ).mean(),
    summary_cal_holdout[summary_cal_holdout['frequency_cal']>0]['monetary_value_cal'].mean()))


# # Conclusions:
# - Repeat purchases could be predicted (holdout set's pearson correlation with predictions) although the probability is very low (expected number of future purchases).
# - Monetary value predictions has good correlation although this includes the non-repeat buyers. 
# - For the repeat buyers, monetary value predictions' correlation is low. The exact monetary value is hard to predict since for an e-commerce site, the variety of items is very large.

# # Regression with item types
# ## Target: Frequency Holdout
# Not good!

# In[ ]:


returning_customers_summary[:2]


# In[ ]:


from sklearn.model_selection import train_test_split

train_cols = ["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]
X_train = returning_customers_summary[train_cols].values
y_train = returning_customers_summary["frequency_cal"].values

test_cols = ["frequency_cal", "recency_cal", "T_cal", "monetary_value_holdout"]
X_test = returning_customers_summary[test_cols].values
y_test = returning_customers_summary["frequency_holdout"].values


# In[ ]:


import statsmodels.api as sm

X_train_constant = sm.add_constant(X_train, prepend=False)

mod = sm.OLS(y_train, X_train_constant)
res = mod.fit()
print(res.summary())


# In[ ]:


from sklearn.svm import SVR

svr_model = SVR()

svr_model.fit(X_train, y_train)
preds = svr_model.predict(X_test)

n = 30

summary = returning_customers_summary.copy()
duration_holdout = summary.iloc[0]['duration_holdout']

summary['model_predictions'] = preds
agg_data = summary.groupby("frequency_cal")[['frequency_holdout', 'model_predictions']].mean()
ax = agg_data.iloc[:n].plot(title="Pearson correlation (calibration & holdout): {:.4f}".format(agg_data.corr().min().min()))
ax.set_xticks(agg_data.iloc[:n].index);


# # Adding the nature of items
# - X attributes: 
#  - number of items purchased, type of items purchased, city, payment history, type of seller, cost of items, 
# - y attributes: how many times the customer returned to purchase an order

# In[ ]:


orders = pd.read_csv('../input/olist_orders_dataset.csv')
customers = pd.read_csv('../input/olist_customers_dataset.csv')
payments = pd.read_csv('../input/olist_order_payments_dataset.csv')
order_items = pd.read_csv("../input/olist_order_items_dataset.csv")


# In[ ]:


print("Orders")
display(orders[:2])
print("Customers")
display(customers[:2])
print("Payments")
display(payments[:2])
print("Orders-Items")
display(order_items[:2])
print("Items")
display(products[:2])
print("Sellers")
display(sellers[:2])


# In[ ]:


import featuretools as ft

es = ft.EntitySet(id = 'customers')

es = es.entity_from_dataframe(entity_id = 'customers', dataframe = customers[["customer_id", "customer_city", "customer_state"]], 
                              index = 'customer_id',)

es = es.entity_from_dataframe(entity_id = 'orders', dataframe = orders.reset_index(),
                              index = 'order_id', 
                              time_index = 'order_purchase_timestamp')

es = es.entity_from_dataframe(entity_id = 'order_products', 
                              dataframe = order_items[["order_id", "product_id", "seller_id", "price", "freight_value"]],
                              index = 'order_product_id', make_index=True)

es = es.entity_from_dataframe(entity_id = 'products', 
                              dataframe = products,
                              index = 'product_id', )

es = es.entity_from_dataframe(entity_id = 'sellers', 
                              dataframe = sellers[["seller_id", "seller_city", "seller_state"]],
                              index = 'seller_id', )

es = es.entity_from_dataframe(entity_id = 'payments', 
                              dataframe = payments[["order_id", "payment_type", "payment_installments", "payment_value"]],
                              index = 'payment_id', make_index=True)


# In[ ]:


# Add the relationship to the entity set
# customer to orders
es = es.add_relationship(ft.Relationship(es['customers']['customer_id'],
                                    es['orders']['customer_id']))
# orders to order products
es = es.add_relationship(ft.Relationship(es['orders']['order_id'],
                                    es['order_products']['order_id']))
# products to order products
es = es.add_relationship(ft.Relationship(es['products']['product_id'], 
                                         es['order_products']['product_id']))
# sellers to order products
es = es.add_relationship(ft.Relationship(es['sellers']['seller_id'],
                                         es['order_products']['seller_id']))

# orders to payments
es = es.add_relationship(ft.Relationship(es['orders']['order_id'],
                                         es['payments']['order_id']))


# In[ ]:


es.add_last_time_indexes()


# In[ ]:


es


# In[ ]:


# We want to know if the customer will buy anything again after cutoff
cutoff_time = pd.Timestamp('July 1, 2018')


# # 60 days

# In[ ]:


# training window of only 2 months, then experiment with 4 months, 6 months
# find out if the customer will buy anything after 
# turns out, sliding windows are not supported yet
features, feature_names = ft.dfs(entityset = es, target_entity = 'customers', verbose=True, 
                                 cutoff_time=cutoff_time,
                                 training_window = ft.Timedelta("60 days"))


# In[ ]:


import missingno as msno
features_null_filtered = msno.nullity_filter(features, p=0.75)


# In[ ]:


features_encoded, features_names_encoded = ft.encode_features(features_null_filtered, feature_names)


# # Regression, finally

# In[ ]:


orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
last_timestamp_per_customer = orders.reset_index().groupby("customer_id")["order_purchase_timestamp"].max()
customer_bought_after_cutoff = last_timestamp_per_customer > cutoff_time


# In[ ]:


from sklearn.model_selection import train_test_split

X = features_encoded
y = customer_bought_after_cutoff

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

imputer = Imputer(strategy='most_frequent')
t_svd = TruncatedSVD(n_components=100)
log_res = LogisticRegression(C=0.1)

pipeline = make_pipeline(imputer, t_svd, log_res)
pipeline.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

def complete_evaluation(y_test, y_probs, y_preds, positive_class=1):
    """
    Complete evaluation
    """
    acc = accuracy_score(y_test, y_preds)
    average_prec = average_precision_score(y_test, y_probs[:, positive_class])
    precs, recs, fscore, support = precision_recall_fscore_support(y_test, y_preds)
    prec_0, prec_1 = precs
    rec_0, rec_1 = recs
    fscore_0, fscore_1 = fscore 
    support_0, support_1 = support
    
    balanced_accuracy = (rec_0 + rec_1) / 2

    log_loss_value = log_loss(y_test, y_probs)
    auc_value = roc_auc_score(y_test, y_probs[:, positive_class])

    return pd.DataFrame([{"accuracy" : acc, "loss" : log_loss_value, "auc" : auc_value, "average_precision" : average_prec,
                  "precision_0" : prec_0, "precision_1" : prec_1, 
                  "recall_0" : rec_0, "recall_1" : rec_1, "fscore_0" : fscore_0, "fscore_1" : fscore_1,
                  "support_0" : support_0, "support_1" : support_1, "balanced_accuracy" : balanced_accuracy}])


# In[ ]:


y_proba = pipeline.predict_proba(X_test)

list_results = []
for threshold in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
    y_preds = y_proba[:, 1] > threshold
    result = complete_evaluation(y_test, y_proba, y_preds)
    result["threshold"] = threshold
    result["number_predictions"] = (y_preds == 1).sum()
    list_results.append(result)
    
df_results_thresholds = pd.concat(list_results)


# In[ ]:


df_results_thresholds


# In[ ]:


width = 0.3
plt.figure(figsize=(15,5))
df_results_thresholds.plot(y=['number_predictions', "precision_1"], x="threshold", position=0, kind='bar',
                                    width=width, secondary_y= 'number_predictions', rot=0, )


# # 120 days

# In[ ]:


# 120 days training window
features, feature_names = ft.dfs(entityset = es, target_entity = 'customers', verbose=True, 
                                 cutoff_time=cutoff_time,
                                 training_window = ft.Timedelta("120 days"))


# In[ ]:


import missingno as msno
features_null_filtered = msno.nullity_filter(features, p=0.75)


# In[ ]:


features_encoded, features_names_encoded = ft.encode_features(features_null_filtered, feature_names)


# # Regression, finally

# In[ ]:


last_timestamp_per_customer = orders.reset_index().groupby("customer_id")["order_purchase_timestamp"].max()
customer_bought_after_cutoff = last_timestamp_per_customer > cutoff_time


# In[ ]:


from sklearn.model_selection import train_test_split

X = features_encoded
y = customer_bought_after_cutoff

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


imputer = Imputer(strategy='most_frequent')
t_svd = TruncatedSVD(n_components=100)
log_res = LogisticRegression(C=0.1)

pipeline = make_pipeline(imputer, t_svd, log_res)
pipeline.fit(X_train, y_train)


# In[ ]:


y_proba = pipeline.predict_proba(X_test)

list_results = []
for threshold in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
    y_preds = y_proba[:, 1] > threshold
    result = complete_evaluation(y_test, y_proba, y_preds)
    result["threshold"] = threshold
    result["number_predictions"] = (y_preds == 1).sum()
    result["average f-score"] = ((result["fscore_0"] + result["fscore_1"])/2).iloc[0]
    list_results.append(result)
    
df_results_thresholds = pd.concat(list_results)


# In[ ]:


df_results_thresholds


# In[ ]:


width = 0.3
plt.figure(figsize=(15,5))
df_results_thresholds.plot(y=['number_predictions', "precision_1"], x="threshold", position=0, kind='bar',
                                    width=width, secondary_y= 'number_predictions', rot=0, )


# ## A more complex model -- Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

imputer = Imputer(strategy='most_frequent')
random_forest = RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=10, 
                                       class_weight='balanced_subsample')
# gbt = GradientBoostingClassifier(subsample=0.5, n_iter_no_change=3, verbose=True,)

pipeline = make_pipeline(imputer, random_forest)
pipeline.fit(X_train, y_train)


# In[ ]:


def name_scores(featurecoef, col_names, label="Score", sort=False):
    """
    Generates a DataFrame with all the independent variables used in the model with their corresponding coefficient
    :param featurecoef: model.coef_ | model.feature_importances_
    :param column: string to be anonymized
    :param label: Name of the column where coefficients will be added
    :param sort: False = Decending, True = Ascending
    :return: pandas DataFrame
    """
    
    df_feature_importance = pd.DataFrame([dict(zip(col_names, featurecoef))]).T.reset_index()
    df_feature_importance.columns = ["Feature", label]
    if sort:
        return df_feature_importance.sort_values(ascending=False, by=label)
    return df_feature_importance


# In[ ]:


pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)


# In[ ]:


name_scores(random_forest.feature_importances_, X_train.columns, sort=True)[:20]


# In[ ]:


y_proba = pipeline.predict_proba(X_test)
sns.distplot(y_proba[:, 1])

list_results = []
for threshold in [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65]:
    y_preds = y_proba[:, 1] > threshold
    result = complete_evaluation(y_test, y_proba, y_preds)
    result["threshold"] = threshold
    result["number_predictions"] = (y_preds == 1).sum()
    result["average f-score"] = ((result["fscore_0"] + result["fscore_1"])/2).iloc[0]
    list_results.append(result)
    
df_results_thresholds = pd.concat(list_results)


# In[ ]:


df_results_thresholds


# In[ ]:


width = 0.3
plt.figure(figsize=(15,5))
df_results_thresholds.plot(y=['number_predictions', "precision_1"], x="threshold", position=0, kind='bar',
                                    width=width, secondary_y= 'number_predictions', rot=0, )


# ## Gradient Boosted Trees

# In[ ]:


imputer = Imputer(strategy='most_frequent')
gbt = GradientBoostingClassifier(verbose=True,)

pipeline = make_pipeline(imputer, gbt)
pipeline.fit(X_train, y_train)


# In[ ]:


name_scores(gbt.feature_importances_, X_train.columns, sort=True)[:20]


# In[ ]:


y_proba = pipeline.predict_proba(X_test)
sns.distplot(y_proba[:, 1])

list_results = []
for threshold in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
    y_preds = y_proba[:, 1] > threshold
    result = complete_evaluation(y_test, y_proba, y_preds)
    result["threshold"] = threshold
    result["number_predictions"] = (y_preds == 1).sum()
    result["average f-score"] = ((result["fscore_0"] + result["fscore_1"])/2).iloc[0]
    list_results.append(result)
    
df_results_thresholds = pd.concat(list_results)


# In[ ]:


df_results_thresholds


# In[ ]:


width = 0.3
plt.figure(figsize=(15,5))
df_results_thresholds.plot(y=['number_predictions', "precision_1"], x="threshold", position=0, kind='bar',
                                    width=width, secondary_y= 'number_predictions', rot=0, )

