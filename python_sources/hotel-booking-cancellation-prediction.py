#!/usr/bin/env python
# coding: utf-8

# # DATA ACQUISITION

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', None)


# In[ ]:


df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


df.head(10)


# In[ ]:


df.info()


# **Data types:**
# * Categorical - hotel, is_canceled, customer_type, is_repeated_guest, meal, country, market_segment, distribution_channel, reserved_room_type, assigned_room_type, deposit_type, agent, company, reservation_status,
# * Numerical - lead_time, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, previous_cancellations, booking_changes, previous_bookings_not_canceled, days_in_waiting_list, adr, required_car_parking_spaces, total_of_special_requests	
# * Ordinal - arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month,  reservation_status_date

# **Mising values:**
# 1. country
# 2. agent
# 3. company

# In[ ]:


df.describe()


# 
# * The following columns previous_cancellations, previous_bookings_not_canceled, booking_changes days_in_waiting_list, required_car_parking_spaces, total_of_special_requests have only a maximum value. This shows that these features contribute to the decision of cancellation only in very few cases.

# **Assumptions about impact of features:**
# * High:
# hotel, lead_time, arrival_date_year, arrival_date_month, stays_in_weekend_nights, stays_in_week_nights, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, assigned_room_type, deposit_type, days_in_waiting_list, customer_type
# 
# * Medium:
# children, babies, distribution_channel, booking_changes, adr
# 
# * Low:
# arrival_date_week_number, arrival_date_day_of_month,country, meal, adults, market_segment, agent, company, required_car_parking_spaces, total_of_special_requests, reservation_status, reservation_status_date

# **Assumptions about cancellation:**
# 1. The type of hotel decides the cancelation rate with higher cancellations in city hotels as compared to resort hotels due to variety of facilities available in resort hotels.
# 2. The earlier the booking made, higher the chances of cancellation.
# 3. Customers who have bookings for longer durations have lesser chance of cancelling their booking. 
# 4. As more children or babies are involved in the booking, higher chances of cancellation.
# 5. Old guest (is_repeated_guest=1) is less likely to cancel current booking.
# 6. If there are high previous cancellations, possibility of cancellation of current booking is also high.
# 7. If room assigned is not the same as reserved room type, customer might positively cancel the booking.
# 8. Higher the number of changes made to the booking, lesser is the chance of cancellation due to the investment of time in curating the booking as per one's requirement.
# 9. Bookings that are refundable or for which deposits were not made at the time of booking stand a high chance of cancelation.
# 10. If the number of days in waiting list is significant, customer might make some other booking due to uncertainty of confirmation of current booking.

# **Target variable:**
# is_canceled

# # EXPLORATORY DATA ANALYSIS

# ***UNIVARIATE ANALYSIS (Checking the validity of assumptions)***

# In[ ]:


is_can = len(df[df['is_canceled']==1])
print("Percentage cancelation= ", is_can/len(df))
df['reservation_status'].value_counts(normalize=True)*100


# In[ ]:


corr= df.corr(method='pearson')['is_canceled'][:]
corr


# * highest positive correlations : lead_time followed by previous_cancellations
# * highest negative correlations : total_of_special_requests, required_car_parking_spaces

# In[ ]:


sns.countplot(data=df, x='hotel', hue='is_canceled')
resort_canceled = df[(df['hotel']=='Resort Hotel') & (df['is_canceled']==1)]
city_canceled = df[(df['hotel']=='City Hotel') & (df['is_canceled']==1)]
print('Cancelations in resort hotel= ', (len(resort_canceled))/(len(df[df['hotel']=='Resort Hotel'])))
print('Cancelations in city hotel= ', (len(city_canceled))/(len(df[df['hotel']=='City Hotel'])))


# Our 1st assumption, city hotels have higher cancelation rate than resort hotels, is valid.

# In[ ]:


grid = sns.FacetGrid(df, col='is_canceled')
grid.map(plt.hist, 'lead_time', width=50)
grid.add_legend()


# Maximum cancelations occur if the booking is made 60-70 days before the checkin date. Longer the lead_time, lower is the cancelation. This invalidates our 2nd assumption. 

# In[ ]:


print(len(df[(df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0)])) 


# 715 bookings don't have both weekday or weekend nights which could be ar error in the data as this is not possible in real life scenario. Therefore these rows can be eliminated from the dataset.

# In[ ]:


((len(df.loc[(df['children']!=0) | (df['babies']!=0)]))/(len(df))) * 100


# The number of customers having children or babies or both are only 8% of the total population. Therefore this information can be ignored as it will not play a significatn role in deciding whether to cancel the booking or not. Assumption 4 can be discarded.

# In[ ]:


sns.countplot(data=df, x='is_repeated_guest', hue='is_canceled')
new_guest = df[(df['is_repeated_guest']==0) & (df['is_canceled']==1)]
old_guest = df[(df['is_repeated_guest']==1) & (df['is_canceled']==1)]
print('Cancelations among new guests= ', (len(new_guest))/(len(df[df['is_repeated_guest']==0])))
print('Cancelations among old guests= ', (len(old_guest))/(len(df[df['is_repeated_guest']==1])))


# As seen in the correlation table, the above graph bolsters the evidence that maximum customers are new comers and they are less likely to cancel their current booking.
# Old guests are less likely to cancel the booking (14%). Assumption 5 holds true.

# In[ ]:


sns.countplot(data=df, x='previous_cancellations', hue='is_canceled')


# Maximum customers have 0 previous cancellations. They are less likely to cancel the current booking. However, customers who have cancelled once earlier are more likely to cancel the current booking. This also matches with the positive correlation between previous_cancellations and is_cancelled and supports Assumption 6.

# In[ ]:


temp = df.loc[df['reserved_room_type']!=df['assigned_room_type']]
temp['is_canceled'].value_counts(normalize=True)*100


# Assumption 7 that there more cancellations when assigned room type is different from reserved room type is not valid. There are only 5% cancellations in such a case.

# In[ ]:


sns.pointplot(data=df, x='booking_changes', y='is_canceled')


# Assumption 8 about the bookings does not hold as there is no trend in it's impact on the cancellation of bookings.

# In[ ]:


sns.countplot(x="deposit_type", hue="is_canceled",data=df);


# Contrary to assumption 9, bookings that are non_refundable are canceled.

# In[ ]:


sns.relplot(data=df, x='days_in_waiting_list', y='is_canceled', kind='line', estimator=None)


# No relation can be established between days_in_waiting_list and is_canceled. Therefore, we will take this feature for further analysis. Assumption 10 can be discarded.

# In[ ]:


sns.countplot(data=df, x='arrival_date_year', hue='is_canceled')


# In[ ]:


chart = sns.catplot(data=df, x='arrival_date_month', hue='is_canceled', kind='count')
chart.set_xticklabels(rotation=65, horizontalalignment='right')


# Maximum bookings occur in 2016 in the months of July and August.

# In[ ]:


year_count = df.groupby(['arrival_date_year', 'is_canceled']).size().to_frame(name='count')
year_perct = year_count.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
print(year_perct)

month_count = df.groupby(['arrival_date_month', 'is_canceled']).size().to_frame(name='count')
month_perct = month_count.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
print(month_perct)


# Percentage of cancellations was higher in 2015 and 2017 despite higher number of bookings in 2016. April and June had the largest cancellations overall.

# In[ ]:


chart = sns.catplot(data=df, x='market_segment', kind='count', hue='is_canceled')
chart.set_xticklabels(rotation=65, horizontalalignment='right')


# In[ ]:


sns.countplot(data=df, x='distribution_channel', hue='is_canceled')


# In[ ]:


print(df['customer_type'].value_counts(normalize=True)*100)
sns.countplot(data=df, x='customer_type', hue='is_canceled')


# 75% bookings occur in Transient category of customers. It also sees the highest cancellation among all the categories.

# In[ ]:


df['reservation_status'].unique()


# ***MULTIVARIATE ANALYSIS***

# In[ ]:


grid = sns.FacetGrid(df, col='arrival_date_year')
grid.map(sns.countplot, 'hotel')


# In all three years city hotels saw more bookings than resort hotels.

# In[ ]:


df['meal'].nunique(), df['customer_type'].nunique()
grid = sns.FacetGrid(df, col='customer_type')
grid.map(sns.countplot, 'meal')


# All kinds of customers prefer BB type meals majorly.

# In[ ]:


df.pivot_table(columns='hotel', values='country', aggfunc=lambda x:x.mode())


# People from country with ISO code 'PRT' made the most number of bookings in both types of hotels.

# In[ ]:


g = sns.countplot(data=df, x='hotel', hue='reserved_room_type')
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# * Resort hotels room preference : A, D, E
# *  City hotels room preference : A, D, F

# In[ ]:


print("TABLE 1")
print(df.groupby(['hotel', 'customer_type']).size())


# For each kind of hotel, Transient type of customers are the highest followed by Transient Party. Group bookings are the least.

# In[ ]:


print(df.groupby(['customer_type', 'deposit_type']).size())


# Each category of customers book hotels without deposit. Surprisingly, between refundable and non-refundable type, higher number of people book hotels that are non-refundable.

# In[ ]:


print(df.groupby(['customer_type', 'distribution_channel']).size())
print("-"*60)
print(df.groupby(['customer_type', 'market_segment']).size())


# In[ ]:


print(df.groupby(['hotel', 'distribution_channel']).size())
print("-"*40)
print(df.groupby(['hotel', 'market_segment']).size())


# Combining table 1 and above table, we see the relation between freqeunt customer types at each hotel and their mode of booking. This information can be used by the hotel to focus on customised publicity stratgies. Similarly, the market segments can be analysed for a more customer centric approach. Hotel type with distribution channel and market segment can also be analysed. 

# In[ ]:


group = df.groupby(['customer_type', 'reservation_status']).size()
group_pcts = group.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
group_pcts


# In[ ]:


df.pivot_table(columns='hotel', values=['stays_in_weekend_nights', 'stays_in_week_nights'], aggfunc=lambda x:x.sum())


# In[ ]:


df.pivot_table(columns='hotel', values='total_of_special_requests', aggfunc=lambda x:x.sum())


# In[ ]:


sns.catplot(data=df, x='hotel', y='days_in_waiting_list', jitter=False)


# As it is seen, city hotels have much larger waiting time in days compared to resort hotels which may signify that their demad is higher.

# In[ ]:


df['country'].value_counts(normalize=True)*100


# In[ ]:


temp = df.loc[(df['country']=='PRT') | (df['country']=='GBR') | (df['country']=='FRA') | (df['country']=='ESP') | (df['country']=='DEU')]
grid = sns.FacetGrid(temp, col='country')
grid.map(sns.countplot, 'distribution_channel')


# Using this information hotels can implement models of publicity for getting more bookings in the top 5 countries from where most of their customers hail.

# In[ ]:


sns.barplot(data=df, x='customer_type', y='total_of_special_requests', ci=None)


# In[ ]:


sns.boxplot(data=df, x='distribution_channel', y='lead_time')


# # FEATURE ENGINEERING

# In[ ]:


month_map = {'January':'01', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09', 'October':'10', 'November':'11', 'December':'12'}
df.arrival_date_month = df.arrival_date_month.map(month_map).astype(int)


# In[ ]:


df['arrival_date'] = df['arrival_date_year'].astype(str)+'-'+df['arrival_date_month'].astype(str)+'-'+df['arrival_date_day_of_month'].astype(str)


# In[ ]:


def roomChange(row):
    if row['assigned_room_type'] == row['reserved_room_type']:
        return False
    else:
        return True

df['change_in_room'] = df.apply(roomChange, axis=1)


# In[ ]:


df['children'] = df['children'].fillna(0)
df['offspring'] = (df['children'] + df['babies']).astype(int)


# In[ ]:


df['total_bookings'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']


# In[ ]:


df['country'].fillna(df['country'].mode()[0], inplace=True)
df['agent'].fillna(df['agent'].mode()[0], inplace=True)
df['company'].fillna(df['company'].mode()[0], inplace=True)


# In[ ]:


for i in range(len(df)):
    if df.loc[i, 'country'] == 'PRT':
        df.at[i, 'country'] = 1
    elif df.loc[i, 'country'] == 'GBR':
        df.at[i, 'country'] = 2
    else:
        df.at[i, 'country'] = 0


# In[ ]:


df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df['arrival_date'] = pd.to_datetime(df['arrival_date'])


# In[ ]:


df['stay_duration'] = df['reservation_status_date'] - df['arrival_date']
df['stay_duration'] = df['stay_duration'] / np.timedelta64(1, 'D')
df['stay_duration'] = df['stay_duration'].astype(int)


# In[ ]:


for i in range(len(df)):
    if df.loc[i, 'stay_duration']<0:
        df.at[i, 'stay_duration'] = -1


# In[ ]:


lb = LabelEncoder()
var = ['hotel', 'customer_type', 'deposit_type', 'change_in_room', 'market_segment', 'distribution_channel', 'country']
for item in var:
    df[item] = lb.fit_transform(df[item])
df = pd.get_dummies(df, columns=['hotel', 'customer_type', 'deposit_type', 'change_in_room', 'market_segment', 'distribution_channel', 'country'])


# In[ ]:


df.drop(['meal', 'assigned_room_type', 'reserved_room_type', 'reservation_status', 'reservation_status_date', 'arrival_date'], axis=1, inplace=True)


# # MODEL AND RESULT

# In[ ]:


train_var = df.drop(['is_canceled'], axis=1)
test_var = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(train_var, test_var, test_size=0.20)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


#Data scaling
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)

mm_scaler = MinMaxScaler()
mm_scaler.fit(X_train)
X_train_mm = mm_scaler.transform(X_train)
X_test_mm = mm_scaler.transform(X_test)


# In[ ]:


#Logistic Regression
logreg = LogisticRegression(max_iter=500).fit(X_train_mm, y_train)
scores = cross_val_score(logreg, X_train_mm, y_train, cv=5)
logreg_pred = logreg.predict(X_test_mm)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(logreg.score(X_test_mm, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, logreg_pred)))
print(confusion_matrix(y_test, logreg_pred))


# In[ ]:


#Linear SVC
svc = LinearSVC().fit(X_train_mm, y_train)
scores = cross_val_score(svc, X_train_mm, y_train, cv=5)
svc_pred = svc.predict(X_test_mm)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(svc.score(X_test_mm, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, svc_pred)))
print(confusion_matrix(y_test, svc_pred))


# In[ ]:


#SGD Classifier
sgd = SGDClassifier(alpha=0.1).fit(X_train_std, y_train)
scores = cross_val_score(sgd, X_train_std, y_train, cv=5)
sgd_pred = sgd.predict(X_test_std)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(sgd.score(X_test_std, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, sgd_pred)))
print(confusion_matrix(y_test, sgd_pred))


# In[ ]:


#Ridge Classifier
rc = RidgeClassifier(alpha=1, normalize=True)
rc.fit(X_train, y_train)
scores = cross_val_score(rc, X_train, y_train, cv=5)
rc_pred = rc.predict(X_test)
print("Normalized data:")
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(rc.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, rc_pred)))
print(confusion_matrix(y_test, rc_pred))

rc = RidgeClassifier(alpha=1)
rc.fit(X_train_std, y_train)
scores = cross_val_score(rc, X_train_std, y_train, cv=5)
rc_pred = rc.predict(X_test_std)
print("Standard scaled data:")
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(rc.score(X_test_std, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, rc_pred)))
print(confusion_matrix(y_test, rc_pred))


# In[ ]:


#KNN
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 6)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
scores = cross_val_score(knn, X_train, y_train, cv=5)
knn_pred = knn.predict(X_test)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(knn.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, knn_pred)))
print(confusion_matrix(y_test, knn_pred))


# In[ ]:


#Decision Tree
tree = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
scores = cross_val_score(tree, X_train, y_train, cv=5)
tree_pred = tree.predict(X_test)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(tree.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, tree_pred)))
print(confusion_matrix(y_test, tree_pred))


# In[ ]:


def classifier(train, test, estimator, param_grid):
    grid_search = GridSearchCV(estimator, param_grid, cv=5)
    grid_search.fit(train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print("Test score: {:.3f}".format(grid_search.score(test, y_test)))

def feature_selection(model):
    select_features = SelectFromModel(estimator=model, threshold='median')
    select_features.fit(X_train, y_train)
    X_train_select = select_features.transform(X_train)
    X_test_select = select_features.transform(X_test)
    return X_train_select, X_test_select

def run_model(model, model_feature, param_grid):
    print("Before feature selection:")
    classifier(X_train, X_test, model, param_grid)
    X_train_select, X_test_select = feature_selection(model_feature)
    print("After feature selection")
    classifier(X_train_select, X_test_select, model, param_grid)


# In[ ]:


#Random Forest
param_grid = {'n_estimators':[50,75,100], 'max_depth':[1,2,5]}
run_model(RandomForestClassifier(), RandomForestClassifier(n_estimators=50, max_depth=2), param_grid)


# In[ ]:


#Gradient Boosted Classifier
param_grid = {'max_depth':[1,2,5], 'learning_rate':[1,0.1,0.001]}
run_model(GradientBoostingClassifier(), GradientBoostingClassifier(learning_rate=0.001), param_grid)


# In[ ]:


#Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
scores = cross_val_score(gnb, X_train, y_train, cv=5)
gnb_pred = gnb.predict(X_test)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(gnb.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, gnb_pred)))
print(confusion_matrix(y_test, gnb_pred))


# In[ ]:


#Multi Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=[35, 20], alpha=0.001, solver='adam', activation='relu')
mlp.fit(X_train_std, y_train)
mlp_pred = mlp.predict(X_test_std)
print("Train score: {:.3f}".format(mlp.score(X_train_std, y_train)))
print("Test accuracy: {:.3f}".format(mlp.score(X_test_std, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, mlp_pred)))
print(confusion_matrix(y_test, mlp_pred))


# In[ ]:


#Adaboost Classifier
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
scores = cross_val_score(ada, X_train, y_train, cv=5)
print("Average cross validation score: {:.3f}".format(scores.mean()))
print("Test accuracy: {:.3f}".format(ada.score(X_test, y_test)))
print("F1 score: {:.3f}".format(f1_score(y_test, ada_pred)))
print(confusion_matrix(y_test, ada_pred))


# In[ ]:




