import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report 

hotel_data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

print(hotel_data.head())

hotel_data['total_stay'] = hotel_data['stays_in_weekend_nights'] + hotel_data['stays_in_week_nights']
hotel_data['family_members_count'] = hotel_data['adults'] + hotel_data['children'] + hotel_data['babies']
hotel_data.drop(['company','agent','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies'], axis=1, inplace=True)
hotel_data.dropna(inplace=True)

fig,axes = plt.subplots(nrows=3, ncols=2, figsize=(15,11))
snp.set_style('darkgrid')
snp.countplot(x='is_canceled', data=hotel_data, hue='hotel', ax=axes[0][0])
snp.countplot(x='arrival_date_month', data=hotel_data, hue='hotel', ax=axes[0][1])
snp.countplot('total_stay', data=hotel_data, ax=axes[1][0])
snp.countplot('family_members_count', data=hotel_data, hue='hotel', ax=axes[1][1])
snp.scatterplot(x='family_members_count', y='previous_cancellations', data=hotel_data, ax=axes[2][0], palette='coolwarm', hue='hotel')
snp.scatterplot(y='total_of_special_requests', x='family_members_count', data=hotel_data, hue='hotel')
snp.jointplot(x='family_members_count', y='total_stay', data=hotel_data)

hotel_type = pd.get_dummies(data=hotel_data['hotel'], drop_first=True)
meal_type = pd.get_dummies(data=hotel_data['meal'], drop_first=True)
deposit_types = pd.get_dummies(data=hotel_data['deposit_type'], drop_first=True)
hotel_data['arrival_date_month'] = hotel_data['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
hotel_data = pd.concat([hotel_data, hotel_type, meal_type, deposit_types], axis=1)
reserved = ['reserved_room_type', 'assigned_room_type', 'customer_type', 'reservation_status']
hotel_data = pd.get_dummies(data=hotel_data, columns=reserved, drop_first=True)
hotel_data.drop(['hotel','meal', 'deposit_type','country', 'market_segment', 'distribution_channel','reservation_status_date'], axis=1, inplace=True)

X = hotel_data.drop('total_of_special_requests', axis=1)
y = hotel_data['total_of_special_requests']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Random Forest Classifier
logmodel = RandomForestClassifier(n_estimators=100)
logmodel.fit(X_train, y_train)
prediction = logmodel.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction)) 

plt.show()
