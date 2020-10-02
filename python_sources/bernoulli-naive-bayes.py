import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB  # Bernoulli distribution naive Bayes classifier
sns.set(style="white", color_codes=True)
seed = 42
np.random.seed(seed)

# Best score so far: 0.77033 without zones,
# 0.75 with zones


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64})

print("\n\nTop of the training data:")
print(train.head(2))
print("\n\nTop of the test data:")
print(test.head(2))

print("\n\nTraining set survival:")
print(train.Survived.value_counts())

#print(train.Parch.value_counts())

# Extract the cabin zone from the cabin numbers
train["Zone"] = train["Cabin"].map(lambda xs: xs[:1] if pd.notnull(xs) else "NA")
print(train.Zone.value_counts(dropna=False))
plt.figure()
sns.factorplot("Zone", "Survived", "Sex", data=train, kind="bar", size=6, palette="muted")
plt.savefig('titanic_zone_survival.png')
# Looks promising. Too bad we have the cabin numbers only for 25% of the data points.
test["Zone"] = test["Cabin"].map(lambda xs: xs[:1] if pd.notnull(xs) else "NA")
# One-hot encode
train = train.join(pd.get_dummies(train['Zone'], 'Zone')) #.drop('Sex', axis=1)
test = test.join(pd.get_dummies(test['Zone'], 'Zone')) #.drop('Sex', axis=1)


# Transform the symbolic values into numbers suitable for the Bayes classifier
#train['Sex'] = pd.factorize(train['Sex'])[0]
#test['Sex'] = pd.factorize(test['Sex'])[0]
# One-hot encode to compensate for the several buckets we have for age and class
train = train.join(pd.get_dummies(train['Sex'], 'Sex')) #.drop('Sex', axis=1)
test = test.join(pd.get_dummies(test['Sex'], 'Sex')) #.drop('Sex', axis=1)


# Factor plot: something to try
# sns.factorplot("Pclass", "Survived", "Sex", \
#                   data=train, kind="bar", \
#                   size=6, palette="muted", legend=False)

# Let's see how the fare relates to Pclass
plt.figure()
plot = sns.FacetGrid(train, hue="Pclass", size=6) \
  .map(sns.kdeplot, "Fare") \
  .add_legend()
plot.fig.get_axes()[0].set_yscale('log')
plt.savefig("titanic_fare_class_kde.png")

# Looks like we have two more sub-classes within Pclass 1 with fare 200-300 and ~500.
# We should probably model them within fare buckets.
# There are a few annoying extremely high and low fare points, but they seem to be relevant.
# It looks like we need about 5-7 buckets to catch all the fare clusters. Have to experiment.
# Fare in the low-mid range does not seem to predict much.
# But some values under ~8 and over 500 look useful, and 100-300 may add value to Pclass.
# So let's use the extreme low and high buckets, and ignore the middle ones.

plt.figure()
#ax = sns.boxplot(x="Pclass", y="Fare", data=train)
ax = sns.stripplot(x="Pclass", y="Fare", data=train, jitter=True, edgecolor="gray")
ax.set_yscale('log')
plt.savefig('titanic_fare_class_points.png')
# It does seem to have a strong correllation, which might make
# the Bayes classifier overemphasize it if we include both. Let's skip it for now.
# There are still interesting clusters at 200-300 and 500 within 1st class,
# which might be relevant. Maybe try to include them in the next version.

# Now what about fare and survival?
plt.figure()
ax = sns.boxplot(x="Survived", y="Fare", data=train)
ax = sns.stripplot(x="Survived", y="Fare", ax=ax, data=train, hue="Sex", jitter=True, edgecolor="gray")
ax.set_yscale('log')
plt.savefig('titanic_fare_survival_points.png')

#print("\n\nDEBUG FARE 1:")

#print(train.Fare.dtype)  # float64
#print(test.Fare.dtype)  # float64

#train.replace([np.inf, -np.inf], np.nan)

#train = train[(train.Fare >= 0) & (train.Fare < 20)]

#train.info()
#print(np.isfinite(train.Fare.sum()))  # True
#print(train.Fare.dtype.char in np.typecodes['AllFloat'])  # True
#print(np.isfinite(train.Fare).all())  # True
#print(train.Fare.value_counts())
#print(train[train.Fare.isnull()].describe())

test['Fare'] = test['Fare'].fillna(0)
#print("\n\nDEBUG FARE 2:")
#print(train.Fare.value_counts())
#print(train[train.Fare.isnull()].describe())

#km1 = KMeans(n_clusters=7).fit(np.nan_to_num(train[['Fare']].as_matrix()))
km = KMeans(n_clusters=7, random_state=seed).fit(train[['Fare']])
fareLabels = km.labels_
fareClustered = pd.DataFrame(data=fareLabels, columns=['FareCluster'], index=train.index)
# show the buckets
train = train.join(fareClustered)
print(train.describe())
plt.figure()
sns.boxplot(x="FareCluster", y="Fare", data=train)
sns.stripplot(x="FareCluster", y="Fare", data=train, jitter=True, edgecolor="gray")
plt.savefig('titanic_fare_buckets.png')

testFareLabels = km.predict(test[['Fare']])
testFareClustered = pd.DataFrame(data=testFareLabels, columns=['FareCluster'], index=test.index)

# Also convert to one-hot encoding
train = train.join(pd.get_dummies(fareClustered['FareCluster'], 'Fare')).drop('Fare', axis=1)
test = test.join(pd.get_dummies(testFareClustered['FareCluster'], 'Fare')).drop('Fare', axis=1)


# Get one-hot encoding for column Pclass
train = train.join(pd.get_dummies(train['Pclass'], 'Pclass')).drop('Pclass', axis=1)
test = test.join(pd.get_dummies(test['Pclass'], 'Pclass')).drop('Pclass', axis=1)

print("\n\nStats before handling the N/A values:")
print(train.describe())

# We don't know the age in ~20% cases.
# It seems to be a little better to fill them with mean values than to drop them.
#train = train[train.Age.notnull()]
train.fillna(train.mean(), inplace=True)

print("\n\nStats after handling the N/A values:")
print(train.describe())

# See later if we can do something more meaningful here
test.fillna(test.mean(), inplace=True)

# Let's take a look at the distribution of survivors by age and by fare
plt.figure()
sns.FacetGrid(train, hue="Survived", size=6) \
  .map(sns.kdeplot, "Age") \
  .add_legend()
plt.savefig("titanic_age_survival_kde.png")

plt.figure()
#sns.boxplot(x="Survived", y="Age", data=train)
sns.stripplot(x="Survived", y="Age", data=train, hue="Sex", jitter=True, edgecolor="gray")
plt.savefig('titanic_age_survival_points.png')

# Disregard the outliers for training: one survivor over 80
train = train[(train.Age < 78)]
plt.figure()
#sns.boxplot(x="Survived", y="Age", data=train)
sns.stripplot(x="Survived", y="Age", data=train, hue="Sex", jitter=True, edgecolor="gray")
plt.savefig('titanic_age_survival_points_dropped_outliers.png')

# Let's map the age into 3 social groups: (kids, adults, seniors) with KMeans
km = KMeans(n_clusters=3, random_state=seed).fit(train[['Age']])
labels = km.labels_
clustered = pd.DataFrame(data=labels, columns=['AgeCluster'], index=train.index)
testLabels = km.predict(test[['Age']])
testClustered = pd.DataFrame(data=testLabels, columns=['AgeCluster'], index=test.index)

# Also convert to one-hot encoding
train = train.join(pd.get_dummies(clustered['AgeCluster'], 'Age')).drop('Age', axis=1)
test = test.join(pd.get_dummies(testClustered['AgeCluster'], 'Age')).drop('Age', axis=1)

columns = ["Age_0", "Age_1", "Age_2", \
  "Pclass_1", "Pclass_2", "Pclass_3", \
# best choice out of 7 fare buckets: the highest and the lowest
  "Fare_2", "Fare_4", \
# zones with 100% female survival:
#  "Zone_A", "Zone_B", "Zone_D", "Zone_F", \
# zones with the most data points
#  "Zone_B", "Zone_C", "Zone_D", "Zone_E", \
  "Sex_male", "Sex_female"]

trainDF = train[columns]
print("\n\nFinal train dataframe stats:")
print(trainDF.describe())

# Format the data and expected values for SKLearn
trainData = trainDF.as_matrix()
trainTarget = pd.DataFrame.as_matrix(train[["Survived"]]).ravel()
testData = pd.DataFrame.as_matrix(test[columns])

classifier = BernoulliNB()
classifier.fit(trainData, trainTarget)

predictedValues = classifier.predict(testData).astype(int)
print("\n\nTest set survival:")
print(itemfreq(predictedValues))

testResults = test[['PassengerId']]
testResults['Survived'] = predictedValues
print(testResults.head())

#Any files you save will be available in the output tab below
#np.savetxt("titanic_prediction.csv", predictedValues, fmt='%i')
testResults.to_csv('titanic_prediction.csv', index=False)