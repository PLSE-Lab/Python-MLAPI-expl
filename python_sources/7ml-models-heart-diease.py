# %% [markdown]
# # Prediction of Heart Disease via 3 ML models

# %% [markdown]
# We will apply 3 different ML models to predict whether patient has HEART DISEASE ( 1) or not (0).
# 1. KNearestNeighbors
# 2. RandomForestClassifier
# 3. DecisionTreeClassifier

# %% [code]
#Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# %% [code]
#Importing Dataset
data = pd.read_csv('dataset.csv')
data.head()

# %% [code]
data.info()

# %% [code]
data.describe()

# %% [code]
for col in data.columns:
    print(col, ': ', len(data[col].unique()),'labels')

# %% [code]
#Lets see correlation
corrmat = data.corr()
corrmat

# %% [code]
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# %% [markdown]
# We can see Target ( dependent variable ) has positive relation with CP and thalach whereas negative relation with oldpeak,exang

# %% [code]
data.hist()

# %% [code]
sns.set_style('whitegrid')
sns.countplot(data=data,x='target')

# %% [markdown]
# Since we can see the dataset is BALANCED and we can see which features are correlated, we can go to data pre processing steps.

# %% [markdown]
# ## Data Pre-processing

# %% [code]
data = pd.get_dummies(data,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

# %% [code]
data.head()

# %% [code]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
col_to_scale = ['age','trestbps','chol','thalach','oldpeak']
data[col_to_scale] = sc.fit_transform(data[col_to_scale])

# %% [code]
data.head()

# %% [code]
#Dataset
X = data.drop(['target'],axis=1)
y = data['target']

# %% [code]
X

# %% [code]
y

# %% [markdown]
# ## Applying ML models 

# %% [markdown]
# ## KNN Classifier

# %% [code]
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())

# %% [code]
plt.figure(figsize=(20,30))
plt.plot([k for k in range(1,21)],knn_scores,color='red')
for i in range(1,21):
    plt.text(i,knn_scores[i-1],(i,knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K_values')

# %% [code]
#we can see 12 is the best K value with 85% accuracy
knn_classifier = KNeighborsClassifier(n_neighbors=12)
knn_score = cross_val_score(knn_classifier,X,y,cv=10)

# %% [code]
KNN_Score = round(knn_score.mean()*100,2)
KNN_Score

# %% [markdown]
# ## RandomForest Classifier

# %% [code]
from sklearn.ensemble import RandomForestClassifier
randomClassifier= RandomForestClassifier(n_estimators=10)
randomForest_score = cross_val_score(randomClassifier,X,y,cv=10)

# %% [code]
Random_Forest_Score=round(randomForest_score.mean()*100,2)
Random_Forest_Score

# %% [markdown]
# ## DecisionTree Classifier

# %% [code]
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
Descion_tree_score = cross_val_score(dtree,X,y,cv=10)

# %% [code]
Desicion_Tree_Score= round(Descion_tree_score.mean()*100,2)
Desicion_Tree_Score

# %% [markdown]
# ## Logistic Regression 

# %% [code]
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X,y)
lg_score = cross_val_score(lg,X,y,cv=10)

# %% [code]
Logistic_Regression_Score = round(lg_score.mean()*100,2)
Logistic_Regression_Score

# %% [code]
#Doing it without cross value
#from sklearn.linear_model import LogisticRegression
#lg_og = LogisticRegression()
#lg_og.fit(X,y)
#lg_og_score = round(lg_og.score(X,y).mean(),2)
#lg_og_score

# %% [code]
coeff_df = pd.DataFrame(data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(lg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# %% [code]


# %% [markdown]
# ## Support Vector Machine 

# %% [code]
from sklearn.svm import SVC
svc = SVC()
svc.fit(X,y)
Support_Vector_Machine_Score = round(svc.score(X,y).mean()*100,2)
Support_Vector_Machine_Score

# %% [markdown]
# ## Naive Bayes 

# %% [code]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X,y)
Naive_Bayes_Score = round(gnb.score(X,y).mean()*100,2)
Naive_Bayes_Score

# %% [markdown]
# ## Perceptron 

# %% [code]
from sklearn.linear_model import Perceptron
p = Perceptron()
p.fit(X,y)
Perceptron_Score = round(p.score(X,y).mean()*100,2)
Perceptron_Score

# %% [markdown]
# ## Model Evaluation

# %% [code]
model = pd.DataFrame( {
    'Model': ['K_Nearest_Neighbors_Classifer',
              'Logistic Regresion',
              'Decision TreeClassifier',
              'Support Vector Machine',
              'Random Forest Classifier',
             'Naive Bayes','Perceptron'],
    'Score': [KNN_Score,
              Logistic_Regression_Score,
              Desicion_Tree_Score,
              Support_Vector_Machine_Score,
              Random_Forest_Score,
              Naive_Bayes_Score,
              Perceptron_Score]
})

# %% [code]
model

# %% [code]
model_sorted = model.sort_values(by='Score',ascending=False)
model_sorted

# %% [code]
#Visualization
plt.figure(figsize=[25,6])
plt.tick_params(labelsize=14)
x = model_sorted['Model']
y = model_sorted['Score']
plt.bar(x,y)
plt.ylim(bottom=70,top=90)

plt.title('Score of 7 Supervised Machine Learning Models')
plt.xlabel('Models')
plt.ylabel('Score, %')
plt.show()

# %% [code]
