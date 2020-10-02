#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input style = "float:right" type="submit" value="Toggle code">''')


# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={
    'figure.figsize':(12,7), 
    'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'},font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)


# In[ ]:


#Reading the data from the CSV file. We use a flag IS_FOR_KAGGLE to toggle the path of the data file.
IS_FOR_KAGGLE = True

if IS_FOR_KAGGLE:
    trd = pd.read_csv('../input/train.csv')
    tsd = pd.read_csv('../input/test.csv')
else:
    trd = pd.read_csv('train.csv')
    tsd = pd.read_csv('test.csv')

td = pd.concat([trd, tsd], ignore_index=True, sort  = False)


# # Titanic: Machine Learning from Disaster
# 
# https://www.kaggle.com/c/titanic/data
#     

# ### 1. Getting the hang of the data

# The data contains 1309 rows and 12 colunmns. Each row signifies a human on the deck and each column is an attribute corresponding to that human.

# In[ ]:


td.shape


# On diving deeper in to the columns, we see five integer type columns, five object types (strings) and two columns with decimal values. Also, all the columns do not have 1309 values which indicates that there are missing values.

# In[ ]:


td.info()


# ### 2. Missing values
# 
# The line plot and the heat map below show the number of missing values in all columns. There are multiple columns that have missing values
# 
# - Age 
# - Cabin
# - Embarked
# - Fare
# - Survived
# 
# **77.5%** of data is NOT available for the column 'Cabin'. Hence, it can be dropped in our analysis and prediction. An alternate approach would have been to predict values for the missing values. However, it is not recommended as the number of values available for Cabin is significantly lower.
# 
# In contrast, Embarked has only 2 missing missing values. As the column contains **0.15%** missing values, the values can be substituted by the mode of the column. 
# 
# The column 'Age' on the other hand has 177 missing values which contribute to **20%** on the total expected values. We will impute the age data by categorising people on the basis of their title and finding the median age.
# 
# Fare has one missing value.

# In[ ]:


pd.DataFrame(td.isnull().sum()).plot.line().set_title("Number of missing values in the given features")
td.isnull().sum()


# In[ ]:


sns.heatmap(td.isnull(), cbar = False).set_title("Missing values heatmap")


# 
# ### 3. Number of unique values
# 
# To comprehend the categorical data in the training data, we list down al the columns with the number of unique values they have. We infer that 
# 
# - survived and Sex can have **two** distinct values
# - Embarked and PClass have **three** distinct values
# 
# So, we have four columns of cateogrical data - **Survived, Sex, Embarked and PClass**.

# In[ ]:


td.nunique()


# # Features
# 
# ### 4. Survived ###
# 
# The horizontal bar plot below shows the percentage of people that survived and percentage of people that unfortunately couldn't make it out alive from the disaster. **More than 60% of the people on the ship died.**

# In[ ]:


(trd.Survived.value_counts(normalize=True) * 100).plot.barh().set_title("Training Data - Percentage of people survived and Deceased")


# ### 5. Pclass ###
# 
# Pclass or passenger class represents the traveling class of commuter. There were three classes. In the combined dataset, a clear majority **(709)** traveled in the third class, followed by the second **(277)** and then the first **(323)**. 
# 
# **The number of passengers in the third class was more than the number of passengers in first and second class combined.**
# 
# **With the missing survived values, More than 40% of the first class passengers were rescued.** The pattern differed for the second and third class survivors as roughly around **70% of the second class passengers lost their lives**. The numbers skyrocketed for the third class passengers. More than **80% of the third class passengers couldn't survive** the disaster.
# 
# 
# 

# In[ ]:


fig_pclass = trd.Pclass.value_counts().plot.pie().legend(labels=["Class 3","Class 1","Class 2"], loc='center right', bbox_to_anchor=(2.25, 0.5)).set_title("Training Data - People travelling in different classes")


# In[ ]:


pclass_1_survivor_distribution = round((trd[trd.Pclass == 1].Survived == 1).value_counts()[1]/len(trd[trd.Pclass == 1]) * 100, 2)
pclass_2_survivor_distribution = round((trd[trd.Pclass == 2].Survived == 1).value_counts()[1]/len(trd[trd.Pclass == 2]) * 100, 2)
pclass_3_survivor_distribution = round((trd[trd.Pclass == 3].Survived == 1).value_counts()[1]/len(trd[trd.Pclass == 3]) * 100, 2)
pclass_perc_df = pd.DataFrame(
    { "Percentage Survived":{"Class 1": pclass_1_survivor_distribution,"Class 2": pclass_2_survivor_distribution, "Class 3": pclass_3_survivor_distribution},  
     "Percentage Not Survived":{"Class 1": 100-pclass_1_survivor_distribution,"Class 2": 100-pclass_2_survivor_distribution, "Class 3": 100-pclass_3_survivor_distribution}})
pclass_perc_df.plot.bar().set_title("Training Data - Percentage of people survived on the basis of class")


# In[ ]:


for x in [1,2,3]:    ## for 3 classes
    trd.Age[trd.Pclass == x].plot(kind="kde")
plt.title("Age density in classes")
plt.legend(("1st","2nd","3rd"))


# In[ ]:


for x in ["male","female"]:
    td.Pclass[td.Sex == x].plot(kind="kde")
plt.title("Training Data - Gender density in classes")
plt.legend(("Male","Female"))


# In[ ]:


pclass_perc_df


# ### 6. Sex
# 
# Roughly around **65% of the tourists were male** while the remaining 35% were female. However, the percentage of female survivors was higher than the number of male survivors.
# 
# More than 80% male passengers had to die as compared to around 70% female passengers.

# In[ ]:


fig_sex = (trd.Sex.value_counts(normalize = True) * 100).plot.bar()
male_pr = round((trd[trd.Sex == 'male'].Survived == 1).value_counts()[1]/len(trd.Sex) * 100, 2)
female_pr = round((trd[trd.Sex == 'female'].Survived == 1).value_counts()[1]/len(trd.Sex) * 100, 2)
sex_perc_df = pd.DataFrame(
    { "Percentage Survived":{"male": male_pr,"female": female_pr},  "Percentage Not Survived":{"male": 100-male_pr,"female": 100-female_pr}})
sex_perc_df.plot.barh().set_title("Percentage of male and female survived and Deceased")
fig_sex


# ### 7. Age

# The description of column 'Age' tells us that the **youngest traveler onboard was aged around 2 months** and the **oldest commuter was 80 years**. The **average age of passengers onboard was just under 30 years.** 
# However, it must be remembered that these observations are with missing values. Since we know that the oldest passenger was 80, we can plot age range and number of people survived or died for that age range.
# 
# Apparently, higher number of children below age 10 were saved than died. For every other age group, the number of casualities was higher than the number of survivors. More than 140 people within the age group 20 and 30 were dead as compared to just around 80 people of the same age range sustained.

# In[ ]:


pd.DataFrame(td.Age.describe())


# In[ ]:


td['Age_Range'] = pd.cut(td.Age, [0, 10, 20, 30, 40, 50, 60,70,80])
sns.countplot(x = "Age_Range", hue = "Survived", data = td, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])


# In[ ]:


sns.distplot(td['Age'].dropna(),color='darkgreen',bins=30)


# ### 8. SibSp
# 
# SibSp is the number of siblings or spouse of a person onboard. A maximum of 8 siblings and spouses traveled along with one of the tourists.
# 
# **More than 90% people traveled alone or with one of their sibling or spouse**.
# 
# The chances of survival dropped drastically if someone traveled with more than 2 siblings or spouse.

# In[ ]:


td.SibSp.describe()


# In[ ]:


ss = pd.DataFrame()
ss['survived'] = trd.Survived
ss['sibling_spouse'] = pd.cut(trd.SibSp, [0, 1, 2, 3, 4, 5, 6,7,8], include_lowest = True)
(ss.sibling_spouse.value_counts()).plot.area().set_title("Training Data - Number of siblings or spouses vs survival count")


# In[ ]:


x = sns.countplot(x = "sibling_spouse", hue = "survived", data = ss, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])
x.set_title("Training Data - survival based on number of siblings or spouses")


# ### 9. Parch
# 
# Similar to the SibSp, this feature contained the number of parents or children each passenger was traveling with. A maximum of 9 parents/children traveled along with one of the passenger.
# 
# We will create two new columns, a column named family will have the sum of the number of siblings/spouse and number of parents/children. 
# 
# People traveling alone had higher chances of survival. So, we also create a column Is_Alone
# 
# 

# In[ ]:


pd.DataFrame(td.Parch.describe())


# In[ ]:


pc = pd.DataFrame()
pc['survived'] = trd.Survived
pc['parents_children'] = pd.cut(trd.Parch, [0, 1, 2, 3, 4, 5, 6], include_lowest = True)
(pc.parents_children.value_counts()).plot.area().set_title("Training Data - Number of parents/children and survival density")


# In[ ]:


x = sns.countplot(x = "parents_children", hue = "survived", data = pc, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])
x.set_title("Training Data - Survival based on number of parents/children")


# In[ ]:


td['Family'] = td.Parch + td.SibSp
td['Is_Alone'] = td.Family == 0


# ### 10. Ticket
# 
# 

# As the feature 'Ticket' does not provide any additional information, we would not consider this feature

# ### 11. Fare
# 
# It is clear that there is a strong correlation between the fare and the survival. **The higher a tourist paid, the higher would be his chances to survive.**

# In[ ]:


td.Fare.describe()


# In[ ]:


td['Fare_Category'] = pd.cut(td['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid',
                                                                                      'High_Mid','High'])


# In[ ]:


x = sns.countplot(x = "Fare_Category", hue = "Survived", data = td, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])
x.set_title("Survival based on fare category")


# ### 12. Cabin
# 
# As Cabin has a lot of missing values, we will impute it in following section and then use it for prediction.
# 

# ### 13. Embarked
# 
# Embarked signifies where the traveler boarded from. There are three possible values for Embark - Southampton,Cherbourg,Queenstown.
# 
# In combined data, more than **70% of the people boarded from Southampton. Just under 20% boarded from Cherbourg and the rest boarded from Queenstown.**
# 
# More People who **boarded from Cherbourg survived than those who died**

# In[ ]:


p = sns.countplot(x = "Embarked", hue = "Survived", data = trd, palette=["C1", "C0"])
p.set_xticklabels(["Southampton","Cherbourg","Queenstown"])
p.legend(labels = ["Deceased", "Survived"])
p.set_title("Training Data - Survival based on embarking point.")


# # Data Imputation

# ### Embarked
# 
# Since embarked only has two missing values and the highest number of people boarded the ship from Southampton, the probablity of boarding from Southampton is high. So, we fill the missing values with Southampton. However, instead of manually putting in Southampton, we would find the mode of the Embarked column and substitute missing values with it.

# In[ ]:


td.Embarked.fillna(td.Embarked.mode()[0], inplace = True)


# ### Age
# 
# Age has **263 missing values**. To deal with missing values, we first try to categorise the people with their titles. There are **17 different titles** in the training data. We group the titles and sex and then we find the median of all the categories and replace the missing values with the median of that category.

# In[ ]:


td['Salutation'] = td.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip()) 
td.Salutation.nunique()
wc = WordCloud(width = 1000,height = 450,background_color = 'white').generate(str(td.Salutation.values))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

td.Salutation.value_counts()


# In[ ]:


grp = td.groupby(['Sex', 'Pclass'])  
td.Age = grp.Age.apply(lambda x: x.fillna(x.median()))

#If still any row remains
td.Age.fillna(td.Age.median, inplace = True)


# In[ ]:


sal_df = pd.DataFrame({
    "Survived":
    td[td.Survived == 1].Salutation.value_counts(),
    "Total":
        td.Salutation.value_counts()
})
s = sal_df.plot.barh()


# ### Cabin
# 
# Assigning NA for non available cabin values. Pulling deck value from Cabin and adding a feature 'Deck'

# In[ ]:


td.Cabin = td.Cabin.fillna('NA')


# # Encoding & dropping columns
# 
# Using Pandas' get dummies we encoded the categorical data. Later, we drop all the columns we encoded.

# In[ ]:


td = pd.concat([td,pd.get_dummies(td.Cabin, prefix="Cabin"),pd.get_dummies(td.Age_Range, prefix="Age_Range"), pd.get_dummies(td.Embarked, prefix="Emb", drop_first = True), pd.get_dummies(td.Salutation, prefix="Title", drop_first = True),pd.get_dummies(td.Fare_Category, prefix="Fare", drop_first = True), pd.get_dummies(td.Pclass, prefix="Class", drop_first = True)], axis=1)
td['Sex'] = LabelEncoder().fit_transform(td['Sex'])
td['Is_Alone'] = LabelEncoder().fit_transform(td['Is_Alone'])


# In[ ]:


td.drop(['Pclass', 'Fare','Cabin', 'Fare_Category','Name','Salutation', 'Ticket','Embarked', 'Age_Range', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)


# # Prediction
# 
# Prediction algorithms used:
# 
# 1. Gaussian Naive Bayes
# 2. Random Forest

# In[ ]:


# Data to be predicted
X_to_be_predicted = td[td.Survived.isnull()]
X_to_be_predicted = X_to_be_predicted.drop(['Survived'], axis = 1)

#Training data
train_data = td
train_data = train_data.dropna()
feature_train = train_data['Survived']
label_train  = train_data.drop(['Survived'], axis = 1)
train_data.shape #891 x 28

##Gaussian
clf = GaussianNB()
x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)
clf.fit(x_train,  np.ravel(y_train))
print("NB Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")
result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for GNB is:',round(result_rf.mean()*100,2))
y_pred = cross_val_predict(clf,x_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix for NB', y=1.05, size=15)


# In[ ]:


##Random forest
clf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)
clf.fit(x_train,  np.ravel(y_train))
print("RF Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")

result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))
y_pred = cross_val_predict(clf,x_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix for RF', y=1.05, size=15)


# In[ ]:


result = clf.predict(X_to_be_predicted)
submission = pd.DataFrame({'PassengerId':X_to_be_predicted.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)


# <footer id="attribution" style="float:right; color:#999; background:#fff;">
# github.com/sumitmukhija
# </footer>
