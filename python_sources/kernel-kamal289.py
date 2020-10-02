#team member= kamal kant, nitish mishra, sunny kumar
# registered email id= kantkamal086@gmail.com

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

##../input/traincsv/train.csv
df1 = pd.read_csv(r"../input/train.csv")
print("readed")

df=df1
df['new']=df['first_name']
for i in range(len(df['first_name'])):
    name=str(df['first_name'][i])
    k=name.split()
    df['new'][i]=k[0]

df['new1']=df['last_name']
for i in range(len(df['last_name'])):
    name=str(df['last_name'][i])
    k=name.split()
    if len(k)>0:
        df['new1'][i]=k[0]
    else:
        df['new1'][i]=" "
print("train data transformed")
df['name'] = df['new'].fillna('') +" "+ df['new1'].fillna('')
df_names=df
df_names.race.replace({'black':0,'hispanic':1,'indian':2,'white':3},inplace=True)
df_names.gender.replace({'f':0,'m':1},inplace=True)
Xfeatures =df_names['name']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
y = df_names.race

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
scr_race=clf.score(X_test,y_test)
print(scr_race)

df2 = pd.read_csv(r"../input/test.csv")
print("readed")

df2['new3']=df2['first_name']
for i in range(len(df2['first_name'])):
    name=str(df2['first_name'][i])
    k=name.split()
    df2['new3'][i]=k[0]
    
df2['new4']=df2['last_name']
for i in range(len(df2['last_name'])):
    name=str(df2['last_name'][i])
    k=name.split()
    if len(k)>0:
        df2['new4'][i]=k[0]
    else:
        df2['new4'][i]=" "
print("test data transformed")

df2['name'] = df2['new3'].fillna('') +" "+ df2['new4'].fillna('')
n=df2['name']
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        str1='black'
        return str1
    elif clf.predict(vector)==1:
        str1='hispanic'
        return str1
    elif clf.predict(vector)==2:
        str1='indian'
        return str1
    else:
        str1='white'
        return str1


list1=[]
for j in range(len(n)):
    k=genderpredictor(n[j])
    list1.append(k)
    
    
my_df_race = pd.DataFrame(list1)
import csv
##my_df_race.to_csv(r"C:\Users\kamal\Desktop\New folder\output5.csv", index=False, header=False)
print("hello")
my_df_race.to_csv(r"output6.csv", index=False)

########################################################################
########################################################################

Xfeatures =df_names['name']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
y = df_names.gender
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
scr_gender=clf.score(X_test,y_test)

df2['name'] = df2['new3'].fillna('') +" "+ df2['new4'].fillna('')

n=df2['name']

def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        str1='f'
        return str1
    else:
        str1='m'
        return str1

list2=[]
for j in range(len(n)):
    k=genderpredictor(n[j])
    list2.append(k)
my_df_gender = pd.DataFrame(list2)
##my_df_gender.to_csv(r"C:\Users\kamal\Desktop\New folder\output5.csv", index=False, header=False)
my_df_gender.to_csv(r"output7.csv", index=False)
#result = pd.concat([my_df_gender, my_df_race], axis=1)
#result.to_csv(r"output10.csv", index=False)
import csv
all6=['id','gender','race']
with open(r"final_output.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(all6)
a1=np.arange(1,len(my_df_race))
a1=list(a1)
a1=pd.DataFrame(a1)

result = pd.concat([a1,my_df_gender, my_df_race], axis=1)
with open(r"final_output.csv", 'a', newline='') as f:
    result.to_csv(f, index=False,header=False)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.