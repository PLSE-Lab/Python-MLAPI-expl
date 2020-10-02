#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
py.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris = load_iris()


# In[ ]:


x=iris.data
y=iris.target


# In[ ]:


data=np.c_[x,y]


# In[ ]:


cols=['sepal_length','sepal_width','petal_length','petal_width']
header=cols+['species']
iris_df=pd.DataFrame(data=data,columns=header)


# In[ ]:


iris_df.head()
iris_df.species.replace(0.0,'iris-sesota',inplace=True)
iris_df.species.replace(1.0,'iris-versicolor',inplace=True)
iris_df.species.replace(2.0,'iris-virginica',inplace=True)


# In[ ]:


iris_df.shape


# In[ ]:


import plotly.graph_objs as go
df1=iris_df.iloc[:50,:]
df2=iris_df.iloc[50:100,:]
df3=iris_df.iloc[100:150,:]
trace1 = go.Scatter(
    x=df1.sepal_length,
    y=df1.sepal_width,
    mode='markers',
    name='iris-setosa',
    marker=dict(color = 'rgba(255, 128, 2, 0.8)'),
    text=df1.species)
trace2 = go.Scatter(
    x=df2.sepal_length,
    y=df2.sepal_width,
    mode='markers',
    name='iris-versicolor',
    marker=dict(color = 'rgba(0, 255, 200, 0.8)'),
    text=df2.species)
trace3 = go.Scatter(
    x=df3.sepal_length,
    y=df3.sepal_width,
    mode='markers',
    name='iris-virginica',
    marker=dict(color = 'rgba(255, 128, 255, 0.8)'),
    text=df3.species)
data_list=[trace1,trace2,trace3]
layout=dict(title='Sepal length and Sepal Width of Species',
               xaxis=dict(title='Sepal Length',ticklen=5,zeroline=False),
               yaxis=dict(title='Sepal Width',ticklen=5,zeroline=False))
fig=dict(data=data_list,layout=layout)
py.offline.iplot(fig)


# In[ ]:


import plotly.graph_objs as go
df1=iris_df.iloc[:50,:]
df2=iris_df.iloc[50:100,:]
df3=iris_df.iloc[100:150,:]
trace1 = go.Scatter(
    x=df1.petal_length,
    y=df1.petal_width,
    mode='markers',
    name='iris-setosa',
    marker=dict(color = 'rgba(255, 128, 2, 0.8)'),
    text=df1.species)
trace2 = go.Scatter(
    x=df2.petal_length,
    y=df2.petal_width,
    mode='markers',
    name='iris-versicolor',
    marker=dict(color = 'rgba(0, 255, 200, 0.8)'),
    text=df2.species)
trace3 = go.Scatter(
    x=df3.petal_length,
    y=df3.petal_width,
    mode='markers',
    name='iris-virginica',
    marker=dict(color = 'rgba(255, 128, 255, 0.8)'),
    text=df3.species)
data_list=[trace1,trace2,trace3]
layout=dict(title='Petal length and Petal Width of Species',
               xaxis=dict(title='Petal Length',ticklen=5,zeroline=False),
               yaxis=dict(title='Petal Width',ticklen=5,zeroline=False)
            )
fig=dict(data=data_list,layout=layout)
py.offline.iplot(fig)


# In[ ]:


corr_martix=iris_df[cols].corr()
sns.heatmap(corr_martix,cbar=True,annot=True,fmt='.1f',cmap='coolwarm');


# In[ ]:


class Question:
    def __init__(self,column,value):
        self.column=column
        self.value=value
    def match(self,data):
        value=data[self.column]
        return value>=self.value
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


# In[ ]:


Question(0,5)
q=Question(0,5)
q.match(x[3])


# In[ ]:


def count_values(rows):
    count={}
    for row in  rows:
        label=row[-1]
        if label not in count:
            count[label]=0
        count[label]+=1
    return count 


# In[ ]:


count_values(data)


# In[ ]:


def partition(rows,question):
    true_row,false_row=[],[]
    for row in rows:
        if question.match(row):
            true_row.append(row)
        else:
            false_row.append(row)
    return true_row,false_row


# In[ ]:


print(Question(0,5))
t_r,f_r=partition(data,Question(0,5))
t_r


# In[ ]:


def gini(rows):
    count=count_values(rows)
    impurity=1
    for label in count:
        probab_of_label=count[label]/float(len(rows))
        impurity-=probab_of_label**2
    return impurity


# In[ ]:


gini(data)


# In[ ]:


def entropy(rows):
    entropy=0
    from math import log
    log2=lambda x:log(x)/log(2)
    count=count_values(rows)
    for label in count:
        p=count[label]/float(len(rows))
        entropy-=p*log2(p)
    return entropy


# In[ ]:


def info_gain_gini(current,left,right):
    p =float(len(left))/len(left)+len(right)
    return current-p*gini(left)-(1-p)*gini(right)


# In[ ]:


def info_gain_entropy(current,left,right):
    p =float(len(left))/len(left)+len(right)
    return current-p*entropy(left)-(1-p)*entropy(right)


# In[ ]:


def best_split(rows):
    best_gain=0
    best_question=None
    current=gini(rows)
    features=len(rows[0])-1
    for col in range(features):
        values=set([row[col] for row in rows])
        for val in values:
            question=Question(col,val)
            true_rows,false_rows=partition(rows,question)
            if len(true_rows)==0 or len(false_rows) ==0:
                continue
            gain=info_gain_gini(current,true_rows,false_rows)
            if gain>=best_gain:
                best_gain,best_question=gain,question
    return best_gain,best_question


# In[ ]:


a,b=best_split(data)
print(b)
print(a)


# In[ ]:


class DecisionNode:
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[ ]:


class Leaf:
    def __init__(self,rows):
        self.predictions=count_values(rows)


# In[ ]:


def build_tree(rows):
    gain,question=best_split(rows)    
    if gain==0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return DecisionNode(question, true_branch, false_branch)    


# In[ ]:


tree=build_tree(data)


# In[ ]:


def print_tree(node,indentation=""):
    if isinstance(node,Leaf):
        print(indentation+"PREDICTION",node.predictions)
        return 
    print(indentation + str(node.question))
    print(indentation+ "True Branch")
    print_tree(node.true_branch,indentation + " ")
    print(indentation+ "False Branch")
    print_tree(node.false_branch,indentation + " ")


# In[ ]:


print_tree(tree)


# In[ ]:


def  split_info(left,right):
    num=float(len(left))
    den=len(left)+len(right)
    p=num/den
    from math import log
    log10=lambda x:log(x)/log(10)
    return -(log10(p)*p+(1-p)*log10(1-p))


# In[ ]:


def print_tree_output(data,level):
    gain,question=best_split(data)
    if gain==0:
        print("Level ",level)
        count=count_values(data)
        for value in count:
            print("Count of ",value," = ",count[value])
        print("Reached Leaf Node") 
        return 
    print('Level ',level)
    count=count_values(data)
    for value in count:
        print("Count of ",value," = ",count[value])
    feature=question.column
    true_rows, false_rows = partition(data, question)
    split=split_info(true_rows,false_rows)
    gain1=info_gain_entropy(entropy(data),true_rows,false_rows)
    print("Spliting on feature ",header[feature])
    print_tree_output(true_rows,level+1)
    print_tree_output(false_rows,level+1)


# In[ ]:


print_tree_output(data,0)


# In[ ]:


def train_test_split(dataset,target,test_size):
 trainSize = int(len(dataset) * test_size)
 trainSet = []
 copy = list(dataset)
 while len(trainSet) < trainSize:
     index = random.randrange(len(copy))
     trainSet.append(copy.pop(index))
 return [trainSet,target]


# In[ ]:


def confusion_matrix(test_data,predicted_data):
    mat=[[0]*3]*3
    for i in range(len(test_data)):
        if(test_data[i]==0.0 and predicted_data==0.0):
            mat[0][0]=mat[0][0]+1
        elif(test_data[i]==0.0 and predicted_data==1.0):
            mat[0][1]=mat[0][1]+1
        elif(test_data[i]==0.0 and predicted_data==2.0):
            mat[0][2]=mat[0][2]+1
        elif(test_data[i]==1.0 and predicted_data==0.0):
            mat[1][0]=mat[1][0]+1
        elif(test_data[i]==1.0 and predicted_data==1.0):
            mat[1][1]=mat[1][1]+1
        elif(test_data[i]==1.0 and predicted_data==2.0):
            mat[1][2]=mat[1][2]+1
        elif(test_data[i]==2.0 and predicted_data==0.0):
            mat[2][0]=mat[2][0]+1
        elif(test_data[i]==2.0 and predicted_data==1.0):
            mat[2][1]=mat[2][1]+1
        elif(test_data[i]==2.0 and predicted_data==2.0):
            mat[2][2]=mat[2][2]+1
    return mat


# In[ ]:


def classification_report(matrix):
    pre0=(matrix[0][0])/(matrix[0][0]+matrix[1][0]+matrix[2][0])
    rec0=(matrix[0][0])/(matrix[0][0]+matrix[0][1]+matrix[0][2])
    pre1=(matrix[1][1])/(matrix[0][1]+matrix[1][1]+matrix[2][1])
    rec1=(matrix[1][1])/(matrix[1][0]+matrix[1][1]+matrix[2][1])
    pre2=(matrix[2][2])/(matrix[0][2]+matrix[1][2]+matrix[2][2])
    rec2=(matrix[2][2])/(matrix[2][0]+matrix[2][1]+matrix[2][2])
    f10=(2*pre0*rec0)/(pre0+rec0)
    f11=(2*pre1*rec1)/(pre1+rec1)
    f12=(2*pre2*rec2)/(pre2+rec2)
    sup0=matrix[0][0]+matrix[0][1]+matrix[0][2]
    sup1=matrix[1][0]+matrix[1][1]+matrix[2][1]
    sup2=matrix[2][0]+matrix[2][1]+matrix[2][2]
    print("\t  precision  recall  f1-score  support")
    print()
    print("\t0\t",pre0,"\t",rec0,"\t",f10,"\t ",sup0)
    print("\t0\t",pre1,"\t",rec1,"\t",f11,"\t ",sup1)
    print("\t0\t",pre2,"\t",rec2,"\t",f12,"\t ",sup2)
    print()
    print("avg / total\t",(pre0+pre1+pre2)/3,"\t",(rec0+rec1+rec2)/3,"\t",(f10+f11+f12)/3,"\t",(sup0+sup1+sup2))


# In[ ]:


X_test, y_test = train_test_split(x, y,0.25)
y_pred=[]
q=Question()

for i in range(len(X_test)):
    y_pred.append(q.match(X_test[i],y_test[i]))

mat=confusion_matrix(y_test, y_pred)
print(mat)
classification_report(mat)

