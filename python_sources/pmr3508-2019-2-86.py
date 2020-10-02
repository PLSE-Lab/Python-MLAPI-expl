#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


trainDF = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv', 
        names=[
          "Age", "Workclass", "Samp_weight", "Education", "Education-Num", "Martial Status",
          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
          "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


trainDF = trainDF.drop(['Id'])
trainDF.head()


# # Analise

# In[ ]:


import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi'] = 100

from matplotlib import pyplot as plt


plt.figure(figsize=(18,5))

fig = plt.bar(sorted(trainDF.Age.unique()), trainDF['Age'].value_counts().sort_index(), alpha=0.8)


plt.xlabel('Age').set_color('black')
plt.xticks(rotation=90)
[i.set_color("black") for i in plt.gca().get_xticklabels()]


plt.ylabel('Frequency').set_color('black')
[i.set_color("black") for i in plt.gca().get_yticklabels()]


plt.title('Age distribution').set_color('black')
plt.margins(x=0, y=None, tight=True)


# In[ ]:


print(trainDF['Age'].mean()) # n sei oq aconteceu aki
trainDF.groupby(by="Age").describe()


# In[ ]:


print(trainDF.shape)
clean_tDF = trainDF.dropna()
print(clean_tDF.shape)


# In[ ]:


fig = plt.bar(clean_tDF.Workclass.unique(), clean_tDF['Workclass'].value_counts(), alpha=0.8)


plt.xlabel('Workclass').set_color('black')
plt.xticks(rotation=90)
[i.set_color("black") for i in plt.gca().get_xticklabels()]


plt.ylabel('Frequency').set_color('black')
[i.set_color("black") for i in plt.gca().get_yticklabels()]


plt.title('Workclass distribution').set_color('black')
plt.margins(x=0, y=None, tight=True)


# In[ ]:


fig = plt.figure(figsize=(25,20))

for ed_num,tmpdf in clean_tDF.groupby(by='Education-Num'):
    plt.scatter(sorted(tmpdf['Age']),sorted(tmpdf['Hours per week']),label=ed_num)

plt.legend()
plt.title('Age x Hours per Week')
plt.xlabel('Age')
plt.ylabel('Hours per week')


# In[ ]:


clean_tDF.Target.value_counts(normalize=True).plot(kind="bar")


# # Treinamento Knn

# In[ ]:


X = clean_tDF[["Age", "Education-Num", "Capital Gain", "Capital Loss" ,"Hours per week"]]
Y = clean_tDF.Target


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

best_meanScore = 0

#for n in range(1,20):
#  neigh = KNeighborsClassifier(n_neighbors=n)
#  neigh.fit(X, Y) 
#  for folds in range(2,10):
#    score_rf = cross_val_score(neigh, X, Y, cv=folds, scoring='accuracy').mean()
#    if score_rf > best_meanScore:
#      best_meanScore = score_rf
#      best_pair = [n, folds] #best pair of n and cv
      
#print(best_pair)
#best_meanScore


# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=14)
neigh.fit(X, Y) 
score_rf = cross_val_score(neigh, X, Y, cv=9, scoring='accuracy')
score_rf


# In[ ]:


testDF =  pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv',
        names=[
          "Age", "Workclass", "Samp_weight", "Education", "Education-Num", "Martial Status",
          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
          "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


testDF = testDF.drop(["Id"])


# In[ ]:


testDF.head()


# In[ ]:


Xtest = testDF[["Age", "Education-Num", "Capital Gain", "Capital Loss" ,"Hours per week"]]


# In[ ]:


Ypred = neigh.predict(Xtest)


# In[ ]:


#Salvando as previsoes do knn
savepath = "predictions_knn.csv" 
prev = pd.DataFrame(Ypred, columns = ["income"]) 
prev.to_csv(savepath, index_label="Id") 


# In[ ]:


prev.income.value_counts(normalize=True).plot(kind="bar")


# O Knn feito no EP1 preve uma proporcao de <=50k maior que a proporcao vista na base de treino, o que pode ser um indicio de que existe margem para melhora

# # Analise mais profunda

# In[ ]:


import seaborn as sns
cat_attributes = trainDF.select_dtypes(include=['object'])


# In[ ]:


sns.set(rc={"font.style":"normal",
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(22.0, 7.0)})
sns.countplot(x='Age', hue='Target', data = cat_attributes)


# * Claramente podemos ver claramente que a distribuicao de quem ganha mais que 50k se aproxima de uma normal platicurtica, se concentrando entre 30 e 60 anos. 
# * Aqueles que recebem menos de 50k tem uma concentracao entre 18 e 40 anos, se aproximando de uma normla com obliquidade negativa.
# * Tambem ve-se que as maiores proporcoes de pessoas que recebem >=50k entre 44 e 54 anos.

# In[ ]:


sns.set(rc={"font.style":"normal",
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(17.0, 7.0)})
oc_plot = sns.countplot(x='Occupation', hue='Target', data = cat_attributes)
oc_plot.set_xticklabels(oc_plot.get_xticklabels(), rotation=40, ha="right")


# * Conclui-se que as maiores proporcoes de >=50k estao nas ocupacoes de: exec-managerial e prof-specialty, enquanto as menores se encontram em: other services e priv-house-serv

# In[ ]:


sns.set(rc={"font.style":"normal",
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(17.0, 7.0)})
oc_plot = sns.countplot(x='Country', hue='Target', data = cat_attributes)
oc_plot.set_xticklabels(oc_plot.get_xticklabels(), rotation=40, ha="right")


# Devido a grande diferenca de quantidade de pessoas da amostra em cada pais, nenhuma conclusao pode ser tirada.

# In[ ]:


sns.set(rc={"font.style":"normal",
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(17.0, 7.0)})
oc_plot = sns.countplot(x='Education-Num', hue='Target', data = cat_attributes)
oc_plot.set_xticklabels(oc_plot.get_xticklabels(), rotation=40, ha="right")


# In[ ]:


sns.set(rc={"font.style":"normal",
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(17.0, 7.0)})
oc_plot = sns.countplot(x='Workclass', hue='Target', data = cat_attributes)
oc_plot.set_xticklabels(oc_plot.get_xticklabels(), rotation=40, ha="right")


# In[ ]:


sns.set(rc={"font.style":"normal",
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(17.0, 7.0)})
oc_plot = sns.countplot(x='Sex', hue='Target', data = cat_attributes)
oc_plot.set_xticklabels(oc_plot.get_xticklabels(), rotation=40, ha="right")


# * Pelo grafico, fica evidente que a proporcao de homens que ganham >=50k se mostra maior que a de mulheres que ganham >=50k

# In[ ]:





# # Knn com Colunas categoricas

# Vamos usar somente as categorias que apresentaram relevancia na diferenciacao entre <=50k e >=50k

# In[ ]:


categorical_subset = clean_tDF.drop(columns =['Target','Age', 'Samp_weight' , 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week'])

categorical_subset.columns


# In[ ]:


# One hot encode
relevant_categorical_subset = pd.get_dummies(categorical_subset.drop(columns =['Country']))


# In[ ]:


missing = clean_tDF[['Target','Age', 'Samp_weight' , 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']]
TurboTrainDF = pd.concat([relevant_categorical_subset,missing], axis=1)
TurboTrainDF.shape


# In[ ]:


# I can use the same X and Y
X = TurboTrainDF.drop(columns =['Samp_weight', 'Target'])
Y = TurboTrainDF.Target
X.shape


# In[ ]:


neigh2 = KNeighborsClassifier(n_neighbors=14)
neigh2.fit(X, Y) 
score_rf = cross_val_score(neigh2, X, Y, cv=9, scoring='accuracy')
score_rf


# In[ ]:


#one hot on test DF
cs_tst= testDF.drop(columns =['Age', 'Samp_weight' , 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week'])
relevant_cs_tst = pd.get_dummies(cs_tst.drop(columns =['Country']))

missing_tst = testDF[['Age', 'Samp_weight' , 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']]
TurboTestDF = pd.concat([relevant_cs_tst,missing_tst], axis=1)

#for some reason 'Workclass_Never-worked' do not appear on X, so i`ll drop it in X test
Xtest = TurboTestDF.drop(columns =['Samp_weight', 'Workclass_Never-worked'])
Xtest.shape


# In[ ]:


Ypred_knn2 = neigh2.predict(Xtest)


# In[ ]:


#Salvando as previsoes do knn
savepath = "predictions_knn2.csv" 
prev = pd.DataFrame(Ypred_knn2, columns = ["income"]) 
prev.to_csv(savepath, index_label="Id") 
prev.income.value_counts(normalize=True).plot(kind="bar")


# Nao esta clara a diferenca entre as proporcoes do knn sem o uso das variaveis categoricas e com o uso das mesmas, apesar de haver um aumento na acuracia no cv do treino

# # Regressao Logistica

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logR = LogisticRegression(random_state=0)
scores = cross_val_score(logR, X, Y, cv=15, scoring='accuracy')
scores


# ### Podemos facilmente ver a incluencia de cada coluna(parametro) no y previsto:

# In[ ]:


import sklearn as sklearn
logReg = sklearn.linear_model.LogisticRegression()

logReg.fit(X, Y)
coefs = pd.Series(logReg.coef_[0], index=X.columns)

coefs.sort_values(ascending = False)


# In[ ]:


coefs = pd.Series(logReg.coef_[0], index=X.columns)
coefs = coefs.sort_values()
plt.subplot(1,1,1)
coefs.plot(kind="bar")
plt.show()
coefs.sort_values(ascending = False)


# * ***Observou-se que, seguindo a intuicao da analise exploratoria, temos que o fator "sexo feminino" tem grande influencia na renda, fazendo com que as pessoas que sao do sexo feminino, tenham maior chance de pertencer ao grupo <=50k ***
# * Por outro lado, vemos que a idade e o numero de horas trabalhadas por semana, ao contrario da intuicao da analise, nao tem grande relevancia para a predicao do modelo

# In[ ]:


#Salvando as previsoes da Regressao logistica
savepath = "predictions_logReg.csv" 
prev = pd.DataFrame(logReg.predict(X), columns = ["income"]) 
prev.to_csv(savepath, index_label="Id") 
prev.income.value_counts(normalize=True).plot(kind="bar")


# Aqui a proporcaose aproxima mais da proporcao da base de treino, em relacao ao knn

# ### Sera que seria interessante retirar essas features que aparentemente nao tem grande influencia?

# In[ ]:


scores2 = cross_val_score(logR, X.drop(columns =['Age', 'Hours per week', "Capital Gain", "Capital Loss"]), Y.drop(columns =['Age', 'Hours per week', "Capital Gain", "Capital Loss"]), cv=15, scoring='accuracy')
scores2


# Nos testes de acuracia, tudo indica que retirar essas features piora o modelo

# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
ranFo = RandomForestClassifier(random_state=60)
ranFo.fit(X, Y)


# In[ ]:


accuracy_score(Y, ranFo.predict(X))


# In[ ]:


cross_val_score(ranFo, X, Y, cv=5, scoring='accuracy')


# In[ ]:


cross_val_score(ranFo, X, Y, cv=10, scoring='accuracy')


# In[ ]:


cross_val_score(ranFo, X, Y, cv=15, scoring='accuracy')


# Ao contrario de outros modelos, nao ve-se um aumento claro nas medias das cross-validations da random forest

# In[ ]:


#Salvando as previsoes do random forest
savepath = "predictions_ranFo.csv" 
prev = pd.DataFrame(ranFo.predict(X), columns = ["income"]) 
prev.to_csv(savepath, index_label="Id") 
prev.income.value_counts(normalize=True).plot(kind="bar")


# Aqui a proporcaose aproxima mais da proporcao da base de treino, em relacao ao knn e a logistic regression

# # Conclusoes e comparacoes

# In[ ]:


from time import time
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from numpy import mean as ar_mean

def train_predict(learner, sample_size, X_train, y_train, X_test): 
    results = {}
    
    start = time() 
    learner =  learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() 
    
    results['train_time'] = end - start
        
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:301])
    end = time() # Get end time
    
    results['pred_time'] = end - start
            
    results['acc_train'] = accuracy_score(y_train[:301], predictions_train)
    results['cv_avg_acc-5folds'] = ar_mean(cross_val_score(learner,  X_train[:sample_size], y_train[:sample_size], cv=5, scoring='accuracy' ))
    results['cv_avg_acc-10folds'] = ar_mean(cross_val_score(learner,  X_train[:sample_size], y_train[:sample_size], cv=10, scoring='accuracy' ))
    
    results['prec_train'] = precision_score(y_train[:301], predictions_train, average=None)
    
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results


# In[ ]:


s100 = int(len(X))
s10 = int(len(X) / 10)
s1 = int(len(X) / 100)

results = {}  
for clf in [neigh2, logR, ranFo]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([s1, s10, s100]):
        results[clf_name][i] =         train_predict(clf, samples, X, Y, Xtest)
              
for i in results.items():
    print (i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))


# ## Conclusoes:
# * Claramente a regressao logistica tem maior velocidade, apesar de que a random forest se compara a regressao logistica, enquanto o Knn tem um tempo muito superior
# * Ela tambem tem maior interpretabilidade, pois conseguimos ver a influencia de cada parametro para a previsao
# * Tanto a randomForest quanto a logistic Regression perdem acuracie e precisao a medida que se aumenta o uso da base
# * A random forest mantem a maior precisao e acuracia em todos os cenarios
# * Somente A random forest tem uma perda consideravel de acuracia com o uso da validacao cruzada

# **Por curiosidade, vou testar retirando as features que tiveram menos influencia na lin reg:**

# In[ ]:


s100 = int(len(X))
s10 = int(len(X) / 10)
s1 = int(len(X) / 100)

results = {}  
for clf in [neigh2, logR, ranFo]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([s1, s10, s100]):
        results[clf_name][i] =         train_predict(clf, samples, X.drop(columns =['Age', 'Hours per week', "Capital Gain", "Capital Loss"]), Y.drop(columns =['Age', 'Hours per week', "Capital Gain", "Capital Loss"]), Xtest.drop(columns =['Age', 'Hours per week', "Capital Gain", "Capital Loss"]))

for i in results.items():
    print (i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))


# #### *Conclusoes 2:*
# * Ao contrario dos outros modelos, estranhamente, os tempos de predicao do Knn aumentaram, apesar da base ser menor
# * Todas as medidas de acuracia e precisao diminuiram para todos os modelos em todos os casos em comparacao com o uso das features retiradas, seguindo a tendencia vista anteriormente na regressao logistica
# 

# In[ ]:




