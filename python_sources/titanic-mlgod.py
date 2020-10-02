# https://www.kaggle.com/startupsci/titanic-data-science-solutions/code
# https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial

# Bibliotecas para análise.
import pandas as pd
import numpy as np
import random as rnd

# Bibliotecas para visualização.
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# Bibliotecas para machine learning.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Lê os arquivos train.csv e test.csv, depois junta os dois em uma lista.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

# Retorna colunas do dataframe train_df para análisar possiveis "Features".
#print(train_df.columns.values)

# Retorna dados do dataframe train_df para entender os dados e verificar possiveis correções que irá precisar.
#train_df.head()
#train_df.tail()

# Retorna os tipos das colunas dos dataframes train_df e test_df.
#train_df.info()
#print('_'*40)
#test_df.info()

# Retorna calculos sobre os valores numéricos do dataframe train_df para ver se os dados são suficientes.
#train_df.describe()

# Retorna calculos sobre os valores categóricos do dataframe train_df para ver se os dados são suficientes.
#train_df.describe(include=['O'])

# Retorna pivoteamentos de features do dataframe train_df para validar a relevância da aplicação dessas features.
#train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Retorna gráfico do dataframe train_df com coluna "Survived" e valor "Age" para análise visual do problema (Númericos).
#g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Age', bins=50)

# Retorna gráfico do dataframe train_df com coluna "Survived", linha "Pclass" e valor "Age" para análise visual do problema (Númericos e ordinais).
#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();

# Retorna gráfico do dataframe train_df com coluna "Pclass", série "Survived" e valor "Age" para análise visual do problema (Númericos e ordinais).
#grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();

# Retorna gráfico do dataframe train_df com coluna "Pclass", série "Survived" e valor "Age" para análise visual do problema (Categóricos).
#grid = sns.FacetGrid(train_df, col='Embarked')
#grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
#grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
#grid.add_legend()

# Retorna gráfico do dataframe train_df com coluna "Embarked", série "Survived", valor de x "Age" e valor de y "Fare" para análise de correlação de algumas features.
#grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'b'})
#grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
#grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
#grid.add_legend()

# Correção dos dados dropando as colunas "Ticket" e "Cabin" pois não serão utilizadas
#print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
#"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

# Engenharia de features para criar uma nova feature "Title" (Miss, Mister, Doctor, Master, etc...) e ver se conseguimos ter alguma relação com sobrevivencia.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
#pd.crosstab(train_df['Title'], train_df['Sex'])

# Renomeando "Title" menos comuns para um nome só, chamado "Rare".
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
#train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Convertendo os "Title" de categoricos para ordinais.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#train_df.head()

# Elimando as colunas "Name" e "PassengerId" que não serão mais necessárias.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
#train_df.shape, test_df.shape

# Convertendo mais uma feature de categorica para ordinal, nesse caso "Sex" (Isso é feito pois a maioria dos algoritimos de modelo requerem que os campos sejam ordinais (números)).
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#train_df.head()

# Analisa visualmente corelação entre idades e outras features para depois gerar idades para os registros que não tem uma idade informada
#grid = sns.FacetGrid(train_df, col='Pclass', hue='Sex')
#grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()

# Criação de uma matriz de zeros para depois colocar os valores de idade gerados.
guess_ages = np.zeros((2,3))
#guess_ages

# Iteração entre "Sex (0 or 1)" e "Pclass (1, 2, 3)" para calcular novos valores de "Age" com seis combinações.
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

#train_df.head()

# Cria agrupamento de idade para verificar correlação com "Survived"
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Reescreve "Age" Como ordinal baseado nos agrupamentos criados acima.
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

#train_df.head()

# Remove "AgeBand" do dataframe train_df.
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
#train_df.head()

# Criando nova feature "FamilySize" a partir das features "SibSp" e "SibSp" para depois pode dropar as colunas.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Criando mais uma nova feature "IsAlone" a partir do "FamilySize".
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# Dropando "Parch", "SibSp" e "FamilySize" pois já conseguimos "IsAlone".
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

#train_df.head()

# Criando uma feature artificial com "Age" e "PClass".
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# Completando a feature categorica "Embarked" com o valor mais comum.
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
#train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Convertendo a feature categorica "Embarked" para ordinal (númerica).
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Completando e convertendo a ultima feature necessaria para rodar o modelo.
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#test_df.head()

# Criando o agrupamento de "Fare'
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# Convertendo Fare baseado no agrupamento de "Fare" e dropando a feature "FareBand"
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
#train_df.head(10)
#test_df.head(10)

##########################################################################

# Inicio do Model, predict and solve.

# Dropando colunas desnecessarias do train_df e test_df.
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
#X_train.shape, Y_train.shape, X_test.shape

# Testando o score da nossa predição com "Logistic Regression".
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#acc_log

# Testando a correlação das nossa features atraves de Regressão logística para saber quais mais impactam em "Survived".
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
#coeff_df.sort_values(by='Correlation', ascending=False)

# Testando o score da nossa predição com "Support Vector Machines".
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#acc_svc

# Testando o score da nossa predição com "Gaussian Naive Bayes".
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#acc_gaussian

# Testando o score da nossa predição com "k-Nearest Neighbors".
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#acc_knn

# Testando o score da nossa predição com "Perceptron".
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#acc_perceptron

# Testando o score da nossa predição com "Linear SVC".
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#acc_linear_svc

# Testando o score da nossa predição com "Stochastic Gradient Descent".
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#acc_sgd

# Testando o score da nossa predição com "Decision Tree".
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#acc_decision_tree

# Testando o score da nossa predição com "XGBoost" 
xgb = xgb.XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
acc_xgb = round(xgb.score(X_train, Y_train) * 100, 2)
#predictions = gbm.predict(test_X)

# Testando o score da nossa predição com "Random Forest"
random_forest = RandomForestClassifier(n_estimators=300)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#acc_random_forest

# Ordenando os modelos de ML para ver qual é o melhor.
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'XGBoost'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_xgb]})
              
#models.sort_values(by='Score', ascending=False)

# Escolhido "Random Forest" como melhor modelo para predizer quem morreu ou não no titanic.
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)
