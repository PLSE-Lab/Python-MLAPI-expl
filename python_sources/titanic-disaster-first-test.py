##############################################################
#################    import libraries     ####################
##############################################################

#Handle table-like data and matrices
import numpy as np
import pandas as pd

#Modelling algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

#Modelling Helpers
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

sns.set()

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


##############################################################
#################    Function definition  ####################
##############################################################

def plot_correlation_map(df):
    corr = df.corr()
    _ , ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    # cmap = sns.cubehelix_palette(light = 1, as_cmap=True)
    _ = sns.heatmap(
    corr,
    cmap = cmap,
    square = True,
    cbar_kws = {'shrink' : 0.9},
    ax=ax,
    annot=True,
    annot_kws={'fontsize' : 12}
    )
    plt.show()



def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()
    plt.show()

def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()
    plt.show()

def plot_variable_importance(x, y):
    tree = DecisionTreeClassifier(random_state = 99)
    tree.fit(x, y)
    plot_model_var_imp(tree, x, y)

def plot_model_var_imp(model, x, y):
    imp = pd.DataFrame(
        model.feature_importances_,
        columns = ['Importance'],
        index = x.columns
    )
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[:10].plot(kind='barh')
    print(model.score(x, y))
    plt.show()


##############################################################
#################    Data Extraction      ####################
##############################################################

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
full = train.append(test, ignore_index=True)


##############################################################
#################       EDA - Data Viz    ####################
##############################################################

print('Datasets', 'train : ', train.shape, 'test : ', test.shape)
print(train.head())
print(train.describe())


# plot_correlation_map(train)
# plot_distribution(train, var='Fare', target='Survived', row='Sex')
# plot_categories(train, cat='Parch', target='Survived')


##############################################################
#################      Data preparation   ####################
##############################################################

#Transform Sex into bianry variable
sex = pd.Series(np.where(full.Sex == 'male', 1, 0), name='Sex')

#Create new variable for every unique value of embvarked
embarked = pd.get_dummies(full.Embarked, prefix='Embarked')

#Create a new variabke for every unique value of Pclass
pclass = pd.get_dummies(full.Pclass, prefix='pclass')

#Fill missing value
print(full.columns[full.isnull().any()])
filling_mean = pd.DataFrame()
filling_mean['Age'] = full.Age.fillna(full.Age.mean())
filling_mean['Fare'] = full.Fare.fillna(full.Fare.mean())

##############################################################
###### Feature Engineering - create new variable #############
##############################################################




##############################################################
#########      Assemble Dataset for Modelling    #############
##############################################################

full_x = pd.concat([filling_mean, sex, embarked, pclass], axis=1)

#Create all datasets: train, validate, test
# print(len(train))
train_valid_x = full_x[:891]
train_valid_y = full[:891].Survived
test_x = full_x[891:]

train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, train_size=.7)


# plot_variable_importance(train_x, train_y)

# model = LogisticRegression()
# model = SVC()
# model = GaussianNB()
# model = RandomForestClassifier(n_estimators=100)
model = GradientBoostingClassifier()


##############################################################
###################     Evaluation          ##################
##############################################################

model.fit(train_x, train_y)
print(model.score(train_x, train_y), model.score(valid_x, valid_y))

#Feature selection
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(train_y, 2), scoring='accuracy')
rfecv.fit(train_x, train_y)

print(rfecv.score(train_x, train_y), rfecv.score(valid_x ,valid_y))
print('Optimal number of feature : %d' % rfecv.n_features_)

#Plot number of feature vs cross valisation scores
# plt.figure()
# plt.xlabel("number of features selected")
# plt.ylabel("Cross validation score (nb of correct classification)")
# plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
# plt.show()


##############################################################
#############     Submission - Deployment      ###############
##############################################################

test_y = model.predict(test_x)
passenger_id = full[891:].PassengerId
test = pd.DataFrame({'PassengerId': passenger_id, 'Survived': test_y})
# print(test.shape)
# print(test.head())
test.to_csv('titanic_pred.csv', index=False)
