# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
#%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

''' Helper Functions '''
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    sns.plt.show()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
    sns.plt.show()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
    #print(corr)
    plt.show()

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = False )
    imp[ :  10].plot( kind = 'barh' )
    plt.show()
    print (model.score( X , y ))

''' End Helper Functions '''

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
full = train.append( test , ignore_index = True )
titanic = full[ :891 ]
# Step 2: Statistical summaries and visualisations
print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
''' A heat map of correlation may give us a understanding of which variables are important '''
#plot_correlation_map( titanic )
''' Plot distributions of Age of passangers who survived or did not survive '''
#plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
''' Plot distributions of Fare of passangers who survived or did not survive '''
#plot_distribution( titanic , var = 'Fare' , target = 'Survived'  )
''' Plot survival rate by Embarked '''
#plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
''' Plot survival rate by Sex '''
#plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )
#plot_categories( titanic , cat = 'SibSp' , target = 'Survived' )
#plot_categories( titanic , cat = 'Parch' , target = 'Survived' )

""" Step 3: Data Preparation  """
# Step 3.1: Categorical variables need to be transformed to numeric variables
# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
#print(embarked.head())
# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
#print(pclass.head())

# Step 3.2: Fill missing values in variables
# Create dataset
imputed = pd.DataFrame()
# Fill missing values of Age with the average of Age (mean)
imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )
# Fill missing values of Fare with the average of Fare (mean)
imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )
#print(imputed.head())

""" Step 4: Feature Engineering -Creating new variables """
# Step 4.1: Extract titles form passenger names
title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title, prefix='Title')
#print(title.head())

# Step 4.2: Extract Cabin category information from the Cabin number
cabin = pd.DataFrame()
# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )
# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
# dummy encoding ...
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )
#print(cabin.head())

# Step 4.3: Extract ticket class from ticket number
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()
# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )
#print(ticket.shape)
#print(ticket.head())

# Step 4.4: Create family size and category for family size
family = pd.DataFrame()
# introducing a new feature : the size of families (including the passenger)

family[ 'FamilySize' ] = full[ 'Parch' ].fillna(0) + full[ 'SibSp' ].fillna(0) + 1
# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
#del family['FamilySize']
#print(family.head())


""" Step 5: Assemble final datasets for modelling """
# Step 5.1 Variable selection
# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket, title
full_X = pd.concat( [ title, pclass, ticket, family, imputed , embarked , cabin , sex ] , axis=1 )
#print(full_X.head())
#print(list(full_X))

# Step 5.2 Create datasets
# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
del titanic
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

''' Use this function if want export matrix to LibSVM format'''
def exportLibSVMFormat(train_X, train_y):
	train_libsvm = pd.DataFrame()
	train_libsvm['label'] = train_y.astype(int)
	row_num = train_X.shape[0]
	for i, col in enumerate(list(train_X)):
		li = []
		col_serie = train_X[col]
		for row in list(train_X.index):
			index = str(i +1)
			value = str(col_serie[row])
			feature = index +":"+value
			li.append(feature)
		newCol = pd.Series(li, name=col, index=list(train_X.index))
		train_libsvm[col] = newCol

	train_libsvm.to_csv('titanic_train_LibSVM.csv',sep=' ', index=False, header=False)
	#train_X.to_csv('titanic_train_X.csv',sep=' ', index=False, header=False)

#exportLibSVMFormat(train_valid_X,train_valid_y )


#print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)

# Step 5.3 Feature importance
# Selecting the optimal features in the model is important.
# We will now try to evaluate what the most important variables are for the model to make the prediction.
#'''plot_variable_importance(train_X, train_y) #DecisionTreeClassifier'''

""" Step 6: Modeling"""
# Step 6.1 Model Selection

# Support Vector Machines
model = SVC()

'''
# DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 99 )
# Random Forests Model
model = RandomForestClassifier(n_estimators=100)
# Gradient Boosting Classifier
model = GradientBoostingClassifier()
# K-nearest neighbors
model = KNeighborsClassifier(n_neighbors = 3)
# Gaussian Naive Bayes
model = GaussianNB()
# Logistic Regression
model = LogisticRegression()
'''
# Step 6.2 Train the selected model
print('------------------------\nTraining SVM model')
print(model.fit( train_X , train_y ))
#print(model.fit(train_valid_X , train_valid_y))
""" Step 7: Evaluation 
We can evaluate the accuracy of the model by using the validation set where we know the actual outcome.
 This data set have not been used for training the model, so it's completely new to the model.
We then compare this accuracy score with the accuracy when using the model on the training data. 
If the difference between these are significant this is an indication of overfitting. We try to avoid this because it means the model will not generalize well to new data and is expected to perform poorly.
"""
print("---------------------------\nEvaluting\n")
# Score the model
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))

print("-----------------------------\nPredicting")
test_Y = model.predict( test_X )
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y.astype(int) } )
test.to_csv( 'titanic_pred.csv' , index = False )

