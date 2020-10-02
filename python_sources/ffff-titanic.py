env="not-local"

####################################
## Import libraries
####################################
import sys
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Handle table-like data and matrices
import numpy as np
import pandas as pd
from math import log2
from collections import Counter
import datetime
import scipy
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ]=10,8
pylab.rcParams[ 'figure.subplot.wspace' ]=.3
pylab.rcParams[ 'figure.subplot.hspace' ]=.6

####################################
## Load datasets
####################################
if env=="local":
    train=pd.read_csv('./input/train.csv')
    test=pd.read_csv('./input/test.csv')
else:
    train=pd.read_csv('../input/train.csv')
    test=pd.read_csv('../input/test.csv')

####################################
## Support functions: Generic
####################################
def someDay():
    # Code source: Gaël Varoquaux
    #              Andreas Müller
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    h = .02  # step size in the mesh
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]
    
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    
    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]
    
    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
    
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
    
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
    
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6)
    
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
    
    plt.tight_layout()
    plt.show()

## Dataframe description 
def describe(df,label="Survived"):
    ### Explore the Data ###
    print("## Attributes")
    print(df.info())   
    #print("## Descriptive statistics")
    #print(df.describe()) 
    print("## Features with null values")
    plotFeaturesWithNullValues(df)
    print("## Correlation map")
    plot_correlation_map(df)
    print("## Correlation with ",label)
    plot_correlation(df,label)
    #print("train.head(20)")
    #print(train.head(20)) ## Show some data... better to see in excel

def plot_correlation_map(df):
    corr=df.corr()
    _,ax=plt.subplots(figsize=(10,10))
    sns.heatmap(corr,
        cmap='YlGnBu',linecolor="white",
        square=True,
        cbar_kws={ 'shrink' : .9 },
        ax=ax,annot=True,
        annot_kws={ 'fontsize' : 8 },
        vmax=.8, linewidths=0.01
        )
    plt.show()

def plot_correlation(df,label):
    pylab.rcParams[ 'figure.figsize' ]=5,4
    corr=df.corr()[label].abs().sort_values(ascending=False)
    corr=corr[~corr.index.isin([label,"PassengerId"])]
    corr.plot(kind='bar',title="Correlation with "+label)
    plt.show()

## Dataframe entropy
def processInfoGainTree(df,label,depth=0,msg="",minInfoGain=0.01,maxDepth=5):
    currentAccuracy=df[label].mean()
    currentAccuracy=max(currentAccuracy,1-currentAccuracy)
    #
    currentEntropy=__entropiesBeforeFiltering(df,label)
    #
    if depth>=maxDepth:
        print ("Accuracy(Depth)",depth,msg,currentAccuracy)
        return currentAccuracy
    #
    gain=studyEntropies(currentEntropy,df,label)
    #gain.plot(y='gain',kind='bar')
    #plt.title("Information gain after grouping by each feature")
    #plt.show()
    #
    selectedFeature=gain["feature"][0]
    gain=gain["gain"][0]
    #
    if gain==0 or gain<minInfoGain:
        print ("Accuracy(gain)",depth,msg,currentAccuracy)
        return currentAccuracy
    #
    print (depth,msg,selectedFeature,gain)
    #
    g1=df.groupby([selectedFeature])##grouped by the selected feature
    dfCount=df[label].count()
    newAccuracy=0
    for valueOfSelectedFeature,g in g1: ##for each value in the selected feature
        #print("Calculating information gain where ",selectedFeature,"==",valueOfSelectedFeature)
        dfaux=df[df[selectedFeature]==valueOfSelectedFeature]
        dfauxCount=dfaux[label].count()
        childAccuracy=processInfoGainTree(dfaux,label,depth+1,msg+","+selectedFeature+"=="+str(valueOfSelectedFeature))
        newAccuracy+=dfauxCount/dfCount*childAccuracy
    print ("Accuracy(Children)",depth,msg,newAccuracy)
    return newAccuracy

def studyEntropies(currentEntropy,df,label):
    #entropy after filtering by features
    featureList=[]
    gainList=[]
    features=list(df.columns.values)
    for feature in features:
        if feature!=label:
            ent,count=__entropy(df,label,feature)
            gain=(currentEntropy-ent)/count 
            featureList.append(feature)
            gainList.append(gain)
    #
    entropies=pd.DataFrame(data={"feature":featureList,"gain":gainList},index=featureList)
    entropies.sort_values("gain",inplace=True, ascending=False)
    return entropies

def __entropy(df,label,category):
	entropy=0
	g2=df.groupby([category,label]).size().unstack()##grouped by category and label
	g1=df.groupby([category])##grouped by label only
	for key,g in g1: ##for each value category
		classWeight=g.count()[label]
		if classWeight!=0:
			entropyClass=0
			g2f=g2.ix[key]
			for q in g2f: ##for each value of the field for cat "key"
				if not np.isnan(q) and q!=0:
					prob=q/classWeight
					entropyClass+=-(prob*log2(prob))
			entropy+=classWeight*entropyClass
	entropy=entropy/g2.sum().sum()
	count=g1[label].size().count()
	if count > 10:
		print("Category has too many possible values,consider grouping them. Category:",category)
	#	print(list(g1.size().index))
	return entropy,count

def __entropiesBeforeFiltering(df,label):
    selectedColumn=df[label]
    entropy=0
    numRows=len(selectedColumn) #Cantidad total de registros
    if numRows==0:
        return 0
    field_counter=Counter(selectedColumn) #Distinct de valores de Survived y su counts
    for labelValue,labelCount in field_counter.items():
        prob=labelCount/numRows #Frecuencia relativa del valor de survived
        if prob!=0:
            entropy+=-prob*log2(prob)
    return entropy

def processAccuracyGainTree(df,label,depth=0,msg="",minGain=0.01,maxDepth=5):
    currentAccuracy=df[label].mean()
    currentAccuracy=max(currentAccuracy,1-currentAccuracy)
    #
    if depth>=maxDepth:
        #print ("Accuracy(Depth)",msg,currentAccuracy)
        return currentAccuracy
    #
    gain=studyAccuracies(currentAccuracy,df,label)
    #
    #print ("##Plot ",msg)
    #gain.plot(y='gain',kind='bar')
    #plt.title("Accuracy gain after grouping by each feature")
    #plt.show()
    #
    selectedFeature=gain["feature"][0]
    accuracyGain=gain["gain"][0]
    #
    if accuracyGain==0 or accuracyGain<minGain:
        #print ("Accuracy(MinG) ",msg)
        return currentAccuracy
    #
    g1=df.groupby([selectedFeature])##grouped by the selected feature
    dfCount=df[label].count()
    newAccuracy=0
    for valueOfSelectedFeature,g in g1: ##for each value in the selected feature
        #print("Calculating information gain where ",selectedFeature,"==",valueOfSelectedFeature)
        dfaux=df[df[selectedFeature]==valueOfSelectedFeature]
        dfauxCount=dfaux[label].count()
        childAccuracy=processAccuracyGainTree(dfaux,label,depth+1,msg+selectedFeature+"=="+str(valueOfSelectedFeature)+",")
        newAccuracy+=dfauxCount/dfCount*childAccuracy
    newAccuracy=max(newAccuracy,1-newAccuracy)
    print ("Accuracy(Child)",msg+selectedFeature,newAccuracy)
    return newAccuracy

def studyAccuracies(currentAccuracy,df,label):
    #entropy after filtering by features
    featureList=[]
    accuracyGainList=[]
    features=list(df.columns.values)
    for feature in features:
        if feature!=label:
            newAccuracy=__accuracy(df,label,feature)
            accuracyGain=newAccuracy-currentAccuracy 
            featureList.append(feature)
            accuracyGainList.append(accuracyGain)
    #
    accuracies=pd.DataFrame(data={"feature":featureList,"gain":accuracyGainList},index=featureList)
    accuracies.sort_values("gain",inplace=True, ascending=False)
    #print ("-->",accuracies)
    return accuracies

def __accuracy(df,label,selectedFeature):
    g1=df.groupby([selectedFeature])##grouped by the selected feature
    dfCount=df[label].count()
    accuracy=0
    for valueOfSelectedFeature,g in g1: ##for each value in the selected feature
        #print("Calculating information gain where ",selectedFeature,"==",valueOfSelectedFeature)
        dfaux=df[df[selectedFeature]==valueOfSelectedFeature]
        dfauxCount=dfaux[label].count()
        childAccuracy=dfaux[label].mean()
        childAccuracy=max(childAccuracy,1-childAccuracy)
        accuracy+=dfauxCount/dfCount*childAccuracy
    accuracy=max(accuracy,1-accuracy)
    return accuracy

## Dummy features
def createDummiesAndDeleteFeature(df,feature):
    aux_dum=pd.get_dummies(df[feature],prefix=feature)
    aux=pd.concat([df,aux_dum],axis=1)
    aux.drop(feature,axis=1,inplace=True)
    return aux

def createDummiesAndDeleteFeatures(df,features):
    for i in range(len(features)):
        df=createDummiesAndDeleteFeature(df,features[i])
    return df

## Plotting 
def plotSelectedCategoricals(df,features,nColumns=3,label="Survived"):
    if nColumns!=1:
        pylab.rcParams['figure.figsize']=10,10
        fig=plt.figure()
        nCharts=0
        for feature in features:
            nCharts+=1
        ii=0
        for feature in features:
            ii+=1
            ax=fig.add_subplot(1+nCharts/nColumns,nColumns,ii)
            g=sns.factorplot(x=feature,y=label,data=df,ci=95,ax=ax)
            plt.close(g.fig)
        plt.show()
    else:
        pylab.rcParams['figure.figsize']=5,4
        for feature in features:
            g=sns.factorplot(x=feature,y=label,data=df,ci=95)
            plt.show()

def plotSelectedNonCategoricals(df,features,nColumns=3,label="Survived"):
    for i in range(len(features)):
        feature=features[i]
        plot_distribution(df,var=feature,target=label)
        plt.show()


#################Revisar desde aqui para abajo

def plot_histograms(df,features,n_rows,n_cols):
    fig=plt.figure( figsize=( 16,12 ) )
    for i,var_name in enumerate(features):
        ax=fig.add_subplot( n_rows,n_cols,i+1 )
        df[ var_name ].hist( bins=10,ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ),) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [],visible=False )
        ax.set_yticklabels( [],visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df,var,target,**kwargs ):
    row=kwargs.get('row',None )
    col=kwargs.get('col',None )
    facet=sns.FacetGrid(df,hue=target,aspect=4,row=row,col=col)
    facet.map(sns.kdeplot,var,shade=True)
    facet.set(xlim=(0,df[var].max()))
    facet.add_legend()

def plot_categories( df,cat,target,**kwargs ):
    row=kwargs.get( 'row',None )
    col=kwargs.get( 'col',None )
    facet=sns.FacetGrid( df,row=row,col=col )
    facet.map( sns.barplot,cat,target )
    facet.add_legend()

def plotFeaturesWithNullValues(df):
    ##List features with null values
    null_columns=df.columns[df.isnull().any()]
    for col in null_columns:
        print ("%s %s" % (col, df[col].isnull().sum()) )
    return
    ##Plot features with null values
    labels = []
    values = []
    null_columns=df.columns[df.isnull().any()]
    for col in null_columns:
        labels.append(col)
        values.append(df[col].isnull().sum())
    ind = np.arange(len(labels))
    width=0.6
    fig,ax = plt.subplots(figsize=(6,5))
    ax.barh(ind,np.array(values),color='purple')
    ax.set_yticks(ind+((width)/2.))
    ax.set_yticklabels(labels,rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_ylabel("Column Names")
    ax.set_title("Variables with missing values")
    plt.plot()

####################################
## Support functions: Specific for Titanic classification
####################################

### Feature Engineering - Functions ###
def cleanCabin(cabin):
    if cabin=="Unk":
        return cabin
    ##En "F Exxx" quita "F "
    pos=cabin.find(" ")
    if pos!=-1 and pos<3:
        cabin=cabin[cabin.find(" ")+1:]
    ##En "Fxxx Exxx" quita "Exxx"
    pos=cabin.find(" ")
    if pos!=-1:
        cabin=cabin.split(' ')[0]
    ##En "D" returna "Unk"
    if len(cabin)==1:
        return "Unk"
    return cabin
def extractDeck(cabin):
    return cabin[0]
def extractDeckNoMod2(cabin):
    if cabin=="Unk":
        return cabin
    aux=cabin[1:]
    return int(aux) % 2
def extractDeckNoDiv10(cabin):
    if cabin=="Unk":
        return cabin
    aux=cabin[1:]
    return int (int(aux)/10)
def extractAgeGroup(age):
    if age=="Unk":
        return age
    if int(age)>=40:
        age=40
    return int(int(age)/10)
def extractMum(sex,parch,age,title):
    return 1 if sex=="female" and parch>0 and age>18 and title!="Miss" else 0    

### Feature Engineering ###
def prepareData(df):
    ##########################
    ### Feature Engineering -- Fill missing data
    ##########################
    ## Average used for numeric features -- Age
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    ## "Average used for numeric features -- Fare
    df['Fare'].fillna(df['Fare'].mean(),inplace=True)
    #
    ## Unk is better than ? for numeric feature creation -- done later on
    ## "Unk" is used for non-numeric features -- Cabin
    df['Cabin'].fillna('Unk',inplace=True)
    ## "Unk" is used for non-numeric features -- Embarked
    df['Embarked'].fillna('Unk',inplace=True)
    #Another approach for Embarked:
    #df[df['Embarked'].isnull()]
    #PassengerId 62 and 830 have missing embarked values. Both have Passenger class 1 and fare $80. Lets plot a graph to visualize and try to guess from where they embarked
    #We can see that for 1st class median line is coming around fare $80 for embarked value 'C'. So we can replace NA values in Embarked column with 'C'
    #sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df);
    #df['Embarked'].fillna('C',inplace=True)

    ##########################
    ### Feature Engineering -- clean data (ensure format)
    ##########################
    ## Clean data -- Cabin
    df['Cabin']=df['Cabin'].map(cleanCabin)

    ##########################
    ### Feature Engineering -- Create features
    ##########################
    #Turning Name into Title
    titleDictionary={
        "Capt":       "Off",
        "Col":        "Off",
        "Major":      "Off",
        "Jonkheer":   "Roy",
        "Don":        "Roy",
        "Sir" :       "Roy",
        "Dr":         "Off",
        "Rev":        "Off",
        "the Countess":"Roy",
        "Dona":       "Roy",
        "Mme":        "Mrs",
        "Mlle":       "Miss",
        "Ms":         "Mrs",
        "Mr" :        "Mr",
        "Mr2" :       "Mr",
        "Mrs" :       "Mrs",
        "Miss" :      "Miss",
        "Master" :    "Master",
        "Lady" :      "Roy"
        }
    #
    #df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0])
    #Lucky last names
    #train['LastName'] = train['Name'].apply(lambda x: x.split(',')[0])
    #luckyLastNames=train.groupby(['LastName'])['Survived'].mean()
    #luckyLastNames=luckyLastNames[luckyLastNames>.9]
    #df['LastName'] = df['LastName'].apply(lambda x: 1 if x in luckyLastNames else 0)
    #
    df['Title']=df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip() )
    df['Title']=df['Title'].map(lambda x: titleDictionary[x])
    df['Title']=df['Title'].fillna('Unk')
    #Turning Cabin number into Deck
    df['IsCabinKnown']=df['Cabin'].map(lambda x: 1 if x!="Unk" else 0)
    #df['Deck']=df['Cabin'].map(extractDeck) ## Commented out as it adds no information
    #df['DeckNoDiv10']=df['Cabin'].apply(extractDeckNoDiv10) ## Commented out as it adds no information
    #df['DeckNoMod2']=df['Cabin'].apply(extractDeckNoMod2) ## Commented out as it adds no information
    #
    #Creating new FamSize column
    df['FamSize']=df['SibSp']+df['Parch']+1
    df['FamSize']=df['FamSize'].map(lambda x: 1 if x==1 else 2 if 2 <=x<= 4 else 3 if 5<=x else 'Unk')
    #Sex is binary. Simplify.
    df['Sex_Male']=df['Sex'].map(lambda x:1 if x=='male' else 0)
    df['Mum_True']=np.vectorize(extractMum)(df['Sex'],df['Parch'],df['Age'],df['Title'])
    #Age groups
    #df['Age']=df['Age'].map(lambda age: int(age))
    df['AgeGroup']=df['Age'].apply(extractAgeGroup)
    #
    bins = (0, 8, 15, 31, 1000) ## train.describe() -> min, 25%, 50%, 75% and max
    group_names = [1, 2, 3, 4]
    df['Fare']=pd.cut(df['Fare'], bins, labels=group_names)

    ### Cleaning Data ###
    # get rid of the useless cols
    df.drop(['Name','Ticket','SibSp','Parch','Sex',"Age","Cabin"],axis=1,inplace=True)

    ### Plot training data ###
    if 'Survived' in df.columns:
        print("## Features before creating dummies")
        describe(df)
        print("## Plotting data before creating dummies")
        plotSelectedCategoricals(df,["Sex_Male","Fare","IsCabinKnown","AgeGroup","Mum_True","Pclass","FamSize","Embarked","Title"])
        #plotSelectedNonCategoricals(df,["FamSize"])

    ### Create dummy columns ###
    df=createDummiesAndDeleteFeatures(df,['Pclass','Embarked','Title','Fare','FamSize','AgeGroup'])

    ### Reindex columns ###
    df=df.reindex(columns=[
        'PassengerId','Survived',
        "IsCabinKnown","Sex_Male","Mum_True",
        "AgeGroup_0","AgeGroup_1","AgeGroup_2","AgeGroup_3","AgeGroup_4",
        "Fare_1","Fare_2","Fare_3","Fare_4",
        "Pclass_1","Pclass_2","Pclass_3",
        "Embarked_C","Embarked_Q","Embarked_S","Embarked_Unk",
        "Title_Master","Title_Miss","Title_Mr","Title_Mrs","Title_Off","Title_Roy",
        "FamSize_1","FamSize_2","FamSize_3","FamSize_Unk"
        ],fill_value=0)

    return df

####################################
## Describe
####################################

describe(train)

#print("## Numeric features histogram map")
#train.ix[:, train.columns.difference(["PassengerId"])].hist(bins=10,grid=False)
#plt.show()

#Feature pair histogram map (categorical, non-categorical)
#print("## Feature pair histogram map (categorical, non-categorical)")
#g=sns.FacetGrid(train,col="Sex",row="Survived",margin_titles=True)
#g.map(plt.hist,"Age",color="purple");
#plt.show()

#Feature pair scatter map (non-categorical, non-categorical)
#g=sns.FacetGrid(train,hue="Survived",col="Pclass",margin_titles=True,palette={1:"seagreen", 0:"gray"})
#g=g.map(plt.scatter,"Fare","Age",edgecolor="w").add_legend();
#plt.show()

#Feature trio scatter map (categorical, non-categorical, non-categorical)
#g=sns.FacetGrid(train,hue="Survived",col="Sex",margin_titles=True,palette="Set1",hue_kws=dict(marker=["^", "v"]))
#g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
#plt.subplots_adjust(top=0.8)
#g.fig.suptitle('Survival by Gender , Age and Fare');
#plt.show()

#Feature pair probability map (categorical, categorical)
#sns.set(font_scale=1)
#g = sns.factorplot(x="Sex", y="Survived", col="Pclass",data=train, saturation=.5,kind="bar", ci=None, aspect=.6)
#(g.set_axis_labels("", "Survival Rate").set_xticklabels(["Men", "Women"]).set_titles("{col_name} {col_var}").set(ylim=(0, 1)).despine(left=True))
#plt.subplots_adjust(top=0.8)
#g.fig.suptitle('How many Men and Women Survived by Passenger Class');
#plt.show()

#Feature box map (categorical)
#ax=sns.boxplot(x="Survived", y="Age", data=train)
#ax=sns.stripplot(x="Survived", y="Age",data=train, jitter=True,edgecolor="gray")
#sns.plt.title("Survival by Age",fontsize=12);
#plt.show()

#Feature histogram (categorical, non-categorical)
#train.Age[train.Pclass == 1].plot(kind='kde')
#train.Age[train.Pclass == 2].plot(kind='kde')
#train.Age[train.Pclass == 3].plot(kind='kde')
#plt.xlabel("Age")
#plt.title("Age Distribution within classes")
#plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;

#Feature violin (categorical, categorical, non-categorical)
#g=sns.factorplot(x="Age", y="Embarked",
#    hue="Sex", row="Pclass",
#    data=train[train.Embarked.notnull()],
#    orient="h", size=2, aspect=3.5,
#    palette={'male':"purple", 'female':"blue"},
#    kind="violin", split=True, cut=0, bw=.2);
#plt.show()

#Feature boxplot (categorical, categorical, non-categorical)
#sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train);

#linear regression charts
#sns.pairplot(train,x_vars=['Fare'],y_vars='Survived',kind='reg')
#sns.pairplot(train,x_vars=['Parch'],y_vars='Survived',kind='reg')

#Visualizing data is crucial for recognizing underlying patterns to exploit in the model.
#sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

#sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
#    palette={"male": "blue", "female": "pink"},
#    markers=["*", "o"], linestyles=["-", "--"]);
            
### Normalize data
train=prepareData(train)
test=prepareData(test)

print("## Features after creating dummies")
describe(train)

pylab.rcParams['figure.figsize']=5,4

print("## Accuracy gain decision tree")
accuracy=processAccuracyGainTree(train[train.columns.difference(["PassengerId"])],"Survived")
print("## Final accuracy",accuracy)
    
#print("## Entropy on Survived after filtering by other categories")
#entropies=entropies(train[train.columns.difference(["PassengerId"])],'Survived')
#entropies.plot(y='entropy',kind='bar')
#plt.title("Entropies after grouping by each feature")
#plt.show()

#Remove PassengerId as it is important for training,but it is only confusing the model.
train_x=train.drop(['Survived','PassengerId'],axis=1)
train_y=train['Survived']
test_ids=pd.DataFrame(test['PassengerId'],columns=['PassengerId'])
test_x=test.drop(['Survived','PassengerId'],axis=1)

print("## Null accuracy",max(train['Survived'].mean(),1-train['Survived'].mean()))

classifiers=[]
classifierParams=[]

classifiers.append(GaussianNB())
classifierParams.append({
    'priors':[None]
    })

classifiers.append(DecisionTreeClassifier())
classifierParams.append({
    'criterion':['gini'],  #entropy
    'max_depth':list(range(1,20,1))
    })

classifiers.append(GradientBoostingClassifier())
classifierParams.append({
      'max_depth': [2,3,4,5,6,7,8], 
      'n_estimators': [20], 
      'min_samples_leaf': [2]
    })

classifiers.append(SVC())
classifierParams.append({
    'C': [0.025], 
    'gamma': [1],
    'kernel': ['linear','poly','rbf'],
    'class_weight':['balanced',None]
    })

classifiers.append(KNeighborsClassifier())
classifierParams.append({
    "n_neighbors":list(range(1,50,2))
    })

classifiers.append(RandomForestClassifier())
classifierParams.append({
      'max_depth': [2,3,4,5,6,7,8], 
      'n_estimators': [20], 
      'max_features': ['log2'], #"sqrt" 
      'criterion': ['gini'], #"entropy"
      'min_samples_leaf': [2],
      'n_jobs': [-1]
    })

classifiers.append(ExtraTreesClassifier())
classifierParams.append({
      'max_depth': [2,3,4,5,6,7,8], 
      'n_estimators': [20], 
      'min_samples_leaf': [2],
      'n_jobs': [-1]
    })

classifiers.append(AdaBoostClassifier())
classifierParams.append({
      'n_estimators': [20], 
      'learning_rate': [.5, .75, 1, 1.5, 2]
    })

classifiers.append(LogisticRegression())
classifierParams.append({
      'C': [0.01, 0.1, 1, 1.5, 2, 3, 4, 10]
    })

def testClassifiers(classifiers,classifierParams,train_x,train_y):
    scores=[]
    best=[]
    bestnames=[]
    i=0
    for classifier in classifiers:
        # Loof for best settings
        paramGrid=classifierParams[i]
        grid=GridSearchCV(classifier,param_grid=paramGrid,cv=10,scoring='accuracy')
        grid.fit(train_x,train_y)
        # Score best classificator
        classifier=grid.best_estimator_
        cvscores=cross_val_score(classifier,train_x,train_y,cv=10,scoring='accuracy')
        score=cvscores.mean()
        print("##",classifier.__class__.__name__,score)
        print("  ",classifier)
        classifier.fit(train_x,train_y)
        #Plot
        __testClassifiersPlot(grid,classifier)
        # Scores
        scores.append(score)
        best.append(classifier)
        bestnames.append(classifier.__class__.__name__)
        # Counter
        i=i+1
    #Print results
    i=0
    print("## Scores")
    for classifier in best:
        score=scores[i]
        print("  ",classifier.__class__.__name__,score)
        print("  ",classifier)
        print(" ")
        i=i+1
    # Chart
    imp=pd.DataFrame(scores,columns=['Score'],index=bestnames)
    imp.plot(kind='barh')
    plt.xlabel("Classifier" )
    plt.ylabel("Score" )
    plt.title("Scores")
    plt.show()
    #
    return best

def __testClassifiersPlot(grid,classifier):
        pylab.rcParams[ 'figure.figsize' ]=10,4
        fig=plt.figure()
        ax=fig.add_subplot(1,2,1)
        # Chart classifier selection
        gridMeanScores=[result.mean_validation_score for result in grid.grid_scores_]
        imp=pd.DataFrame(gridMeanScores,columns=['Score'])
        imp.plot(kind='bar',ax=ax)
        plt.xlabel("Settings" )
        plt.ylabel("Score" )
        plt.title("Scores " + classifier.__class__.__name__)
        # Plot feature importance if applicable to the classifier
        if hasattr(classifier, 'feature_importances_'):
            ax=fig.add_subplot(1,2,2)
            imp=pd.DataFrame(classifier.feature_importances_,columns=['Importance'],index=train_x.columns)
            imp=imp.sort_values(['Importance'],ascending=True)
            imp.plot(kind='barh',ax=ax)
            plt.title("Feature importance" + classifier.__class__.__name__)
        plt.show()
        
best=testClassifiers(classifiers,classifierParams,train_x,train_y)

trainExt_x = pd.DataFrame()
testExt_x = pd.DataFrame()
for classifier in best:
    trainExt_x[classifier.__class__.__name__]=classifier.predict(train_x)
    testExt_x[classifier.__class__.__name__]=classifier.predict(test_x)

from sklearn import tree
print("## Decision tree")
print(tree.export_graphviz(best[1],out_file=None,feature_names=train_x.columns,label="none",impurity=False))

def printTree(tree,features,node=0,depth=0):
    print(depth,node,features[tree.feature[node]],"<=",tree.threshold[node])
    if tree.children_left[node]!=-1:
        printTree(tree,features,node=tree.children_left[node],depth=depth+1)
    if tree.children_right[node]!=-1:
        printTree(tree,features,node=tree.children_right[node],depth=depth+1)

printTree(best[1].tree_,train_x.columns)

#print("## Plot number of features VS. cross-validation scores")
#rfecv=RFECV(estimator=classifier,step=1,cv=StratifiedKFold(train_y,10),scoring='accuracy')
#rfecv.fit(train_x,train_y)
#plt.figure()
#plt.title("RFECV")
#plt.xlabel("Number of features selected" )
#plt.ylabel("Cross validation score (nb of correct classifications)" )
#plt.plot( range( 1,len( rfecv.grid_scores_ ) + 1 ),rfecv.grid_scores_ )
#plt.show()
#print("## RFECV score",rfecv.score(train_x,train_y))
#print("   Optimal number of features : %d" % rfecv.n_features_ )

#best=testClassifiers(classifiers,classifierParams,trainExt_x,train_y)

#Score classifier
classifier=LogisticRegression(C=1)
scores=cross_val_score(classifier,trainExt_x,train_y,cv=10,scoring='accuracy')
print("## LogisticRegression on top of predicted values",scores.mean())
#Train & oredict
classifier.fit(trainExt_x,train_y)
prediction=classifier.predict(testExt_x)

#Save output
prediction=pd.DataFrame(prediction,columns=['Survived'])
output=pd.concat([test_ids,prediction],axis=1,join_axes=[test_ids.index])
if env=="local":
    output.to_csv('./output/prediction.'+datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")+'.csv',index=False)
else:
    output.to_csv('prediction.'+datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")+'.csv',index=False)

