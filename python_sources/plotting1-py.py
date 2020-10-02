# -*- coding: utf-8 -*-
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


from scipy.stats import mode

def cleandf(df):

    #cleaning fare column
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    classmeans = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
    
    #cleaning the age column
    meanAge=np.mean(df.Age)
    df.Age=df.Age.fillna(meanAge)
    
    #cleaning the embarked column
    df.Cabin = df.Cabin.fillna('Unknown')
    modeEmbarked = df.Embarked.mode()[0]
    print('mode:'+str(modeEmbarked))
    df.embarked = df.Embarked.fillna(modeEmbarked)

    return df

def cleaneddf(no_bins=0):
    #you'll want to tweak this to conform with your computer's file system
    trainpath = '../input/train.csv'
    testpath = '../input/test.csv'
    traindf = pd.read_csv(trainpath)
    testdf = pd.read_csv(testpath)
    
    #discretise fare
    if no_bins==0:
        return [cleandf(traindf), cleandf(testdf)]
    traindf=cleandf(traindf)
    testdf=cleandf(testdf)
    bins_and_binned_fare = pd.qcut(traindf.Fare, no_bins, retbins=True)
    bins=bins_and_binned_fare[1]
    traindf.Fare = bins_and_binned_fare[0]
    testdf.Fare = pd.cut(testdf.Fare, bins)
    
    #discretise age
    bins_and_binned_age = pd.qcut(traindf.Age, no_bins, retbins=True)
    bins=bins_and_binned_age[1]
    
    traindf.Age1 = traindf.Age
    testdf.Age1 = testdf.Age
    
    traindf.Age = bins_and_binned_age[0]
    testdf.Age = pd.cut(testdf.Age, bins)
    
    #create a submission file for kaggle
    predictiondf = pd.DataFrame(testdf['PassengerId'])
    predictiondf['Survived']=[0 for x in range(len(testdf))]
    predictiondf.to_csv('prediction.csv',
                  index=False)
    return [traindf, testdf]


data = cleaneddf()
traindf, testdf =data[0], data[1]


def proportionSurvived(discreteVar):
    by_var = traindf.groupby([discreteVar,'Survived'])
    table = by_var.size().unstack()
    normedtable = table.div(table.sum(1), axis=0)
    return normedtable

discreteVarList = ['Sex', 'Pclass', 'Embarked']

fig1, axes1 = plt.subplots(3,1)

for i in range(3):
    var = discreteVarList[i]
    table = proportionSurvived(var)
    table.plot(kind='barh', stacked=True, ax=axes1[i])
#fig1.show()

fig2, axes2 = plt.subplots(2,3)
genders=traindf.Sex.unique()
classes=traindf.Pclass.unique()

#normalise rgb
def normrgb(rgb):
    rgb = [float(x)/255 for x in rgb]
    return rgb
    

#list our colors
darkpink, lightpink =normrgb([255,20,147]), normrgb([255,182,193])
darkblue, lightblue = normrgb([0,0,128]),normrgb([135,206, 250])
for gender in genders:
    for pclass in classes:
        if gender=='male':
            colorscheme = [lightblue, darkblue]
            row=0
        else:
            colorscheme = [lightpink, darkpink]
            row=1
        group = traindf[(traindf.Sex==gender)&(traindf.Pclass==pclass)]
        group = group.groupby(['Embarked', 'Survived']).size().unstack()
        group = group.div(group.sum(1), axis=0)
        group.plot(kind='barh', ax=axes2[row, (int(pclass)-1)], color=colorscheme, stacked=True, legend=False).set_title('Class '+str(pclass)).axes.get_xaxis().set_ticks([])

plt.subplots_adjust(wspace=0.4, hspace=1.3)

fhandles, flabels = axes2[1,2].get_legend_handles_labels()
mhandles, mlabels = axes2[0,2].get_legend_handles_labels()
plt.figlegend(fhandles, ('die', 'live'), title='Female', loc='center', bbox_to_anchor=(0.06, 0.45, 1.1, .102))
plt.figlegend(mhandles, ('die', 'live'), 'center', title='Male',bbox_to_anchor=(-0.15, 0.45, 1.1, .102))

#fig2.show()

def removeBadStringFromString(string, badStringList):
    for badString in badStringList:
        string = string.replace(badString, '')
    return string

def removeBadStringFromLabels(ax, badStringList):
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [removeBadStringFromString(label, badStringList) for label in labels]
    return labels

def plotVarsGroupedBySex(varList, varNames, badStringList):
    fig, axes = plt.subplots(1,2)
    for i in range(2):
        group = traindf.groupby([varList[i], 'Sex', 'Survived'])
        group = group.size().unstack()
        group = group.div(group.sum(1), axis=0)
        cols = [[darkpink, darkblue],[lightpink, lightblue]]
        group.plot(kind='barh', stacked=True, ax=axes[i],legend=False, color=cols)
        labels = removeBadStringFromLabels(axes[i], badStringList)
        axes[i].set_yticklabels(labels)
        axes[i].get_xaxis().set_ticks([])
        axes[i].set_ylabel('') 
        axes[i].set_title(varNames[i])
            
        if i ==1:
            axes[i].yaxis.tick_right()
            axes[i].yaxis.set_label_position("right")
            
    handles, labels = axes[0].get_legend_handles_labels()
    plt.figlegend(handles[0], ['die', 'die'], loc='lower left')
    plt.figlegend(handles[1], ['live','live'], loc='lower right')
    #fig.show()

bins = [0,5,14, 25, 40, 60, 100]
binNames =['Young Child', 'Child', 'Young Adult', 'Adult', 'Middle Aged', 'Older']
binAge = pd.cut(traindf.Age, bins, labels=binNames)
binFare = pd.qcut(traindf.Fare, 3, labels=['Cheap', 'Medium', 'Expensive'])
binVars = [binAge, binFare]
varNames = ['Age', 'Fare']
badStringList=['(', ')', 'female', 'male', ',']
plotVarsGroupedBySex(binVars, varNames, badStringList)

varNames = ['# Parents or Children', '# Siblings or Spouses']
varList = ['Parch', 'SibSp']
plotVarsGroupedBySex(varList, varNames, badStringList)

plt.show()
