#!/usr/bin/env python
# coding: utf-8

# This notebook contains an approach to predict the survivors of the Titanic ship sinking using Neural Networks. Every step in the process, from getting the data to predicting the survivors is thoroughly documented and explained. 

# # 1. Reading the data and setting up the environment

# The first step to analyzing the data is to load all the libraries we are going to use. This is performed at the start so that we can know at any point which libraries are loaded in the notebook. 

# In[ ]:


get_ipython().run_cell_magic('capture', '', "import numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\nfrom pathlib import Path #flexible path files\nimport matplotlib.pyplot as plt #plotting\nfrom fastai import *  \nfrom fastai.tabular import *\nimport torch #Pytorch\nimport missingno as msno #library for missing values visualization\nimport warnings #ignoring warnings\nwarnings.filterwarnings('ignore')\n\n%matplotlib inline")


# Data file locations:

# In[ ]:


# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Now we can read the data into Pandas dataframes. A copy of the original data is kept should we require it later. Both training and test datasets are put together in a list so that we can iterate over both at the same time during data cleaning. 

# In[ ]:


path = Path('/kaggle/input/titanic')
trpath = path/'train.csv'
cvpath = path/'test.csv'

df_train_raw = pd.read_csv(trpath)
df_test_raw = pd.read_csv(cvpath)

df_train = df_train_raw.copy(deep = True)
df_test  = df_test_raw.copy(deep = True)

data_cleaner = [df_train_raw, df_test_raw] #to clean both simultaneously


# # 2. Undestanding the Data

# Let's first take a look at the first couple of rows of the training data, as well as the types of variables that the dataframe posesses and their corresponding value types.

# In[ ]:


df_train.head(n=10)


# In[ ]:


df_train.info()


# In[ ]:


varnames = list(df_train.columns)
for name in varnames:
    print(name+": ",type(df_train.loc[1,name]))


# The variables included in the data are:
# 
# * ***PassengerId***: Passenger index
# * Survived: Whether the passenger in the accident. Possible values:
#     * 0 = died , 
#     * 1 = survived
# * ***Pclass***: Passenger class. Possible values:
#     * 1 = First class
#     * 2 = Second class
#     * 3 = Third class
# * ***Name***: Passenger name
# * ***Sex***: Passenger gender. Possible values:
#     * male
#     * female
# * ***Age***: Passenger Age
# * ***SibSp***: Number of siblings/spouses on board
# * ***Parch***: Number of parents/children on board
# * ***Ticket***: Ticket number
# * ***Fare***: Ticket cost
# * ***Cabin***: Cabin number
# * ***Embarked***: Port of Embarkation. Possible values:
#     * C = Cherbourg
#     * Q = Queenstown
#     * S = Southampton

# It is very important to understand whether and where there are missing values in the data (both train and test). This will help us determine a strategy for filling in the missing values.

# In[ ]:


print("Training Set")
print(df_train.isnull().sum(axis=0))
print("Test Set")
print(df_test.isnull().sum(axis=0))


# In[ ]:


msno.matrix(df_train)


# In[ ]:


msno.bar(df_test)


# We can see that there are 2 major categories with missing data, as well as a couple of missing values in other two.
# 
# **Major missing value variables: **
# * Age
# * Cabin
# 
# **Minor missing value variables: **
# * Fare
# * Embarked

# # 3. Exploratory Data Analysis

# Before we start cleaning up the data, it is important to see which variables are of relevance, which can be ignored  and what is the most appropriate way to fill in the missing values. As we can see in the charts above, there are 3 variables with missing values in the training set(Age,Cabin and Embarked) and only 2 in the test set (Age,Cabin). In the test set, there is also 1 fare entry missing, which we will fill later on. We shall now try and decide what we are going to do with those values.

# In[ ]:


print('Overall survival quota:')
df_train['Survived'].value_counts(normalize = True)


# **Setting up plotting parameters:**

# In[ ]:


plt.style.use('seaborn')


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

plt.figure()
fig = df_train.groupby('Survived')['Age'].plot.hist(histtype= 'bar', alpha = 0.7)
plt.legend(('Died','Survived'), fontsize = 13)
plt.xlabel('Age', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.suptitle('Histogram of the ages of survivors and decased ones',fontsize =22)
plt.show()


# We see that the ages distribution between those who survived and those who did not is similar. We see, however, that more young-aged passengers were saved. This was expected, since it lines up with ship evacuation policies. Other than that, age is probably not a major factor that determined who survived the accident.

# Let's now explore the impact that the amount of relatives on board had on survival. For that, we create a new feature called 'Family onboard', which is the sum of parents/children/siblings/spouses (variables Parch and SibSp).

# In[ ]:


df_train['Family onboard'] = df_train['Parch'] + df_train['SibSp']
plt.rcParams['figure.figsize'] = [20, 7]
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

fig, axes = plt.subplots(nrows=1, ncols=3)
df_train.groupby(['Parch'])['Survived'].value_counts(normalize=True).unstack().plot.bar(ax=axes[1],width = 0.85)
df_train.groupby(['SibSp'])['Survived'].value_counts(normalize=True).unstack().plot.bar(ax=axes[2],width = 0.85)
df_train.groupby(['Family onboard'])['Survived'].value_counts(normalize=True).unstack().plot.bar(ax=axes[0],width = 0.85)

axes[0].set_xlabel('Family onboard',fontsize = 18)
axes[1].set_xlabel('parents / children aboard',fontsize = 18)
axes[2].set_xlabel(' siblings / spouses aboard',fontsize = 18)

for i in range(3):
    axes[i].legend(('Died','Survived'),fontsize = 13, loc = 'upper left')
axes[0].set_ylabel('Survival rate',fontsize = 18)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=0)

plt.suptitle('Survival rates over Number of relatives onboard',fontsize =22)
plt.show()


# We see a clear trend that passengers with a family size between 1 and 3 had the higher the chance of survival. They are the only columns where the survivors are more than the deceased ones. Family size also combines the other 2 variables nicely and gives a more clear picture of the survival chances. Therefore, we conclude that this is an interesting feature to include in our training data.

# In[ ]:


plt.rcParams['figure.figsize'] = [7, 5]
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

plt.figure()
fig = df_train.groupby(['Sex'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.5)
plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')
plt.xlabel('Gender',fontsize =18)
plt.xticks(rotation=0)
plt.ylabel('Survival rate',fontsize = 18)


plt.suptitle('Survival rates over Gender',fontsize =22)
plt.show()


# We also see that female passengers had a higher chance of survival than male ones. It was expected that females and children would be more likely to survive, as the evacuation protocol of the ship was instructing accordingly. Let us now compare the survival chances and the passengers' ticket class.

# In[ ]:


plt.rcParams['figure.figsize'] = [8, 5]
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

plt.figure()
fig = df_train.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.5)
plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')
plt.xlabel('Ticket Class',fontsize =18)
plt.ylabel('Survival rate',fontsize = 18)
plt.suptitle('Survival rate over Ticket class', fontsize = 22)
plt.xticks(rotation=0)
plt.show()


# As expected, first class passengers have a higher survival rate, meaning they were either given priority during evacuation or they were closer to the lifeboats. THis can be double-checked through the cabin feature that will be discussed later. 
# 
# We would now to check if the title name of a person can be useful in determining whether that person survived or not. This assumption stems from the idea that people of higher status could have been given higher priority during the ship's evacuation.  Therefore, we create a new variable called 'Title'.

# In[ ]:


df_train['Title'] = df_train['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()
varnames = list(df_train.columns)
    
print("Training set: " ,list(df_train['Title'].unique()))    
df_test['Title'] = df_test['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()
print("Test set: " ,list(df_test['Title'].unique()))    


# Some of these titles can be grouped up, since they mean the same thing. For example, "Mrs", "Miss", "Ms" will be grouped together under the label "Mrs". There are also some titles that appear to actually be a name instead of a title (Mlle, Mme, Dona) that will also be mapped to the same value. "Don" is probably an abbreviation to a male name and will be mapped to "Mr". The rest of the titles denote nobility, military or clergy service and doctors. To avoid sparse categories, they are all grouped under the title 'Notable'. Finally, 'Master' is kept as a standalone title that was given to men under 26 years of age.

# In[ ]:


def new_titles(df):
    new_titles = dict()
    assert 'Title' in df.columns
    for key in df['Title'].unique():
        females = ['Mrs','Miss','Ms','Mlle','Mme','Dona']
        males = ['Mr','Don']
        notable = ['Jonkheer','the Countess','Lady','Sir','Major','Col','Capt','Dr','Rev','Notable']
        titles = [females,males,notable,'Master']
        newtitles = ['Mrs','Mr','Notable','Master']
        idx = [key in sublist for sublist in titles]
        idx = np.where(idx)[0] 
        new_titles[key] = newtitles[idx[0]]
    return new_titles


new_titles_dict = new_titles(df_train)
df_train['Title'] = df_train['Title'].replace(new_titles_dict)


# We can now check the survival rates for each title to see if there is some useful information here.

# In[ ]:


plt.rcParams['figure.figsize'] = [12, 5]
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

plt.figure()
fig = df_train.groupby(['Title'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.7)
plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')
plt.xlabel('Title',fontsize =16)
plt.xticks(rotation=0)


plt.suptitle('Survival rates over Title',fontsize =22)
plt.show()


# Again, we see that different titles have different survival probabilities. A small surprising result is that people under the title 'Notable' have a low survival rate. One would expect that 'notable' people would travel first class and would therefore have a higher survival chance (see above), but is appears that this is not the case. This result indicates that the higher survival rates of the first class passengers' have to do with their positioning on the ship. We shall now examine that. To do that, we only keep the cabin deck portion of the 'cabin' variable and, since there are a lot of missing cabin information, the missing values are denoted as 'M' for missing. 

# In[ ]:


df_train['Cabin'][df_train['Cabin'].isnull()]='Missing'
df_train['Cabin'] = df_train['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]


# In[ ]:


plt.rcParams['figure.figsize'] = [12, 5]
plt.figure()
fig = df_train.groupby(['Cabin'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.9)
plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')
plt.xlabel('Cabin Deck',fontsize =18)
plt.suptitle('Survival rates over Cabin Deck',fontsize =22)
plt.xticks(rotation=0)
plt.show()


# We see that the cabin decks have different survival rates. As for the ones where the data was missing, the rates line up with the overall survival rate of the ship (~68%-32%).
# 
# 

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
plt.figure()
fig = df_train.groupby(['Embarked'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.7)
plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')
plt.xlabel('Embarking Port',fontsize =18)
plt.suptitle('Survival rates over embarking port',fontsize =22)
plt.xticks(rotation=0)
plt.show()


# Finally, we initially thought that the embarking port should be irrelevant to the task. However, passengers that embarked the ship in Cherbourg were more likely to survive. An explanation for that could be that more rich people embarked the ship and were travelling in a better class.

# In[ ]:


df_train.groupby(['Embarked'])['Pclass'].value_counts(normalize=True).unstack()


# # 4. Data Cleaning

# We are now going to ensure that there are no missing values in the dataset and prepare it for training our model. The 4 categories that have missing values in the train and test sets are:
# * Age 
# * Cabin 
# * Embarked 
# * Fare
# 
# In order to ease the documents' readability, any extra variables created above will be recreated here from scratch and will be encapsulated in a function. This is done to make it easier to the reader to find all feature engineering procedures in one place.

# First, explore how to fill in the missing ages. Several strategies pinpoint to replace the missing values with the mean or median of the whole distribution, which in our eyes doesn't seem a good choice. Instead, let's look into the correlation of age with the other variables.

# In[ ]:


df_train.corr(method='pearson')['Age'].abs()


# We see that the strongest correlation of the variable age is with the variable Pclass (passenger class). Therefore, it is appropriate to use this information in order to sample the missing ages according to the pclass. We can either take the median of each Pclass group or sample a random value from that group. We are going to try both and see which one yields better results. Sampling from a distribution, however, seems like the more viable option, since we have a lot of missing values to replace and setting all of them to the same value would skew the distribution massively. For the other missing variables ( Fare, Embarked), the analysis we performed above leads us to believe that sampling according to the passengers' class is a viable method.

# In[ ]:


def df_fill(datasets, mode):
    assert mode =='median' or mode =='sampling'
    datasets_cp =[]
    np.random.seed(2)
    varnames = ['Age','Fare']
    for d in datasets:
        df = d.copy(deep = True)
        for var in varnames:
            idx = df[var].isnull()
            if idx.sum()>0:
                if mode =='median':
                    medians = df.groupby('Pclass')[var].median()
                    for i,v in enumerate(idx):
                        if v:
                            df[var][i] = medians[df['Pclass'][i]]
                else:
                    g = df[idx==False].groupby('Pclass')[var]
                    for i,v in enumerate(idx):
                        if v:
                            df[var][i] = np.random.choice((g.get_group(df['Pclass'][i])).values.flatten())
    #Embarked                 
        idx = df['Embarked'].isnull()
        g = df[idx==False].groupby('Pclass')['Embarked']
        for i,v in enumerate(idx):
            if v:
                df['Embarked'][i] = np.random.choice((g.get_group(df['Pclass'][i])).values.flatten())                   
    #Cabin
        df['Cabin'][df['Cabin'].isnull()]='Missing'
        df['Cabin'] = df['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]
        datasets_cp.append(df)
    return datasets_cp

data_clean = df_fill(data_cleaner,'median')


# In[ ]:


def prepare_data(datasets):
        datasets_cp = []
        for d in datasets:
            df = d.copy(deep = True)
            df['Family onboard'] = df['Parch'] + df['SibSp']
            df['Title'] = df['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()
            new_titles_dict = new_titles(df)
            df['Title'] = df['Title'].replace(new_titles_dict)
            df.drop(columns = ['PassengerId','Name','Ticket'],axis = 1, inplace = True)
            datasets_cp.append(df)
        return datasets_cp
        


# In[ ]:


train,test =prepare_data(df_fill(data_cleaner,mode = 'sampling'))  
print("Training data")
print(train.isnull().sum())
print("Test data")
print(test.isnull().sum())


# There are no more missing values in our data. We are now ready to create a model and start training!

# # 5. Train model

# We would like to see what performance a neural network can achieve. Neural networks are usually preferable when the data is high dimensional and the training set size is big, which is not the case here. However, we would like to determine whether a respectable result can be achieved with the feature engineering that was performed. To train the model, we will be using the fast.ai library which provides some very neat features to tune the hyperparameters of the model. To perform training, we need to define which variables are continuous and which are categorical and will be trained using embeddings. The target variable is 'Survived'. 20% of the training set has been chosen for validation. Before training, the data is also normalized to minimize the impact of outliers on the training and speeding-up the network.

# In[ ]:


cont_names = ['Fare','Age','Pclass','SibSp','Parch','Family onboard']
cat_names = ['Sex','Cabin','Embarked']
procs = [Categorify,Normalize]
dep_var = 'Survived'

data_test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)

data = (TabularList.from_df(train, path='/kaggle/working', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_rand_pct(0.2)
                           .label_from_df(cols = dep_var)
                           .add_test(data_test, label=0)
                           .databunch()
       )


# The layer architecture is a hyperparameter that the user can freely choose. After experimenting with it, a 3-layer network  with declining size has been chosen. Every layer has batch normalization and weight decay on,while the dropout probability is 10%. 

# In[ ]:


learn = tabular_learner(data, 
                        layers=[500,200,100],
                        metrics=accuracy,
                        emb_drop=0.1,
                       )

learn.model


# Training will be conducted using acyclical learning rates. To begin training we will fit 2 cycles, meaning 2 full epochs through the data with a learning rate of 0.025 . Then, we will call the learning rate finder that plots the training loss for differnt learning rates in order to pick the best value for further training.

# In[ ]:


torch.device('cuda')
learn.fit_one_cycle(2, 2.5e-2)
learn.save('stage1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Now we want to pick a learning rate where the loss decay is the steepest. Looking at the plot, we train 3 more cycles at a learning rate of 4e-1. And call the learning rate finder again.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(4e-1))
learn.save('stage2')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# We will now train a little bit more at the point of steepest loss decline and repeat the same procedure, until the training and validation loss stabilize.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-2))
learn.save('stage3')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(5e-3))
learn.save('stage4')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(9e-4))
learn.save('stage5')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(5e-5))
learn.save('stage6')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


# learn.load('stage6')
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':labels})


# In[ ]:


submission.to_csv('submission-fastai.csv', index=False)


# # 6. Veridct
# 
# While during some runs of the model we were able to achieve ~87% accuracy on the validation set, running the model multiple times yields varying results. It appears that the obstacles of the small dataset and the missing values cannot be overcome. A different ML algorithm like Decision trees or SVM could achieve a much higher accuracy with the same feature engineering.
