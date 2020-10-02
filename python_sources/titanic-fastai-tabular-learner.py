#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.tabular import * 


# In[ ]:


# Define the path
path = Path('/kaggle/input/titanic')
path.ls()


# In[ ]:


# Import the datasets
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')


# In[ ]:


# Check the length of the dataset
print(train_df.shape)
print(test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# * Now let's explore the training set a little bit

# In[ ]:


print(train_df.columns)


# In[ ]:


train_df.describe()


# In[ ]:


# We can see that some age value are lost, let's check for any null value
train_df.isnull().sum()


# In[ ]:


# Let's check for the test set as well
test_df.isnull().sum()


# * We can see that there are 4 columns that have missing values: **Age, Fare, Cabin, Embarked**

# In[ ]:


# Fill missing "Cabin" values with "N" as there are too many missing values
def process_cabin(df):
    df['Cabin'].fillna('N', inplace=True)


# In[ ]:


# Fill missing "Embarked" values of dataset set with the most frequent value - mode
def process_embarked(df):
    df['Embarked'].fillna(df['Embarked'].mode().iloc[0], inplace=True)


# In[ ]:


# Fill the missing "Age" values with median
def process_age(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)


# In[ ]:


# Fill the missing "Fare" values with median
def process_fare(df):
    df['Fare'].fillna(df['Fare'].median(), inplace=True)


# * Now let's apply our knowledge to have some feature engineering 

# In[ ]:


# Extract some information about the name & tile
def process_title(df):
    title_dict = {
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
    df['Title'] = df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    df['Title'] = df['Title'].map(title_dict)


# In[ ]:


def process_df(df):
    # Can add more feature engineering if you want
    func_list = [process_title, process_age, process_fare, process_embarked, process_cabin]
    for func in func_list:
        func(df)


# In[ ]:


# Apply the feature engineering on both the training set and the testing set
process_df(train_df)
process_df(test_df)


# In[ ]:


# Let's check the training set again
print(train_df.isnull().sum())
train_df.head()


# In[ ]:


# Preprocessing 
# Actually, I didn't really need to manually process the missing data if using "FillMissing"
# Let's try to remove the manual process later
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


# Split our variables into target, categorical and continuous variables
dep_var = 'Survived'

# There were too many missing "Cabin" values, so we will ignore that
# The "Name" column has already been replaced by the "Title" column
cat_names = train_df.drop(['Cabin', 'Name'], axis=1).select_dtypes(exclude='number').columns.tolist()

cont_names = train_df.drop('Survived', axis=1).select_dtypes(include='number').columns.tolist()

print(cat_names)
print(cont_names)


# In[ ]:


test = TabularList.from_df(df=test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)


# In[ ]:


np.random.seed(42)
data = (TabularList.from_df(df=train_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
                   .split_by_rand_pct()
                   .label_from_df(cols=dep_var)
                   .add_test(test)
                   .databunch())


# In[ ]:


data.show_batch(10)


# * Now we can start building our model using FastAI Tabular Learner

# In[ ]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


learn.model_dir = '/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(10, min_grad_lr)


# In[ ]:


learn.recorder.plot_losses()


# * Judging from the learning curve, our model seems to **overfit** a little bit!
# * In the end, our model was able to reach around **84%** acccuracy!

# In[ ]:


# Getting prediction
preds, targets = learn.get_preds(DatasetType.Test)
labels = [p.argmax().item() for p in preds]

# Create "submission.csv" file
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': labels})
submission.to_csv('submission.csv', index=False)
submission.head()

