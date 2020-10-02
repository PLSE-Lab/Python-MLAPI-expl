#!/usr/bin/env python
# coding: utf-8

# # Fastai Tabular V1 Titanic Classification
# 
# Use fastai tabular for the titanic dataset with limited preprocessing steps

# In[ ]:


from fastai.tabular import *

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


path = "/kaggle/input/titanic/"
base_path="../output"
BATCH_SIZE=128


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#Missing values in training set
print(train.isnull().sum())
print(train.shape)
#Missing values in test set
print(test.isnull().sum())
print(test.shape)


# In[ ]:


train.describe(include='all')


# There is a null in Fare, lets remove that

# In[ ]:


test = test.fillna({'Fare':0})


# # EDA and Pre Proccessing

# In[ ]:


train['isChild'] = train['Age'].apply(lambda x: True if x <=16 else False)
train['isOld'] = train['Age'].apply(lambda x: True if x >=50 else False)

test['isChild'] = test['Age'].apply(lambda x: True if x <=16 else False)
test['isOld'] = test['Age'].apply(lambda x: True if x >=50 else False)
test.head()


# In[ ]:


train[['Last','First']] = train['Name'].str.split(",",expand=True)
train[['Title','First']] = train['First'].str.split(".",1,expand=True)

test[['Last','First']] = test['Name'].str.split(",",expand=True)
test[['Title','First']] = test['First'].str.split(".",1,expand=True)

train.head(2)


# # Data Bunch

# In[ ]:


dep_var = 'Survived'
cat_names = ['Sex', 'Pclass', 'Embarked','isChild','isOld', 'Last', 'First', 'Title']
cont_names = ['Age', 'Parch','Fare','SibSp']
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


data_test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names)


# In[ ]:


data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,200)))
                           .label_from_df(cols=dep_var)
                           .add_test(data_test, label=0)
                           .databunch())
data.show_batch(rows=5)


# In[ ]:


np.random.seed(42)
learn = tabular_learner(data, layers=[200,100], metrics=[accuracy])


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5, 3e-2)


# In[ ]:


learn.save('fit_head')


# In[ ]:


learn.load('fit_head');


# In[ ]:


learn.unfreeze()
#learn.lr_find()
#learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(3e-3))


# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)


# In[ ]:


test_id = test['PassengerId']


# In[ ]:


# create submission file to submit in Kaggle competition
submission = pd.DataFrame({'PassengerId': test_id, 'Survived': labels})
submission.to_csv('submission.csv', index=False)

