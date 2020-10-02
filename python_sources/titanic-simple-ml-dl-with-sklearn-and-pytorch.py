#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('markdown', '', '# **Titanic notebook** <br> \nTitanic competition in Kaggle *https://www.kaggle.com/c/titanic/*<br>\n## **Overview:**\n### *I- Descriptive analysis and Data visualization*\n### *II- Data preprocessing*\n### *III- Machine Learning algorithms with sklearn and xgboost*\n### *IV- Deep Learning Classification with Pytorch*')


# In[ ]:


### Imports
## Utils
import pandas as pd #For dataframes manipulations
import numpy as np 
from IPython.display import display
from collections import namedtuple

##Data visualizations
import seaborn as sns 
import matplotlib.pyplot as plt

#########################

##Machine learning imports
#Preprocessing
from sklearn.preprocessing import LabelEncoder # label encoder for our categorical data
from sklearn.model_selection import train_test_split # Data splitting into train and validation sets 
#Classifiers
from sklearn.linear_model import LogisticRegression # Logistic Regression 
from sklearn.ensemble import RandomForestClassifier # Random Forest 
from sklearn.tree import DecisionTreeClassifier # Decision tree
import xgboost # Xgboost  
#Accuracy Metric 
from sklearn.metrics import accuracy_score # To calculate accuracies

#########################

##Deep learning imports (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Import our datasets : training and testing
input_path = '/kaggle/input/titanic/'
train_set = pd.read_csv(input_path+'train.csv')
test_set = pd.read_csv(input_path+'test.csv')
dataset = [train_set, test_set]


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '## **Descriptive analysis and Data visualization**')


# In[ ]:


print("Columns of training dataset")
print(train_set.columns.values)
print("----------------------------------------------\n\n")
print("Training dataset info")
display(train_set.info())
print("----------------------------------------------\n\n")
print("training dataset description")
display(train_set.describe())
print("----------------------------------------------\n\n")
print("training dataset description of columns of type object")
display(train_set.describe(include=['object'])) #description of categorical columns


# In[ ]:


"""
the commented lines are another way to visualise the data which bigger plots 
"""
##Survived by Pclass
#plt.figure(figsize=(30,20))
#plt.subplot(2, 2, 1)
plt.figure(figsize=(25,10))
plt.subplot(2, 3, 1)
ax = sns.countplot(x="Pclass", hue="Survived", data=train_set).set_title('Ticket class')

##Survived by Sex
#plt.subplot(2, 2, 2)
plt.subplot(2, 3, 2)
sns.countplot(train_set['Sex'], hue="Survived", data=train_set).set_title('Sex')
#plt.show()

##Survived by SibSp
#plt.figure(figsize=(30,20))
#plt.subplot(2, 2, 1)
plt.subplot(2, 3, 3)
sns.countplot(train_set['SibSp'], hue="Survived", data=train_set).set_title('Number of siblings / spouses aboard the Titanic')

##Survived by Parch
#plt.subplot(2, 2, 2)
plt.subplot(2, 3, 4)
sns.countplot(train_set['Parch'], hue="Survived", data=train_set).set_title('Number of parents / children aboard the Titanic')
#plt.show()

##Survived or not
#plt.figure(figsize=(30,20))
#plt.subplot(2, 2, 1)
plt.subplot(2, 3, 5)
sns.countplot(train_set['Survived']).set_title('Survival')

##Survived by Embarked
#plt.subplot(2, 2, 2)
plt.subplot(2, 3, 6)
sns.countplot(train_set['Embarked'], hue="Survived", data=train_set).set_title('Port of Embarkation')
plt.show()


# In[ ]:


g = sns.FacetGrid(train_set,height=4, col="Sex", row="Survived", margin_titles=True, hue = "Survived")
g = g.map(plt.hist, "Age", edgecolor = 'white', bins = 8)
g.fig.suptitle("Survived by Sex and Age", size = 30)
plt.subplots_adjust(top=0.85)


# In[ ]:


g = sns.FacetGrid(train_set,height=4, col="Pclass", row="Survived", margin_titles=True, hue = "Survived")
g = g.map(plt.hist, "Age", edgecolor = 'white', bins = 8)
g.fig.suptitle("Survived by Ticket class and Age", size = 35)
plt.subplots_adjust(top=0.85)


# In[ ]:


#missing values
def missing_values_df(df):
    """
    Check missing values in our dataset
    :input df: (Dataframe) input dataframe 
    :output output_df: (Dataframe) dataframe of missing values and there percentage 
    """
    missing_values = df.isnull().sum().sort_values(ascending = False)
    missing_values = missing_values[missing_values>0]
    ratio = missing_values/len(df)*100
    output_df= pd.concat([missing_values, ratio], axis=1, keys=['Total missing values', 'Percentage'])
    return output_df
print('Missing values in the columns of training dataset with percentage')
display(missing_values_df(train_set))
print('\n------------------------------------------------\n')
print('Missing values in the columns of test dataset with percentage')
display(missing_values_df(test_set))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '## **Data Preprocessing**\n1) fill the *NaN* values with mean (*Age feature*) our more frequent values (*Embarked feature*) <br>\n2) Add *titles* of passengers from names then delete names <br>\n3) Encode categorical features into integers <br>\n4) make 4 bins of age to categorize it <br>\n5) Normalize our datasets(training_set and test_set)<br>\n6) Split our training set into training and validation sets (80%/20%) <br>')


# In[ ]:


for i in range(len(dataset)):
    freq_port = dataset[i]['Embarked'].dropna().mode()[0]
    dataset[i]['Embarked'] = dataset[i]['Embarked'].fillna(freq_port)
    dataset[i] = dataset[i].fillna(dataset[i].mean())


# In[ ]:


print("Titels of passengers by sex")
dataset[0]['Title'] = dataset[0].Name.str.extract(' ([A-Za-z]+)\.', expand=False)
display(pd.crosstab(dataset[0]['Sex'], dataset[0]['Title']))


# In[ ]:


for i, data in enumerate(dataset):
    dataset[i]['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset[i]['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
                                                 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset[i]['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset[i]['Title'] = data['Title'].replace('Mme', 'Mrs')
    print(dataset[i]['Title'].unique())


# In[ ]:


encoder = LabelEncoder()
categoricalFeatures = dataset[0].select_dtypes(include=['object']).columns
for i, data in enumerate(dataset):
    data[categoricalFeatures]=data[categoricalFeatures].astype(str)
    encoded = data[categoricalFeatures].apply(encoder.fit_transform)
    for j in categoricalFeatures:
        dataset[i][j]=encoded[j]
dataset[0].head()


# In[ ]:


bins = [0,18,60,80]
labels = [1,2,3]
for i, data in enumerate(dataset):
    dataset[i] = dataset[i].drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    dataset[i]['Age']=pd.cut(dataset[i]['Age'],bins=bins ,labels=labels)
    dataset[i]['Age']=dataset[i]['Age'].astype('int64')
print('training dataset:')
display(dataset[0].head())
print('testing dataset:')
display(dataset[1].head())


# In[ ]:


#Normalizing our inout data in training and testing dataset
X=dataset[0].iloc[:, 1:]
Y=dataset[0].iloc[:, 0]
x_test=dataset[1].iloc[:, 0:]
normalized_data = X
normalized_data=normalized_data.append(x_test)
normalized_x_train = normalized_data.values
normalized_x_train /= np.max(np.abs(normalized_x_train),axis=0)

X = pd.DataFrame(normalized_x_train[:891,:], 
                      columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title'])
print(X.head())
print(len(X))
x_test = pd.DataFrame(normalized_x_train[891:,:], 
                      columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title'])
display(x_test.head())
print(len(x_test))
display(X.columns)


# In[ ]:


# Split our training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.20)
print("Training set shape: "+str(X_train.shape))
print("Validation set shape: "+str(X_val.shape))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '## Machine learning algorithms for classification\n1) Logistic Regression <br>\n2) Decision Tree <br>\n3) Random Forest <br>\n4) XGBoost <br>')


# In[ ]:


accuracies_list = list()
accuracies = namedtuple('accuracies',('Model', 'accuracy'))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '#### Logistic Regression')


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_val)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print('accuracy: {}'.format(acc_log))
accuracies_list.append(accuracies('Logistic Regression', acc_log))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '#### Decision Tree')


# In[ ]:


decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pred = decisiontree.predict(X_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print('accuracy: {}'.format(acc_decisiontree))
accuracies_list.append(accuracies('Decision Tree', acc_decisiontree))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '#### Random Forest')


# In[ ]:


clf = RandomForestClassifier(max_depth=10, max_leaf_nodes =20,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_val)
acc_random_forest = round(accuracy_score(y_pred, y_val) * 100, 2)
print('accuracy: {}'.format(acc_random_forest))
accuracies_list.append(accuracies('Random Forest', acc_random_forest))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '#### XGBoost')


# In[ ]:


xgb = xgboost.XGBClassifier(random_state=5,learning_rate=0.01)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)
acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)
print('accuracy: {}'.format(acc_xgb))
accuracies_list.append(accuracies('XGBoost', acc_xgb))


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '## **Deep learning for binary classification with pytorch**\n1) Declare consts <br>\n2) Training Set && Testing Set preparation for pytorch <br>\n3) Define our DL model class <br>\n4) Instantiate our model, loss and optimizer <br>\n5) Define fit function <br>\n6) Training process <br>\n7) Define Predict Function <br>\n8) Preprare for submission <br>')


# In[ ]:


#Constants
BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 800
INPUT_NODES = 8


# In[ ]:


## Training Set && Testing Set preparation for pytorch 

#Create Tensors from our dataframes
X_train_torch = torch.from_numpy(X_train.values).type(torch.FloatTensor) # Train X
y_train_torch = torch.from_numpy(y_train.values).type(torch.LongTensor) # Train Y
X_val_torch = torch.from_numpy(X_val.values).type(torch.FloatTensor) # Validate X
y_val_torch = torch.from_numpy(y_val.values).type(torch.LongTensor) # Validate Y
x_test_torch = torch.from_numpy(x_test.values).type(torch.FloatTensor) # Test X

#Create Tensordatasets for pytorch
train = torch.utils.data.TensorDataset(X_train_torch,y_train_torch) # Train
val = torch.utils.data.TensorDataset(X_val_torch, y_val_torch) # Validate
test = torch.utils.data.TensorDataset(x_test_torch) # Test

#Create data loaders
data_loader = torch.utils.data.DataLoader(train) # Train
val_loader = torch.utils.data.DataLoader(val) # Validate
test_loader = torch.utils.data.DataLoader(test) # Test


# In[ ]:


get_ipython().run_cell_magic('markdown', '', '### **Define model Class**\nInput Features --> Fully Connected layer(512 nodes) --> Dropout(50%) --> Fully Connected layer(256 nodes) --> Dropout(50%) --> Fully Connected layer(128 nodes) --> Dropout(50%) --> Fully Connected layer(1 node)')


# In[ ]:


class Titanic_NN(nn.Module):
    """
    Class of our Neural network for the titanic dataset
    3 fully connected hidden Layers + output Layer with dropout of 0.5 probability each.
    We have 8 input nodes and 1 output node (1 if >0.5 else 0)
    NB: We didn't use batch normalization because we didn't use batches.
    """
    def __init__(self, INPUT_NODES):
        super(Titanic_NN, self).__init__()
        self.fc1 = nn.Linear(INPUT_NODES,512)
        #self.batcTitanic_NNh_norm1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(512,256)
        #self.batch_norm2 = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,1)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        
        return F.sigmoid(x)


# In[ ]:


#Model - Loss - Optimizer
# Instantiate our class Titanic_NN
model = Titanic_NN(INPUT_NODES)
try: #if we have a trained model, we load it
    model.load_state_dict(torch.load(input_path+'titanic_model_4layers'))
except:
    pass

#Binary Classification Entropy Loss
error = nn.BCELoss()

#SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
print(model)


# In[ ]:


def fit(model, data, phase='training', batch_size = 1, is_cuda=False, input_dim = 8):
    """
    Method to train the model
    :input model:(Titanic_NN) Our model (in this case is the Titanic_NN model class)
    :input data:(dataLoader) our training / validation data to train the model on and validate
    :input phase: (String) 'training' to train model 'validation' to make predictions and validate the model
    :input batch_size: (int) batch size we feed to our neural network
    :input is_cuda: (Bool) wheather to use cuda or not (GPU) (still not implemented)
    :input input_dim: (int) number of input nodes
    :return loss: (float) loss value for one epoch
    :return accuracy: (float) accuracy value for one epoch
    """
    if phase == 'training':
        model.train()
    elif phase == 'validation':
        model.eval()
    loss_values = 0.0
    correct_values = 0
    for _, (features, label) in enumerate(data):
        if is_cuda:
            features, label = features.cuda(), label.cuda()
        features, label = Variable(features.view(batch_size, 1, input_dim)), Variable(label.float().view(-1, 1))
        if phase == 'training':
            optimizer.zero_grad() 
        output = model(features) #make one forward pass
        loss = error(output, label) #calculate loss for one forward pass
        loss_values += loss.data
        if output[0] > 0.5:
            predictions = torch.Tensor([1])
        else:
            predictions = torch.Tensor([0])
        correct_values += predictions.eq(label.data.view_as(predictions)).cpu().sum()
        if phase == 'training':
            loss.backward() 
            optimizer.step()  # make one gradient step
    loss = loss_values / len(data.dataset) # calculate loss mean for the epoch
    accuracy = 100. * correct_values / len(data.dataset) #calculate accuracy
    #if phase == 'validation':
        #print(f'\n{phase} loss is {loss:{5}.{2}} and {phase} accuracy is \
              #{accuracy:{10}.{4}}\n=============================================') #we can choose to print both phases
    return loss, accuracy


# In[ ]:


#Training process
#validation / traing lists for accuracy and loss values during training
train_loss_list, val_loss_list = [], []
train_accuracy_list, val_accuracy_list = [], []

for epoch in range(EPOCHS):
    #print(epoch)
    #Training
    train_epoch_loss, train_epoch_accuracy = fit(model, data_loader, batch_size=BATCH_SIZE, input_dim=INPUT_NODES) 
    #Validating
    val_epoch_loss, val_epoch_accuracy = fit(model, val_loader, phase='validation', batch_size=BATCH_SIZE, input_dim=INPUT_NODES)
    if epoch % 50 == 0:
        torch.save(model.state_dict(), 'titanic_model_4layers')
    train_loss_list.append(train_epoch_loss)
    train_accuracy_list.append(train_epoch_accuracy)
    val_loss_list.append(val_epoch_loss)
    val_accuracy_list.append(val_epoch_accuracy)


# In[ ]:


accuracies_list.append(accuracies('Neural Network __Validation_Set__', val_accuracy_list[-1]))


# In[ ]:


def predict(model, data):
    """
    Make predictions on the test dataset
    :input model: (Titanic_NN) our model DL to make predictions with
    :input data: (torch.utils.data.DataLoader) our test data loader
    :output test_predictions: (list()) list of predictions
    """
    model.eval()
    test_predictions = list()
    for _, (feature,) in enumerate(data):
        feature = Variable(feature.view(1, 1, INPUT_NODES))
        output = model(feature)
        if output[0] > 0.5:
            prediction = 1
        else:
            prediction = 0
        test_predictions.append(prediction)
    return test_predictions


# In[ ]:


#create our dataframe of predictions and indices from test_set
#our_predictions = predict(model, test_loader)
pred_df = pd.DataFrame(np.c_[np.arange(892, len(test_set)+892)[:,None], predict(model, test_loader)], 
                      columns=['PassengerId', 'Survived'])
#save results to a csv file
pred_df.to_csv('titanic_submission.csv', index=False)


# In[ ]:


print(accuracies_list)

