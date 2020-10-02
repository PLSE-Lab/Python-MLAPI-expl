# Importing and Settings
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn

pd.set_option("display.width", None)

# Reading Data
df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")

# Data Analysis&Cleaning
df_all = pd.concat([df_train, df_test], sort=False)


def preprocess(One_hot_fea, Sca_fea, mode_fea, mean_fea, df):
    df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    enc = preprocessing.LabelEncoder()
    sca = preprocessing.MinMaxScaler()
    for col in mode_fea:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in mean_fea:
        df[col] = df[col].fillna(df[col].mean())
    for col in One_hot_fea:
        df[col] = enc.fit_transform(df_all[[col]])  # Todo: One hot coding problems
    for col in Sca_fea:
        df[[col]] = sca.fit_transform(df[[col]])
    return df


One_hot_features = df_all[["Sex", "Embarked", "Pclass"]]
Sca_features = df_all[["Fare", "Age", "SibSp", "Parch"]]
Nan_mode_features = df_all[["Embarked"]]
Nan_mean_features = df_all[["Age", "Fare"]]
df_all = preprocess(One_hot_features, Sca_features, Nan_mode_features, Nan_mean_features, df_all)

df_train = df_all[:df_train.shape[0]]
df_test = df_all[df_train.shape[0]: df_all.shape[0]]

X_train = np.array(df_train.drop(["Survived", "PassengerId"], axis=1))  # Todo: to .values ?
y_train = np.array(df_train["Survived"])
X_test = np.array(df_test.drop(["Survived", "PassengerId"], axis=1))  # Todo: to .values ?

X_train = torch.from_numpy(X_train).float().cuda()
y_train = torch.from_numpy(y_train).long().cuda()
X_test = torch.from_numpy(X_test).float().cuda()


# Model
class Net(nn.Module):

    def __init__(self, n_input, n1_hidden, n2_hidden, n3_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n1_hidden).cuda()
        self.fc2 = nn.Linear(n1_hidden, n2_hidden).cuda()
        self.fc3 = nn.Linear(n2_hidden, n3_hidden).cuda()
        self.fc4 = nn.Linear(n3_hidden, n_output).cuda()
        self.relu = nn.ReLU().cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


alpha = 0.01
n_epochs = 50000

model = Net(7, 300, 400, 300, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)  # Todo: parameter?
loss = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    pred = model(X_train)
    loss_value = loss(pred, y_train)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    loss_show = loss_value.cpu().detach().numpy()

    Acc = ((pred.softmax(1)[:, 1] > 0.5).float() == y_train).float().mean().cpu().detach().numpy()
    if epoch % 100 == 0:
        print("Epoch: ", epoch, "loss: ", loss_show, "Acc: ", Acc)


# Submission

prediction = (model(X_test).softmax(1)[:, 1] > 0.5).int().cpu().detach().numpy()
output = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": prediction})
output.to_csv("sub.csv", index=False)
