from collections import deque

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

import pandas as pd
import numpy as np
train_df = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

test_df = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")

train_df2 = train_df[['id']]
val_train_df2 = train_df2.values
val_train_df2 = train_df2
val_train_df2 = np.subtract(val_train_df2, 1)
agent1 = np.remainder(val_train_df2, 7).values

train_df["Agent"] = agent1

test_df2 = test_df[['id']]
df2_val_test = test_df2.values
df2_val_test = test_df2
df2_val_test = np.subtract(df2_val_test, 1)
agent1_test = np.remainder(df2_val_test, 7).values

test_df["Agent"] = agent1_test

train_df = train_df.drop(columns=['id', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

test_df = test_df.drop(columns=['id', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

part_remove = []
seven_c = []

for col in train_df:
    count = len(train_df[col].unique())
    if count == 1:
      print(col + ": " + str(count))
      part_remove.append(col)
    if count == 7:
      seven_c.append(col)

train_df1_c = train_df.copy()
train_df1_c = train_df1_c.drop(columns = part_remove)
df1_test = test_df.copy()
df1_test = df1_test.drop(columns = part_remove)

train_df2 = train_df1_c.copy()

test_df2 = df1_test.copy()

agent1_df = train_df2[train_df2['Agent'] == 1]

for item in agent1_df.columns:
  y = agent1_df[item].values
  x = agent1_df['time']

labels_y = train_df2['label'].values
agent_x = train_df2[['Agent']].values
time_x = train_df2[['time']].values
df3 = train_df2.copy()
df3 = df3.drop(columns=['Agent', 'time', 'label'])

df3_test = test_df2.copy()
df3_test = df3_test.drop(columns=['Agent', 'time'])

scalar = StandardScaler() 
scalar.fit(df3) 
scaled_data = scalar.transform(df3) 
pca = PCA().fit(scaled_data)

scaled_data_test = scalar.transform(df3_test)

pca = PCA(n_components = 40) 
pca.fit(scaled_data) 
x_pca = pca.transform(scaled_data) 

x_pca_test = pca.transform(scaled_data_test)

df4 = pd.DataFrame(x_pca)

df4_test = pd.DataFrame(x_pca_test)

df4["Agent"] = train_df1_c["Agent"]
df4["label"] = train_df1_c['label']

df4_test['Agent'] = df1_test['Agent']

from sklearn.tree import DecisionTreeRegressor
data_output = []
pred_rmse = []
window_size = 6
final_data = []
labels_final = []
for agent_number in range(7):
  print(agent_number)
  df_agent = df4[df4['Agent'] == agent_number]
  df_agent_label = df_agent[['label']].values
  df_agent = df_agent.drop(columns=['Agent', 'label'])
  #defining deque
  li = deque()
  for i in range(window_size):
    row = df_agent.iloc[i].tolist()
    li.extend(row)
  #making new data format
  data_new = []
  labels = []
  for i in range(window_size, df_agent.shape[0]):
    row_new = []
    values = df_agent.iloc[i].tolist()
    row_new.extend(li)
    row_new.extend(values)
    row_new.append(agent_number)
    labels.append(df_agent_label[i][0])
    for j in range(40):
      li.popleft();
    li.extend(values)
    data_new.append(row_new)
    #print(df_agent_label[i])
    #data_new.extend(df_agent_label[i])
  labels_final.extend(labels)

  #print(len(data_new[0]))
  final_data.extend(data_new)


df_age1 = df4[df4['Agent'] == agent_number]

final_data_test = []
labels_final_test = []
for agent_number in range(7):
  print(agent_number)
  df_agent = df4_test[df4_test['Agent'] == agent_number]
  df_age1 = df4[df4['Agent'] == agent_number]
  df_age1 = df_age1.drop(columns=['Agent', 'label'])
  leng = df_age1.shape[0]
  #print(leng-window_size)
  #print(leng)
  #df_agent_label = df_agent[['label']].values
  df_agent = df_agent.drop(columns=['Agent'])
  #defining deque
  li = deque()

  for i in range(leng - window_size, leng):
    row = df_age1.iloc[i].tolist()
    #print(row)
    li.extend(row)
  #making new data format
  #print("a")
  #print(len(li))
  data_new = []
  labels = []
  for i in range(0, df_agent.shape[0]):
    row_new = []
    values = df_agent.iloc[i].tolist()
    row_new.extend(li)
    row_new.extend(values)
    row_new.append(agent_number)
    for j in range(40):
      li.popleft();
    li.extend(values)
    data_new.append(row_new)
  final_data_test.extend(data_new)

x_val = []
for i in range(3, 9):
  x_val.append(i)

X = np.array(final_data)
Y = np.array(labels_final)
train_pct_index = int(0.8*len(X))

x_train, x_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]

X = np.array(final_data)
Y = np.array(labels_final)
x_train = X
y_train = Y

from sklearn.ensemble import ExtraTreesRegressor
rf = ExtraTreesRegressor(n_estimators=100, verbose=1, n_jobs=-1, warm_start=True, max_leaf_nodes=10000)
rf.fit(x_train, y_train)

final_data_test_reshuffled = []

for i in range(5756):
  for j in range(7):
    #print(j*5756 + i)
    final_data_test_reshuffled.append(final_data_test[j*5756 + i])

x_test = final_data_test_reshuffled
y_pred_test = rf.predict(x_test)



ouput = pd.DataFrame({'label':y_pred_test})
ouput['id'] = ""
cols = ['id', 'label']
ouput = ouput[cols]

for index, row in ouput.iterrows():
  ouput.at[index, 'id'] = index + 1

ouput.to_csv("ouput.csv", index=False)