#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
X_train, X_val, y_train, y_val = train_test_split(data_train.drop('label', axis=1), data_train['label'], test_size=0.2, random_state=42)
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255


# In[ ]:


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


# In[ ]:


image_batch = np.array_split(X_train, 500)
label_batch = np.array_split(y_train, 500)


# In[ ]:


for i in range(len(image_batch)):
    image_batch[i] = torch.from_numpy(image_batch[i].values).float()
    image_batch[i] = image_batch[i].cuda()
for i in range(len(label_batch)):
    label_batch[i] = torch.from_numpy(label_batch[i].values)
    label_batch[i] = label_batch[i].cuda()

X_val = torch.from_numpy(X_val.values).float()
X_val = X_val.cuda()
y_val = torch.from_numpy(y_val.values)
y_val = y_val.cuda()


# In[ ]:


model = Classifier()
model.cuda()
ps = torch.exp(model(image_batch[0]))
print(ps.shape)


# In[ ]:


model = Classifier()
model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

epochs = 100

train_losses, test_losses = [], []
for e in range(epochs):
    model.train()
    train_loss = 0
    for i in range(len(image_batch)):
        optimizer.zero_grad()
        output = model(image_batch[i])
        loss = criterion(output, label_batch[i])
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            model.eval()
            log_ps = model(X_val)
            test_loss += criterion(log_ps, y_val)
                
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y_val.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(train_loss/len(image_batch))
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(image_batch)),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}".format(accuracy))
        torch.save(model.state_dict(), '{}.pwf'.format(e+1))


# In[ ]:


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)


# In[ ]:


test = torch.from_numpy(X_test.values).float()
test = test.cuda()

model.load_state_dict(torch.load('79.pwf'))
with torch.no_grad():
    model.eval()
    output = model.forward(test)

ps = torch.exp(output)
ps = ps.cpu()
prediction_labels = ps.numpy().argmax(axis=1)
len(prediction_labels)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['Label'] = prediction_labels
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe
create_download_link(submission)


# In[ ]:




