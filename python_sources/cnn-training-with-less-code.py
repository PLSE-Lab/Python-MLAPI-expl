#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install neural-pipeline==0.1.0')


# In[ ]:


from IPython.display import HTML
from neural_pipeline import DataProducer, AbstractDataset, TrainConfig, TrainStage,    ValidationStage, Trainer, FileStructManager, Predictor

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


def read_data(path: str, with_labels: bool = True) -> []:
    content  = np.genfromtxt(path, delimiter=',')[1:]
    data = content.astype(np.float32)
    
    if not with_labels:
        return data.reshape((data.shape[0], 28, 28))
    else:
        data = data[:, 1:]
        data = data.reshape((data.shape[0], 28, 28))
    
    labels = content[:, 0].astype(np.long)
    return data, labels


# In[ ]:


data, labels = read_data('../input/train.csv')
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

class MNISTDataset(AbstractDataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        self._data = data
        self._labels = labels

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        data = transformations(np.expand_dims(self._data[item], axis=2))
        if self._labels is None:
            return data
        return {'data': data, 'target': self._labels[item]}

train_dataset = DataProducer([MNISTDataset(data_train, labels_train)], batch_size=8, num_workers=8)
validation_dataset = DataProducer([MNISTDataset(data_test, labels_test)], batch_size=8, num_workers=8)

model = Net()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from neural_pipeline.builtin.monitors.mpl import MPLMonitor
fsm = FileStructManager(base_dir='data', is_continue=False, exists_ok=True)

train_stage = TrainStage(train_dataset)
train_stage.enable_hard_negative_mining(0.1)
train_config = TrainConfig([train_stage, ValidationStage(validation_dataset)], torch.nn.NLLLoss(),
                           torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5))

trainer = Trainer(model, train_config, fsm, torch.device('cuda:0')).set_epoch_num(30)
mpl_monitor = MPLMonitor()
mpl_monitor.realtime(False)
trainer.monitor_hub.add_monitor(mpl_monitor)
trainer.enable_lr_decaying(coeff=0.5, patience=4, target_val_clbk=lambda: np.mean(train_stage.get_losses()))
trainer.train()


# In[ ]:


test_data = read_data('../input/test.csv', with_labels=False)


# In[ ]:


fsm = FileStructManager(base_dir='data', is_continue=True)
predictor = Predictor(model, fsm, device=torch.device('cuda:0'))

dataset = MNISTDataset(test_data)

with open('submission.csv', 'w') as out:
    out.write('ImageId,Label\n')
    for i, data in enumerate(dataset):
        res = predictor.predict({"data": data.unsqueeze(0)})
        res = res.cpu().numpy()
        out.write('{},{}\n'.format(i + 1, int(np.argmax(res))))

print('Predict cancel. Predicted', i + 1, 'images')

