import torch
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

max_epoch = 1000
lr = 1e-2
print_step = 100

class Net(torch.nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.layer_1 = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.layer_1(x)

def read_dataset():
    data = load_boston()
    features = np.array(data.data, dtype=np.float32)
    target = np.array(data.target, dtype=np.float32)

    return torch.from_numpy(features), torch.from_numpy(target)

def features_normalize(features):
    return (features - features.mean(0)) / features.std(0)

def main():
    features, target = read_dataset()
    data_count, n_features = features.shape
    print('Data count: {}, number of features: {}'.format(data_count, n_features))

    # normalize
    features = features_normalize(features)

    # start training
    net = Net(n_features=n_features)
    opt = torch.optim.SGD(params=net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    loss_list = list()

    for epoch in range(max_epoch):
        output = net(features)
        loss = loss_func(output, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_list.append(loss.data.numpy())
        if epoch % print_step == 0:
            print('epoch {}: loss = {}'.format(epoch, loss.data.numpy()))
    print('Final: loss = {}'.format(loss.data.numpy()))

    plt.plot(loss_list)
    plt.show()

if __name__ == '__main__':
    main()