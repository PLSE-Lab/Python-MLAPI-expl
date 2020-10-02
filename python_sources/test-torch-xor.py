import torch
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = torch.nn.Linear(2, 2)
        self.layer_2 = torch.nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.layer_2(x)
        x = torch.nn.functional.sigmoid(x)
        
        return x
    
class TrainingParams:
    data_XOR_x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_XOR_y = torch.FloatTensor([[0], [1], [1], [0]])
    max_epoch  = 10000
    lr         = 1e-1
    print_step = 1000
    
def loss_func(output, target):
    return torch.mean((target * torch.log(output) + (1.0 - target) * torch.log(1.0 - output)) * -1)
    
def main():
    net = Net()
    opt = torch.optim.SGD(params=net.parameters(), lr=TrainingParams.lr)#, momentum=0.8)
    #loss_func = torch.nn.CrossEntropyLoss()
    loss_list = list()
    
    for epoch in range(TrainingParams.max_epoch):
        output = net(TrainingParams.data_XOR_x)
        loss = loss_func(output, TrainingParams.data_XOR_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % TrainingParams.print_step == 0:
            print('epoch {}: loss = {}, pred = {}'.format(epoch, loss.data.numpy(), output.data.numpy().reshape(-1)))
        loss_list.append(loss.data.numpy())
    print('Final: loss = {}, pred = {}'.format(loss.data.numpy(), output.data.numpy().reshape(-1)))
    plt.plot(loss_list)
    plt.show()
    
if __name__ == '__main__':
    main()