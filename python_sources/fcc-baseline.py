import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions.categorical as categorical
import torch.distributions.categorical as one_hot_categorical
import torch.distributions.bernoulli as bernoulli

import child_model as CM
import utils
import time

def sample_operation(P_op, num_samples=1):
    return categorical.Categorical(P_op).sample([num_samples])

def mapped_sampler(P_net, num_samples, map_ops):
    chops = categorical.Categorical(P_net).sample([num_samples])
    for i, v in enumerate(map_ops):
        chops[chops == i] = v
    return chops

def map_to_ops(tens, map_ops):
    for i, v in enumerate(map_ops):
        tens[tens == i] = v
    return tens

def sample_model(Pop, map_ops):
    num_nodes = Pop.size(0)
    model_skips = torch.zeros(num_nodes*(num_nodes - 1)//2)
    model_ops = sample_operation(Pop).squeeze()
    for i, v in enumerate(map_ops):
        model_ops[model_ops == i] = v
    cm = CM.ChildModel(model_ops, model_skips)
    return cm
    

# returns the accuracies of a given list of torch child models on one random test data batch
def test_one_batch(models, test_set):
    accuracies = []
    test_size = len(test_set)
    randbatch_ind = np.random.randint(test_size)
    bdata, btarget = test_set[randbatch_ind]
    for model in models:
        model.eval()
        correct = 0
        with torch.no_grad():
            #t1 = time.time()
            output = model(bdata)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(btarget.view_as(pred)).sum().item()
            
            accuracy = correct/len(bdata)
            accuracies.append(accuracy)
            #print("testing time = {:.0f}s".format(time.time() - t1))
    return accuracies

# trains the given model for one pass/epoch through the given training data
def train1(model, train_set, optimizer, loss_func=F.nll_loss, log_interval=10, max_batches=None):
    
    batch_size = len(train_set[0][1])
    # only train on the first 'max_samples' images
    if max_batches is None:
        num_batches = len(train_set)
    else:
        # TODO choose batches randomly
        train_set = train_set[0:max_batches]
        num_batches = len(train_set)
    train_size = num_batches*batch_size
	
    model.train()
    t1 = time.time()
    for batch_idx, (data, target) in enumerate(train_set):
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0: 
            print('Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Time: {:.3f}'.format(batch_idx*batch_size, train_size,
                100*batch_idx/num_batches, loss.item(), time.time() - t1))
            t1 = time.time()

# initialization routine for the controller weights
def controller_init(m):
    if isinstance(m, nn.LSTMCell):
        for p in m.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

class FCController(nn.Module):
    def __init__(self, num_nodes, num_child_samples=100, learning_rate=0.001, gamma=0.9, input_amplitude=0.01, allowed_ops=[0, 4], layer_sizes=[100, 500, 10]):
        super(FCController, self).__init__()
        
        #initialize hyper parameters
        self.num_nodes = num_nodes
        self.allowed_ops = allowed_ops
        self.layer_sizes = layer_sizes
        self.num_child_samples = num_child_samples
        self.input_amplitude = input_amplitude
        
        # define layers
        layers = []
        for lid in range(len(self.layer_sizes)):
            if lid == len(self.layer_sizes) - 1:
                outsize = len(self.allowed_ops)*self.num_nodes
                layers.append(nn.Linear(self.layer_sizes[lid], outsize))
            else:    
                layers.append(nn.Linear(self.layer_sizes[lid], self.layer_sizes[lid + 1]))
                layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        # hyperparameters
        self.gamma = gamma # exponential baseline decay
        self.learning_rate = learning_rate

        self.timestep = 1
        
        self.apply(controller_init)
        self.to(utils.device)
 
    def backward(self, childmodel, Pop):
        for node_ind in range(Pop.size(0)):
            op = childmodel.ops[node_ind].item()
            opid = np.where(np.array(self.allowed_ops) == op)[0][0]
            prob = Pop[node_ind, opid]
            weight = 1/prob.clone().detach()
            prob.backward(weight, retain_graph=True)
        
    def update_step_naive(self, R, baseline=True, log=False):
        num_models = len(R)
        b = 0 # baseline to reduce variance
        if baseline:
            b = self.gamma #(1 - self.gamma)*r + self.gamma*b # update baseline
        with torch.no_grad():
            for pi, (name, p) in enumerate(self.named_parameters()):
                if log:
                    pbefore = p.clone()
                for n in range(num_models):
                    r = R[n]
                    p += self.learning_rate*(r - b)*p.grad/num_models
                if log:
                    print("mean relative update for {} = {}".format(name, ((pbefore - p)/p).mean()))

        self.timestep += 1
    
    def forward(self, uniform=True):
        if uniform:
            inp = nn.init.uniform_(torch.zeros(1, self.layer_sizes[0], device=utils.device), -self.input_amplitude, self.input_amplitude)
        else:
            inp = nn.init.normal_(torch.zeros(1, self.layer_sizes[0], device=utils.device), std=self.input_amplitude)
        
        out = self.fc(inp).view(self.num_nodes, -1)
        P_net = F.softmax(out, dim=1)
        # sample operation
        #model_ops = torch.zeros(self.num_nodes)
        #model_skips = torch.zeros(self.num_nodes*(self.num_nodes - 1)//2)
        #model_ops = sample_operation(P_net)
        #cm = CM.ChildModel(model_ops, model_skips)
        return P_net#, cm
    
    def step1(self, epoch, train_set, shared_parameters, optimizer=optim.SGD, opt_args={"lr": 0.01, "momentum": 0.8, "weight_decay": 1e-4, "nesterov": True},
              train_args={"loss_func": F.cross_entropy, "log_interval": 10, "max_batches": None}):
        ts = time.time()
        print("Starting step 1 of epoch {}".format(epoch))
        P_net = self.forward()
        
        cm = sample_model(P_net, self.allowed_ops)
        
        tcm = cm.to_torch_model(shared_parameters)
        opt = optimizer(tcm.parameters(), **opt_args)
        train1(tcm, train_set, opt, **train_args)
        print("End of step 1, took {:.0f}s".format(time.time() - ts))
        return tcm.get_shared_weights(shared_parameters)
    
    def step2(self, epoch, test_set, shared_parameters):
        print("Starting step 2 of epoch {}".format(epoch))
        ts = time.time()
        print("Generating {} child models ...".format(self.num_child_samples))

        tcms = []        
        for _ in range(self.num_child_samples):
            self.zero_grad()
            Pop = self.forward()
            cm = sample_model(Pop, self.allowed_ops)
            tcm = cm.to_torch_model(shared_parameters)
            tcms.append(tcm)
            self.backward(tcm.childmodel, Pop)
        
        print("Validating child models ...")
        acc = test_one_batch(tcms, test_set) # test child model performance
        
        print("Mean validation accuracy = {:.1f}%".format(np.mean(acc)*100))
        print("Updating controller weights ...")
        self.update_step_naive(acc) # update controller weights naively
        print("End of step 2 of epoch {} took {:.0f} seconds to complete".format(epoch, time.time() - ts))