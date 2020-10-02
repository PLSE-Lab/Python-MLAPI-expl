import child_model as CM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions.categorical as categorical
import torch.distributions.categorical as one_hot_categorical
import torch.distributions.bernoulli as bernoulli

import numpy as np

import utils
import time

# returns a ChildModelBatch containing num_samples models according to probabilities P_op and P_skip
def sampler(P_op, P_skip, num_samples):
    cat_op = categorical.Categorical(P_op)
    cat_sk = bernoulli.Bernoulli(P_skip)
    ops = cat_op.sample([num_samples])
    sks = cat_sk.sample([num_samples])
    #print(ops.shape)
    #print(sks.shape)
    return CM.ChildModelBatch(ops, sks)

def sample_operation(P_op, num_samples=1):
    return categorical.Categorical(P_op).sample([num_samples])

def sample_skip_connections(P_sk, num_samples=1):
    return bernoulli.Bernoulli(P_sk).sample([num_samples])

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
def train1(model, train_set, optimizer, loss_func, log_interval=10, max_batches=None):
    
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

class Controller(nn.Module):
    def __init__(self, num_nodes, num_child_samples=100, num_hidden=100, dim_w=64, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, gamma=0.9, kl_weight=0.8, sk_prob_target=0.4):
        super(Controller, self).__init__()
        
        #initialize hyper parameters
        self.num_nodes = num_nodes
        self.num_hidden = num_hidden
        self.num_ops = len(CM.OPERATION_NAMES)
        self.num_child_samples = num_child_samples
        
        # define LSTM cells and output layers
        self.op_cell = nn.LSTMCell(self.num_hidden, self.num_hidden)
        self.op_out = nn.Linear(self.num_hidden, self.num_ops)
        self.sk_cell = nn.LSTMCell(self.num_hidden, self.num_hidden)
        
        #self.op_cell.to(utils.device)
        #self.op_out.to(utils.device)
        #self.sk_cell.to(utils.device)
        
        #sample weights
        wprev_init = nn.init.normal_(torch.Tensor(dim_w, self.num_hidden), std=0.1)
        wcurr_init = nn.init.normal_(torch.Tensor(dim_w, self.num_hidden), std=0.1)
        v_init = nn.init.normal_(torch.Tensor(dim_w, 1), std=0.1)
        
        #wprev_init.to(utils.device)
        #wcurr_init.to(utils.device)
        #v_init.to(utils.device)
        
        self.W_prev = nn.Parameter(data=wprev_init)
        self.W_curr = nn.Parameter(data=wcurr_init)
        self.v = nn.Parameter(data=v_init)
        
        #self.W_prev.to(utils.device)
        #self.W_curr.to(utils.device)
        #self.v.to(utils.device)
        
        # hyperparameters
        self.gamma = gamma # exponential baseline decay
        self.kl_weight = kl_weight # weight of the KL divergence in the reward
        self.sk_prob_target = sk_prob_target # target probability of sampling a skip connection
        
        # ADAM parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        params = [p for p in self.parameters()]
        #print('Before params ', len([p for p in params]))

        # initialize optimizer
        self.timestep = 1
        
        self.moment1 = [
            torch.zeros(p.size(), requires_grad=False, device=utils.device)
            for p in params
        ]
        self.moment2 = [
            torch.zeros(p.size(), requires_grad=False, device=utils.device)
            for p in params
        ]
        # register momentum buffers to be able to load them via the module state_dict
        for i, t in enumerate(self.moment1):
            self.register_buffer("moment1_"+str(i), t)
            
        for i, t in enumerate(self.moment2):
            self.register_buffer("moment2_"+str(i), t)
            
        #self.moment2 = nn.ParameterList(parameters=self.moment2)
    
        self.apply(controller_init)
        self.to(utils.device)
    
    def backward(self, childmodel, Pop, Psk):
        #print(Psk)
        num_nodes = Pop.size(0)
        pgrad = [torch.zeros((*list(p.size())), 
                            device=utils.device) for p in self.parameters()]
            
        for node_ind in range(num_nodes):
            self.zero_grad()
            
            op = childmodel.ops[node_ind].int()
            prob = Pop[node_ind, op]
            v = torch.zeros(self.num_ops, device=utils.device); v[op] = 1
            Pop[node_ind].backward(v, retain_graph=True)
            
            with torch.no_grad():
                for pi, p in enumerate(self.parameters()):
                    if not p.grad is None:
                        pgrad[pi] += p.grad.clone().detach()/prob
            
            self.zero_grad()
            
            hood = node_ind*(node_ind - 1)//2
            skip = childmodel.skips[hood:hood + node_ind]
            prob = Psk[hood:hood + node_ind]
            prob.backward(skip, retain_graph=True)
            with torch.no_grad():
                for pi, p in enumerate(self.parameters()):
                    if not p.grad is None:
                        pgrad[pi] += p.grad.clone().detach()/torch.prod(prob)
            
            # calculate skip connection penalty
            norm_prob = self.sk_prob_target*torch.ones(Psk.size(), device=utils.device)
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            kl_skip_loss = kl_loss(torch.log(Psk), norm_prob)
            
        return pgrad, kl_skip_loss
    
    def update_step_adam(self, R, grads, kl_skips, baseline=True):
        num_models = len(R)
        
        dtheta = [torch.zeros(p.size(),device=utils.device) for p in self.parameters()]
        b = 0 # baseline to reduce variance
        for n in range(num_models):
            r = R[n] + self.kl_weight*kl_skips[n]
            grad = grads[n]
            for pi, p in enumerate(self.parameters()):
                dtheta[pi] += (r - b)*grad[pi]
            if baseline:
                b = r + self.gamma*b # update baseline
        
        # do ADAM update step
        with torch.no_grad():
            for pi, p in enumerate(self.parameters()):
                g = dtheta[pi]/num_models # Obacht!
                
                self.moment1[pi] = self.beta1*self.moment1[pi] + (1 - self.beta1)*g
                self.moment2[pi] = self.beta2*self.moment2[pi] + (1 - self.beta2)*g**2
                m1_hat = self.moment1[pi]/(1 - self.beta1**self.timestep)
                m2_hat = self.moment2[pi]/(1 - self.beta2**self.timestep)
                dp = self.learning_rate*m1_hat/(torch.sqrt(m2_hat) + self.epsilon)
                #pprev = p.clone()
                p += dp
                #print(p - pprev)
        
        self.timestep += 1
        
    def update_step_naive(self, R, grads, kl_skips, baseline=True):
        num_models = len(R)
        
        dtheta = [torch.zeros(p.size(),device=utils.device) for p in self.parameters()]
        b = 0 # baseline to reduce variance
        for n in range(num_models):
            r = R[n] + self.kl_weight*kl_skips[n]
            grad = grads[n]
            for pi, p in enumerate(self.parameters()):
                dtheta[pi] += (r - b)*grad[pi]
            if baseline:
                b = r + self.gamma*b # update baseline
        
        # do ADAM update step
        with torch.no_grad():
            for pi, p in enumerate(self.parameters()):
                g = dtheta[pi]/num_models # Obacht!
                p += self.learning_rate*g
        
        self.timestep += 1
                 
    def forward(self, g_emb):
        batch_size = g_emb.shape[0]
        h_prev = []
        h_prev.append(torch.zeros(batch_size, self.num_hidden, device=utils.device))
        c_prev = []
        c_prev.append(torch.zeros(batch_size, self.num_hidden, device=utils.device))
        P_skips = torch.zeros((int((self.num_nodes - 1)*self.num_nodes/2)), device=utils.device)
        P_ops = torch.zeros((self.num_nodes, self.num_ops), device=utils.device)
        
        sk_ind = 0
        for cell_i in range(2*self.num_nodes): # iterate over cells
            i = cell_i//2 # node index
            if cell_i % 2 == 0:
                # operation cell
                h_out, c_out = self.op_cell(g_emb, (h_prev[-1], c_prev[-1]))

                # calculate prob distribution operation at this node
                P_op = F.softmax(self.op_out(h_out), dim=1)
                #print(P_op)
                P_ops[i] = P_op
            else:
                # skip connection cell
                h_out, c_out = self.sk_cell(g_emb, (h_prev[-1], c_prev[-1]))
                
                # calculate prob distribution for skip connections to this node
                for j in range(i):
                    Pij = torch.sigmoid(self.v.t() @ torch.tanh(self.W_prev @ h_prev[j].t() + self.W_curr @ h_out.t()))
                    P_skips[sk_ind] = Pij
                    #print(Pij)
                    sk_ind += 1
            # store hidden and cell state
            h_prev.append(h_out)
            c_prev.append(c_out)
            
        return P_ops, P_skips
    
    def forward_with_feedback(self):
        g_emb = nn.init.uniform_(torch.zeros(1, self.num_hidden, device=utils.device), -0.1, 0.1)
        batch_size = g_emb.size(0)
        h_prev = []
        h_prev.append(torch.zeros(batch_size, self.num_hidden, device=utils.device))
        c_prev = []
        c_prev.append(torch.zeros(batch_size, self.num_hidden, device=utils.device))
        P_skips = torch.zeros((int((self.num_nodes - 1)*self.num_nodes/2)), device=utils.device)
        P_ops = torch.zeros((self.num_nodes, self.num_ops), device=utils.device)
        
        model_ops = torch.zeros(self.num_nodes)
        model_skips = torch.zeros(self.num_nodes*(self.num_nodes - 1)//2)
        
        sk_ind = 0
        for cell_i in range(2*self.num_nodes): # iterate over cells
            i = cell_i//2 # node index
            if cell_i % 2 == 0:
                # operation cell
                h_out, c_out = self.op_cell(g_emb, (h_prev[-1], c_prev[-1]))
                #print(h_out)
                # calculate prob distribution operation at this node
                op_out_out = self.op_out(h_out).squeeze(0)
                #print(op_out_out.squeeze(0))
                P_op = F.softmax(op_out_out, dim=0)
                #print(P_op)
                P_ops[i] = P_op
                
                # sample operation
                opid = sample_operation(P_op)
                model_ops[i] = opid
                
                # define next input
                g_emb = torch.zeros(g_emb.size(), device=utils.device)
                g_emb[0][opid] = 1
            else:
                # skip connection cell
                h_out, c_out = self.sk_cell(g_emb, (h_prev[-1], c_prev[-1]))
                
                # calculate prob distribution for skip connections to this node
                
                for j in range(i):
                    Pij = torch.sigmoid(self.v.t() @ torch.tanh(self.W_prev @ h_prev[j].t() + self.W_curr @ h_out.t()))
                    P_skips[sk_ind] = Pij
                    sk_ind += 1
                # get current skip probabilties
                hood = i*(i - 1)//2
                p_sk = P_skips[hood:hood + i]
                skips = sample_skip_connections(p_sk)
                model_skips[hood:hood + i] = skips
                
                # define next input
                g_emb = torch.zeros(g_emb.size(), device=utils.device)
                g_emb[0][hood:hood + i] = skips
                
            # store hidden and cell state
            h_prev.append(h_out)
            c_prev.append(c_out)
        
        cm = CM.ChildModel(model_ops, model_skips)
        return P_ops, P_skips, cm