
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import utils
import child_model as CM
import child_visualizer as V
import time

def test_child(model, test_loader, output_interval=10):
    
    model.eval()
    correct = 0
    size_dataset = len(test_loader.dataset)
    
    t1 = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
        	output = model(data)
        	pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        	correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct/size_dataset
    
    msg = 'Testing child model on dataset size %d \t' % size_dataset
    msg += "testing time = {:.0f}s".format(time.time() - t1)
    print(msg)
    return accuracy

def train_child(model, train_set, valid_set, optimizer, loss_func, epochs=5, log_interval=100, test_interval=1):
    batch_size = train_set.batch_size
    num_batches = len(train_set)
    train_size = num_batches*batch_size
	
    model.train()
    t1 = time.time()
    
    for epoch in range(epochs):
        print('Epoch ', epoch, ' / ', epochs )
        
        for batch_idx, (data, target) in enumerate(train_set):
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = loss_func(output, target)
    
            loss.backward()
            optimizer.step()
    
            if batch_idx % log_interval == 0:
                msg = 'Train [{}/{} ({:.0f}%)]'.format(batch_idx*batch_size, train_size, 100*batch_idx/num_batches)
                msg += '\tLoss: {:.6f}, Time: {:.3f}'.format(loss.item(), time.time() - t1) 
                print(msg)
        if epoch % test_interval == 0:
            acc = test_child(model, valid_set)
            print('Accuracy {:.5f}'.format(acc))
        
        t1 = time.time()


def pick_best_from_checkpoint(experiment_name=None, checkpoint=None, weight_init="old", channels=9):
    ' never set experiment_name and checkpoint at the same time!'
    
    if checkpoint is None:
        #load checkpoint
        checkpoint = utils.load_checkpoint(experiment_name, file_name=None)
    else:
        child = checkpoint['best_child_model']
        state = checkpoint['best_child']
        omega = checkpoint['shared_weights']
        episode = checkpoint['epoch']
        accuracy = max( checkpoint['best_accs'] )
    
    #show child model
    V.draw_child(child)
    print('Reward signal:', accuracy)
    print('Controller episode:', episode)
    
    
    #Initialization strategies
    if weight_init == 'new':
        child = child.to_torch_model(channels=channels)
        
    elif weight_init == 'current':
        child = child.to_torch_model(omega, channels=channels)
        
    elif weight_init == 'old':
        child = child.to_torch_model(omega, channels=channels)
        child.load_state_dict(state)
        
    elif weight_init == 'nudge':
        child = child.to_torch_model(channels=channels)
        weights = child.state_dict()
        for key in state:
            state[key] = state[key] + weights[key]
        child.load_state_dict(state)
    else:
        assert False, 'invalid initialization strategy ' + str(weight_init)
    return child
            
    
