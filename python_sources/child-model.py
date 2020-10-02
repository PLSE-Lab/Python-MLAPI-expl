import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

from collections import OrderedDict

# global variables & hyperparameters
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32        # batch size of training set
TEST_BATCH_SIZE = 1000 # batch size of test set
CHANNELS = 240           # number of (output) channels, constant throughout the network


## define possible layer operations

#def weight_sharing(f):
#    def g(node, *param):
#        layer = f(*param)    
#        if node ...:
#            layer.weight = load    
#
#@weight_sharing
def conv(in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=1, groups=1):
    same_padding = kernel_size//2
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=same_padding, groups=groups)
    
def conv3(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=3)

def conv5(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=5)

def depth_conv3(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=3, groups=in_channels)

def depth_conv5(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=5, groups=in_channels)

def max_pool(kernel_size=3, stride=3, padding=0):
    return nn.MaxPool2d(kernel_size=kernel_size, padding=padding)

def avg_pool(kernel_size=3, stride=3, padding=0):
    return nn.AvgPool2d(kernel_size=kernel_size, padding=padding)

def batch_norm(in_channels=CHANNELS):
    return nn.BatchNorm2d(in_channels, track_running_stats=False)

def relu():
    return nn.ReLU()

OPERATION_NAMES = ["conv3", "conv5", "dconv3", "dconv5", "maxpool", "avgpool"]
OPERATIONS = [conv3, conv5, depth_conv3, depth_conv5, max_pool, avg_pool]
#OPERATION_CATEGORIES = { 'conv ': [1,2], 'depth': [3,4]], 'pool': [5,6]}

#convert between string name and integer numbers
op_name_dict = {name: i for i,name in enumerate(OPERATION_NAMES) }
def enumerate_operation_names(operation_names):
    return [op_name_dict[name] for name in operation_names]

# returns pre and post concat image sizes of the given child model (and input size)
def generate_image_sizes1(child, input_size=32):
    N = child.number_of_nodes()
    
    current_size = input_size
    pre_concat_sizes = torch.zeros((N), device=utils.device)
    post_concat_sizes = torch.zeros((N), device=utils.device) 
    
    for node in range(N):        
        #pooling layers resize the input
        if child.ops[node] == 4 or child.ops[node] == 5:
            if current_size % 3 != 0:
                current_size += 2
            current_size = current_size//3

        pre_concat_sizes[node] = current_size
        
        #concatenations from skip connections
        if node > 0:
            hood = node*(node - 1)//2 
            links = child.skips[hood:hood+node]
            links = links.float()
            
            max_neighboring_sizes = links*pre_concat_sizes[:node]
            max_neighboring_size = torch.max(max_neighboring_sizes, dim=0)[0]
        else:
            max_neighboring_size = 0
        
        current_size = max(max_neighboring_size, current_size)
        
        post_concat_sizes[node] = current_size
    
    return pre_concat_sizes, post_concat_sizes

# Module implementing skip connections by concatenation
class SkipLayer(nn.Module):
    def __init__(self, node_index, links, pre_imgsizes, channels=CHANNELS):
        super(SkipLayer, self).__init__()

        self.node_index = node_index
        self.pre_imgsizes = pre_imgsizes
        self.link_indices = (links == 1).nonzero().squeeze(dim=1)
        self.channels = channels

        
        if self.link_indices.size(0) > 0:
            self.node_inds = torch.tensor(# indices of nodes to be linked
                    [*self.link_indices, self.node_index], device=utils.device) 
            
            sizes = self.pre_imgsizes[self.node_inds] # relevant pre concat image sizes
            
            maxind = torch.max(sizes, dim=0)[1]
            self.out_size = sizes[maxind].int() # output image size must be the maximum
            
            self.pad_inds = [] # indices of nodes to be padded
            self.pad_list = nn.ModuleList() # list of padding modules
            
            for sind, size in enumerate(sizes):
                size = size.int()
                if size < self.out_size: # if size is smaller than output needs to be
                    shape_diff = int(self.out_size - size) # calculate difference
                    #print(shape_diff)
                    if shape_diff % 2 == 0: # even image dim difference
                        pad = shape_diff//2 # padding is exactly half of the difference
                        constpad = nn.ConstantPad2d(pad, 0)
                        #constpad = nn.ConstantPad2d((pad, pad, pad, pad), 0)
                    else: # odd image dim difference, TODO test ODD image dimensions
                        # need by-one-different padding for each side
                        pad_topleft = shape_diff//2
                        pad_bottomright = pad_topleft + 1
                        constpad = nn.ConstantPad2d((pad_topleft, pad_bottomright, pad_topleft, pad_bottomright), 0)
                    self.pad_list.append(constpad) # register padding operation
                    self.pad_inds.append(self.node_inds[sind])
            self.pad_inds = torch.tensor(self.pad_inds, device=utils.device)
            self.conv1 = conv(in_channels=self.channels*len(sizes), out_channels=self.channels) # define conv1x1 to bring channel number back to CHANNELS
    
    def __call__(self, all_inputs):
        #print([inp.shape for inp in all_inputs])
        if len(self.link_indices) > 0: # there is at least one skip connection
            concat_inputs = []
            for inp_ind, inp in enumerate(all_inputs):
                if inp_ind in self.pad_inds: # if input needs to be padded
                    inpcopy = inp.clone() # copy input (because different paddings might be needed at two layers)

                    pad_ind = torch.where(self.pad_inds == inp_ind, 
                                    torch.ones_like(self.pad_inds),
                                    torch.zeros_like(self.pad_inds) )
                    pad_ind = pad_ind.nonzero().squeeze()
                    
                    padop = self.pad_list[pad_ind] # get padding operation

                    padout = padop(inpcopy) # apply padding
                    concat_inputs.append(padout) # add padded output to concatenation list
                elif inp_ind in self.node_inds: # if this input is involved (but doesnt need padding)
                    concat_inputs.append(inp) # add to concat list
            catout = torch.cat(concat_inputs, dim=1) # concatenate all involved inputs
            convout = self.conv1(catout) # cross-correlate to change to CHANNELS channels
            return convout
        else: # return unchanged input
            return all_inputs[-1]

# PyTorch module for child model (shared_weights enables warm start, input_size is side length of the square input images, output_size is number of classes to be detected)
class TorchChildModel(nn.Module):
    def __init__(self, childmodel, input_size=32, output_size=10, channels=CHANNELS):
        super(TorchChildModel, self).__init__()
        
        self.childmodel = childmodel
        self.num_nodes = len(self.childmodel.ops)
        self.channels = channels
        
        self.pre_imgsizes, self.post_imgsizes = generate_image_sizes1(self.childmodel, input_size)
        
        # module containers for each layer and skip layer
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        for nodeind in range(len(self.childmodel.ops)): # iterate over nodes
            opid = self.childmodel.ops[nodeind].int() # get current nodes operation
            op = OPERATIONS[opid]

            if opid == 0 or opid == 1: # conv3x3, conv5x5
                if nodeind == 0:
                    layer = nn.Sequential(op(in_channels=3, out_channels=self.channels),
                                          batch_norm(in_channels=self.channels)
                    )
                else:
                    layer = nn.Sequential(relu(),
										op(in_channels=self.channels, out_channels=self.channels),
										batch_norm(in_channels=self.channels)
					)
            elif opid == 4 or opid == 5: # maxpool3x3, avgpool3x3
                padding = 1 # padding of 1 to avoid 2 columns/rows to be ignored
                if nodeind == 0:
                    layer = nn.Sequential(op(padding=padding),
                                          conv(in_channels=3, out_channels=self.channels),
                                          batch_norm(in_channels=self.channels)
                    )
                else:
                    curr_post_imgsize = self.post_imgsizes[nodeind - 1]
                    if curr_post_imgsize % 3 == 0: # if img size is divisible by 3 there is no need for padding
                        padding = 0
                    layer = nn.Sequential(relu(),
                                        op(padding=padding),
                                        batch_norm(in_channels=self.channels)
                    )
            else: # depthwise separable 3x3, 5x5
                if nodeind == 0:              
                    layer = nn.Sequential(op(in_channels=3, out_channels=self.channels),
                                          conv(in_channels=self.channels, out_channels=self.channels), # additional conv1x1 for separable conv
                                          batch_norm(in_channels=self.channels)
                    )
                else:
                    layer = nn.Sequential(relu(),
                                        op(in_channels=self.channels, out_channels=self.channels), 
                                        conv(in_channels=self.channels, out_channels=self.channels), 
                                        batch_norm(in_channels=self.channels)
                    )

            self.layers.append(layer)
            
            hood = nodeind*(nodeind - 1)//2 
            links = self.childmodel.skips[hood:hood + nodeind]
            skip_layer = SkipLayer(nodeind, links, self.pre_imgsizes, channels=self.channels)
            self.skip_layers.append(skip_layer)
            
        self.output_layer = nn.Linear(in_features=self.channels, out_features=output_size) # fully connected layer to classes
        
        self.apply(childmodel_init) # initialize weights depending on layer types
        self.to(utils.device) # move this module to the correct device
        
        #set_shared_weights -> for now moved outside
    
    def forward(self, x):
        outputs = []

        for lid, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(x)
            skip_layer = self.skip_layers[lid]
            x = skip_layer(outputs)
        global_avg_pool_out = torch.flatten(x, start_dim=2, end_dim=-1).mean(dim=2, keepdim=False)

        output = self.output_layer(global_avg_pool_out)

        return output 
    
    # set shared weights if corresponding entries in given list are not all nan (not trained yet)
    def set_shared_weights(self, SW):
        if not SW is None:
            model_dict = self.state_dict()
            
            share_dict = OrderedDict()
            for nodeind in range(self.num_nodes):
                opid = self.childmodel.ops[nodeind].int()
                
                layer_keys = [key for key in self.state_dict().keys() if key.startswith("layers." + str(nodeind) + ".")]

                for i_key, key in enumerate(layer_keys):
                    swl = SW.layer_weights[nodeind][opid][i_key]
                    if torch.isnan(swl).sum() != np.prod(swl.size()):
                        share_dict[key] = swl
                        
            out_weight = SW.output_weights[0]
            out_bias = SW.output_weights[1]
            if torch.isnan(out_weight).sum() != np.prod(out_weight.size()) and torch.isnan(out_bias).sum() != np.prod(out_bias.size()):
                share_dict["output_layer.weight"] = out_weight
                share_dict["output_layer.bias"] = out_bias
    
            model_dict.update(share_dict)
            self.load_state_dict(model_dict)
    
    # modifies shared weights object by updating the tensors used by this model
    def get_shared_weights(self, SW):
        for nodeind in range(self.num_nodes):
            opid = self.childmodel.ops[nodeind].int()
            layer_keys = [key for key in self.state_dict().keys() if key.startswith("layers." + str(nodeind) + ".")]

            assert len(layer_keys) == len(SW.layer_weights[nodeind][opid]), "Not as many weights as should be, op {0}.".format(opid)
            
            for i_key, key in enumerate(layer_keys):
                SW.layer_weights[nodeind][opid][i_key] = self.state_dict()[key] 
            
        SW.output_weights[0] = self.state_dict()["output_layer.weight"]
        SW.output_weights[1] = self.state_dict()["output_layer.bias"]
        return SW

# class holding a batch of child models
class ChildModelBatch:
    def __init__(self, operations, skip_connections):
        self.ops = operations
        self.skips = skip_connections
        
        #sanity checks
        N = self.ops.size(1)
        n = self.skips.size(1)
        assert n == (N-1) * N / 2, 'size of skip connections incompatible with number of nodes'
        assert self.ops.size(0) == self.skips.size(0), 'batch size incoherent'
    
    def number_of_nodes(self): 
        return self.ops.size(1)
    
    def batch_size(self):
        return self.ops.size(0)

    def get_childmodel(self, i):
        assert i < self.batch_size(), 'index out of batch bounds'
        return ChildModel(self.ops[i], self.skips[i])

    def to_torch_models(self, weights=None, channels=CHANNELS):
        tm_list = []
        for i in range(self.batch_size()):
            cm = self.get_childmodel(i)
            tcm = cm.to_torch_model(channels)
            tcm.set_shared_weights(weights)
            tm_list.append(tcm)
        return tm_list

# class holding the definition of a child model (operations and skip connections)
class ChildModel:
    def __init__(self, operations, skip_connections):
        self.ops = torch.as_tensor(operations, device=utils.device)
        self.skips = torch.as_tensor(skip_connections, device=utils.device)
        
        #sanity checks
        N = self.ops.size(0)
        n = self.skips.size(0)
        assert n == (N-1) * N / 2, 'size of skip connections incompatible with number of nodes'
    
    def number_of_nodes(self): 
        return self.ops.size(0)

    def to_torch_model(self, weights=None, channels=CHANNELS):
        tcm = TorchChildModel(self, channels=channels)
        tcm.set_shared_weights(weights)
        return tcm

# extracts the correct tensor sizes for all shared weights by using a dummy child model with all possible operations
def get_weight_sizes():
    input_weight_sizes = []
    for j in range(len(OPERATIONS)):
        all_ops = torch.tensor([j, 1, 2])
        ch2 = ChildModel(all_ops, torch.zeros(3))
        tcm = TorchChildModel(ch2)
        w_sizes_perop = []
        for key in tcm.state_dict().keys():
            if key.startswith("layers.0."):
                w_sizes_perop.append(tcm.state_dict()[key].size())
        input_weight_sizes.append(w_sizes_perop)
    
    all_ops = torch.Tensor([0, 0, 1, 2, 3, 4, 5])
    len_skips = all_ops.size(0)*(all_ops.size(0) - 1)//2
    ch2 = ChildModel(all_ops,  torch.zeros(len_skips))
    tcm = TorchChildModel(ch2, input_size=32)
    layer_weight_sizes = []
    for j in range(len(OPERATIONS)):
        w_sizes_perop = []
        for key in tcm.state_dict().keys():
            if key.startswith("layers." + str(j+1)):
                #print(tcm.state_dict()[key].size())
                w_sizes_perop.append(tcm.state_dict()[key].size())
        layer_weight_sizes.append(w_sizes_perop)
        
    output_weight_sizes = []
    output_weight_sizes.append(tcm.output_layer.weight.size()) # output fully connected 
    output_weight_sizes.append(tcm.output_layer.bias.size())
    
    return input_weight_sizes, layer_weight_sizes, output_weight_sizes

# He initialization for a weights of a Conv2d layer (fan_in depends on input size)
def He_init(ten, fan_in=32*32*CHANNELS): #He init: 2/FAN_IN with FAN_IN = 32*32*9
    std = np.sqrt(2/fan_in)
    return torch.nn.init.normal_(ten, std=std)

# initialization routine for the layer types contained in TorchChildModel
def childmodel_init(m):
    if isinstance(m, nn.Conv2d):
        He_init(m.weight, fan_in=m.in_channels*32*32)
        He_init(m.bias, fan_in=m.in_channels*32*32)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# class for storing all shared weight tensors in one place
class SharedWeights:
    def __init__(self): 
        self.output_weights = []
        self.layer_weights = []
    
    def init(self, layer, out):
        self.layer_weights = layer
        self.output_weights = out

    def clone(self):          
        # clone input weights
        cloned_output_weights = []
        for i in range(len(self.output_weights)):
            cloned_output_weights.append(self.output_weights[i].clone())
                      
        # clone layer weights
        cloned_layer_weights = []
        for node in range(len(self.layer_weights)):
            node_W = []
            for op in range(len(self.layer_weights[node])):
                op_W = []
                for w in range(len(self.layer_weights[node][op])):
                    op_W.append(self.layer_weights[node][op][w].clone())
                node_W.append(op_W)
            cloned_layer_weights.append(node_W)
        # create new shared weights object
        cloned_shared_weights = SharedWeights()
        cloned_shared_weights.init(cloned_layer_weights, cloned_output_weights)
        return cloned_shared_weights
    
    def diff(self, sw):
        # diff input weights
        diff_output_weights = []
        for i in range(len(self.output_weights)):
            diff_output_weights.append(self.output_weights[i] - sw.output_weights[i])
                      
        # diff layer weights
        diff_layer_weights = []
        for node in range(len(self.layer_weights)):
            node_W = []
            for op in range(len(self.layer_weights[node])):
                op_W = []
                for w in range(len(self.layer_weights[node][op])):
                        op_W.append(self.layer_weights[node][op][w] - sw.layer_weights[node][op][w])
                node_W.append(op_W)
            diff_layer_weights.append(node_W)
        # create new shared weights object
        diff_shared_weights = SharedWeights()
        diff_shared_weights.init(diff_layer_weights, diff_output_weights)
        return diff_shared_weights
    

# used to be used for initialization, now possibly deprecated
#def initialize_weights(num_nodes, weight_sizes=None, init_func=He_init):
#    if weight_sizes is None:
#        weight_sizes = get_weight_sizes()
#    inp_sizes, layer_sizes, out_sizes = weight_sizes
#
#    W = SharedWeights()
#    # initialize output layer
#    for size in out_sizes:
#        W.output_weights.append(init_func(torch.zeros(size, requires_grad=True, device=utils.device)))
#    # initialize hidden layers
#    for node_ind in range(num_nodes):
#        add_lst = []
#        if node_ind == 0:
#            fan_in = 32 * 32 * CHANNELS
#            for op_ind in range(len(OPERATIONS)):
#                add_lst2 = []
#                for size in inp_sizes[op_ind]:
#                    if len(size) == 4: # input conv weight
#                        inp_channels = size[1]
#                        fan_in = 32*32*inp_channels # adjust weight fan_in
#                    add_lst2.append(init_func(torch.zeros(size, requires_grad=True), fan_in=fan_in))
#                add_lst.append(add_lst2)
#        
#        else:
#            for op_ind in range(len(OPERATIONS)):
#                add_lst2 = []
#                for size in layer_sizes[op_ind]:
#                    add_lst2.append(init_func(torch.zeros(size, requires_grad=True)))
#                add_lst.append(add_lst2)
#        W.layer_weights.append(add_lst)
#    return W

# creates a SharedWeights object for a num_nodes sized network filled with nan
def create_shared_weights(num_nodes, weight_sizes=None):
    if weight_sizes is None:
        weight_sizes = get_weight_sizes()
    inp_sizes, layer_sizes, out_sizes = weight_sizes

    W = SharedWeights()
    # create output layer
    for size in out_sizes:
        W.output_weights.append(torch.zeros(size, requires_grad=True, device=utils.device)/0)
    # create hidden layers
    for node_ind in range(num_nodes):
        add_lst = []
        if node_ind == 0:
            for op_ind in range(len(OPERATIONS)):
                add_lst2 = []
                for size in inp_sizes[op_ind]:
                    add_lst2.append(torch.zeros(size, requires_grad=True, device=utils.device)/0)
                add_lst.append(add_lst2)
        else:
            for op_ind in range(len(OPERATIONS)):
                add_lst2 = []
                for size in layer_sizes[op_ind]:
                    add_lst2.append(torch.zeros(size, requires_grad=True, device=utils.device)/0)
                add_lst.append(add_lst2)
        W.layer_weights.append(add_lst)
    return W