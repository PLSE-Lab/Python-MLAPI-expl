#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install git+https://github.com/haonguyen1915/hao-lib.git')
import os
import sys
if '/kaggle/working' not in sys.path:
    sys.path.append("/kaggle/working")


# In[ ]:


get_ipython().system('cp -rf "../input/10k-mortobike-data/10k_mortobike_data/TFHub" "./"')
get_ipython().system('cp -rf "../input/10k-mortobike-data/10k_mortobike_data/data" "./"')
get_ipython().system('cp -rf "../input/10k-mortobike-data/10k_mortobike_data/model_src" "./"')
get_ipython().system('cp -rf "../input/10k-mortobike-data/10k_mortobike_data/sync_batchnorm" "./"')
get_ipython().system('ls')


# In[ ]:


import torch
import torch.nn as nn
import functools
from haolib.lib_ai.torch_utils import show_batch
from tqdm import tqdm
from model_src.my_utils import _get_data_loaders, get_defaults_configure, update_config_roots,     prepare_root, seed_rng, name_from_config
from model_src import BigGAN as model
from model_src import inception_utils
from model_src import train_fns
from sync_batchnorm.replicate import patch_replication_callback
from model_src.my_utils import *


# In[ ]:


activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True), }

def run(config):
    
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    
    config['G_activation'] = activation_dict[config['G_nl']]
    config['D_activation'] = activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    seed_rng(config['seed'])

    # Prepare root folders if necessary
    prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    # model = __import__("utils." + config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = ema_(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    # exit()
    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD = model.G_D(G, D)
#     print(G)
#     print(D)
#     print('Number of params in G: {} D: {}'.format(
#         *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        load_weights(G, D, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GD = nn.DataParallel(GD)
        if config['cross_replica']:
            patch_replication_callback(GD)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    write_metadata(config['logs_root'], experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])
    loaders = _get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                   'start_itr': state_dict['itr']})
#     for xs, ys in loaders[0]:
#         print("xs: {}".format(xs.shape))
#         show_batch(xs, ys)

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'],
                                                                      config['inception_path'], config['no_fid'])

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
 
    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(sample_,
                               G=(G_ema if config['ema'] and config['use_ema']
                                  else G),
                               z_=z_, y_=y_, config=config)

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    # exit()

    for epoch in range(state_dict['epoch'], config['num_epochs']):
#         print('Epoch: {}'.format(epoch))
        pbar_name = "Epoch {}".format(epoch)
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0], desc=pbar_name)
#             pbar = tqdm(loaders[0])
        # pbar = tqdm(loaders[0])
        for i, (x, y) in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']),
                              **{**get_SVs(G, 'G'), **get_SVs(D, 'D')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                   for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                          state_dict, config, experiment_name)

            # Test every specified interval
            if not (state_dict['itr'] % config['test_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                               get_inception_metrics, experiment_name, test_log)
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


# In[ ]:


config = get_defaults_configure()
config['save_every'] = 1000
config['test_every'] = 2000
config['resolution'] = 128
config['batch_size'] = 32
config['n_classes'] = 1
config['num_epochs'] = 200
config['shuffle'] = True
config['num_D_steps'] = 2
config['num_D_accumulations'] = 2
config['num_G_accumulations'] = 2
config['dataset'] = 'MOTO128'
config['pbar'] = None
config['inception_path'] = 'data/MOTO128_inception_moments.npz'
config['data_root'] = '/kaggle/input/10k-mortobike-images-jpeg/10k_mortobike_images/'
print(config)
# exit()
run(config)


# In[ ]:




