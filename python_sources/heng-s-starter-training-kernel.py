#!/usr/bin/env python
# coding: utf-8

# This kernel is based on [Heng's Starter code](https://www.kaggle.com/c/bengaliai-cv19/discussion/123757). I have already published a [kernel](https://www.kaggle.com/bibek777/heng-starter-inference-kernel) doing the inference using his models. In this kernel, we will use his codes to do the training.
# 
# I have uploaded the necessary codes from Heng's starter and opensourced it as a [kaggle dataset](https://www.kaggle.com/bibek777/hengcodes) so that you can use it to train in kaggle kernels/google-colabs. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.


# In[ ]:


os.listdir('../input')


# Add heng's code to our envionment and import the modules

# In[ ]:


hengs_path = '../input/hengcodes'
sys.path.append(hengs_path)


# In[ ]:


from common  import *
from model   import *
from kaggle import *


# Heng uses his own version of datasplit(which I have uploaded [here](https://www.kaggle.com/bibek777/hengdata)) in his codes. I tried using it but get memory error, maybe it's too large to load. So I have edited the dataloader in his code and use different split for train and valid dataset. The codes/ideas for dataloader is taken from this [kernel](https://www.kaggle.com/backaggle/catalyst-baseline). Also the dataset used in this kernel is taken from [here](https://www.kaggle.com/pestipeti/bengaliai), uploaded by Peter

# In[ ]:


data_root = "../input/bengaliai/256_train/256/"
df = pd.read_csv("../input/bengaliai-cv19/train.csv")
train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=2411)

TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]
NUM_TASK = len(TASK_NAME)


# In[ ]:


class KaggleDataset(Dataset):

    def __init__(self, df, data_path, augment=None):
        self.image_ids = df['image_id'].values
        self.grapheme_roots = df['grapheme_root'].values
        self.vowel_diacritics = df['vowel_diacritic'].values
        self.consonant_diacritics = df['consonant_diacritic'].values

        self.data_path = data_path
        self.augment = augment

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        return string


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, index):
        # print(index)
        image_id = self.image_ids[index]
        grapheme_root = self.grapheme_roots[index]
        vowel_diacritic = self.vowel_diacritics[index]
        consonant_diacritic = self.consonant_diacritics[index]

        image_id = os.path.join(self.data_path, image_id + '.png')

        image = cv2.imread(image_id, 0)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image = image.astype(np.float32)/255
        label = [grapheme_root, vowel_diacritic, consonant_diacritic]

        infor = Struct(
            index    = index,
            image_id = image_id,
        )

        if self.augment is None:
            return image, label, infor
        else:
            return self.augment(image, label, infor)


# In[ ]:


def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    input = np.stack(input)
    #input = input[...,::-1].copy()
    input = input.transpose(0,3,1,2)

    label = np.stack(label)

    #----
    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(label).long()
    truth0, truth1, truth2 = truth[:,0],truth[:,1],truth[:,2]
    truth = [truth0, truth1, truth2]
    return input, truth, infor


##############################################################

def tensor_to_image(tensor):
    image = tensor.data.cpu().numpy()
    image = image.transpose(0,2,3,1)
    #image = image[...,::-1]
    return image


##############################################################

def do_random_crop_rotate_rescale(
    image,
    mode={'rotate': 10,'scale': 0.1,'shift': 0.1}
):

    dangle = 0
    dscale_x, dscale_y = 0,0
    dshift_x, dshift_y = 0,0

    for k,v in mode.items():
        if   'rotate'== k:
            dangle = np.random.uniform(-v, v)
        elif 'scale' == k:
            dscale_x, dscale_y = np.random.uniform(-1, 1, 2)*v
        elif 'shift' == k:
            dshift_x, dshift_y = np.random.uniform(-1, 1, 2)*v
        else:
            raise NotImplementedError

    #----

    height, width = image.shape[:2]

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift_x*width, dshift_y*height

    src = np.array([[-width/2,-height/2],[ width/2,-height/2],[ width/2, height/2],[-width/2, height/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+width/2 +tx
    y = (src*[sin, cos]).sum(1)+height/2+ty
    src = np.column_stack([x,y])

    dst = np.array([[0,0],[width,0],[width,height],[0,height]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)
    image = cv2.warpPerspective( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1))

    return image


def do_random_log_contast(image, gain=[0.70, 1.30] ):
    gain = np.random.uniform(gain[0],gain[1],1)
    inverse = np.random.choice(2,1)

    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image,0,1)
    return image


#https://github.com/albumentations-team/albumentations/blob/8b58a3dbd2f35558b3790a1dbff6b42b98e89ea5/albumentations/augmentations/transforms.py
def do_grid_distortion(image, distort=0.25, num_step = 10):

    # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    distort_x = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]
    distort_y = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]

    #---
    height, width = image.shape[:2]
    xx = np.zeros(width, np.float32)
    step_x = width // num_step

    prev = 0
    for i, x in enumerate(range(0, width, step_x)):
        start = x
        end   = x + step_x
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + step_x * distort_x[i]
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur


    yy = np.zeros(height, np.float32)
    step_y = height // num_step

    prev = 0
    for idx, y in enumerate(range(0, height, step_y)):
        start = y
        end = y + step_y
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + step_y * distort_y[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1))

    return image



# ##---
# #https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
def do_random_contast(image, alpha=[0,1]):
    beta  = 0
    alpha = random.uniform(*alpha) + 1
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image,0,1)
    return image



# In[ ]:


################################################################################################

def train_augment(image, label, infor):
    if np.random.rand()<0.5:
        image = do_random_crop_rotate_rescale(image, mode={'rotate': 17.5,'scale': 0.25,'shift': 0.08})
    if np.random.rand()<0.5:
        image = do_grid_distortion(image, distort=0.20, num_step = 10)
    return image, label, infor



def valid_augment(image, label, infor):
    return image, label, infor


# In[ ]:


#------------------------------------
def do_valid(net, valid_loader, out_dir=None):

    valid_loss = np.zeros(6, np.float32)
    valid_num  = np.zeros_like(valid_loss)

    valid_probability = [[],[],[],]
    valid_truth = [[],[],[],]

    for t, (input, truth, infor) in enumerate(valid_loader):

        #if b==5: break
        batch_size = len(infor)

        net.eval()
        input = input.cuda()
        truth = [t.cuda() for t in truth]

        with torch.no_grad():
            logit = data_parallel(net, input) #net(input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, truth)
            correct = metric(probability, truth)

        #---
        loss = [l.item() for l in loss]
        l = np.array([ *loss, *correct, ])*batch_size
        n = np.array([ 1, 1, 1, 1, 1, 1  ])*batch_size
        valid_loss += l
        valid_num  += n

        #---
        for i in range(NUM_TASK):
            valid_probability[i].append(probability[i].data.cpu().numpy())
            valid_truth[i].append(truth[i].data.cpu().numpy())

        #print(valid_loss)
        print('\r %8d /%d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/(valid_num+1e-8)

    #------
    for i in range(NUM_TASK):
        valid_probability[i] = np.concatenate(valid_probability[i])
        valid_truth[i] = np.concatenate(valid_truth[i])
    recall, avgerage_recall = compute_kaggle_metric(valid_probability, valid_truth)


    return valid_loss, (recall, avgerage_recall)


# In[ ]:


def run_train():
    out_dir = '/kaggle/working'
    initial_checkpoint = None

    schduler = NullScheduler(lr=0.01)
    iter_accum = 1
    batch_size = 64 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
        
    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = KaggleDataset(
        df = train_df, 
        data_path = data_root,
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    valid_dataset = KaggleDataset(
        df = valid_df, 
        data_path = data_root,
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = 64,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        # for k in list(state_dict.keys()):
        #      if any(s in k for s in ['logit',]): state_dict.pop(k, None)
        # net.load_state_dict(state_dict,strict=False)

        net.load_state_dict(state_dict,strict=True)  #True
    else:
        net.load_pretrain(pretrain_file = '../input/pytorch-pretrained-models/densenet121-a639ec97.pth', is_print=False)


    log.write('net=%s\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0, weight_decay=0.0)

    #num_iters   = 3000*1000 # use this for training longer
    num_iters   = 1000  # comment this for training longer
    iter_smooth = 50
    iter_log    = 250
    iter_valid  = 500
    iter_save   = [0, num_iters-1]                   + list(range(0, num_iters, 1000))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                    |----------------------- VALID------------------------------------|------- TRAIN/BATCH -----------\n')
    log.write('rate    iter  epoch | kaggle                    | loss               acc              | loss             | time       \n')
    log.write('----------------------------------------------------------------------------------------------------------------------\n')
              #0.01000  26.2  15.1 | 0.971 : 0.952 0.992 0.987 | 0.22, 0.07, 0.07 : 0.94, 0.98, 0.98 | 0.37, 0.13, 0.13 | 0 hr 13 min

    def message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'):
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '
            loss = train_loss

        text =             '%0.5f %5.1f%s %4.1f | '%(rate, iter/1000, asterisk, epoch,) +            '%0.3f : %0.3f %0.3f %0.3f | '%(kaggle[1],*kaggle[0]) +            '%4.2f, %4.2f, %4.2f : %4.2f, %4.2f, %4.2f | '%(*valid_loss,) +            '%4.2f, %4.2f, %4.2f |'%(*loss,) +            '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    #----
    kaggle = (0,0,0,0)
    valid_loss = np.zeros(6,np.float32)
    train_loss = np.zeros(3,np.float32)
    batch_loss = np.zeros_like(train_loss)
    iter = 0
    i    = 0



    start_timer = timer()
    while  iter<num_iters:
            
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch
            
            #if 0:
            if (iter % iter_valid==0):
                valid_loss, kaggle = do_valid(net, valid_loader, out_dir) #
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                log.write(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='log'))
                log.write('\n')

            #if 0:
            if iter in iter_save:
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth = [t.cuda() for t in truth]

            logit = data_parallel(net, input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, truth)

            (( 2*loss[0]+loss[1]+loss[2] )/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  --------
            loss = [l.item() for l in loss]
            l = np.array([ *loss, ])*batch_size
            n = np.array([ 1, 1, 1 ])*batch_size
            batch_loss      = l/(n+1e-8)
            sum_train_loss += l
            sum_train      += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0


            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --
    
    log.write('\n')


# We will run the training but stop it, after 1000 iterations as the purpose of this kernel is to show you how to use Heng's code to train your models in kaggle kernels. You can train longer.

# In[ ]:


run_train()


# Like I said before, this is a very strong baseline from Heng. It can be developed in many ways, some of which are:
# * Add mixup augmentation
# * Train with `serex50` model

# I hope this kernel was helpful to you in someways. I published this because I did not want Heng's Starter to be used by only those with compute power. Hopefully this kernel allows you to train your models using his codes using Kaggle GPUS, Google Colabs, and/or [Paperspace free GPUs](https://gradient.paperspace.com/free-gpu). You can follow this pipeline to train your models:
# > train in one kernel(for two hours) -> save model -> load model in another kernel -> train longer
