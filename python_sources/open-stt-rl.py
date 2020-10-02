#!/usr/bin/env python
# coding: utf-8

# ### Reinforcement Learning to fine-tune the RNN-Transducer model
# 
# It is my favorite part from series of end-to-end acoustic models in PyTorch. In the following links you can find the previous parts:
# 1. [open-stt-lm](https://www.kaggle.com/sorokin/open-stt-lm) - Character-based RNN language model
# 2. [open-stt-ctc](https://www.kaggle.com/sorokin/open-stt-ctc) - CNN-RNN acoustic model with CTC loss
# 3. [open-stt-rnnt](https://www.kaggle.com/sorokin/open-stt-rnnt) - Character-based RNN language model and CNN-RNN acoustic model with RNN-T loss
# 
# In this kernel I will apply a REINFORCE algorithm with simple sampling strategy. For each example from the training set, N hypothesis will be generated through a decoder. These hypothesis will be used to estimate the expected reward, which will be maximized during the training procedure.
# 
# As a result of this fine-tuning approach, it is possible to reduce WER of the original model by a few absolute percent. You can find more information at the github [project](https://github.com/1ytic/open_stt_e2e).
# 
# **V1**: Initial version with Expected Loss implementation.
# 
# **V2**: Used a reward engineering method, called Symmetric Accuracy (SymAcc), taken from [work](http://www.apsipa.org/proceedings/2018/pdfs/0001934.pdf). SymAcc discounts the accuracy when the length of the hypothesis is shorter than the reference, which motivates the system to try longer hypothesis.
# 
# **V5**: Added pytorch-edit-distance package to speed up training and accurately calculate WER during evaluation. Increased learning rate and stabilized training with ELU(rewards, alpha=0.5).

# In[ ]:


get_ipython().system('pip install warp-rnnt')


# In[ ]:


get_ipython().system('pip install pytorch-edit-distance')


# In[ ]:


import time
import torch
from warp_rnnt import rnnt_loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import log_softmax, relu, elu
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from open_stt_utils import Labels, Transducer, AudioDataset, BucketingSampler, AverageMeter, collate_fn_rnnt
from pytorch_edit_distance import remove_blank, wer, AverageWER, AverageCER


# In[ ]:


labels = Labels()
print(len(labels))


# In[ ]:


model = Transducer(128, len(labels), 512, 256, am_layers=3, lm_layers=3, dropout=0.4)
model.load_state_dict(torch.load('/kaggle/input/open-stt-rnnt/asr.bin', map_location='cpu'))
model.cuda()


# In[ ]:


train = [
    ['open-stt-public-youtube1120-hq', 'data.csv']
]

test = [
    ['open-stt-val', 'asr_calls_2_val.csv'],
    ['open-stt-val', 'buriy_audiobooks_2_val.csv'],
    ['open-stt-val', 'public_youtube700_val.csv']
]

train = AudioDataset(train, labels)
test = AudioDataset(test, labels)

train.filter_by_conv(model.encoder.conv)
train.filter_by_length(500)

test.filter_by_conv(model.encoder.conv)
test.filter_by_length(1000)

sampler = BucketingSampler(train, 32)

train = DataLoader(train, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_sampler=sampler)
test = DataLoader(test, pin_memory=True, num_workers=4, collate_fn=collate_fn_rnnt, batch_size=32)


# In[ ]:


optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=250, gamma=0.99)


# In[ ]:


N = 10
alpha = 0.01
blank = torch.tensor([labels.blank()], dtype=torch.int).cuda()
space = torch.tensor([labels.space()], dtype=torch.int).cuda()


# In[ ]:


for epoch in range(10):
    
    start = time.time()

    sampler.shuffle(epoch)

    grd_train = AverageMeter('gradient')
    rwd_train = AverageMeter('reward')
    err_train = AverageMeter('train')
    err_valid = AverageMeter('valid')
    cer_valid = AverageCER(blank, space)
    wer_valid = AverageWER(blank, space)

    num_batch = 0

    for xs, ys, xn, yn in train:

        optimizer.zero_grad()

        xs = xs.cuda(non_blocking=True)
        ys = ys.cuda(non_blocking=True)
        xn = xn.cuda(non_blocking=True)
        yn = yn.cuda(non_blocking=True)

        model.train()

        zs, xs, xn = model(xs, ys, xn, yn)
        
        model.eval()

        with torch.no_grad():

            ys = ys.t().contiguous()

            xs_e = xs.repeat(N, 1, 1)
            xn_e = xn.repeat(N)
            ys_e = ys.repeat(N, 1)
            yn_e = yn.repeat(N)

            hs_e = model.greedy_decode(xs_e, sampled=True)

            remove_blank(hs_e, xn_e, blank)

            Err = wer(hs_e, ys_e, xn_e, yn_e, blank, space)

            xn_e_safe = torch.max(xn_e, torch.ones_like(xn_e)).float()

            SymAcc = 1 - 0.5 * Err * (1 + yn_e.float() / xn_e_safe)

            rewards = relu(SymAcc).reshape(N, -1)

            hs_e = hs_e.reshape(N, len(xs), -1)
            xn_e = xn_e.reshape(N, len(xs))
            
        model.train()
        
        rewards = rewards.cuda()
            
        rwd_train.update(rewards.mean().item())

        rewards -= rewards.mean(dim=0)
        
        # Stabilize training
        elu(rewards, alpha=0.5, inplace=True)

        total_loss = 0

        if alpha > 0:

            nll = rnnt_loss(zs, ys, xn, yn)

            loss = alpha * nll.mean()
            loss.backward(retain_graph=True)

            total_loss = loss.item()

        for n in range(N):

            ys = hs_e[n]
            yn = xn_e[n]

            # Cut unnecessary padding
            ys = ys[:, :yn.max()].contiguous()

            zs = model.forward_decoder(xs, ys.t(), yn)

            nll = rnnt_loss(zs, ys, xn, yn)

            loss = nll * rewards[n]

            loss = loss.mean() / N

            loss.backward(retain_graph=True)

            total_loss += loss.item()

        grad_norm = clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        scheduler.step()

        err_train.update(total_loss)
        grd_train.update(grad_norm)

        num_batch += 1
        if num_batch == 500:
            break
    
    model.eval()

    with torch.no_grad():
        for xs, ys, xn, yn in test:

            xs = xs.cuda(non_blocking=True)
            ys = ys.cuda(non_blocking=True)
            xn = xn.cuda(non_blocking=True)
            yn = yn.cuda(non_blocking=True)

            zs, xs, xn = model(xs, ys, xn, yn)

            ys = ys.t().contiguous()

            loss = rnnt_loss(zs, ys, xn, yn, average_frames=False, reduction="mean")
            err_valid.update(loss.item())
            
            xs = model.greedy_decode(xs)

            remove_blank(xs, xn, blank)

            cer_valid.update(xs, ys, xn, yn)
            wer_valid.update(xs, ys, xn, yn)
    
    minutes = (time.time() - start) // 60
    
    with open('asr.log', 'a') as log:
        log.write('epoch %d lr %.6f %s %s %s %s %s %s time %d\n' % (
            epoch + 1, scheduler.get_lr()[0],
            grd_train, rwd_train, err_train,
            err_valid, cer_valid, wer_valid,
            minutes
        ))
    
    torch.save(model.state_dict(), 'asr.bin')

