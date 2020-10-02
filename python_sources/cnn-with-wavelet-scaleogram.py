#!/usr/bin/env python
# coding: utf-8

# # Example CNN solution with wavelet pre-processing for the challenge

# In[ ]:


from customdatasets import TrainDataset, TestDataset


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from types import SimpleNamespace
import matplotlib.pyplot as plt
import csv
import librosa
import scipy as sc


# In[ ]:


# Hyperparameters
args = SimpleNamespace(batch_size=64, test_batch_size=64, epochs=1,
                       lr=0.01, momentum=0.5, seed=1, log_interval=200)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# In[ ]:


print("Using device: {}".format(device))


# In[ ]:


import numpy as np

toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)

trainDataset = TrainDataset("../input/oeawai/train/kaggle-train", transform=toFloat)
trainDataset = TrainDataset("../input/oeawai/train-small/train-small", transform=toFloat)
print(len(trainDataset))

testDataset = TestDataset("../input/oeawai/kaggle-test/kaggle-test", transform=toFloat)
print(len(testDataset))

train_loader = torch.utils.data.DataLoader(trainDataset,
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testDataset,
        batch_size=args.test_batch_size, shuffle=False) #Shuffle should be false!


# # Data preprocessing
# 
# Below I made some changes of how to preprocess the data. I decided to us a wavelet transform in form of a scaleograms. Some of the utilities are inspired/taken from [this github project](https://github.com/alsauve/scaleogram)

# In[ ]:


import pywt

def periods2scales(periods, wavelet, dt=1.0):
    return (periods/dt) * pywt.central_frequency(wavelet)


def fastcwt(data, scales, wavelet, sampling_period=1.0, method='auto'):
    
    # accept array_like input; make a copy to ensure a contiguous array
    data = np.array(data)
    if not isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
        wavelet = pywt.DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.array([scales])
    dt_out = None  # currently keep the 1.0.2 behaviour: TODO fix in/out dtype consistency
    if data.ndim == 1:
        if wavelet.complex_cwt:
            dt_out = complex
        out = np.zeros((np.size(scales), data.size), dtype=dt_out)
        precision = 10
        int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
        
        if method in ('auto', 'fft'):
            # - to be as large as the sum of data length and and maximum wavelet
            #   support to avoid circular convolution effects
            # - additional padding to reach a power of 2 for CPU-optimal FFT
            size_pad = lambda s: 2**np.int(np.ceil(np.log2(s[0] + s[1])))
            size_scale0 = size_pad( (len(data), 
                                     np.take(scales, 0) * ((x[-1] - x[0]) + 1)) )
            fft_data = None
        elif not method == 'conv':
            raise ValueError("method must be in: 'conv', 'fft' or 'auto'")

        for i in np.arange(np.size(scales)):
            step = x[1] - x[0]
            j = np.floor(
                np.arange(scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
            if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
            int_psi_scale = int_psi[j.astype(np.int)][::-1]
               
            if method == 'conv':
                conv = np.convolve(data, int_psi_scale)
            else:
                size_scale = size_pad( (len(data), len(int_psi_scale)) )
                if size_scale != size_scale0:
                    # the fft of data changes when padding size changes thus
                    # it has to be recomputed
                    fft_data = None
                size_scale0 = size_scale
                nops_conv = len(data) * len(int_psi_scale)
                nops_fft  = (2+(fft_data is None)) * size_scale * np.log2(size_scale)
                if (method == 'fft') or ((method == 'auto') and (nops_fft < nops_conv)):
                    if fft_data is None:
                        fft_data = np.fft.fft(data, size_scale)
                    fft_wav = np.fft.fft(int_psi_scale, size_scale)
                    conv = np.fft.ifft(fft_wav*fft_data)
                    conv = conv[0:len(data)+len(int_psi_scale)-1]
                else:
                    conv = np.convolve(data, int_psi_scale)
                
            coef = - np.sqrt(scales[i]) * np.diff(conv)
            if not np.iscomplexobj(out):
                coef = np.real(coef)
            d = (coef.size - data.size) / 2.
            if d > 0:
                out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
            elif d == 0.:
                out[i, :] = coef
            else:
                raise ValueError(
                    "Selected scale of {} too small.".format(scales[i]))
        frequencies = pywt.scale2frequency(wavelet, scales, precision)
        if np.isscalar(frequencies):
            frequencies = np.array([frequencies])
        for i in np.arange(len(frequencies)):
            frequencies[i] /= sampling_period
        return out, frequencies
    else:
        raise ValueError("Only dim == 1 supported")


# In[ ]:


def log_scaleogram (audio) :
    audio_input = audio[0::500]
    dt = np.linspace(0,4,len(audio_input))
    scales = periods2scales( np.arange(1, 129,1) ,wavelet='cmor1-1.5')
    out = fastcwt(audio_input, scales=scales,wavelet='cmor1-1.5')
    scaleogram_log = np.log(np.abs(out[0])+np.exp(-12))
    return scaleogram_log


# In[ ]:


number_of_examples_to_plot = 20
scaleograms = np.zeros((number_of_examples_to_plot,128,128))
for samples, instrumentsFamily in train_loader:
    for index in range(number_of_examples_to_plot):
#         print (samples.shape)
        scaleograms[index] = log_scaleogram(audio=samples[index])
    family = trainDataset.transformInstrumentsFamilyToString(instrumentsFamily.numpy().astype(int))
    break # SVM is only fitted to a fixed size of data

import matplotlib.pyplot as plt
    
for i in range(number_of_examples_to_plot):
    print(scaleograms[i].shape)
    plt.imshow(scaleograms[i])
    print(family[i])
    plt.colorbar()
    plt.show()


# # Net
# 
# In this class, you can modify the network's structure to try to improve the performance. 

# In[ ]:


# NN architecture (three conv and two fully connected layers)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.first_conv = nn.Conv2d(1, 20, 5, 1)
        self.second_conv = nn.Conv2d(20, 50, 5, 2)
        self.third_conv = nn.Conv2d(50, 50, 5, 2)
        self.fc1 = nn.Linear(50*2*2, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        
        scaleograms = np.zeros((len(x),128,128))
        for index, audio in enumerate(x.cpu().numpy()):
            scaleograms[index] = log_scaleogram(audio=audio) 
        
        x = torch.from_numpy(scaleograms[:, np.newaxis, :, :]).to(device).float()
        
        # x.size is (batch_size, 1, 256, 252)
        x = F.relu(self.first_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.second_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.third_conv(x))
        x = F.max_pool2d(x, 2, 2)
        # x.size is (batch_size, 50, 6, 6)
        x = x.view(-1, 2*2*50)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# In[ ]:


# This function trains the model for one epoch
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# In[ ]:


# This function evaluates the model on the test data
def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        familyPredictions = np.zeros(len(test_loader.dataset), dtype=np.int)
        for index, samples in enumerate(test_loader):
            samples = samples.to(device)
            familyPredictions[index*len(samples):(index+1)*len(samples)] = model(samples).max(1)[1].cpu() # get the index of the max log-probability
    
    familyPredictionStrings = trainDataset.transformInstrumentsFamilyToString(familyPredictions.astype(int))

    with open('NN-submission-' +str(epoch)+'.csv', 'w', newline='') as writeFile:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for index in range(len(testDataset)):
            writer.writerow({'Id': index, 'Predicted': familyPredictionStrings[index]})
    print('saved predictions')


# In[ ]:


# Main
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                      momentum=args.momentum)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader, epoch)

