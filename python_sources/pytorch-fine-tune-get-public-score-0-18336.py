# encoding=utf-8
import sys
import argparse
import numpy as np
import re
import math
import os
import datetime
import logging
import logging.handlers
import redis
import traceback
import operator
import requests
import bisect
import json
import hashlib
import random
import inspect
import itertools
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
import csv
import torchvision.models as models

from torch.autograd import Variable
from PIL import Image
from operator import itemgetter
from tqdm import *
from subprocess import Popen
from subprocess import PIPE
from threading import Lock
from threading import Thread
from urllib import urlencode
from Queue import Queue
from conf import *
from gensim.models import KeyedVectors
from torch.autograd import Variable
from torchvision import transforms, datasets

'''
the directory struct is as follow.
./train
    dogs/
        1.jpg
        2.jpg
        .
        .
        .
    cats/
        1.jpg
        2.jpg

./val
    dogs/
        1.jpg
        2.jpg
        .
        .
        .
    cats/
        1.jpg
        2.jpg
'''

def lp(content):
    content = content.__str__()
    file_path = os.path.abspath(__file__)
    noline = str(inspect.currentframe().f_back.f_lineno)
    print file_path + ':' + noline + ' ' + content


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TRAIN_ROOT = './train'
VAL_ROOT = './val'
TEST_ROOT = './test'
MODEL_PATH='./output/model_9'
RESULT_PATH='result'
EPOCH_NUM = 100
# PREDICT = False
PREDICT = True
MODEL_SAVE_INTERVAL = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 2048)
        self.resnet50.fc.weight.data = torch.eye(2048)
        for p in self.resnet50.parameters():
            p.requires_grad = False

        self.vgg16 = models.vgg16(pretrained=True)
        mod = list(self.vgg16.classifier.children())
        mod.pop()
        self.vgg16.classifier = torch.nn.Sequential(*mod)
        for p in self.vgg16.parameters():
            p.requires_grad = False

        self.densenet161 = models.densenet161(pretrained=True)
        self.densenet161.classifier = nn.Linear(2208, 2208)
        self.densenet161.classifier.weight.data = torch.eye(2208)
        for p in self.densenet161.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(8352, 1000)
        self.dp1 = nn.Dropout()
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):

        x1 = self.resnet50(x)
        x2 = self.vgg16(x)
        x3 = self.densenet161(x)

        x = torch.cat([x1, x3, x4], 1)

        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x)

        return x


def train():
    data_transform = transforms.Compose([
        transforms.Scale((224, 224), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=TRAIN_ROOT,
                                   transform=data_transform)
    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=25, shuffle=True, num_workers=10)

    dataset = datasets.ImageFolder(root=VAL_ROOT,
                                   transform=data_transform)
    val_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=50, shuffle=True, num_workers=10)

    model = Model().cuda()

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

    model.train()
    for i in range(EPOCH_NUM):
        loss_list = []
        for batch in tqdm(train_data_loader):
            x = Variable(batch[0].cuda())

            img = transforms.ToPILImage()(batch[0][0])
            label = Variable(batch[1].cuda())

            output = model(x)
            output = F.log_softmax(output)

            optimizer.zero_grad()
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data[0])

        total_list = []
        correct_list = []
        for batch in (val_data_loader):
            x = Variable(batch[0].cuda())
            label = batch[1].cuda()
            output = model(x)

            _, predicted = torch.max(output.data, 1)
            total = label.shape[0]
            correct = (predicted == label).sum()

            total_list.append(total)
            correct_list.append(correct)

        print 'epoch:%d avg_loss=%lf acc=%lf' % (i, sum(loss_list) / len(loss_list), sum(correct_list) * 1.0 / sum(total_list))
        if i % MODEL_SAVE_INTERVAL == 0:
            torch.save(model, '%s/model_%d' % ('./output', i))


class TestImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.imgs = []
        for filename in os.listdir('./test'):
            if filename.endswith('jpg'):
                self.imgs.append('{}'.format(filename))

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
            return img, filename

    def __len__(self):
        return len(self.imgs)


def predict():
    data_transform = transforms.Compose([
        transforms.Scale((224, 224), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    dataset = TestImageFolder(TEST_ROOT, data_transform)
    test_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=50, shuffle=True, num_workers=10)

    model = torch.load(MODEL_PATH)
    model.eval()
    f_test_predict = open(RESULT_PATH, 'w')
    f_test_predict.write('id,label\n')

    for batch in tqdm(test_data_loader):

        x = Variable(batch[0].cuda())
        output = model(x)

        for i in range(output.data.shape[0]):
            filepath = batch[1][i]
            filepath = os.path.splitext(os.path.basename(filepath))[0]
            filepath = int(filepath)

            f_test_predict.write(str(filepath) + ',' +
                                 str(math.exp(output.data[i][1])) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()

    if not PREDICT:
        train()
    else:
        predict()