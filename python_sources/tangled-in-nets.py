#!/usr/bin/env python
# coding: utf-8

# 

# # Tangled in nets
# ## What I have learnt working with my first "real life" neural networks
# 
# 
# I'd like to tell you the story of my submission for *ML1819 - What's Cooking?* competition. It was held during my Machine Learning classes, thus we were strongly encouraged to experiment with different techincs. That was even more important to me, since I had no experience in the field of machine learnig. However, I've always been interested in neural networks, so the moment I've heard we can try beat this problem using PyTorch, I knew what I would do.
# 
# In this kernel (or however it's called - I don't even know Kaggle's terminology) I'd like to present you my solution to the problem stated it this competition. Along with all the problems I've encountered, all the mistateks I've made, and all the things I've learnt. The main purpose of this noetbook is for me to learn and organize the knowledge I've obtained while working on this contest.
# 
# First of all, I'd like to apologise every person that is reading this, especially the experienced ones. I'm still learing so I might have done mistakes that I haven't mentioned here. Moreover, I don't know the jargon used in machine learnig communit, so I might struggle to express myself clearly. Additionally, this notebook doesn't contain any spell check, so I apologise for all the typos.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[ ]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}
    print("CUDA")
else:
    device = torch.device("cpu")
    kwargs = {}
    print("CPU")


# In[ ]:


train_file = '../input/cooking_train.json'
test_file = '../input/cooking_test.json'
train_batch = 128
test_batch = 500
log_interval = 10


# First things first, we're importing some libraries. *matplotlib* for plots, *pandas* for reading the data, *torch* for the nets. We divide data into training and verification dataset with `train_test_split()` and prepare the input and target with `CountVectorizer()`. And for data preparation I also need *numpy*, because we'll have to apply the function to the *Y* vector returned by `CountVectorizer()`. 
# 
# After imports we have CUDA. I think you always would like to use CUDA, unless you don't have support for it. Honestly, I don't precisely understand what does the `kwargs` do, but we'll use it to create `torch.utils.data.DataLoader`. From the documentation, I understand that `pin_memory` means that before returning the data, the `DataLoader` will copy everything to GPU. `num_workers` is in my oppinion a little bit more vague, but it might mean the number of GPUs, we use. You can read more about this in the [documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). For the record, I've copied the `kwargs` part from one of the MNIST example codes.
# 
# In the end we have some parameters, like batch size and files destination. I'm not sure, whether the batch size is ok. These are the values that I was using on my PC. I have 4GB GPU, but I'm not quite sure how to correctly compute the amount of memory needed. From my computations I assume, that it's not too much, but I may have used even bigger batch. I guess I should change it for the sake of this notebook, because it's running on CPU, but let's leave it this way. From what I know, it will only hurt the performance.
# 
# Right now, we can load some data and see what we have there.

# In[ ]:


dataset = pd.read_json(train_file)
testset = pd.read_json(test_file)


# In[ ]:


def preproc(line):
    return ' '.join(line).lower()


# In[ ]:


vect_x = CountVectorizer(preprocessor=preproc)
vect_x.fit(dataset.ingredients)
data_x = vect_x.transform(dataset.ingredients).todense()
data_x.shape


# In the above code, I have loaded the data and prepared *X* vector. I've used `CountVectorizer`, because it was used in example solution we were provided during our classes. The vector *X* contains the number of appearences of each word in the ingredients lits. I have to mention, that I'm not sure if this method is correct. On the one hand, it's good because the ingredeints contain for example _olive oil_ and _extra-virgin olive oil_, which I would consider the same in terms of cuisine. On the other hand, we also have _sesame oil_ in the ingredients, which is used in completely different cuisines. Everything depends on how the network will learn this features. Maybe there is a better way to prepare the *X* vector, unfortuantely, I wasn't looking for it. Before we move on, note that `data_x` is `30000x2866` this means that we have 30000 examples and 2866 different ingredients (our features).

# In[ ]:


def elem_wrap(sthg):
    return sthg.tolist()[0].index(1)


# In[ ]:


vect_y = CountVectorizer()
data_y = vect_y.fit_transform(dataset.cuisine.values).todense()
data_y = np.apply_along_axis(elem_wrap, 1, data_y)

cuisine = sorted(set(dataset.cuisine))
len(cuisine)


# As you can see above, we've just prepared the *Y* vector. I'm sure that it isn't the best way to approach this problem, becase I have converted the cuisine names to vectors encoding them and then I've decoded it using my `elem_warp()` function. I think I could have done someting like this:

# In[ ]:


def smart_elem_wrap(sthg):
    return cuisine.index(sthg)

vectorize_elem_wrap = np.vectorize(smart_elem_wrap)
data_y2 = vectorize_elem_wrap(dataset.cuisine.values)
data_y2 == data_y


# As you can see, the result is the same, but in my oppinion the latter code looks better than the former. Plus it should work a little bit faster, I think, because `CountVectorizer` doesn't have to learn the data.
# 
# The last thing that I would like to mention before moving on is the `len(cuisine)`. As you can see, it has 20 different cuisines in it - this is our number of classes we should consider.

# Now, we can look at architectres I've used. As it's mentioned in the subtitle, this is my first encounter with "real life" neural networks. Before that, I was only learnig about them. And I ran some example code on MNIST dataset. That's why designing my own architecture was quite some fun. I didn't know what I was doing and if I were to do this again, I still wouldn't know what to do. **BUT!** I've gained some basic grasp of the architectues. Let's look at my first self-designed neural network:

# In[ ]:


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(2866, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# What have I done? I literally have no idea. I wanted to use only a few layers, which in the end appeared to be a good idea. Actually, this first design might have been the best I've used for this problem (according to score on verification set). We start with 2866 features. The first layer contains 5000 neurons, because I've heard somewhere that it should be roughly 2x the number of features. Because of this, maybe 6000 would be a better number, but I didn't want to create too huge layer. Next we have 3000 neurons and 20 neurons in the end - each coressponding to the output class. I used ReLU, because I think it's the most popular. I was thinking about LeakyReLU, but it appeared that I'm too lazy to do this.
# In the end, I used `log_softmax()` function, becaue it was used in one of the example codes I've seen, plus it makes sense I think. As a loss function I'm using `nll_loss()`, so combined with `log_softmax()` I get cross entropy. I don't know whether this is the best loss function for this problem, but [Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy) article on this function convinced me that it's not a bad idea.

# In[ ]:


class Net1_1(nn.Module):
    def __init__(self):
        super(Net1_1, self).__init__()
        self.fc1 = nn.Linear(2866, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# The above network is `Net1` in version 1.1. I've added dropout, because I've heard it's good to use dropout. In case of this network, I didn't see much of an improvement, but maybe I've done the benchmark wrong. The problem might lay in my usage of the dropout. I don't know the theory behind it, but I think that it's better to use bigger probability of dropout with bigger number of neurons. So I think that I might be better to use `p=0.5` in the first dropout and `p=0.2` in the second. However, I've come up with this idea, right now, so I haven't tested it out. You'll see that in `Net4` I've used decaying dropout probability.

# In[ ]:


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(2866, 5932)
        self.fc2 = nn.Linear(5932, 3000)
        self.fc3 = nn.Linear(3000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# Here, the only difference is the number of neurons in the first layer. I wanted to use exactly 2 times more than feature, but while calculating the number I entered 2966.

# In[ ]:


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(2866, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 1000)
        self.fc4 = nn.Linear(1000, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


# `Net4` as mentioned before has decaying probability of dropout. Additionally, I've added one more layer in comparison to previous architectures. I think that because of this, it might need a little more epochs to learn, because it has worse results than other architectures after the same number of epochs.
# 
# You might have noticed, that I skipped `Net3`. We'll check it out after `Net5`.

# In[ ]:


class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(2866, 5732)
        self.fc2 = nn.Linear(5732, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# In `Net5` I wanted to use as little layers as I could. I wanted this design to overfit less. Unfortunately, it wasn't also providing as good results as other architectures.

# In[ ]:


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(2866, 6000)
        self.fc2 = nn.Linear(6000, 5000)
        self.fc3 = nn.Linear(5000, 4000)
        self.fc4 = nn.Linear(4000, 2000)
        self.fc5 = nn.Linear(2000, 500)
        self.fc6 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)


# And here we have it! The monstrosity called `Net3`. I think it has way too many layers and it overfits faster than other networks (in comparison to it's score on verification dataset). In the end, I haven't used this network in my final submission, but we will train it anyway - just so we can see the results.

# In[ ]:


def train(log_interval, model, device, train_loader, optimizer, epoch, verbose=False):
    model.train()
    dataset_len = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and verbose:
            print('Train Epoch:', epoch)
            print('\t{}/{}  -  {:.0f}%'.format(batch_idx * len(data), dataset_len,
                                               100 * batch_idx / len(train_loader)))
            print('\tLoss: {:.4f}'.format(loss.item()))


# In[ ]:


def test(model, device, test_loader, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    dataset_len = len(test_loader.dataset)
    test_loss /= dataset_len
    if verbose:
        print("Testing:")
        print("\tLoss: {:.4f}".format(test_loss))
        print("\tAccuracy: {}/{}  -  {:.3f}%".format(correct, dataset_len,
                                               100 * correct / dataset_len))
    return test_loss, correct


# In[ ]:


def conduct_experiment(model, device, epochs, log_interval, train_loader, optimizer, verification_loader):
    train_loss = []
    verify_loss = []
    train_scored = []
    verify_scored = []

    for ep in range(epochs):
        train(log_interval, model, device, train_loader, optimizer, ep + 1)
        ts, t = test(model, device, train_loader)
        vl, c = test(model, device, verification_loader)
        train_loss.append(ts)
        verify_loss.append(vl)
        train_scored.append(t)
        verify_scored.append(c)

    print("Trainset:")
    test(model, device, train_loader, True)
    print("Verifyset:")
    test(model, device, verification_loader, True)
    plt.plot(train_loss)
    plt.title('Train loss')
    plt.show()
    plt.plot(verify_loss)
    plt.title('Verify loss')
    plt.show()
    plt.plot(train_scored)
    plt.title('Train scored')
    plt.show()
    plt.plot(verify_scored)
    plt.title('Verify scored')
    plt.show()


# The above functions are quite abvious. `train()` trains, `test()` tests. I'm using `conduct_experiment()`, becuase I'm training several networks in this notebook. Additionally, I used this function to test some of my ideas, find the best hyperparameters and so on.
# 
# The only thing that is worth mentioning here is `correct += pred.eq(target.view_as(pred)).sum().item()` from `test()` function. I've seen it used in MNIST example code and it is a kind of magic to me. If I were to write it myself I wouldn't be able to do it in only a single line of code (which you could see before white creating `data_y` and `data_y2`).
# 
# I would like to mention, that I don't remember how to write the `train()` and `test()` functions, so I will have to look up this code everytime I have to write it again.

# In[ ]:


train_x, verify_x, train_y, verify_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)


# In the above line I create train and verification datasets. `random_state` is set to 42 as it is **the best** number to use everywhere. I think you should also use it **everywhere**.
# 
# You might be thinking, why I'm using training and verification datasets instead of using cross validation. The answer is simple: the idea of cross validation didn't cross my mind. I think this is a mistake and in the future I will certainly use this method. Fortunately, I don't think this had much impact on my score.

# In[ ]:


test_x = vect_x.transform(testset.ingredients).todense()
ids = testset.id.values


# Here, I prepared input for test dataset. I think that it's important to note, that I've used `vect_x` to do this. This is `CountVectorizer` trained on `dataset.ingredients` - data we use for training and verification. I think this is very important, because in the beginning I was using new `CountVectorizer`. Since test dataset contains different ingredeints, the dimension of `test_x` doesn't match with the size of input accepted by my networks. It might seem obvious to you, but it wasn't for me at the time. So, the very important thing I've learnt is: (it doesn't apply only to neural networks, but to the whole machine learnig field)
# > `testset` might contain different values than `trainset`. What is more, in the `testset` there might be values which your model has never seen. You should always take that into consideration while designing your solutions to machine learnig problems.

# In[ ]:


class FoodTrain(torchdata.Dataset):
    def __init__(self, ingredients, cuisine):
        self.ingredients = torch.tensor(ingredients, dtype=torch.float32)
        self.cuisine = torch.tensor(cuisine, dtype=torch.int64)

    def __len__(self):
        return len(self.ingredients)

    def __getitem__(self, index):
        return self.ingredients[index], self.cuisine[index]


# In[ ]:


class FoodTest(torchdata.Dataset):
    def __init__(self, dataset, idx):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.index = torch.tensor(idx, dtype=torch.int32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.index[index]


# In[ ]:


train_loader = torchdata.DataLoader(FoodTrain(train_x, train_y), batch_size=train_batch, shuffle=True, **kwargs)
verification_loader = torchdata.DataLoader(FoodTrain(verify_x, verify_y), batch_size=test_batch, shuffle=True, **kwargs)
test_loader = torchdata.DataLoader(FoodTest(test_x, ids), batch_size=test_batch, shuffle=False, **kwargs)


# In the above code, I've created two classes. The first one for training and verification, the second one for generating submission. Then, I've created `DataLoader`s with these data. There is one **very** important thing to mention. In the train loader, I've used `train_x` and `train_y` - only the part of the dataset. While it's the right thing thing to do while working on my answer, I shouldn't have done that while generating submission. And yes, **I did it**. I think this is the biggest mistake I have made. After the competition, I've checked the result of networks trained on whole dataset. It was `0.81244` instead of `0.80616`. I have learnt a lesson:
# > While creating the submission, relearn your network on the whole dataset provided.

# Right now, we can move on to the training. I won't discuss anyting network specific this time, because all the networks are trained the same way. The only difference is the number of epochs I used for each network for the submission. For this demonstartion, I'm using 10 epochs for each network, so you can see the results more clearly. You can find the number of epochs used for each training in the comments. I've choosen the numbers by looking at the plots. Unfortunately, I can't tell how I was choosing the correct sopt. I was just going with the gut feeling. I don't know, if it was good idea, because this way I might have overfitted my solution to the verification dataset, which is not very good. But since it was only the number of epochs, I don't think it had bad impact on the final score.
# 
# As you can see, I'm using `Adagrad` optimizer with learnig rate equal to `0.01`. In the beginning I was using `torch.optim.SGD` with `momentum=0.5`. I was quite happy about that, but then I tested `Adam`, `Adadelta` and `Adagrad`. I was quite surprised how good these optimizers were working in comparison to simple `SGD`. I got not only the improvement in learing speed, but also in accuracy. All in all, `Adagrad` appeared to be the best. I wasn't really fine-tuning the learing rate. I only compared three values: `0.1`, `0.01`, and `0.001`. The value in the middle appeared to be the best of them.
# 
# You might ask, why there isn't any L2 normalization. The answer is simple, I couldn't make it work. I boldly starter with `0.2`, but I got stuck at 20% score on **both** training and verification datasets. Then, I tried `0.02` and `0.002`. The latter was better, but I had to increas number of epochs and learning rate. All in all, I wasn't able to achieve as good results on verification dataset with L2 as without it, so I decided not to use it. Because of this you might notice that my networks are **overfitted AF**. The good thing is, that it works, the bad, that it can work better I think. I plan on working a little bit more and this problem, so I can finally achieve better result with L2 than without it.
# 
# The last thing before training: The first network has different seed than the others. I hoped that it would make the network better, but results of this single network looked better with seed `42`.

# In[ ]:


torch.manual_seed(2137)
model1 = Net1().to(device)
optimizer = optim.Adagrad(model1.parameters(), lr=0.01)
# 10 epochs in final submission
conduct_experiment(model1, device, 10, log_interval, train_loader, optimizer, verification_loader)


# In[ ]:


torch.manual_seed(42)
model1_1 = Net1_1().to(device)
optimizer = optim.Adagrad(model1_1.parameters(), lr=0.01)
# 7 epochs in final submission
conduct_experiment(model1_1, device, 10, log_interval, train_loader, optimizer, verification_loader)


# In[ ]:


model2 = Net2().to(device)
optimizer = optim.Adagrad(model2.parameters(), lr=0.01)
# 4 epochs in final submission
conduct_experiment(model2, device, 10, log_interval, train_loader, optimizer, verification_loader)


# In[ ]:


model4 = Net4().to(device)
optimizer = optim.Adagrad(model4.parameters(), lr=0.01)
# 8 epochs in final submission
conduct_experiment(model4, device, 10, log_interval, train_loader, optimizer, verification_loader)


# In[ ]:


model5 = Net5().to(device)
optimizer = optim.Adagrad(model5.parameters(), lr=0.01)
# 5 epochs in final submission
conduct_experiment(model5, device, 10, log_interval, train_loader, optimizer, verification_loader)


# So, since we've trained the networks used to predict the submission, let's look, how my little monster (`Net3`) works:

# In[ ]:


model3 = Net3().to(device)
optimizer = optim.Adagrad(model3.parameters(), lr=0.01)
conduct_experiment(model3, device, 10, log_interval, train_loader, optimizer, verification_loader)


# As you can see, it's nowhere near the result given by the rest of the nets. That's why I think this isn't good arhcitecture. Because of this, I haven't used it in my final submission.
# 
# I think we can finally predict the values for test dataset. I have used mean of all the predictions made by my models. I may have done it differently and tried to create another model, which would learn wights for all the models, but I was too lazy to do it. Additionally, I don't think whether it would change much in this case.

# In[ ]:


model1.eval()
model1_1.eval()
model2.eval()
model4.eval()
model5.eval()


# In[ ]:


test_loss = 0
correct = 0
with torch.no_grad():
    for data_batch, target in verification_loader:
        data_batch, target = data_batch.to(device), target.to(device)
        output1 = model1(data_batch)
        output1_1 = model1_1(data_batch)
        output2 = model2(data_batch)
        output4 = model4(data_batch)
        output5 = model5(data_batch)
        outputs = torch.stack((output1, output1_1, output2, output4, output5))
        output = torch.mean(outputs, 0)
        test_loss += F.nll_loss(output, target, reduction='sum')
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

dataset_len = len(verification_loader.dataset)
test_loss /= dataset_len
print("Average loss: {:.4f}".format(test_loss))
print("Accuracy: {}/{}  -  {:.3f}%".format(correct, dataset_len, 100 * correct / dataset_len))


# As you can see, even though, the networks are **overfitted AF**, they have rather high score on verification dataset. The last thing left to do is genreating the submission.

# In[ ]:


result = pd.DataFrame({'Id': [], 'cuisine': []}, dtype=np.int32)


# In[ ]:


with torch.no_grad():
    for data_batch, idx in test_loader:
        data_batch = data_batch.to(device)
        output1 = model1(data_batch)
        output1_1 = model1_1(data_batch)
        output2 = model2(data_batch)
        output4 = model4(data_batch)
        output5 = model5(data_batch)
        outputs = torch.stack((output1, output1_1, output2, output4, output5))
        output = torch.mean(outputs, 0)
        pred = output.argmax(dim=1)
        result = pd.concat([result, pd.DataFrame({'Id': idx.cpu(), 'cuisine': pred.cpu()}, dtype=np.int32)])


# In[ ]:


def choose(sthg):
    return cuisine[sthg]


# In[ ]:


result['cuisine'] = result['cuisine'].apply(choose, 0)


# And that's my solution. It was quite a journey for me. I think I learnt a lot during this task. And I hope you did too, while reading this notebook. Especially from my mistakes. As I've mentioned before, I wil try to create a better solution with L2 normalization. We'll see how it ends.
# 
# I appreciate all the feedback and hints on my notebook and/or solution. The main purpose of this is for me to learn, so I'm open to the criticism.
