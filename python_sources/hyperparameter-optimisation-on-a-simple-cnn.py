#!/usr/bin/env python
# coding: utf-8

# In this notebook I'll run through the following topics:
# - Loading and preparing MNIST handwritten digits data.
# - Building and training a simple CNN to recognise the digits.
# - Checking the model/training performance with 5-fold cross validation.
# - Implementing and comparing 3 methods of hyperparameter optimisation: grid search, random search, Bayesian Optimisation
# 
# I haven't optimised this notebook for easy reading or a wide audience. You might like to use this notebook if:
# - You are hunting for code snippets in general
# - You want to see the comparison between the three hyperparameter optimisation methods with painstakingly running it yourself.
# 
# #### **Full disclosure:**
# _Before doing this exercise I had never applied deep learning in any practical sense, and only had a surface level understanding of what it is. This notebook could be useful for those of you just starting out. At the same time, take everything you see with a pinch of salt. The starter code for this project was borrowed from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/_

# # 1. Import libraries and load data
# 
# - I use Keras on Tensorflow for building and training the model.
# 
# - I use Hyperopt for hyperparameter optimisation
# 
# - The dataset used here can be found at https://www.kaggle.com/oddrationale/mnist-in-csv

# In[ ]:


# IMPORTS

# basic utility libraries
import numpy as np
import pandas as pd
import time
import datetime
import pickle
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

# keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.models import load_model

# learning and optimisation helper libraries
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, Trials, hp, rand
from hyperopt.pyll.stochastic import sample


# LOAD DATA

def load_dataset():
    train_set = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
    test_set = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')
    # training features, normalized and reshaped
    trainX = np.array(train_set.iloc[:, 1:])
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    # testing features, normalized and reshaped
    testX = np.array(test_set.iloc[:, 1:])
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # training features: extract the labels, reshape, and one hot encode
    trainY = train_set.label
    trainY = np.array(trainY).reshape(-1, 1)
    trainY = to_categorical(trainY)
    # test features: extract the labels, reshape, and one hot encode
    testY = test_set.label
    testY = np.array(testY).reshape(-1, 1)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# load dataset (60k training samples, and 10k test samples)
load_trainX, load_trainY, load_testX, load_testY = load_dataset()


# # 2. Prepare functions for training and optimisation
# 
# Some functions worth calling out:
# 
# - **define_model** - This is where we define the architecture of our CNN. I've decided to stick to a fairly simple one in order to limit computational resources required.
# 
# - **evaluate_model_cross_validation** - This is for running k-fold cross validation. We take the training data, shuffle it, and split it into k groups. We train a new model on k-1 groups, and validate it on the remaining group. We do this k times (starting with freshly initialized model weights each time), giving each group a chance to be the validation group. Then we take an average over the validation accuracies of each trial to get an estimate for the performance of the model/training parameters.
# 
# - **grid_search** - We trial a preset selection of hyperparameter combinations, evaluating the performance of the model with each run. This lets us get a coarse map of model performance over hyperparameter space. In the interest of saving time and resources we only do single evaluations of the model at each selected point in hyperparameter space. Although given more time and resources, we could do k-fold cross validation.
# 
# - **selecive_search** - We run either Random Search, or Bayesian Optimisation over a selected range of hyperparameters. 
#   - Random search involves randomly selecting hyperparameters in the given range and trialling them. 
#   - Bayesian Optimisation on the other hand, uses knowledge of the performance of previous trials to construct probabilisitic surrogate functions for the real objective function (or put simply - it guesses the relationship between hyperparameter choice and model perormance). It then uses this knowledge to make a best guess for which hyperparameters to choose next.

# In[ ]:


#### CUSTOM CALLBACK

class PlotProgress(Callback):
    def on_train_begin(self, logs={}):
        self.fig, self.ax1, self.ax2 = self.prepare_plot()
        self.loss_history = list()
        self.val_loss_history = list()
        self.accuracy_history = list()
        self.val_accuracy_history = list()
        
    # plot diagnostic learning curve
    def prepare_plot(self):
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title('Cross Entropy Loss')
        ax2.set_title('Accuracy')
        plt.tight_layout()
        return fig, ax1, ax2

    def update_plot(self):
        # plot loss
        self.ax1.plot(self.loss_history, color='blue', label='train_acc')
        self.ax1.plot(self.val_loss_history, color='orange', label='val_acc')
        # plot accuracy
        self.ax2.plot(self.accuracy_history, color='blue', label='train_acc')
        self.ax2.plot(self.val_accuracy_history, color='orange', label='val_acc')
        self.fig.canvas.draw()

    def on_epoch_end(self, epoch, logs=None):
        self.loss_history.append(logs.get('loss'))
        self.val_loss_history.append(logs.get('val_loss'))
        self.accuracy_history.append(logs.get('accuracy'))
        self.val_accuracy_history.append(logs.get('val_accuracy'))
        self.update_plot()

        
        
#### PREPROCESSING
            
# this is actually meant to be called by prep_data
def prep_pixels(dataset):
    dataset_norm = dataset.astype('float32')
    dataset_norm = dataset_norm / 255.0
    return dataset_norm


def prep_data(training_size=0):
    global load_trainX, load_trainY, load_testX, load_testY
    trainX, trainY, testX, testY = np.copy(load_trainX), np.copy(load_trainY), np.copy(load_testX), np.copy(load_testY)
    trainX, testX = prep_pixels(trainX), prep_pixels(testX)
    if training_size == 0:
        training_size = trainX.shape[0]
    trainX = trainX[0:training_size]
    trainY = trainY[0:training_size]
    return trainX, trainY, testX, testY



#### DEFINE MODEL AND TRAINING/EVALUTION METHODS

def define_model(learning_rate, momentum):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(trainX, trainY, testX, testY, max_epochs, learning_rate, momentum, batch_size, model=None, callbacks=[]):
    if model == None:
        model = define_model(learning_rate, momentum)
    history = model.fit(trainX, trainY, epochs=max_epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0, callbacks = callbacks)
    return model, history


def evaluate_model_cross_validation(trainX, trainY, max_epochs, learning_rate, momentum, batch_size, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for trainFold_ix, testFold_ix in kfold.split(trainX):
        # select rows for train and test
        trainFoldsX, trainFoldsY, testFoldX, testFoldY = trainX[trainFold_ix], trainY[trainFold_ix], trainX[testFold_ix], trainY[testFold_ix]
        # fit model
        model = define_model(learning_rate, momentum)
        history = model.fit(trainFoldsX, trainFoldsY, epochs=max_epochs, batch_size=batch_size, validation_data=(testFoldX, testFoldY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testFoldX, testFoldY, verbose=0)
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories



#### VISUALISATION OF TRAINING OUTCOME

def summarize_diagnostics(histories):
    fig, (ax1, ax2) = plt.subplots(2,1)
    for i in range(len(histories)):
        # plot loss
        ax1.set_title('Cross Entropy Loss')
        ax1.plot(histories[i]['loss'], color='blue', label='train_loss')
        ax1.plot(histories[i]['val_loss'], color='orange', label='val_loss')
        # plot accuracy
        ax2.set_title('Accuracy')
        ax2.plot(histories[i]['accuracy'], color='blue', label='train_acc')
        ax2.plot(histories[i]['val_accuracy'], color='orange', label='val_acc')
    fig.canvas.draw()
    
    
    
#### HYPERPARAMETER OPTIMIZATION

def grid_search(trainX, trainY, testX, testY, max_epochs, learning_rates, momentums, batch_sizes):
    hyperparameter_sets, scores = list(), list()
    callbacks = [PlotProgress()]
    i = 1
    total_runs = len(learning_rates) * len(momentums) *  len(batch_sizes)
    running_time = 0
    start = time.time()
    for lr in learning_rates:
        for momentum in momentums:
            for bs in batch_sizes:
                if i > 1:
                    time_remaining = running_time/(i-1)*(total_runs-(i-1))
                    print('{} of {} done so far so far in {}s. Estimated time remaining: {}s'.format(i-1, total_runs, running_time, time_remaining))
                print('Starting run {} of {}'.format(i, total_runs))
                print('Evaluating Hyperparameters: learning_rate: {}, momentum: {}, batch_size: {}'.format(lr, momentum, bs))
                accuracies, _ = evaluate_model_cross_validation(trainX, trainY, max_epochs=max_epochs, learning_rate=lr, momentum=momentum, batch_size=bs, n_folds=5)
                score = np.log10(1 - np.mean(accuracies))
                scores.append(score)
                with open('grid_scores.pickle', 'wb') as file:
                    pickle.dump(scores, file)
                hyperparameter_sets.append({'learning_rate': lr, 'momentum': momentum, 'batch_size': bs})
                with open('grid_hpsets.pickle', 'wb') as file:
                    pickle.dump(hyperparameter_sets, file)
                running_time = time.time() - start
                i+=1
    return hyperparameter_sets, scores


def selective_search(kind, space, max_evals, batch_size=32, plot=False):
    
    trainX, trainY, testX, testY = prep_data()
    
    try:
        hyperparameter_sets = pickle.load(open('{}_hpsets.pickle'.format(kind), 'rb'))
    except:
        hyperparameter_sets = list()
    try:
        scores = pickle.load(open('{}_scores.pickle'.format(kind), 'rb'))
    except:
        scores = list()

    if plot:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('{} search'.format(kind))
        ax.set_xlabel('# Iteration')
        ax.set_ylabel('log(1 - accuracy)')
        
    def objective(params):
        lr, momentum = params['lr'], params['momentum']
        if plot:
            ax.set_title('Evaluating: lr: {:0.4f}, momentum: {:0.4f}, bs: {}'.format(lr, momentum, batch_size))
        accuracies, _ = evaluate_model_cross_validation(trainX, trainY, max_epochs=1, learning_rate=lr, momentum=momentum, batch_size=batch_size, n_folds=5)
        score = np.log10(1 - np.mean(accuracies))
        scores.append(score)
        with open('{}_scores.pickle'.format(kind), 'wb') as file:
            pickle.dump(scores, file)
        hyperparameter_sets.append({'learning_rate': lr, 'momentum': momentum, 'batch_size': batch_size})
        with open('{}_hpsets.pickle'.format(kind), 'wb') as file:
            pickle.dump(hyperparameter_sets, file)
        if plot:
            x = range(len(scores))
            ax.plot(x, scores, 'bo--')
            fig.canvas.draw()
        return score
    
    def run_trials(kind, objective, space, max_evals):
        evals_step = 20

        try:
            trials = pickle.load(open('{}_trials.pickle'.format(kind), 'rb'))
            max_evals = len(trials.trials) + evals_step
            print('Starting new set of trials')
        except:
            trials = Trials()

        if kind == 'bayesian':
            best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
        elif kind == 'random':
            best = fmin(fn=objective, space=space, algo=rand.suggest, trials=trials, max_evals=max_evals)
        else:
            raise BaseError('First parameter "kind" must be either "bayesian" or "random"')

        with open('{}_trials.pickle'.format(kind), 'wb') as file:
            pickle.dump(trials, file)

    while True:
        run_trials(kind, objective, space, max_evals)
        
    return histories, hyperparameter_sets, scores


# # 3. Do a one time training run
# 
# Let's just pick some hyperparameters and try to do a training run. In this example we'll use:
# 
# - A batch size of 32
# - Learning rate of 1e-2
# - Momentum of 0.9
# - 10 epochs
# 
# (Bottom axis on the charts is for epoch #)

# In[ ]:


trainX, trainY, testX, testY = prep_data()
_, history = evaluate_model(trainX, trainY, testX, testY, 10, learning_rate = 1e-2, momentum=0.9, batch_size = 32, callbacks = [PlotProgress()])
print('Final validation accuracy is: {:0.2f}%'.format(100 * history.history['val_accuracy'][-1]))


# # 4. Try K-Fold cross validation
# 
# Let's try 5-fold cross validation with the same hyperparameters as above to get a sense for the standard deviation and average. We'll do it for 1 epoch (we'll end up using this later) and 10 epochs.

# In[ ]:


trainX, trainY, testX, testY = prep_data()
scores_1ep, _ = evaluate_model_cross_validation(trainX, trainY, max_epochs=1, learning_rate=1e-2, momentum=0.9, batch_size=32, n_folds=5)
scores_10ep, _ = evaluate_model_cross_validation(trainX, trainY, max_epochs=10, learning_rate=1e-2, momentum=0.9, batch_size=32, n_folds=5)
plt.boxplot([scores_1ep, scores_10ep], labels=['1 epoch','10 epochs'])
plt.title('Box whisker plot of accuracies')
plt.ylabel('Accuracy')
plt.show()


# # 5. Grid search to map the terrain
# 
# Now we run a coarse grid search over hyperparameters to get a rough idea of where our good values might be. Later on we will be running lengthy optimisation algorithms, so in the interest of saving time we need to limit ourselves to 1 epoch which as we will see, still provides meaningful results. We'll stick with a batch size of 32 while searching over learning rate and momentum. 
# 
# We'll also need to apply 5-fold cross-validation to drown out the noise. It's not entirely necessary for this grid search as the spread is small enough such that the results would still be informative, but it will be needed later for the Bayesian Optimization run.
# 
# I will not actually run the code in this notebook, but instead I'll show the results from what I ran on a Kaggle kernel using a GPU for the heavy lifting. Note that even though in some regimes it looks like our minimum might fall outside of the grid search range, I've checked those areas offline and found that the score got worse.
# 
# What explains the shifting of the minimum as momentum changes you might ask? In a stable configuration, momentum tends to accelerate the convergence towards a minimum, therefore allowing for smaller learning rates to be used. On the other hand, if learning rate is maintained as momentum increases, the system tends to become unstable. Try this interactive simulation to get a feeling for it https://distill.pub/2017/momentum/

# In[ ]:


with open('../input/hyperparameter-optimisation-on-a-cnn/grid_scores.pickle', 'rb') as file:
    scores = pickle.load(file)
    
with open('../input/hyperparameter-optimisation-on-a-cnn/grid_hpsets.pickle', 'rb') as file:
    hp_sets = pickle.load(file)

results_df = pd.DataFrame(hp_sets)
results_df['score'] = scores

learning_rates = [np.sqrt(10)*1e-4, 1e-3, np.sqrt(10)*1e-3, 1e-2, np.sqrt(10)*1e-2, 1e-1]
momentums = [1-np.sqrt(10)*1e-1, 1-1e-1, 1-np.sqrt(10)*1e-2, 1-1e-2, 1-np.sqrt(10)*1e-3]
batch_sizes = [32]

def plot2D_lr(batch_size=32):
    nplts = len(momentums)
    nrows = np.ceil(nplts/2)
    fig = plt.figure(figsize=(8, 3*nrows))
    i = 1
    for momentum in momentums:
        plot_df = results_df[(results_df['batch_size'] == batch_size) & (results_df['momentum'] == momentum)]
        ax = fig.add_subplot(nrows, 2, i)
        x = {'series': np.log10(plot_df['learning_rate']), 'label': 'log(learning_rate)'}
        y = {'series': plot_df['score'], 'label': 'log(1 - acc)'}
        ax.scatter(x['series'], y['series'])
        ax.set_xlabel(x['label'])
        ax.set_ylabel(y['label'])
        ax.set_title('bs = {}, momentum = {}'.format(batch_size, momentum))
        i += 1
    plt.tight_layout()
    plt.show()

def plot2D_momentum(batch_size=32):
    nplts = len(learning_rates)
    nrows = np.ceil(nplts/2)
    fig = plt.figure(figsize=(8, 3*nrows))
    i = 1
    for learning_rate in learning_rates:
        plot_df = results_df[(results_df['batch_size'] == batch_size) & (results_df['learning_rate'] == learning_rate)]
        ax = fig.add_subplot(nrows, 2, i)
        x = {'series': np.log10(1 - plot_df['momentum']), 'label': 'log(1 - momentum)'}
        y = {'series': plot_df['score'], 'label': 'log(1 - acc)'}
        ax.scatter(x['series'], y['series'])
        ax.set_xlabel(x['label'])
        ax.set_ylabel(y['label'])
        ax.set_title('bs = {}, learning_rate = {}'.format(batch_size, learning_rate))
        i += 1
    plt.tight_layout()
    plt.show()
    
plot2D_lr()


# # 6. Random search
# 
# We run random search for 1000 iterations over the same ranges in the grid search (but now the ranges are continuous instead of discrete). This is a control experiment for the subsequent Bayesian Optimization experiment. Notice that the scores tend to vary evenly throughout the runs, and that the spread of trialled hyperparameters stays even throughout the experiment.

# In[ ]:


# this is the function we'll use to plot the results of the random search as well as the bayesian optimization
def plot_selective_search_results(scores, hpsets):
    scores = scores[:1000]
    hpsets = hpsets[:1000]
    fig = plt.figure(figsize=(18, 25))
    x = range(len(scores))
    ax = fig.add_subplot(6,1,1)
    ax.plot(x, scores, 'b.--')
    poly = np.poly1d(np.polyfit(x, scores, 1))
    ax.plot(x, poly(x), 'r-')
    ax.set_title('1. Score over iterations')
    ax.set_xlabel('# iterations')
    ax.set_ylabel('log10(1 - validation_accuracy)')
    ax.legend(labels=['signal', 'fit: {:0.1E}*x {:0.1f}'.format(poly[1], poly[0])], loc='upper right')
    
    ax = fig.add_subplot(6,1,2)
    ax.plot(x, np.minimum.accumulate(scores), 'b.')
    ax.set_title('2. Running best score over iterations (best of all was {})'.format(np.min(scores)))
    ax.set_xlabel('# iterations')
    ax.set_ylabel('log10(1 - validation_accuracy)')

    learning_rates = np.log10([hpsets[i]['learning_rate'] for i in x])
    ax = fig.add_subplot(6,1,3)
    ax.plot(x, learning_rates, 'bo')
    ax.set_title('3. Learning rate over iterations')
    ax.set_xlabel('# iterations')
    ax.set_ylabel('log10(learning_rate)')

    bs = 100
    len_series = int(np.ceil(len(scores)/bs))
    batched_learning_rates = [learning_rates[i*bs:i*bs+bs] for i in range(len_series)]
    ax = fig.add_subplot(6,1,4)
    ax.boxplot(batched_learning_rates, positions=np.arange(len_series))
    ax.set_title('4. Box and whisker plots of learning rates for batched iterations')
    ax.set_xlabel('# batch (each batch contains {} iterations)'.format(bs))
    ax.set_ylabel('log10(learning_rate)')

    momentums = np.log10([1 - hpsets[i]['momentum'] for i in x])
    ax = fig.add_subplot(6,1,5)
    ax.plot(x, momentums, 'bo')
    ax.set_title('5. Momentum over iterations')
    ax.set_xlabel('# iterations')
    ax.set_ylabel('log10(1 - momentum)')

    batched_momentums = [momentums[i*bs:i*bs+bs] for i in range(len_series)]
    ax = fig.add_subplot(6,1,6)
    ax.boxplot(batched_momentums, positions=np.arange(len_series))
    ax.set_title('6. Box and whisker plots momentums for batched iterations')
    ax.set_xlabel('# batch (each batch contains {} iterations)'.format(bs))
    ax.set_ylabel('log10(1 - momentum)')

    plt.tight_layout()
    
with open('../input/hyperparameter-optimisation-on-a-cnn/random_scores.pickle', 'rb') as file:
    rand_scores = pickle.load(file) 
with open('../input/hyperparameter-optimisation-on-a-cnn/random_hpsets.pickle', 'rb') as file:
    rand_hpsets = pickle.load(file)

plot_selective_search_results(rand_scores, rand_hpsets)


# # 7. Bayesian optimisation
# 
# Now let's try running Bayesian optimisation . We can make observations about each chart produced below:
# 
# 1. **Score over iterations** and 2. **Running best score over iterations**
# 
#   - We do observe a trend towards better accuracy as seen by the fitted line. This is probably less because the minimum is improving, and more because the algorithm spends less time evaluating hyperparameters which are clearly not candidates for optimal performance.
#   
#   
# 3. **Learning rate over iterations** and 4. **the corresponding box and whisker plots**
#   - A strange thing we see here is convergence and divergence of the trialled hyperparameters. My guess is because the noise from the statistical deviations are not allowing the algorithm to map out the terrain reliably. It can't settle on a minimum because each time it tests a certain set of hyperparameters it gets a slightly different answer.
#   - Nevertheless we do see hints that the algorithm restricts its search space to a narrower neighborhood than the full range, as expected.
#   
#   
# 5. **Momentum over iterations** and 6. **the corresponding box and whisker plots**
#   - Here we make similar observations to the learning rates. What's interesting, is the way the average tends to converge and diverge with the learning rate. Remember I mentioned earlier that as momentum increases, we need to decrease learning rate to maintain good model training performance. So there is somehow a coupling between momentum and learning rate if we try to maintain good performance. This is what the optimisation algorithm is demonstrating for us here!

# In[ ]:


with open('../input/hyperparameter-optimisation-on-a-cnn/bayesian_scores.pickle', 'rb') as file:
    bo_histories = pickle.load(file) 
with open('../input/hyperparameter-optimisation-on-a-cnn/bayesian_hpsets.pickle', 'rb') as file:
    bo_hpsets = pickle.load(file)

plot_selective_search_results(bo_histories, bo_hpsets)


# # 8. Final remarks about hyperoptimisation (in this specific use case)
# 
# Although we were able to demonstrate Bayesain Optimisation in action, the results are far from inspiring. The chart below shows us the running best scores from each of the methods. Bayesian optimisation gets the best results but only beats random search by 0.04% in accuracy. Our course grid seach got 97.8% accuracy in 30 iterations whereas it took Bayesian Optmisation 500 trials to get 97.9% accuracy.
# 
# It's definitely worth noting that the statistical noise may have spoiled the ability of Bayesian Optimisation to do its job.
# 
# In any case, for the practical purposes of this example, it would have been suffiecient to run grid search and Bayesian Optimisation for say 100 iterations each to get a decent idea of how the model might peform.

# In[ ]:


with open('../input/hyperparameter-optimisation-on-a-cnn/grid_scores.pickle', 'rb') as file:
    grid_scores = pickle.load(file)
with open('../input/hyperparameter-optimisation-on-a-cnn/grid_hpsets.pickle', 'rb') as file:
    grid_hpsets = pickle.load(file)  
with open('../input/hyperparameter-optimisation-on-a-cnn/random_scores.pickle', 'rb') as file:
    rand_scores = pickle.load(file)
with open('../input/hyperparameter-optimisation-on-a-cnn/random_hpsets.pickle', 'rb') as file:
    rand_hpsets = pickle.load(file)
with open('../input/hyperparameter-optimisation-on-a-cnn/bayesian_scores.pickle', 'rb') as file:
    bayesian_scores = pickle.load(file) 
with open('../input/hyperparameter-optimisation-on-a-cnn/bayesian_hpsets.pickle', 'rb') as file:
    bayesian_hpsets = pickle.load(file) 

print('Grid best', grid_hpsets[np.argsort(grid_scores)[0]])
print('Random best', rand_hpsets[np.argsort(rand_scores)[0]])
print('Bayesian best', bayesian_hpsets[np.argsort(bayesian_scores)[0]])
    
fig = plt.figure(figsize=(9.5,4))
ax = fig.add_subplot(1,1,1)
ax.plot(range(len(rand_scores)), np.minimum.accumulate(rand_scores), 'r.')
ax.plot(range(len(bayesian_scores)), np.minimum.accumulate(bayesian_scores), 'b.')
ax.plot(range(len(grid_scores)), np.minimum.accumulate(grid_scores), 'g.')
ax.set_title('Running best scores of each method of hyperparameter optimisation')
ax.set_ylabel('log10(1 - val_accuracy)')
ax.set_xlabel('# iteration')
ax.legend(['random search', 'bayesian optimization', 'grid_search'])
ax.set_ylim(-1.7, -1.6)


# In[ ]:




