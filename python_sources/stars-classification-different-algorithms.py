#!/usr/bin/env python
# coding: utf-8

# # Exoplanet Hunting in Deep Space
# ## Classification of Kepler labelled time series data
# ***Francesco Pudda, 21/05/2020***

# ### Abstract
# An exoplanet or extrasolar planet is a planet outside the Solar System. For centuries scientists, philosophers, and writers suspected that extrasolar planets existed, but with the technology of the time there was no way of knowing that or how similar they might have been to the planets of the Solar System. The first possible evidence of an exoplanet was noted in 1917, but was not recognized as such. The first confirmation of detection occurred in 1992 and hundreds and hundreds more exoplanets have been confirmed since then. As of May, 16 2020, 4266 exoplanets have been confirmed [1][nexo]. The discovery of exoplanets has made interest for extraterrestrial life rise up. In particular, special interest is paid to planets that orbit in a star's habitable zone, where it is possible for liquid water to exist on the surface [2][life]. In this project I am going to show some characteristics of stars with exoplanet and some possible models to classify them.
# 
# [nexo]: exoplanet.eu/catalog/
# [life]: www.nytimes.com/2015/01/07/science/space/as-ranks-of-goldilocks-planets-grow-astronomers-consider-whats-next.html

# ## Introduction
# This was project was carried out by using the dataset provided at Kaggle [3][data]. The dataset was recorded by the space telescope Kepler during its last mission before malfunctioning, but some stars from previous missions were added to increase the number of stars with confirmed exoplanets given the huge class imbalance of this dataset.
# 
# Libraries used for this project are common ones. I used <i>Pandas</i> to easily import the published <i>.csv</i> files, <i>SciPy</i> for time signal processing, <i>MatplotLib</i> for plotting graphs, and finally <i>SciKit-Learn</i> and <i>Keras</i> respectively for machine and deep learning.
# 
# [data]: www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

# In[ ]:


from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score, make_scorer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


# As last preliminary step I define a function that will be used to plot the confusion matrix. <i>SciKit-Learn</i> already provides such function that could work for the <i>Keras</i> models too. Unfortunately, the original code checks whether the passed estimator is a <i>SciKit-Learn</i> model, making it crash with a <i>Keras</i> one.

# In[ ]:


# Function for plotting the confusion matrix. It makes use of the Sklearn class ConfusionMatrixDisplay
def plot_confusion_matrix(cm, labels):
    display_labels = labels
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=display_labels)
    return display.plot(include_values=True,
                        cmap='viridis', ax=None, xticks_rotation='horizontal',
                        values_format=None)


# ### Importing datasets

# In[ ]:


rawTrain = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
rawTest = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')


# ### Inspection and visualisation
# 
# Let's now print and check dataset structure and column types.

# In[ ]:


print(rawTrain.head())
print(rawTrain.dtypes)


# It can be seen that time signals are the rows of the dataframe whereas columns are the time samples stored as <i>float</i>. First column stores, on the other hand, class labels as <i>int</i>. Measure units are not provided so it was necesarry to have a look at Kaggle discussions forum [4][disc] and Wikipedia's page about Kepler telescope [5][kepl]. From there, I was able to find the sampling time interval $1765.5 s$, from which I calculated the time span of the signals, roughly sixty-five days. Regarding the y-axis measure unit, in Kaggle it is stated it is <i>flux</i> without specifying any further (maybe supposing that insiders would know it precisely). Although not important as far as classification is concerned, in this context flux has units of $W m^{-2}$ [6][flux] and probably such signals are given in logarithmic scale even if I was not able to confirm it.
# 
# I am now splitting the datasets based on this piece of information and converting it to a numpy array, more practical for numerical elaboration. Labels are also converted to boolean. According to the forum <i>'2'</i> was the label for <i>confirmed exoplanet</i> so I set it to <i>True</i> and <i>'1'</i> viceversa.
# 
# [disc]: www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/discussion
# [kepl]: en.wikipedia.org/wiki/Kepler_space_telescope
# [flux]: lonewolfonline.net/luminosity-flux-stars/

# In[ ]:


train_orig = rawTrain.iloc[:, 1:].to_numpy()
test_orig = rawTest.iloc[:, 1:].to_numpy()
train_y = rawTrain.iloc[:, 0].to_numpy() == 2
test_y = rawTest.iloc[:, 0].to_numpy() == 2


# Now a couple of stars from the two classes will be plotted to see how they look like and get an idea about preprocessing and classification steps. I will be also computing the time vector given the sampling interval provided above.

# In[ ]:


x_t = np.array(list(i*1765.5 for i in range(train_orig.shape[1]))) / 60 / 60 / 24

fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle("Stars comparison")
np.random.seed(4)
axs[0,0].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[1,0].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[0,0].set_title("Confirmed exoplanet")
axs[1,0].set_xlabel("Time (day)")
axs[0,0].set_ylabel("Star flux")
axs[1,0].set_ylabel("Star flux")
axs[0,0].grid()
axs[1,0].grid()
np.random.seed(3)
axs[0,1].plot(x_t, train_orig[np.random.choice(np.where(~train_y)[0]), :])
axs[1,1].plot(x_t, train_orig[np.random.choice(np.where(~train_y)[0]), :])
axs[0,1].set_title("No exoplanet")
axs[1,1].set_xlabel('Time (day)')
axs[0,1].grid()
axs[1,1].grid()
plt.show()


# Finally, it was reported in the dataset description that classes were unbalanced, so I need to know how much to decide metrics and/or further preprocessing steps.

# In[ ]:


print("Proportion of confirmed exoplanet in training set: %0.3f%%" %(sum(train_y)/len(train_y)*100))
print("Proportion of confirmed exoplanet in test set: %0.3f%%" %(sum(test_y)/len(test_y)*100))
print("Cconfirmed exoplanet in training set: %d" %sum(train_y))
print("Confirmed exoplanet in test set: %d" %sum(test_y))


# We can see that dataset is hugely unbalanced in favor of (fittingly) stars without exoplanet and that absolute number of positive samples is so low that it might be necessary some resampling strategy to increase. This will be decided later.

# ### Preprocessing
# 
# As seen in the image above, the scale of the signals are very different one another so, for classification's sake, it is important to scale them to Z-Score units to make them coherent. In addition, some stars present constant or linear trends during their time span. These will not bring information to the classification process and may actually even hinder it. So, I am also going to remove them by dividing every signal in thirty equals parts and detrending them piecewisely. As segments increase the more detrending will be affected by noise and might <i>over-average</i> the signal loosing pieces of information, on the other hand with few segments detrending may not be precise because it could affect segments with different trends (e.g. quick ups and downs).
# It is important to state that generaly such preprocessing step are carried out over the columns, on the other hand, I need to perform them over the rows since these are time signals.

# In[ ]:


train_time = scale(ss.detrend(train_orig, bp=np.linspace(0, train_orig.shape[1], 50, dtype=int)), axis=1)
test_time = scale(ss.detrend(test_orig, bp=np.linspace(0, test_orig.shape[1], 50, dtype=int)), axis=1)

fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle("Before (top) and after (bottom) scaling and detrending")
np.random.seed(4)
axs[0, 0].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[0, 1].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[0, 0].set_ylabel('Star flux')
axs[0, 0].grid()
axs[0, 1].grid()
np.random.seed(4)
axs[1, 0].plot(x_t, train_time[np.random.choice(np.where(train_y)[0]), :])
axs[1, 1].plot(x_t, train_time[np.random.choice(np.where(train_y)[0]), :])
axs[1, 0].set_ylabel('Normalised star flux')
axs[1, 0].set_xlabel('Time (day)')
axs[1, 1].set_xlabel('Time (day)')
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()


# Although detrending and scaling went good, and signals are now centered around zero, time representation may not be a very good candidate for classification. Next step will then be to compute the power spectrum density of each signal and filter it. The idea behind filtering is that dimmings in flux come at a certain period which can be very different in range, but each one happens relatively quickly and we must be able to capture it, hence the necessity to keep some high frequencies.
# The filter will be an IIR bandpass butterworth filter of eighth order with cutoff frequency $0.00001 Hz$ and $0.00013 Hz$. The reason for this filter is that I prefer a less sloped transition band than ripples in the passing or stopping band (see [7][butter] and [8][cheby] for more information). The window choice is pretty standard in signal processing because the passband and stopband ripple size are not affected by the window length but by how the coefficients roll off (more at [9][kaiser]). Finally I used <i>filtfilt</i> function to apply the filter twice, forwards and backwards, in order to have a zero phase filter; for this reason I had to set the filtering order to four so that applied twice yieds an eighth order one.
# 
# [butter]: https://en.wikipedia.org/wiki/Butterworth_filter
# [cheby]: https://en.wikipedia.org/wiki/Chebyshev_filter
# [kaiser]: https://www.quora.com/Why-is-kaiser-window-superior-to-other-window-functions?share=1

# In[ ]:


x_f, train_freq = ss.periodogram(train_time, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)
test_freq = ss.periodogram(test_time, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)[1]

fig, axs = plt.subplots(2, 2)
fig.suptitle("Power spectrum density of two stars")
np.random.seed(4)
_next = np.random.choice(np.where(train_y)[0])
axs[0, 0].plot(x_t, train_time[_next, :])
axs[1, 0].plot(x_f, train_freq[_next, :])
axs[0, 0].set_title('With exoplanet')
axs[0, 0].set_xlabel('Time')
axs[1, 0].set_xlabel('Frequency')
axs[0, 0].set_ylabel('Star flux')
axs[1, 0].set_ylabel('Power per frequency')
axs[0, 0].grid()
axs[0, 1].grid()
_next = np.random.choice(~np.where(train_y)[0])
axs[0, 1].plot(x_t, train_time[_next, :])
axs[1, 1].plot(x_f, train_freq[_next, :])
axs[0, 1].set_title('No exoplanet')
axs[0, 1].set_xlabel('Time')
axs[1, 1].set_xlabel('Frequency')
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()


# In[ ]:


b, a = ss.butter(N=4, Wn=(0.00001, 0.00013), btype='bandpass', fs=1 / 1765.5)
train_time_filt = ss.filtfilt(b, a, train_time, axis=1)
test_time_filt = ss.filtfilt(b, a, test_time, axis=1)
train_freq_filt = ss.periodogram(train_time_filt, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)[1]
test_freq_filt = ss.periodogram(test_time_filt, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)[1]

fig, axs = plt.subplots(2, 2)
fig.suptitle("Time and spectrum comparison before and after filtering")
np.random.seed(4)
_next = np.random.choice(np.where(train_y)[0])
axs[0, 0].plot(x_t, train_time[_next, :])
axs[1, 0].plot(x_f, train_freq[_next, :])
axs[0, 0].set_title('Original')
axs[0, 0].set_xlabel('Time')
axs[1, 0].set_xlabel('Frequency')
axs[0, 0].set_ylabel('Star flux')
axs[1, 0].set_ylabel('Power per frequency')
axs[0, 0].grid()
axs[0, 1].grid()
axs[0, 1].plot(x_t, train_time_filt[_next, :])
axs[1, 1].plot(x_f, train_freq_filt[_next, :])
axs[0, 1].set_title('Filtered')
axs[0, 1].set_xlabel('Time')
axs[1, 1].set_xlabel('Frequency')
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()


# ## Classification phase
# 
# It is now time to start the classification phase. The reason why I haven't provided any statistical analysis in the previous section is because I am about to classify time signals over rows and not columns. Every row is thus completely independent from the others and there is no meaning in computing, for example, a correlation matrix between columns. So feature engineering in terms of columns selection is not possible here because significative time ranges for a star classification may be useless for another. Only meaningful features engineering is signal processing, like computing the power spectrum or filtering as I did above.
# 
# So first of all, what makes a star worth considering the possibility of an exoplanet? Imagine a star with a planet orbiting around it. When the planet gets in between the camera and the star the perceived luminosity dims and we see dropdowns in the star flux signal and the star is considered a candidate for further evaluation. Classifiers could then dicriminate whether a star has an exoplanet by looking at such drops as in the image above. A further evaluation criterion can definetely be the periodicity of such dimmings. A planet does, in fact, orbit with a fixed orbital period and will then dim the flux at periodic intervals. The way to look for periodic patterns is to get the spectral estimates of the signals and use it as features. In this regard another fact must be pointed out. As already said, signals span over approximately only sixty-five days due to malfuctioning. Thus, it will be impossible to find periodic patterns or even a single dimming if the orbital period is greater than such time span.

# ### Support vector machine
# 
# First estimator that will be tried is <i>Support Vector Machine</i>. It is a quite popular estimator that perform pretty well in most case and it is often used for time series classification as in in the popular MatLab toolbox Brainstorm where it implemented by default [10][SVM].
# 
# There is one major problem that could ruin the accuracy of the models: the huge imbalance. Positive samples are, in fact, less than 1% in both training and test set. Unfortunately little can be done in this regard. Oversampling the positive samples may cause overfitting, on the other hand down sampling may bring to underfitting. There are also mixed technique that were not tried out because final results were overall good.
# 
# <b>NOTE:</b> The dataset is not that huge but the number of features is, so the process will be very slow. I ran the original script on my machine with a wider range of hyperparameters but for the sake a clean representation and a fast execution time I kept only the best one preserving the logic.
# 
# [SVM]: https://neuroimage.usc.edu/brainstorm/Tutorials/Decoding

# In[ ]:


svc_estimator = SVC(tol=1e-4, class_weight='balanced', max_iter=1e+5, random_state=1)

svc_tuning_grid = {"C": (0.12,)}


# Now I am going to create the crossvalidation instances using a stratified shuffle split. Shuffle splits were chosen so as to decide the ratio of randomly sample the stars to keep in the validation set and the number of iterations indipendently one another. Given the huge number of features, I had to use as much of the training set as possible and keep the validation set size to a minimum. Considering that there are only thirty-seven positive samples I couldn't opt for less then ten percent, though, yielding three or four positive samples for each iteration.
# 
# In addition, since <i>Scikit-Learn</i> estimators' fit function doesn't return a copy of the instance but <i>self</i> I used <i>deepcopy</i> to create several instances the the cross-validation one.

# In[ ]:


svc_cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
svc_time_gridcv = GridSearchCV(estimator=svc_estimator, param_grid=svc_tuning_grid,
                               scoring='f1', cv=svc_cv, return_train_score=True)

svc_time_filt_gridcv = deepcopy(svc_time_gridcv)
svc_freq_gridcv = deepcopy(svc_time_gridcv)
svc_freq_filt_gridcv = deepcopy(svc_time_gridcv)


# Finally let's fit our different signals to check which one may provide the best model and print results.

# In[ ]:


svc_time_model = svc_time_gridcv.fit(train_time, train_y)
svc_time_filt_model = svc_time_gridcv.fit(train_time_filt, train_y)
svc_freq_model = svc_freq_gridcv.fit(train_freq, train_y)
svc_freq_filt_model = svc_time_gridcv.fit(train_freq_filt, train_y)


# In[ ]:


print("Time signal best parameters: %s" %svc_time_model.best_params_)
print("Filtered time signal best parameters: %s" %svc_time_filt_model.best_params_)
print("Frequency signal best parameters: %s" %svc_freq_model.best_params_)
print("Filtered frequency signal best parameters: %s" %svc_freq_filt_model.best_params_)


# In[ ]:


print("Time model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_time_model.cv_results_['mean_test_score'][0],
                                                svc_time_model.cv_results_['std_test_score'][0]))

print("-----\nFiltered time model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_time_filt_model.cv_results_['mean_test_score'][0],
                                                svc_time_filt_model.cv_results_['std_test_score'][0]))

print("-----\nFrequency model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_freq_model.cv_results_['mean_test_score'][0],
                                                svc_freq_model.cv_results_['std_test_score'][0]))

print("-----\nFiltered frequency model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_freq_filt_model.cv_results_['mean_test_score'][0],
                                                svc_freq_filt_model.cv_results_['std_test_score'][0]))


# It is clear that none of the above estimators is a reliable one, in fact they all yield very different scores on different iterations often yielding same results. This is surely due to the very small positive sample size causing very different f1 scores if only one sample is wrongly classified. This step was not useless anyway, in fact, we understood that filtering this data didn't bring anything good to classification, so I won't be using filtered signals in later processing stages.

# ### Principal component analysis
# 
# It may be that the poorly reproducible results in the previous step are caused by the huge amount of features and small sample size. It may be a good idea then to try to reduce them via PCA or SVD.

# In[ ]:


pca_time = PCA(svd_solver='full').fit(train_time)
pca_freq = PCA(svd_solver='full').fit(train_freq)


# In[ ]:


plt.plot(np.cumsum(pca_time.explained_variance_ratio_), label='Time PCA')
plt.plot(np.cumsum(pca_freq.explained_variance_ratio_), label='Freq. PCA')
plt.xlabel("Number of components")
plt.ylabel("% of variance explained")
plt.legend()
plt.show()


# In[ ]:


plt.plot(range(150,600), np.cumsum(pca_freq.explained_variance_ratio_)[150:600])
plt.xlabel("Number of components")
plt.ylabel("% of variance explained")
plt.title("Frequency PCA knee")
plt.show()


# Interestingly, the explained variance raises much quicker in the frequency domain than in the time domain. We would barely need 200 components to have more than 90% of explained variance in contrast to approximately 2000. Part of such difference surely lies in the fact that the frequency domain diplays half the features of the other, but the shape of the curve is much more steeper anyway. That will be our candidate in next classifications steps. Given the above graph I arbitrarily choose as many components to obtain 95% of explained variance.

# In[ ]:


train_pca = PCA(n_components=0.95).fit_transform(train_freq)
print("Number of PCA components to get 95%%: %d" %train_pca.shape[1])


# In[ ]:


pca_cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
pca_gridcv = GridSearchCV(estimator=svc_estimator, param_grid=svc_tuning_grid, scoring='f1',
                          cv=pca_cv, return_train_score=True)
pca_model = pca_gridcv.fit(train_pca, train_y)


# In[ ]:


print("\nMean score: %.3f\tSt. dev.: %.3f" %(pca_model.cv_results_['mean_test_score'][0],
                                             pca_model.cv_results_['std_test_score'][0]))


# Execution time was several tens times faster thanks to the reduced features and average score is precisely the same. This means that in all likely the score is higly dependent on the random number generator.

# ### Deep learning networks
# 
# We saw that SVC performs pretty good even though not outstandingly. It could be possible then to extend the scikit-learn possibilities by using the Keras deep learning library able to build efficient and complex neural networkd. The approach will be similar to the previous one. I am going to to build a model and tune its hyperparameters by the combine use of <i>Keras + SciKit-Learn</i> cross-validation techniques. Unfortunately I will not be able to build very complex model that were reported to be very good for time series classification [11][nnmod] because they are too slow to run on my machine and almost impossible on IBM Watson Studio.
# 
# Training will be a little bit trickier here. Indeed, I need to try different networks that must be built separately, each one with different hyper parameters. I am going to use the <i>KerasClassifier</i> class that will let me take advantage of scikit handy crossvalidation functions.
# 
# [nnmod]: https://arxiv.org/abs/1809.04356

# In[ ]:


def train_dl_model(x, y, network, tuning_grid=dict()):
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
    grid_searcher = GridSearchCV(estimator=network,
                                 param_grid=tuning_grid,
                                 cv=splitter,
                                 scoring='f1',
                                 return_train_score=True)
    np.random.seed(1)
    model = grid_searcher.fit(x, y,
                              batch_size=32,
                              epochs=50,
                              verbose=0)
    return model


# In[ ]:


# Source: https://stackoverflow.com/a/45305384/4180457

from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


def build_mlp(n_units):
    model = Sequential()
    model.add(Dropout(0.1))
    model.add(Dense(n_units, input_shape=nn_shape))
    model.add(Dropout(0.2))
    model.add(Dense(n_units))
    model.add(Dropout(0.2))
    model.add(Dense(n_units))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1])
    return model


# Let's try the multi-layer perceptron with the pca transformed data and the frequency one.

# In[ ]:


nn_shape = train_freq[1:]

mlp_tuning_grid = {"n_units": (300,)}

mlp_freq_model = KerasClassifier(build_fn=build_mlp)
mlp_freq_model = train_dl_model(train_freq, train_y, mlp_freq_model, mlp_tuning_grid)


# In[ ]:


nn_shape = train_pca[1:]

mlp_pca_model = KerasClassifier(build_fn=build_mlp)
mlp_pca_model = train_dl_model(train_pca, train_y, mlp_pca_model, mlp_tuning_grid)


# In[ ]:


print("Frequency domain best parameters: %s" %mlp_freq_model.best_params_)
print("PCA doman best parameters: %s" %mlp_pca_model.best_params_)


# In[ ]:


print("Frequency domain model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(mlp_freq_model.cv_results_['mean_test_score'][0],
                                                mlp_freq_model.cv_results_['std_test_score'][0]))

print("-----\nPCA domain model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(mlp_pca_model.cv_results_['mean_test_score'][0],
                                                mlp_pca_model.cv_results_['std_test_score'][0]))


# Interestingly, even though SVC performed decently over the pca components, it seems that they are completely unusable with neural networks. On the other hand the network over the frequency domain signals performs decently but yet worse than SVC.

# ## Final model
# 
# Given the results above I decided to use the SVC classifier with the PCA features as final model.

# In[ ]:


final_model = SVC(C=0.12, tol=1e-4, class_weight='balanced', max_iter=1e+5,
                  random_state=1).fit(train_freq_filt, train_y)
predictions = final_model.predict(test_freq_filt)


# In[ ]:


false_negatives = [i for i, v in enumerate(test_y) if v and test_y[i] != predictions[i]]
false_positives = [i for i, v in enumerate(test_y) if not v and test_y[i] != predictions[i]]
print("False negatives: %d" %len(false_negatives))
print("False positives: %d" %len(false_positives))


# In[ ]:


print("f1-score:  %.3f" %f1_score(test_y, predictions))
print("Balanced accuracy:  %.3f" %balanced_accuracy_score(test_y, predictions))

plot_confusion_matrix(confusion_matrix(test_y, predictions, labels=[True, False]), labels=[True, False])


# ## Discussion
# 
# This classification task has been very hard. The fact that classification was about time series excluded every typical feature engineering technique given that we had to consider the whole signals. I was only able to filter or reduce dimensionality to try different models. Such number of features might have probably caused overfitting that I tried to reduct by regularization, and keeping the training set size as large as possible during cross-validation.
# 
# Except for the neural network with PCA components, similar cross-validation scores were found by using SVC or MLP with different features, but their variability was so high to make me think that they were completely sample dependendent and so dependent on the randomicity of the cross-validation procedure.
# 
# Even if not shown in the report I also tried out more techniques like different kind of filtering that changed performances little no zero (and if so, in negative), and simultaneous over-/undersampling techniques like <i>SMOTENN</i> in the <i>imbalanced-learn</i> library that did not help but increasing overfitting. I also tried more networks but power demands was too high to run them on my laptop.
# 
# Given the small number of falses it would be interesting to inspect them to see why the classifier have wrongly labelled them.

# In[ ]:


fig, axs = plt.subplots(1, len(false_negatives), sharey=True)
fig.suptitle("Time series false negatives")
for i in range(len(false_negatives)):
    axs[i].plot(x_t, test_time[false_negatives[i],:])
axs[0].set_xlabel('Time')
axs[1].set_xlabel('Time')
axs[0].set_ylabel('Star flux')
axs[0].grid()
axs[1].grid()
plt.show()


# In[ ]:


fig, axs = plt.subplots(1, len(false_negatives), sharey=True)
fig.suptitle("Frequency domain false negatives")
for i in range(len(false_negatives)):
    axs[i].plot(x_f, test_freq_filt[false_negatives[i],:])
axs[0].set_xlabel('Frequency')
axs[1].set_xlabel('Frequency')
axs[0].set_ylabel('Power per frequency')
axs[0].grid()
axs[1].grid()
plt.show()


# In conclusion using the power spectrum density as feature combined with SVC or FNN looks like an encouraging approach to classifify stars with exoplanet given also the very few positive samples the classifiers had to learn from. Unfortunately I was not able to visually discriminate why some stars were wrongly labelled, but considering that signals were pretty noisy and computational resources were very limited to not let me try more complex models or more sophisticated hyperparameters tuning, I consider this model a good starting point in pointing out that frequency
