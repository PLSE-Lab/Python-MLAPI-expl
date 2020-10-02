# By Shapedsundew9 31-Dec-2016
# NB: Takes about 10 min to run
# There are lots of good kernels and posts exploring the Titanic dataset. I
# learnt a lot from these:
# https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic
# https://www.kaggle.com/c/titanic/prospector#208
# My goals were:
#   a) Learn python
#   b) Learn a bit about tensorflow
#   c) Learn how to create & submit a Kaggle kernel
# I have tried to document some of my learnings with regards to tensorflows
# quirks (one thing I have learnt is that this excellent library does need some
# time to harden). I am new to python so please forgive my naive code -
# feedback actively encouraged.
# Finally I have not really tried to optimise my results. I have made a few
# notes on future tinkerings should anyone care.

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# Little function to make normalizer lambda functions for real feature
# normalisation. The mormalizer output is in the range [-1.0, 1.0]
def makeNorm(mx, mn): return lambda x: ((2.0 * (x - mn) / (mx - mn)) - 1.0)


# Helper function to remove multiple items from a dictionary avoiding errors
def delDictItems(d, l): return {k: d[k] for k in d if k not in l}


# Helper function to remove items from a list with a list of indices
def delListItems(l, ind): return [i for j, i in enumerate(l) if j not in ind]


# Create an input function to the requirements of tensorflow contrib
# https://www.tensorflow.org/tutorials/input_fn/
def gen_input_fn(c, r, df, t=None): return lambda: create_input_fn(c, r, df, t)


# Helper function to split fields like ticket number and cabin number into
# a set of single digit columns that can be treated categorically
def numStr(name, tmp):
    maxDigits = int(tmp[name].str.len().max())
    tmp[name] = tmp[name].str.zfill(maxDigits)
    df = pd.DataFrame()
    for i in range(0, maxDigits):
        col = name + '_Digit_' + str(i)
        df[col] = tmp[name].str[i].fillna('10')
    return df


# A generalised version of the tensorflow input_fn. I use a function generator
# to remove the parameters. (See gen_input_fn)
def createInputs(field, cats, reals, df):
    # Another quirk of tensorflow is that the target category of a classifier
    # cannot be a str or a float i.e. it has to be an int
    # http://stackoverflow.com/questions/37769860/tensorflow-sklearn-deep-neural-network-classifier-type-error
    # It is also irksome NaN is not supported by pandas for int as that
    # gives us an annoying combination of support gaps. Here I dispose of the
    # NaN's (the data set we need to predict) then convert to int if the
    # target is a category column
    allColumns = list(cats.keys()) + list(reals.keys())
    predictDataSet = df.copy()[allColumns][df[field].isnull()]
    dataSet = df.copy()[allColumns].dropna()
    if field in cats:
            c = cats[field]
            dataSet[field] = dataSet[field].replace(c, range(0, len(c)))
            dataSet[field] = dataSet[field].astype(int)
    # The target column (field) must not be part of the input so I make
    # sure it is expunged here
    cats = delDictItems(cats, [field])
    reals = delDictItems(reals, [field])
    # A useful little pandas function to extract a random subset of samples
    # based just on a fractional amount. I have fixed the random seed so the
    # results are reproducable and extendable.
    trainDataSet = dataSet.sample(frac=0.70, random_state=2)
    testDataSet = dataSet.loc[~dataSet.index.isin(trainDataSet.index)]
    train_fn = gen_input_fn(cats, reals, trainDataSet,
                            trainDataSet[field].values)
    test_fn = gen_input_fn(cats, reals, testDataSet,
                           testDataSet[field].values)
    predict_fn = gen_input_fn(cats, reals, predictDataSet)
    print("    Training set: {:4d} samples".format(len(trainDataSet.index)))
    print("    Test set:     {:4d} samples".format(len(testDataSet.index)))
    print("    Predict set:  {:4d} samples".format(len(predictDataSet.index)))
    return {'train': [train_fn, trainDataSet],
            'test': [test_fn, testDataSet],
            'predict': [predict_fn, predictDataSet]}


# An generic input function that conforms to tensorflows return requirements
# df is a pandas dataframe consisting of columns of real and or category data
# real is a dict of tuples where the key is df
# column names. For real the tuple is made up of two floating point values
# the first the maximum and the second the minimum to normalise the column to.
# cat is a dict where the key is also df column names. The value is the
# complete list of possible category names.
# target is a single list of target values.
def create_input_fn(cat, real, df, target=None):
    featColMap = {k: tf.constant(df[k].values) for k in real}
    catColMap = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)], values=df[k].values,
      shape=[df[k].size, 1]) for k in cat}
    featColMap.update(catColMap)
    if target is not None:
        target = tf.constant(target)
    return featColMap, target


# Create the feature columns
# This is vanilla from the tutorial
# https://www.tensorflow.org/tutorials/wide/
def featCols_fn(cat, real):
    featCols = [tf.contrib.layers.real_valued_column(
                k, normalizer=makeNorm(real[k][0], real[k][1]))
                for k in real]
    catCols = [tf.contrib.layers.embedding_column(
               tf.contrib.layers.sparse_column_with_keys(
                column_name=k, keys=cat[k], combiner='sqrtn'),
               dimension=int(np.trunc(np.log2(len(cat[k]))) + 2),
               combiner='sqrtn') for k in cat]
    featCols.extend(catCols)
    return featCols


# The model function to be passed to the estimator.
# I learnt how to define one of these here:
# https://www.tensorflow.org/tutorials/estimators/
# It is parameterised through 'params' but I never really got time to play
# with them.
#    "learning_rate": Learning rate for the optimizer
#    "categories": Category columns
#    "continuous": Reals columns
#    "hidden": List of ints defining the number and size of hidden layers
#    "output": The number of outputs from the DNN
#    "optimizer": The optimizer to use (any valid string from the docs)
def model_fn(features, labels, mode, params):
    featCols = featCols_fn(params["categories"], params["continuous"])
    input_layer = tf.contrib.layers.input_from_feature_columns(
     columns_to_tensors=features, feature_columns=featCols)
    previousLayer = input_layer
    for i in params["hidden"]:
        layer = tf.contrib.layers.fully_connected(
            inputs=previousLayer, num_outputs=i, activation_fn=tf.nn.relu)
        previousLayer = layer
    output_layer = tf.contrib.layers.fully_connected(
        inputs=previousLayer, num_outputs=params["output"],
        activation_fn=None)
    predictions = {"Value": tf.reshape(output_layer, [-1])}
    loss = tf.contrib.losses.mean_squared_error(predictions["Value"], labels)
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss, global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"], optimizer=params["optimizer"])
    return predictions, loss, train_op


# Estimate 'field' using cats and reals as the input columns. df has the data.
def estimate(field, cats, reals, df, seed=1):
    # Create the training, test and prediction datsSets. The full data set for
    # estimating the field is just all the rows that do not have a 'NaN' in
    # any of the columns we care about.
    # The training set will be used for training the estimator. A random 65%
    # of the full data set.
    # The test set will be used for early stopping. The other 35% of the full
    # data set.
    # The prediction set is the set of data for which we want to make a
    # prediction
    print("ESTIMATING ", field.upper())
    inputs = createInputs(field, cats, reals, df)
    cats = delDictItems(cats, [field])
    reals = delDictItems(reals, [field])

    # Inspiration for building this estimator came from the tensorflow guide
    # https://www.tensorflow.org/tutorials/estimators/
    # I have not spent much (any) time pondering the best type or configuration
    # for estimating the fares. Very happy to be educated...
    # For an explaination of these hyper-parameters see the definition of the
    # model_fn()
    model_params = {"learning_rate": 0.0001,
                    "categories": cats,
                    "continuous": reals,
                    "hidden": [28, 16],
                    "output": 1,
                    "optimizer": "Adam"}

    # There seems to be a tensorflow buggette here. If I do not run a fit()
    # prior to a predict() I get errors.
    # https://github.com/tensorflow/tensorflow/issues/5548
    # Hence here we check to see if a model has been saved. If it has we train
    # for just 0 steps then get on with the prediction. Not ideal as we are
    # changing the model each time
    if os.path.isdir('./' + field):
        print("    Skipping " + field + " training...")
        nn = tf.contrib.learn.Estimator(
              model_fn=model_fn, params=model_params, model_dir='./' + field)
        nn.fit(input_fn=inputs['train'][0], steps=0)
    else:
        # The juicy bit: We create the model and configure its hyper parameters
        # as detailed above. A validation monitor is added that runs the test
        # set through every 1000 steps to give us a sense of how we are doing.
        # If the test set does not improve (reduce in loss) for 200 steps in a
        # row then we stop, otherwise we keep going until we get to 50000
        # steps. Learnt about monitoring using this guide:
        # https://www.tensorflow.org/tutorials/monitors/
        # You can also inspect the training curve by running
        # 'tensorboard --logdir="."' in your execution directory and going to
        # http://127.0.1.1:6006/ in a browser. I learnt asbout that here:
        # https://www.tensorflow.org/how_tos/summaries_and_tensorboard/
        # NB: Due to (what I would say is an oddness in) the contrib API for
        #  r0.12 you have to specify save_checkpoints_secs=None to use
        # save_checkpoints_steps=N otherwise you get a value error. Also, make
        # sure you set eval_steps to 1 in the validationMonitor otherwise it
        # defaults to infinity. (?)
        print("    Training " + field + " estimator...")
        nn = tf.contrib.learn.Estimator(
              model_fn=model_fn, params=model_params, model_dir='./' + field,
              config=tf.contrib.learn.RunConfig(
               save_checkpoints_steps=1000, save_checkpoints_secs=None,
               tf_random_seed=seed))
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=inputs['test'][0], every_n_steps=1000,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True, early_stopping_rounds=5,
            eval_steps=1)
        nn.fit(input_fn=inputs['train'][0], max_steps=100000,
               monitors=[validation_monitor])

    # So we have our prediction - but should we believe it? Always wise to
    # your answers to lets take a look at a histogram of the field
    # and compare that to a histogram of our DNN estimates
    testDataSet = inputs['test'][1]
    sanity_fn = gen_input_fn(cats, reals, testDataSet)
    print("    Making prediction for " + field + "...")
    sanity = nn.predict(input_fn=inputs['test'][0], as_iterable=False)
    bins = np.arange(0, testDataSet[field].max(),
                     testDataSet[field].max() / 30)
    p = plt.xlim(0, testDataSet[field].max())
    p = plt.xticks(bins, rotation='vertical')
    p = plt.hist(sanity["Value"], alpha=0.5, label="Estimated", bins=bins)
    p = plt.hist(testDataSet[field], alpha=0.5, label="Actual", bins=bins)
    p = plt.title('Comparison of Estimated versus Test Data Actuals for ' +
                  field)
    p = plt.xlabel(field)
    p = plt.ylabel('# Samples')
    p = plt.savefig(field+'Distribution.png')
    p = plt.legend(loc='upper right')
    plt.show()
    plt.clf()

    # The histogram only shows if the distribution looks OK or not. Really we
    # want to see the distribution of error in our estimates
    testDataSet['sanity'] = sanity['Value']
    testDataSet = testDataSet[testDataSet[field] != 0.0]
    fractionError = ((testDataSet[field] - testDataSet['sanity']) /
                     testDataSet[field])
    fractionSample = np.zeros_like(fractionError) + 1.0 / fractionError.size
    avFractionError = ((testDataSet[field] - testDataSet['sanity']).abs() /
                       testDataSet[field]).sum() / len(testDataSet.index)
    fracVeryWrong = (len(fractionError[fractionError.abs() > 1.0].index) /
                     float(len(fractionError.index)))
    fracClose = (len(fractionError[fractionError.abs() < 0.1].index) /
                 float(len(fractionError.index)))
    bins = np.arange(-1.0, 1.0, 0.1)
    p = plt.xlim(-1.0, 1.0)
    p = plt.xticks(bins, rotation='vertical')
    p = plt.hist(fractionError, weights=fractionSample, bins=bins)
    p = plt.text(0, 0.090,
                 "Av. fractional error = {:1.2f}".format(avFractionError))
    p = plt.text(0, 0.085,
                 "Fraction > +/-1.0 error = {:1.2f}".format(fracVeryWrong))
    p = plt.title('Distribution of Percentage Error for Estimated ' +
                  field + ' on Test Data')
    p = plt.xlabel('Percentage Error in ' + field)
    p = plt.ylabel('Fraction of Samples')
    p = plt.savefig(field+'PercentageErrorDistribution.png')
    plt.show()
    plt.clf()

    # Out to the console too
    print("    Average fractional error: {:1.2f}".format(avFractionError))
    print("    Fraction of samples > +/-1.0 error: {:1.2f}"
          .format(fracVeryWrong))
    print("    Fraction of samples < +/-0.1 error: {:1.2f}"
          .format(fracClose))

    # Now we have a trained DNN we can make a prediction on the missing fare
    # NB: I had trouble with predict(as_iterable=True). I seemed to end up with
    # an endless list. Documentation for
    # tf.contrib.learn.Estimator.predict(*args, **kwargs) from here:
    # https://www.tensorflow.org/api_docs/python/contrib.learn/estimators#Estimator
    # indicates this is a result of not setting num_epochs=1 but my tensorflow
    # fu is not up to knowing where to do that...answers on a postcard.
    prediction = nn.predict(input_fn=inputs['predict'][0], as_iterable=False)
    df = inputs['predict'][1]
    df[field] = prediction['Value']
    return df[field]


# Classify 'field' using cats and reals as the input columns. df has the data.
def classify(field, cats, reals, master, seed=1):
    print("CLASSIFYING ", field.upper())
    inputs = createInputs(field, cats, reals, master)

    # Whilst technically it is fine to have the target column in with the
    # input columns for the feature_columns mapping (as it is not used) it does
    # cause a type error in the case of category columns because
    # sparse_column_with_keys() in the featCols_fn forces the data values to
    # string but the keys remain whatever type they were and you get a type
    # error. Hence it is expunged here.
    cats2 = delDictItems(cats, [field])
    reals2 = delDictItems(reals, [field])

    # If training has already happened don't do it again (see estimate() for
    # why we have to do a fit with 0 steps - which incidently seems to result
    # in 1 training step from watching the logging when set to INFO)
    if os.path.isdir('./' + field):
        print("    Skipping " + field + " training...")
        nn = tf.contrib.learn.DNNClassifier(
            feature_columns=featCols_fn(cats2, reals2),
            hidden_units=[28, 16], model_dir='./' + field)
        nn.fit(input_fn=inputs['train'][0], steps=0)
    else:
        # I have used a fixed random seed for training for reproducibility
        # (same in the estimator run_config()). I did notice I got quite
        # varied results before I fixed the seed. I have not tried to a good
        # once due to time.
        print("    Training " + field + " classifier...")
        nn = tf.contrib.learn. DNNClassifier(
            feature_columns=featCols_fn(cats2, reals2),
            hidden_units=[28, 16], model_dir='./' + field,
            config=tf.contrib.learn.RunConfig(
             save_checkpoints_steps=100, save_checkpoints_secs=None,
             tf_random_seed=seed))
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=inputs['test'][0], every_n_steps=100,
            early_stopping_metric="accuracy",
            early_stopping_metric_minimize=False, early_stopping_rounds=5,
            eval_steps=1)
        nn.fit(input_fn=inputs['train'][0], monitors=[validation_monitor])
    testDataSet = inputs['test'][1]
    sanity_fn = gen_input_fn(cats, reals, testDataSet)

    # So to calculate accuracy I think I should be able to do this:
    # accuracy = nn.evaluate(input_fn=sanity_fn)["accuracy"]
    # which I got from https://www.tensorflow.org/tutorials/tflearn/
    # but when I do I get this:
    # ValueError: Missing loss.
    # Run out of time to work out why
    testDataSet['p'] = list(nn.predict(input_fn=sanity_fn, as_iterable=False))
    correct = testDataSet[testDataSet['p'] == testDataSet[field]]
    score = len(correct.index) / float(len(testDataSet.index))
    print("    Accurancy on test set: {:1.2f}".format(score))
    # Make our prediction
    prediction = nn.predict(input_fn=inputs['predict'][0],
                            as_iterable=False)

    # Convert our target category column back into the original format (see
    # createInputs() for why we have to do this)
    df = inputs['predict'][1]
    df[field] = prediction
    df = df.replace(range(0, len(cats[field])), cats[field])
    return df[field]


# There are several columns that are sparsely populated in the titanic
# dataset. To get a good prediction of survival rates we can help ourselves
# by filling the gaps in the columns that are only missing a few values.
# In ths script I use tensorflow to create DNNs
# to classify and estimate the missing values. I then create another DNN to
# make the prediction (classification of survived or not)
# To populate the missing values we can use the training and test sets as a
# master set.
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
master = pd.concat([train, test], ignore_index=True)

# Set tensorflow logging level. In API r0.12 I do get warnings from internal
# functions regarding deprecated options and forced conversions. Inspecting
# the code confirms there is nothing that can be done from the contrib
# API level and they do not appear to impact performance of the DNNs
# used here. However, I would be careful using this code with tensorflow
# versions other than r0.12
tf.logging.set_verbosity(tf.logging.ERROR)

# ====== Name ======
# Here we extract the title of the passenger. Title gives a hint at status
# (class) it also suggests age which is a sparse column that we will have to
# estimate.
regex = '(?P<N>[a-zA-Z ,\.\'-]*)?(\(.*\))?'
tmp = master['Name'].str.extract(regex, expand=False)
regex = '(?P<Surname>.*)(?P<Delimiter>, )(?P<Title>.*\.)(?P<Forename>.*)'
master = pd.concat([master, tmp['N'].str.extract(regex, expand=False)], axis=1)
del master['Delimiter']

# ====== Siblings, Spouses, Parents and Children ======
# Now there is probably a correlation within families for survival. If the
# children did not make it it is unlikely that the parents did etc. We can
# infer family relationships through surname and the SibSp and Parch columns.
# This is not 100% accurate as if there were two families with the surname
# "Smith" were have now combined thier fates...but lets assume that is a
# small factor. Below we calculate a Surname Survival Rate, SSR.
# NB: I think this is a poor way to do it in hindsight. I am now thinking
# of creating a father, mother and N sibling categorical columns populated with
# X = Not applicable, U = unknown, S = Survived, D = Died would be better
master['SSR'] = 0.0
families = master.loc[(master['SibSp'] > 0) & (master['Parch'] > 0), 'Surname']
uf = families.value_counts()
uf = uf[uf > 1]
for i in uf.index:
    st = master.loc[master['Surname'] == i, 'Survived']
    sr = 0.0
    if st.notnull().sum() != 0:
        sr = st.sum() / (st.notnull().sum() * 4.0 / 3.0) + 0.25
    master.loc[master['Surname'] == i, 'SSR'] = sr

# ====== Tickets ======
# What can we get from tickets? Well people who have the same ticket number
# are likely closely bonded so there may well be a correlation in survival
# rates between people with the same ticket number.
# The 'Ticket Survival Rate', TSR, is a value from 0.25 to 1.0 which linearly
# equates to the % of survivors for that ticket group 0.25 = 0%, 1.0 = 100%
# Have avoided using 0.0 as that represents no data / not a group.
# NB: I think this is a poor way to do it in hindsight. I am now thinking
# of creating N roommate categorical columns populated with
# X = Not applicable, U = unknown, S = Survived, D = Died would be better
master['TSR'] = 0.0
ut = master['Ticket'].value_counts()
ut = ut[ut > 1]
for i in ut.index:
    st = master.loc[master['Ticket'] == i, 'Survived']
    sr = 0.0
    if st.notnull().sum() != 0:
        sr = st.sum() / (st.notnull().sum() * 4.0 / 3.0) + 0.25
    master.loc[master['Ticket'] == i, 'TSR'] = sr

# Tickets are also grouped by number. I am guessing this would be related to
# the ticket office that were purchased from but it could be some other
# grouping. Some tickets have a prefix but they are varied and sparse.
# I am going to break up tickets into several categorical columns. One
# for the prefix (with some inconsistent formatting removed) and the label
# 'No Prefix' where there was none. I also break the ticket number into
# digits which get a column each. 10 represents no number.
regex = "(?P<Prefix>[^0-9].* )?(?P<TicketNumber>[0-9]*)"
tmp = master['Ticket'].str.extract(regex, expand=False)
master['Prefix'] = tmp['Prefix'].fillna('No Prefix').str.replace('[ \.\/]', '')
master = pd.concat([master, numStr('TicketNumber', tmp)], axis=1)

# ====== Cabin ======
# The cabin data is sparsely populated but there may be some useful
# correlations as it is an indication of location.
# If an individual has multiple rooms I will just look at the first room
# As with tickets I have broken out the deck and each digit of the room number
# into its own categorical column. 10 in a digit column means there was no room
# Z for the deck means that no deck was recorded.
regex = "(?P<Deck>[A-Z])(?P<Number>[0-9]*)"
tmp = master['Cabin'].str.extract(regex, expand=False)
master['Deck'] = tmp['Deck'].fillna('Z')
tmp['Number'][tmp['Number'].str.len() == 0] = np.NaN
master = pd.concat([master, numStr('Number', tmp)], axis=1)

# ====== Type conversions ======
# We are treating both Pclass as a category. Tensorflow requires
# it to be string in that case
master['Pclass'] = master['Pclass'].astype(str)
master[['Parch', 'SibSp']] = master[['Parch', 'SibSp']].astype(float)

# Now that we have pre-processed the data we need to convert it into
# tensorflow feature columns. Data is either a category (labeled) or
# real (continuous). Here we assign each data column a type and provide a
# little bit of meta-data to help with the creation of feature columns.
# For categorical data I pass a list of all the category values.
# For real data a 2 element list of the [max, min] range for that column
cats = {'Embarked': master['Embarked'].dropna().unique().tolist(),
        'Sex': master['Sex'].dropna().unique().tolist(),
        'Title': master['Title'].dropna().unique().tolist(),
        'Pclass': master['Pclass'].dropna().unique().tolist(),
        'Survived': master['Survived'].dropna().unique().tolist(),
        'Prefix': master['Prefix'].dropna().unique().tolist(),
        'Deck': master['Deck'].dropna().unique().tolist()}
reals = {'Age': [master['Age'].max(), 0.0],
         'Fare': [master['Fare'].max(), 0.0],
         'Parch': [master['Parch'].max(), 0],
         'SibSp': [master['SibSp'].max(), 0.0],
         'SSR': [1.0, 0.0],
         'TSR': [1.0, 0.0]}
for col in master.columns:
    if 'Digit' in col:
        cats[col] = [str(i) for i in range(0, 11)]

# Some of our chosen columns have missing data (NaN's in the dataframe) which
# includes our ultimate goal of 'Survived'. The general approach here is to
# populate the missing data starting with the most populous columns (the ones
# with least data missing) first up to an including our target column of
# 'Survived'.

# ====== FARE =======
# There is one fare missing (for passenger ID 1044). Total unnecessary overkill
# to create a DNN estimator for it but I am here to learn python and tensorflow
# as well so...
# To predict the missing fare there are a few things that are unlikely to
# matter so we disregard them from our input data. Note that one could quite
# rightly argue 'age' is a factor but there are so many ages missing that I
# think the dataset is better off without it and a coarse proxy for age is
# 'Title' (though 'Miss' is a bit iffy)
# There is unlikely to be enough information here to get a decent estimate
nCats = delDictItems(cats, ['Survived'])
nReals = delDictItems(reals, ['Age', 'SSR', 'TSR'])
master.update(estimate('Fare', nCats, nReals, master, seed=30))

# ====== EMBARKED ======
# There are two missing embarkation points. In this case we use a classifier.
master.update(classify('Embarked', nCats, nReals, master, seed=12))

# ====== AGE ======
# There are lots of ages missing in the dataset and not much to go on to
# give a good idea of what they should be...hey ho...prepare for iffy results
ageCats = delDictItems(cats, ['Survived'])
ageReals = delDictItems(reals, ['SSR', 'TSR'])
master.update(estimate('Age', ageCats, ageReals, master, seed=19))

# ====== SURVIVED ======
# Why we are here...
master.update(classify('Survived', cats, reals, master, seed=9))

# Output to CSV in the correct format
master['Survived'] = master['Survived'].astype(int)
master.to_csv('fullDataSet.csv', index=False)
submission = master[['PassengerId', 'Survived']]
submission = submission[submission['PassengerId'] > 891]
submission.to_csv('titanic.csv', index=False)

# Ta da