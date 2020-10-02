#!/usr/bin/env python
# coding: utf-8

# # Market Data Only Baseline
# 
# Market data (2007 to present)
# 
# News data (2007 to present)
# 
# volume(float64) - trading volume in shares for the day
# 
# close(float64) - the close price for the day (not adjusted for splits or dividends)
# 
# open(float64) - the open price for the day (not adjusted for splits or dividends)
# 
# returnsClosePrevRaw1(float64) - return from previous day based on close prices
# 
# returnsOpenPrevRaw1(float64) - return form previous day based on open prices
# 
# returnsClosePrevMktres1(float64) - return from previous day based on close prices, adjusted to market movements
# 
# returnsOpenPrevMktres1(float64) - return from previous day based on open prices, adjusted to market movements
# 
# returnsClosePrevRaw10(float64) - return from previous 10 day based on close prices
# 
# returnsOpenPrevRaw10(float64) -  return from previous 10 day based on open prices
# 
# returnsClosePrevMktres10(float64) -  return from previous 10 day based on close prices, adjusted to market movements
# 
# returnsOpenPrevMktres10(float64) -  return from previous 10 day based on open prices, adjusted to market movements
# 
# returnsOpenNextMktres10(float64) - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy, mse
from keras import backend as K
import tensorflow as tf
import os

session_conf = tf.ConfigProto()
tf.set_random_seed(42)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

os.environ['PYTHONHASHSEED'] = '0'


# In[ ]:


# comment these out for random individuals each time
#np.random.seed(1)
#random.seed(1)

# randomly initialize data
num_samples = 20
data = np.concatenate((
    np.array([random.randrange(2, 10, 2) for _ in range(num_samples)]).reshape(-1,1), 
    np.random.randint(64,128,size=(num_samples, 1)),
    np.random.randint(0,3,size=(num_samples, 1)), # There are 7 different optimization functions
    np.random.randint(0,4,size=(num_samples, 1)), # There are 11 different activation functions
    np.random.randint(0,1,size=(num_samples, 1))), axis=1)
# create dataframe
df = pd.DataFrame(data, columns = ['n_layers','n_nodes','optimization','activation','score'])
df.sort_values('score',ascending=True)


# In[ ]:


# define mutations
def create_mutations(df):    
    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    # select random chromo to mutate
    df_mut = df
    # select random column to mutate
    param_to_mut = df_mut.columns.values[np.random.choice(len(df.columns.values)-1)]
    # identify which param is being mutated
    if param_to_mut == 'activation':
        df.iloc[0][param_to_mut]=random.randint(0, 2) # Can give same value
    if param_to_mut == 'optimization':
        df.iloc[0][param_to_mut]=random.randint(0, 2) # Can give same value
    if param_to_mut == 'n_layers':
        flip = random.randint(0,1)
        if flip == 0:
            df.iloc[0][param_to_mut] += 1
        elif flip == 1 and df.iloc[0][param_to_mut] > 1:
            df.iloc[0][param_to_mut] -= 1
    if param_to_mut == 'n_nodes':
        flip = random.randint(0,1)
        if flip == 0:
            df.iloc[0][param_to_mut] += 2
        elif flip == 1 and df.iloc[0][param_to_mut] > 2:
            df.iloc[0][param_to_mut] -= 2
    return df


# In[ ]:


# define Gaussian mutations
def gaussian_mutations(df, dfc, generation, n_generations):    
    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    # select random chromo to mutate
    df_mut = df
    # select random column to mutate
    param_to_mut = df_mut.columns.values[np.random.choice(len(df.columns.values)-1)]
    print(param_to_mut)
    print('before: ', df.iloc[0][param_to_mut])
    
    # determine stepsize for mutation
    p_success = float(generation)/float(n_generations)
    tau = 0.5 # using formula tau = 1/n^0.5, Rechenberg's 1/5 rule
    if p_success > 0.2:
        tau = 2 #stdev/tau
    elif p_success < 0.2:
        tau = 0.5 #stdev*tau
    elif p_success == 0.2:
        tau = 1 #stdev*1
    
    # identify which param is being mutated
    if param_to_mut == 'activation':
        act_stepsize = dfc[param_to_mut].std() * tau
        act_mean = dfc[param_to_mut].mean()
        act_val = random.gauss(act_mean,act_stepsize)
        act_current = df.iloc[0][param_to_mut]
        act_mod = act_val - act_mean #modifier value
        
        df.iloc[0][param_to_mut] = round(act_current + act_mod) #final val mutated
        
        if df.iloc[0][param_to_mut] < 0:
            df.iloc[0][param_to_mut] = 0
        elif df.iloc[0][param_to_mut] > 3:
            df.iloc[0][param_to_mut] = 3
    if param_to_mut == 'optimization':
        opt_stepsize = dfc[param_to_mut].std() * tau
        opt_mean = dfc[param_to_mut].mean()
        opt_val = random.gauss(opt_mean,opt_stepsize)
        opt_current = df.iloc[0][param_to_mut]
        opt_mod = opt_val - opt_mean #modifier value
        
        df.iloc[0][param_to_mut] = round(opt_current + opt_mod)
        
        if df.iloc[0][param_to_mut] < 0:
            df.iloc[0][param_to_mut] = 0
        elif df.iloc[0][param_to_mut] > 2:
            df.iloc[0][param_to_mut] = 2
    if param_to_mut == 'n_layers':
        lay_stepsize = dfc[param_to_mut].std() * tau
        lay_mean = dfc[param_to_mut].mean()
        lay_val = random.gauss(lay_mean,lay_stepsize)
        lay_current = df.iloc[0][param_to_mut]
        lay_mod = lay_val - lay_mean #modifier value
        
        df.iloc[0][param_to_mut] = round(lay_current + lay_mod)
        
        if df.iloc[0][param_to_mut] < 0:
            df.iloc[0][param_to_mut] = 1
    if param_to_mut == 'n_nodes':
        nod_stepsize = dfc[param_to_mut].std() * tau
        nod_mean = dfc[param_to_mut].mean()
        nod_val = random.gauss(nod_mean,nod_stepsize)
        nod_current = df.iloc[0][param_to_mut]
        nod_mod = nod_val - nod_mean #modifier value
        
        df.iloc[0][param_to_mut] = round(nod_current + nod_mod)
        
        if df.iloc[0][param_to_mut] < 0:
            df.iloc[0][param_to_mut] = 0
            
    print('after: ', df.iloc[0][param_to_mut])
    return df
#display(df)
#test = gaussian_mutations(df,0,5)
#display(test)


# In[ ]:


def uniform_crossover(df):
    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    # select two random chromosomes to perform crossover on
    df_cross = df.sample(n=2).copy()
    # create copy of first individual to be used as offspring
    offspring = df_cross.iloc[[0]]
    offspring.iat[0, df.columns.get_loc("score")] = 0
    # create binary array representing crossover points
    split_bits = np.random.randint(0,2,4)
    
    #no duplicates
    while np.sum(split_bits) == 0 or np.sum(split_bits) == 4: #[0,0,0,0] or [1,1,1,1]
        split_bits = np.random.randint(0,2,4)
        
    # iterate through bit array to perform uniform crossover
    for index, val in enumerate(split_bits):
        if split_bits[index] == 1:
            offspring.iat[0,index] = df_cross.iat[1,index]
    return offspring


# In[ ]:


def breed(df, n_individuals=15, mut_percent=5): 
    for i in range(n_individuals):
        offspring_df = uniform_crossover(df.head())
        if random.randint(1, 100) <= mut_percent:
            create_mutations(offspring_df)
        new_df = [df, offspring_df]
        df = pd.concat(new_df, ignore_index=True)
    print("\n------------- Updated DataFrame -------------\n")
    
    return df


# In[ ]:


def breed_es(df, generation, n_generations, n_individuals=15, mut_percent=20): 
    for i in range(n_individuals):
        offspring_df = uniform_crossover(df)
        display(offspring_df)
        if random.randint(1, 100) <= mut_percent:
            gaussian_mutations(offspring_df, df, generation, n_generations) 
        new_df = [df, offspring_df]
        df = pd.concat(new_df, ignore_index=True)
    print("\n------------- Updated DataFrame -------------\n")
    return df

#test = breed_es(df.head(), 0, 5, mut_percent=20)
#display(df)
#display(test)


# In[ ]:


env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()


# In[ ]:


market_train.head()


# In[ ]:


market_train_subset = market_train.sample(n=30000).copy()
market_train_subset = market_train_subset.reset_index(drop=True)

cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']

market_train_subset.head(20)


# In[ ]:


from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)
subset_indices, subset_val_indices = train_test_split(market_train_subset.index.values, test_size=0.25, random_state=42)

print(subset_indices)


# # Handling categorical variables

# In[ ]:


def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]
subset_encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    subset_encoders[i] = {l: id for id, l in enumerate(market_train_subset.loc[subset_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    market_train_subset[cat] = market_train_subset[cat].astype(str).apply(lambda x: encode(subset_encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets


# # Handling numerical variables

# In[ ]:


from sklearn.preprocessing import StandardScaler
 
market_train[num_cols] = market_train[num_cols].fillna(0)
market_train_subset[num_cols] = market_train_subset[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
market_train_subset[num_cols] = scaler.fit_transform(market_train_subset[num_cols])

market_train_subset.tail()


# In[ ]:


def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d


# In[ ]:


def load_neural_net(df):    
    num_layers = df.iloc[0]['n_layers']
    num_nodes = df.iloc[0]['n_nodes']
    activation = df.iloc[0]['activation']
    optimization = df.iloc[0]['optimization']
    
    if activation == 0:
        activation = 'relu'
    elif activation == 1:
        activation = 'selu'
    elif activation == 2:
        activation = 'sigmoid'
    else:
        activation = 'elu'
    
    if optimization == 0:
        optimization = 'adam'
    elif optimization == 1:
        optimization = 'SGD'
    else:
        optimization = 'adadelta'
    
    print(num_layers)
    print(num_nodes)
    print(activation)
    print(optimization)
    categorical_inputs = []
    for cat in cat_cols:
        categorical_inputs.append(Input(shape=[1], name=cat))
            
    categorical_embeddings = []
    for i, cat in enumerate(cat_cols):
        categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

    categorical_logits = Flatten()(categorical_embeddings[0])
    categorical_logits = Dense(32,activation='relu')(categorical_logits)

    numerical_inputs = Input(shape=(11,), name='num')
    numerical_logits = numerical_inputs
    numerical_logits = BatchNormalization()(numerical_logits)
    
    # ADD LOOP HERE
    for i in range(num_layers):
        numerical_logits = Dense(num_nodes,activation=activation)(numerical_logits)

    logits = Concatenate()([numerical_logits,categorical_logits])
    logits = Dense(64,activation='relu')(logits)
    out = Dense(1, activation='sigmoid')(logits)

    model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
    model.compile(optimizer=optimization,loss=binary_crossentropy)
    
    # r, u and d are used to calculate the scoring metric
    X_train,y_train,r_train,u_train,d_train = get_input(market_train_subset, subset_indices)
    X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train_subset, subset_val_indices)

    neural_net = model.fit(X_train,y_train.astype(int),
              validation_data=(X_valid,y_valid.astype(int)),
              epochs=1,
              verbose=True)
    
    new_score = min(neural_net.history['val_loss']) * 100000
    #if new_score < df.iloc[0]['score'] or df.iloc[0]['score'] == 0:
    df.iloc[0]['score'] = new_score
        
    print(neural_net.history['val_loss'])
    print(min(neural_net.history['val_loss']))
    display(df)
    return df


# In[ ]:


# Used for testing
#initial_weights = ""
#for generation in range(0,2):
#    for index, row in df.head(1).iterrows():
#        np.random.seed(1)
#        df.loc[[index]], initial_weights = load_neural_net(df.loc[[index]], initial_weights)
#display(df.head(3))


# In[ ]:


gen_results = pd.DataFrame()
whole_population = pd.DataFrame()
display(df)

def evaluate(df, n_generations=5):
    for generation in range(0,n_generations):
        for index, row in df.iterrows(): 
            #np.random.seed(1)
            df.loc[[index]] = load_neural_net(df.loc[[index]])
        display(df)
        whole_population[generation] = df['score']
        df = df.sort_values('score',ascending=True).head()
        df = df.reset_index(drop=True)
        df = breed(df)
        gen_results[generation] = df['score'].head()
        display(df)
    return df

#df = evaluate(df)
#gen_results = pd.concat([gen_results, df.head()], axis=1)
#display(gen_results)
#display(whole_population)


# In[ ]:


gen_results = pd.DataFrame()
whole_population = pd.DataFrame()
display(gen_results)

display(df)
def evaluate_es(df, n_generations=5):
    for generation in range(0,n_generations):
        for index, row in df.iterrows(): 
            #np.random.seed(1)
            df.loc[[index]] = load_neural_net(df.loc[[index]])
        display(df)
        whole_population[generation] = df['score']
        df = df.sort_values('score',ascending=True).head()
        df = df.reset_index(drop=True)
        df = breed_es(df,generation, n_generations)
        gen_results[generation] = df['score'].head()
        display(df)
    return df

#df = evaluate_es(df)
#gen_results = pd.concat([gen_results, df.head()], axis = 1)
#display(gen_results)
#display(whole_population)


# # Define NN Architecture

# Todo: add explanation of architecture

# # Make sure you don't overwrite previous csv files!

# In[ ]:


file_name = "../working/gen_scores_es.csv" # increment this per run that you save
gen_results.to_csv(file_name)


# In[ ]:


categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
print(categorical_logits)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(11,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

#for i in range(6):
numerical_logits = Dense(128,activation="relu")(numerical_logits)
numerical_logits = Dense(128,activation="relu")(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)


# In[ ]:


# Lets print our model
model.summary()


# In[ ]:


def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)


# # Train NN model

# In[ ]:


"""
from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
neural_net = model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=3,
          verbose=True,
          callbacks=[check_point]) 
print(neural_net.history['val_loss'])
"""


# # Evaluation of Validation Set

# In[ ]:


"""
# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()
"""


# In[ ]:


"""
# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)
"""


# # Prediction

# In[ ]:


#days = env.get_prediction_days()


# In[ ]:


"""
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()

    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values
    
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test)[:,0]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
"""


# In[ ]:


"""
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()
"""


# In[ ]:




