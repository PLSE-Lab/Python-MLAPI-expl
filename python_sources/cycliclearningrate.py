#!/usr/bin/env python
# coding: utf-8

# ### Acknowledgements:<Br>[](http://)
# Data preparation for BERT model is based on Abhishek Thakur's Notebook.
# 

# ### Overall Model highlights:<br>
# 1.BERT base pre trained model from Hugging face is used.<br>
# 2.Softmax layer head used on top of the BERT base output.<Br>
# 3.Cyclic Learning rate is used as a learning rate scheduler.<br>
# 4.Training data split into 5 folds and model is saved for each fold.<Br>
# 5.Results of all the 5 models are averaged.<Br>
# 6.A pair (start_index, end_index) is selected such that the probability of start_index + probability of end_index is maximum.<br>

# #### Achieved Jaccard score of 0.708 on the test set

# ### Triangular Cyclic Learning Rate

# In[ ]:



from IPython.display import Image
Image("../input/cycliclearningrate-1/keras_clr_triangular.png")


# **Max_lr :** maximum learning rate<br>
# **lr - **minimum learning rate or base learning rate<Br>
# **step_size -** number of batches that it takes to reach the maximum learning rate.<br>

# In[ ]:


import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)


# In[ ]:


MAX_LEN = 96
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
# train_pos = train[train.sentiment == "positive"].reset_index()
train_neg = train[train.sentiment == "negative"].reset_index()
# print(train_pos.shape)
# print(train_neg.shape)
#train = train_neg


# In[ ]:


e = tokenizer.encode("going")
print(e.ids)


# In[ ]:


print(train.shape)


# In[ ]:


ct = train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(ct):
      
    
    # FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split()).strip()
    text2 = " ".join(train.loc[k,'selected_text'].split()).strip()
    
    text1 = text1.replace("_","")
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1)
    enc2 = tokenizer.encode(text2)
   
        
    # ID_OFFSETS
    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    
    print(offsets)
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
        
    s_tok = sentiment_id[train.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[k,:len(enc.ids)+5] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1
    
    


# # Test Data
# We must tokenize the test data exactly the same as we tokenize the training data

# In[ ]:


test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

ct = test.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test.shape[0]):
    
#     if test.loc[k, 'sentiment'] == "neutral":
#         continue
        
    # INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    text1 = text1.replace("_","")
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1


# SoftMax Layer head is used on top of the BERT model to predict the index of start words and stop words in the answer

# In[ ]:



def build_model():
    
    '''
    Function which builds the model [ BERT + 2 Softmax layers head ] for span prediction
    '''
    
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
     
    x_bert = bert_model([ids,att,tok])

    
    logits = tf.keras.layers.Dense(2, activation="linear", kernel_initializer="he_normal")(x_bert[0])
    print(x_bert[0].shape)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)   
   
    start_scores = tf.keras.layers.Activation('softmax')(start_logits)
    end_scores = tf.keras.layers.Activation('softmax')(end_logits)    

   
    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[start_scores,end_scores])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


def jaccard(str1, str2): 
    
    '''
    Function which computes the jaccard score
    '''
    
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


print(start_tokens.shape)
print(end_tokens.shape)


# In[ ]:


from keras.callbacks import *
class CyclicLR(Callback):
    '''
    Class which implements the cyclic Learning rate
    
    '''
    

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.scale_mode = 'cycle'
        print(self.scale_mode)
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
       
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            print("Setting the lr to {}".format(self.base_lr))
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        #print("learning rate is {} ".format(K.get_value(self.model.optimizer.lr))) 
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
        




# In[ ]:


clr = CyclicLR(
	mode='triangular',
	base_lr= 1e-6,
	max_lr= 4e-5,
	step_size= 320,
    scale_mode='cycle')


# In[ ]:


jac = []; VER='v4'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
# oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
# oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
preds_start = np.zeros((input_ids.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids.shape[0],MAX_LEN))

def scheduler(epoch):
    if epoch <1:
      return 0.00003
    else:
      return 0.00003 * tf.math.exp(-0.1 * (10 - epoch))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
from sklearn.model_selection import KFold

n_splits = 5

skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):
    print(idxT,idxV)
    print("Training for fold {}" .format(fold))  
         
    K.clear_session()
    model = build_model()
   
    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')
        
    hist = model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv,reduce_lr],
        validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        [start_tokens[idxV,], end_tokens[idxV,]]))
        
  
    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    preds_start[idxV,],preds_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=1)

    all = []
    for k in idxV:
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a>b: 
            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
        else:
            text1 = " "+" ".join(train.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
            #print(st)
        all.append(jaccard(st,train.loc[k,'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
    


# Training on complete DataSet

# In[ ]:


preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
DISPLAY=1
for i in range(n_splits):   
    
    print('#'*25)
    print('### MODEL %i'%(i+1))
    print('#'*25)
    
    K.clear_session()
    model = build_model()
    model.load_weights('v4-robertafulldata-%i.h5'%i)

    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/n_splits
    preds_end += preds[1]/n_splits


# A pair (start_index, end_index) is selected such a way that probability of start_index+end_index is maximum

# In[ ]:


#Train
all = []
jaccard_score = 0
for k in range(input_ids_t.shape[0]):
    
    score_array = np.zeros((MAX_LEN,MAX_LEN),dtype='float32')
    for i in range(MAX_LEN):
        score_array[i,:] = preds_start[k,i] + preds_end[k,]
    
    
    # Find the combination of indices whose combined probability is maximum
    result = np.argmax(score_array)
    (a,b) = np.unravel_index(result, score_array.shape)
   
  
    if a>b:
        print("Result {}".format(a))
        print("Truth {}".format(np.argmax(start_tokens[k,])))
        st = test.loc[k,'text']
        
    else:
        
        if test.loc[k, 'sentiment'] == "neutral":
            st = test.loc[k,'text']
            
        else:
     
            text1 = " "+" ".join(test.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
    
    all.append(st)


# In[ ]:


test['selected_text'] = all
test[['textID','selected_text']].to_csv('submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.head(25)

