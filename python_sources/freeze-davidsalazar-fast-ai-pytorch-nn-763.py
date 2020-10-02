#!/usr/bin/env python
# coding: utf-8

# # Fast.ai/Pytorch Starter
# 
# Although most people (including me) seem to be doing much better with Boosting Trees, I think it is worth the time to explore a Neural Network solution to the  Home Credit Default Risk competition. Besides, a Kaggle competition is always a good opportunity to test what one is currently learning. I did two "cool"  things in this kernel:
# 
# * Categorical Embeddings.
# * Custom loss functions with different weights for each class to try to manage the imbalance in `TARGET`. 
# 
# I had to tweak the fast.ai library a little bit, but all things considered, it is extraordinary how little code you actually have to write to get some model going. 
# 
# ## Load Data
# 
# To aggregate the various tables available in the competition I followed the next heuristic:
# 
# * If the variable was continuous, I aggregated it using its mean. 
# * If the variable was categorical, I aggregated it using its mode. If I were to one-hot-encode the variables and aggregate them using their mean, there wouldn't be categorical variables (besides the one in the main table) for which to create categorical embeddings. 
# 
# The mode computation is very, very slow, so I did it all of this in another Kaggle kernel.
# 
# ## Imports
# 

# In[ ]:


from fastai.imports import *
from fastai.structured import *
from fastai.column_data import *
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Define custom loss function to account for two ouput nodes
def roc_auc_own(y_score, y_true):
    y_score = np.exp(y_score[:,1])
    return roc_auc_score(y_true, y_score)


# In[ ]:


df_train = pd.read_feather('../input/home-credit-data-processing-for-neural-networks/tables_merged_train')
df_test = pd.read_feather('../input/home-credit-data-processing-for-neural-networks/tables_merged_test')


# ## What type of variables do we have?

# In[ ]:


df_train.dtypes.value_counts()


# In[ ]:


cat_vars = [col for col in df_train if df_train[col].dtype.name != 'float64' and df_train[col].dtype.name != 'float32' and len(df_train[col].unique()) < 150]
cat_vars.remove('TARGET')


# In[ ]:


cat_sz = [(c, len(df_train[c].unique())+1) for c in cat_vars]


# Which variables are we going to treat as categorical?

# In[ ]:


cat_vars


# ## Pre-processing 
# 
# The fast.ai library handles NA values for us. For categorical variables, missing variables are encoded as a level within the categories of their own; in this case, with a zero. For continuous variables, if a given variable has missing variables, we create an extra dummy variable recording which of the observations were missing and, in the original given variable, we impute the missing values with the median. 
# 
# Thus, the algorithm will be able to encode missingness in any way it chooses. Also, we will normalize continuous variables for ease of optimization. 

# In[ ]:


# Train validation-split
y = np.array(df_train['TARGET'])
df_train.drop('TARGET', axis = 1, inplace=True)
df_to_nn_train, df_to_nn_valid, y_train, y_valid = train_test_split(df_train, y, test_size=0.33, random_state=23, stratify = y)


# In[ ]:


def preprocess_fast_ai(df_to_nn_train, df_to_nn_valid, cat_vars):
    # Declare categorical variables
    for v in cat_vars: df_to_nn_train[v] = df_to_nn_train[v].astype('category').cat.as_ordered()
    apply_cats(df_to_nn_valid, df_to_nn_train)

    # Deal with missingness and put everything as numbers
    df, _, nas, mapper = proc_df(df_to_nn_train, do_scale=True, skip_flds=['SK_ID_CURR'])
    df_valid, _, nas, mapper = proc_df(df_to_nn_valid, do_scale=True, na_dict=nas, mapper=mapper, skip_flds=['SK_ID_CURR'])
    return df, df_valid


# In[ ]:


df, df_valid = preprocess_fast_ai(df_to_nn_train, df_to_nn_valid, cat_vars)


# The embedding sizes we are going to use for each category:

# In[ ]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# ## PyTorch/Fast.ai
# 
# Define the data loader:

# In[ ]:


md  = ColumnarModelData.from_data_frames('', trn_df = df, val_df = df_valid, 
                                         trn_y = y_train.astype('int'), val_y = y_valid.astype('int'), 
                                         cat_flds=cat_vars, bs=512, is_reg= False)


# There's no easy way of using the fast.ai library (that I know) to predict structured data in a classification problem. Besides, the fast.ai package that Kaggle is running is not the same as the source code in GitHub. Thus, I read a little bit of the code and tweaked it to create the model that we are going to use.

# In[ ]:


class MixedInputModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                 y_range=None, use_bn=False, is_reg=True, is_multi=False):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_szs])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont= n_emb, n_cont
        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: kaiming_normal(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        kaiming_normal(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn,self.y_range = use_bn,y_range
        self.is_reg = is_reg
        self.is_multi = is_multi

    def forward(self, x_cat, x_cont):
        x = []
        for i,e in enumerate(self.embs):
            x.append(e(x_cat[:,i]))
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn(x_cont)
        x = torch.cat([x, x2], 1)
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        x = F.log_softmax(x)
        return x


# Besides the embedding, 3 fully connected layers:

# In[ ]:


# Define Model
m = MixedInputModel(emb_szs, n_cont = len(df.columns)-len(cat_vars),
                   emb_drop = 0.05, out_sz = 2, szs = [500, 250, 250], drops = [0.1, 0.1, 0.1], 
                   y_range = None, use_bn = False, is_reg = False, is_multi = False)
bm = BasicModel(m.cuda(), 'binary_classifier')


# We define our learner's loss function:

# In[ ]:


# Define Learner
class StructuredLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.nll_loss
# Instantiate learner
learn = StructuredLearner(md, bm)


# Now, let's do some fitting:

# In[ ]:


learn.lr_find(1e-4, 1)
learn.sched.plot(100)


# In[ ]:


lr = 1e-2
learn.fit(lr, 3, metrics=[roc_auc_own])


# ## Let's Understand our predictions

# In[ ]:


# predictions 
logpreds = learn.predict() # final output log_softmax
preds = np.exp(logpreds[:,1])


# In[ ]:


logpreds_valid = learn.predict(is_test = False)
preds_valid = np.exp(logpreds_valid[:,1])
preds_binary = (preds_valid >= 0.5).astype(np.int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, preds_binary)
plot_confusion_matrix(cm, [0, 1])


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_valid,
                            preds_binary,
                            target_names= ['0', '1']))


# In[ ]:


from sklearn.metrics import roc_curve
false_positive_rate, true_positive_rate, threshold = roc_curve(y_valid,
                                                               preds_valid)
# Plot ROC curve
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# Although our AUC is not that bad, our predictions are extremely naive. They are driven by the huge imbalance in the dataset. This can be seen by the confusion matrix: we are hardly predicting for any of the observations the `1` class. This results in a lousy recall. We need our model to learn better what indentifies the people who belong to the `1` class.

# ## Tackling the imbalance problem
# 
# Seems we've exhausted what this model can learn, as the changes from `val_loss` and `roc` have hit decreasing returns and our predictions aren't that intelligent. Let's try to correct for the imbalance in `TARGET`.

# In[ ]:


class ColumnarDataset(Dataset):
    def __init__(self, cats, conts, y, is_reg, is_multi):
        n = len(cats[0]) if cats else len(conts[0])
        self.cats  = np.stack(cats,  1).astype(np.int64)   if cats  else np.zeros((n,1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n,1))
        self.y     = np.zeros((n,1))                       if y is None else y
        if is_reg:
            self.y =  self.y[:,None]
        self.is_reg = is_reg
        self.is_multi = is_multi

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont, y=None, is_reg=True, is_multi=False):
        cat_cols = [c.values for n,c in df_cat.items()]
        cont_cols = [c.values for n,c in df_cont.items()]
        return cls(cat_cols, cont_cols, y, is_reg, is_multi)

    @classmethod
    def from_data_frame(cls, df, cat_flds, y=None, is_reg=False, is_multi=False):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y, is_reg, is_multi)

class ColumnarModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs, test_ds=None, shuffle=True):
        test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
        super().__init__(path, DataLoader(trn_ds, bs, shuffle=shuffle, num_workers=1),
            DataLoader(val_ds, bs*2, shuffle=False, num_workers=1), test_dl)
    @classmethod
    def from_data_frames(cls, path, trn_df, trn_y, cat_flds, bs, val_df = None, val_y = None,  is_reg = False, is_multi = False, test_df=None):
        trn_ds  = ColumnarDataset.from_data_frame(trn_df,  cat_flds, trn_y, is_reg, is_multi)
        val_ds  = ColumnarDataset.from_data_frame(val_df,  cat_flds, val_y, is_reg, is_multi) if val_df is not None else None
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds, None,  is_reg, is_multi) if test_df is not None else None
        return cls(path, trn_ds, val_ds, bs, test_ds=test_ds)
    
    @classmethod
    def from_data_frame(cls, path, val_idxs, df, y, cat_flds, bs, is_reg=True, is_multi=False, test_df=None):
        ((val_df, trn_df), (val_y, trn_y)) = split_by_idx(val_idxs, df, y)
        return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg, is_multi, test_df=test_df)


# Our model is having trouble identifying the people with class `1`. The main problem is that there is hardly any of them in the dataset. Let's change that by oversampling with replacement these people and creating and augmented dataset such that they appear the same number of times as the people with class `0`. 

# In[ ]:


train_df, test_df = preprocess_fast_ai(df_train, df_test, cat_vars)


# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
df_resampled, y_resampled = ros.fit_sample(df, y_train)
df_resampled = pd.DataFrame(df_resampled, columns = df.columns)
y_valid.mean(), y_resampled.mean()


# Let's redefine the data loader with the new observations added. Given that our training stops being representative of our validation, let's do some heavy dropout regularization. 

# In[ ]:


md  = ColumnarModelData.from_data_frames('', trn_df = df_resampled, 
                                         val_df = df_valid, trn_y = y_resampled.astype('int'),
                                         val_y = y_valid.astype('int'), cat_flds=cat_vars, bs=1024, is_reg = False,
                                         test_df = test_df)
# Define Learner
class StructuredLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.nll_loss
m = MixedInputModel(emb_szs, n_cont = len(df.columns)-len(cat_vars),
                   emb_drop = 0.4, out_sz = 2, szs = [1000, 500], 
                   drops = [0.6, 0.6],y_range = None, use_bn = False, is_reg = False)
bm = BasicModel(m.cuda(), 'binary_classifier')
# Instantiate learner
learn = StructuredLearner(md, bm)


# In[ ]:


learn.lr_find(1e-2, 2)
learn.sched.plot(100)


# In[ ]:


lr = 0.1
learn.fit(lr, 3, metrics=[roc_auc_own])


# In[ ]:


learn.fit(lr, 2, metrics=[roc_auc_own], cycle_len=1, cycle_mult=2)


# In[ ]:


logpreds = learn.predict()
preds = np.exp(logpreds[:,1])

logpreds_valid = learn.predict(is_test = False)
preds_valid = np.exp(logpreds_valid[:,1])
preds_binary = (preds_valid >= 0.5).astype(np.int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, preds_binary)
plot_confusion_matrix(cm, [0, 1])


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_valid,
                            preds_binary,
                            target_names= ['0', '1']))

from sklearn.metrics import roc_curve
false_positive_rate, true_positive_rate, threshold = roc_curve(y_valid,
                                                               preds_valid)


# In[ ]:


# Plot ROC curve
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


#  Even though we are regularizing heavily, the model is still overfitting and the ROC has improved a little bit. Let's try to take this model to the leaderboard.

# In[ ]:


logpreds = learn.predict(True)
preds = np.exp(logpreds[:,1])

submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'],
              'TARGET': preds})
submission.to_csv('submission.csv', index=False, float_format='%.8f')

