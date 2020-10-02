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

# In[27]:


from fastai.imports import *
from fastai.structured import *
from fastai.column_data import *
from torch.nn import functional as F
import gc 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# In[28]:


get_ipython().system('ls ../input/')


# In[29]:


df_train = pd.read_csv('../input/application_train.csv')
df_test = pd.read_csv('../input/application_test.csv')
df_train.head()


# ## What type of variables do we have?

# In[30]:


df_train.dtypes.value_counts()


# In[31]:


cat_vars = [col for col in df_train if df_train[col].dtype.name != 'float64' and df_train[col].dtype.name != 'float32' and len(df_train[col].unique()) < 150]
cat_vars.remove('TARGET')
for v in cat_vars: df_train[v] = df_train[v].astype('category').cat.as_ordered()
cat_sz = [(c, len(df_train[c].cat.categories)+1) for c in cat_vars]


# Which variables are we going to treat as categorical?

# In[32]:


cat_sz


# In[33]:


apply_cats(df_test, df_train)


# ## Pre-processing 
# 
# The fast.ai library handles NA values for us:

# In[34]:


get_ipython().run_line_magic('time', "df, y, nas, mapper = proc_df(df_train, 'TARGET', do_scale=True, skip_flds=['SK_ID_CURR'])")
get_ipython().run_line_magic('time', "df_test_md, _, nas, mapper = proc_df(df_test, do_scale=True, na_dict=nas, mapper=mapper, skip_flds=['SK_ID_CURR'])")


# In[35]:


df_to_nn_train, df_to_nn_valid, y_train, y_valid = train_test_split(df, y, test_size=0.33, random_state=23, stratify = y)


# In[36]:


for v in cat_vars: df_to_nn_train[v] = df_to_nn_train[v].astype('category').cat.as_ordered()
for v in cat_vars: df_to_nn_valid[v] = df_to_nn_valid[v].astype('category').cat.as_ordered()    


# The embedding sizes we are going to use for each category:

# In[37]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# ## PyTorch/Fast.ai
# 
# Define the data loader:

# In[38]:


md  = ColumnarModelData.from_data_frames('', trn_df = df_to_nn_train, val_df = df_to_nn_valid, trn_y = y_train.astype('int'), val_y = y_valid.astype('int'), cat_flds=cat_vars, bs=128, is_reg= False)


# There's no easy way of using the fast.ai library (that I know) to predict structured data in a classification problem. Besides, the fast.ai package that Kaggle is running is not the same as the source code in GitHub. Thus, I read a little bit of the code and tweaked it to create the model that we are going to use.

# In[39]:


def roc_auc_own(y_score, y_true):
    y_score = np.exp(y_score[:,1])
    return roc_auc_score(y_true, y_score)


# ## Tackling the imbalance problem
# 
# Seems we've exhausted what this model can learn, as the changes from `val_loss` and `roc` have hit decreasing returns. Let's try to correct for the imbalance in `TARGET` by creating a loss function with weights. 

# In[40]:


class StructuredLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = torch.nn.NLLLoss(weight= torch.FloatTensor([0.1, 0.9]).cuda())


# In[41]:


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
    def from_data_frame(cls, df, cat_flds, y=None, is_reg=True, is_multi=False):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y, is_reg, is_multi)

class ColumnarModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs, test_ds=None, shuffle=True):
        test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
        super().__init__(path, DataLoader(trn_ds, bs, shuffle=shuffle, num_workers=1),
            DataLoader(val_ds, bs*2, shuffle=False, num_workers=1), test_dl)
    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg = False, is_multi = False, test_df=None):
        trn_ds  = ColumnarDataset.from_data_frame(trn_df,  cat_flds, trn_y, is_reg, is_multi)
        val_ds  = ColumnarDataset.from_data_frame(val_df,  cat_flds, val_y, is_reg, is_multi)
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds, None,  is_reg, is_multi) if test_df is not None else None
        return cls(path, trn_ds, val_ds, bs, test_ds=test_ds)


# Let's define the data loader, this time with the test set:

# In[42]:


md  = ColumnarModelData.from_data_frames('', trn_df = df_to_nn_train, val_df = df_to_nn_valid, trn_y = y_train.astype('int'), val_y = y_valid.astype('int'), cat_flds=cat_vars, bs=128, is_reg = False, test_df=df_test_md)


# Same model as before, this time with some dropout:

# In[44]:


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


# In[45]:


m = MixedInputModel(emb_szs, n_cont = len(df.columns)-len(cat_vars),
                   emb_drop = 0.05, out_sz = 2, szs = [100, 100, 100], drops = [0.05, 0.05, 0.05],y_range = None, use_bn = False, is_reg = False, is_multi = False)
bm = BasicModel(m.cuda(), 'binary_classifier')
learn = StructuredLearner(md, bm)


# Let's do the fitting:

# In[22]:


learn.lr_find()
learn.sched.plot(100)


# In[34]:


lr = 1e-2
learn.fit(lr, 3, metrics=[roc_auc_own])


# In[35]:


learn.fit(lr, 5, metrics=[roc_auc_own])


# Seems that is the best the model can deliver without torturing too much. The loss function with weights does seem to help it to improve the AUC in validation. However, not good enough compared to Boosting trees. Let's evaluate our predictions a little bit more in depth:

# In[36]:


logpreds = learn.predict(is_test=True)
preds = np.exp(logpreds[:,1])
preds


# In[37]:


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


# In[38]:


logpreds_valid = learn.predict(is_test = False)
preds_valid = np.exp(logpreds_valid[:,1])
preds_binary = (preds_valid >= 0.5).astype(np.int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, preds_binary)
plot_confusion_matrix(cm, [0, 1])


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_valid,
                            preds_binary,
                            target_names= ['0', '1']))


# In[40]:


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


# Even though the loss function with weights did help, the model is still having trouble predicting the positive class: it is predicting it way too much.  Maybe the weight in the positive class was too big. We could always set it as any other hyperparameter by tweaking and comparing the different results in the validation set.
# 
# ## Wrap it up!

# In[41]:


submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'],
              'TARGET': preds})
submission.to_csv('submission.csv', index=False, float_format='%.8f')


# In[ ]:





# In[ ]:





# In[ ]:




