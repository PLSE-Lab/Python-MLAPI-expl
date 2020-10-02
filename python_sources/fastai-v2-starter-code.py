#!/usr/bin/env python
# coding: utf-8

# # fastai v2 Kernel Starter Code
# 
# The goal of this kernel is to show how to train a neural network using fastai 2.0 for this Kaggle Competition

# ## Grabbing the Library

# First we need to enable internet access within this kernel and then `!pip install git+https://github.com/fastai/fastai_dev             > /dev/null` the dev repository for us to import from.

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai_dev > /dev/null')


# We're going to need a variety of imports, most importantly the `tabular.core` module for building the dataset (the rest deal with training the model)
# 

# In[ ]:


from fastai2.data.all import *
from fastai2.tabular.model import *
from fastai2.optimizer import *
from fastai2.learner import *
from fastai2.metrics import *
from fastai2.callback.all import *


# We'll also use some functionality from the `_40_tabular_core_alt` notebook for memory usage

# In[ ]:


class _TabIloc:
    "Get/set rows by iloc and cols by name"
    def __init__(self,to): self.to = to
    def __getitem__(self, idxs):
        df = self.to.items
        if isinstance(idxs,tuple):
            rows,cols = idxs
            cols = df.columns.isin(cols) if is_listy(cols) else df.columns.get_loc(cols)
        else: rows,cols = idxs,slice(None)
        return self.to.new(df.iloc[rows, cols])
    
class Tabular(CollBase, GetAttr, FilteredBase):
    "A `DataFrame` wrapper that knows which cols are cont/cat/y, and returns rows in `__getitem__`"
    _default='items'
    def __init__(self, df, procs=None, cat_names=None, cont_names=None, y_names=None, type_y=Category, splits=None, do_setup=True):
        if splits is None: splits=[range_of(df)]
        df = df.iloc[sum(splits, [])].copy()
        super().__init__(df)
        
        self.y_names = L(y_names)
        if type_y is not None: procs = L(procs) + getattr(type_y, 'create', noop)
        self.cat_names,self.cont_names,self.procs = L(cat_names),L(cont_names),Pipeline(procs, as_item=True)
        self.split = len(splits[0])
        if do_setup: self.setup()

    def subset(self, i): return self.new(self.items[slice(0,self.split) if i==0 else slice(self.split,len(self))])
    def copy(self): self.items = self.items.copy(); return self
    def new(self, df): return type(self)(df, do_setup=False, type_y=None, **attrdict(self, 'procs','cat_names','cont_names','y_names'))
    def show(self, max_n=10, **kwargs): display_df(self.all_cols[:max_n])
    def setup(self): self.procs.setup(self)
    def process(self): self.procs(self)
    def iloc(self): return _TabIloc(self)
    def targ(self): return self.items[self.y_names]
    def all_col_names (self): return self.cat_names + self.cont_names + self.y_names
    def n_subsets(self): return 2

properties(Tabular,'iloc','targ','all_col_names','n_subsets')

class TabularPandas(Tabular):
    def transform(self, cols, f): self[cols] = self[cols].transform(f)
        
def _add_prop(cls, nm):
    @property
    def f(o): return o[list(getattr(o,nm+'_names'))]
    @f.setter
    def fset(o, v): o[getattr(o,nm+'_names')] = v
    setattr(cls, nm+'s', f)
    setattr(cls, nm+'s', fset)

_add_prop(Tabular, 'cat')
_add_prop(Tabular, 'cont')
_add_prop(Tabular, 'y')
_add_prop(Tabular, 'all_col')

class TabularProc(InplaceTransform):
    "Base class to write a non-lazy tabular processor for dataframes"
    def setup(self, items=None):
        super().setup(getattr(items,'train',items))
        # Procs are called as soon as data is available
        return self(items.items if isinstance(items,DataSource) else items)
    
def _apply_cats (voc, add, c): return c.cat.codes+add if is_categorical_dtype(c) else c.map(voc[c.name].o2i)
def _decode_cats(voc, c): return c.map(dict(enumerate(voc[c.name].items)))

class Categorify(TabularProc):
    "Transform the categorical variables to that type."
    order = 1
    def setups(self, to):
        self.classes = {n:CategoryMap(to.iloc[:,n].items, add_na=(n in to.cat_names)) for n in to.cat_names}
    def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))
    def decodes(self, to): to.transform(to.cat_names, partial(_decode_cats, self.classes))
    def __getitem__(self,k): return self.classes[k]
    
@Categorize
def setups(self, to:Tabular): 
    if len(to.y_names) > 0: self.vocab = CategoryMap(to.iloc[:,to.y_names[0]].items)
    return self(to)

@Categorize
def encodes(self, to:Tabular): 
    to.transform(to.y_names, partial(_apply_cats, {n: self.vocab for n in to.y_names}, 0))
    return to
  
@Categorize
def decodes(self, to:Tabular): 
    to.transform(to.y_names, partial(_decode_cats, {n: self.vocab for n in to.y_names}))
    return to

class Normalize(TabularProc):
    "Normalize the continuous variables."
    order = 2
    def setups(self, dsrc): self.means,self.stds = dsrc.conts.mean(),dsrc.conts.std(ddof=0)+1e-7
    def encodes(self, to): to.conts = (to.conts-self.means) / self.stds
    def decodes(self, to): to.conts = (to.conts*self.stds ) + self.means
        
class FillStrategy:
    "Namespace containing the various filling strategies."
    def median  (c,fill): return c.median()
    def constant(c,fill): return fill
    def mode    (c,fill): return c.dropna().value_counts().idxmax()
    
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    def __init__(self, fill_strategy=FillStrategy.median, add_col=True, fill_vals=None):
        if fill_vals is None: fill_vals = defaultdict(int)
        store_attr(self, 'fill_strategy,add_col,fill_vals')

    def setups(self, dsrc):
        self.na_dict = {n:self.fill_strategy(dsrc[n], self.fill_vals[n])
                        for n in pd.isnull(dsrc.conts).any().keys()}

    def encodes(self, to):
        missing = pd.isnull(to.conts)
        for n in missing.any().keys():
            assert n in self.na_dict, f"nan values in `{n}` but not in setup training set"
            to[n].fillna(self.na_dict[n], inplace=True)
            if self.add_col:
                to.loc[:,n+'_na'] = missing[n]
                if n+'_na' not in to.cat_names: to.cat_names.append(n+'_na')
                    
class ReadTabBatch(ItemTransform):
    def __init__(self, to): self.to = to
    # TODO: use float for cont targ
    def encodes(self, to): return tensor(to.cats).long(),tensor(to.conts).float(), tensor(to.targ)

    def decodes(self, o):
        cats,conts,targs = to_np(o)
        vals = np.concatenate([cats,conts,targs], axis=1)
        df = pd.DataFrame(vals, columns=self.to.all_col_names)
        to = self.to.new(df)
        to = self.to.procs.decode(to)
        return to
    
@typedispatch
def show_batch(x: Tabular, y, its, max_n=10, ctxs=None):
    x.show()
    
@delegates()
class TabDataLoader(TfmdDL):
    do_item = noops
    def __init__(self, dataset, bs=16, shuffle=False, after_batch=None, num_workers=0, **kwargs):
        after_batch = L(after_batch)+ReadTabBatch(dataset)
        super().__init__(dataset, bs=bs, shuffle=shuffle, after_batch=after_batch, num_workers=num_workers, **kwargs)

    def create_batch(self, b): return self.dataset.iloc[b]

TabularPandas._dl_type = TabDataLoader


# ## Setting Up Our Data

# Let's make a `Path` object to our data and combine the `train.csv` with the `building_metadata.csv` to grab some more information about these meter readings. For simplicity we will use the first 1000 samples from the training set. For the `DataFrame` preperation please see ryches Kernel [here](https://www.kaggle.com/ryches/simple-lgbm-solution)

# In[ ]:


path = Path('/kaggle/input/ashrae-energy-prediction')


# In[ ]:


df = pd.read_csv(path/'train.csv', nrows=3000)
train = df.iloc[:2000]
test = df.iloc[2000:]
bldg = pd.read_csv(path/'building_metadata.csv')
weather_train = pd.read_csv(path/"weather_train.csv")
weather_test = pd.read_csv(path/"weather_test.csv")


# In[ ]:


len(train), len(test)


# Next we'll get rid of any missing `y` variables

# In[ ]:


train = train[np.isfinite(train['meter_reading'])]
test = test[np.isfinite(test['meter_reading'])]


# Then we'll merge our data with the provided metadata

# In[ ]:


train.head()


# In[ ]:


train = train.merge(bldg, left_on = 'building_id', right_on = 'building_id', how = 'left')
test = test.merge(bldg, left_on = 'building_id', right_on = 'building_id', how = 'left')


# In[ ]:


train = train.merge(weather_train, left_on = ['site_id', 'timestamp'], right_on = ['site_id', 'timestamp'])


# There is some type discrepencies in the test data so we need to convert it first

# In[ ]:


test = test.merge(weather_train, left_on = ['site_id', 'timestamp'], right_on = ['site_id', 'timestamp'])


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


del weather_train, weather_test, bldg


# In[ ]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train["hour"] = train["timestamp"].dt.hour
train["day"] = train["timestamp"].dt.day
train["weekend"] = train["timestamp"].dt.weekday
train["month"] = train["timestamp"].dt.month
test["timestamp"] = pd.to_datetime(test["timestamp"])
test["hour"] = test["timestamp"].dt.hour
test["day"] = test["timestamp"].dt.day
test["weekend"] = test["timestamp"].dt.weekday
test["month"] = test["timestamp"].dt.month


# In[ ]:


train.drop('timestamp', axis=1, inplace=True)
test.drop('timestamp', axis=1, inplace=True)
train['meter_reading'] = np.log1p(train['meter_reading'])
test['meter_reading'] = np.log1p(test['meter_reading'])


# ## Making the DataBunch

# Next, just like in fastai v1 we need to declare a few things. Specifically our Categorical and Continuous variables, our preprocessors (Normalization, Categorification, and FillMissing), along with how we want to split our data. `fastai` v2 now includes a `RandomSplitter` which is similar to `.split_by_rand_pct()` but now we can specify a custom range for our data (hence `range_of(train)`)

# In[ ]:


cat_vars = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]
cont_vars = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature"]
dep_var = 'meter_reading'


# In[ ]:


procs = [Normalize, Categorify, FillMissing]
splits = RandomSplitter()(range_of(train))


# Now that those are defined, we can create a `TabularPandas` object by passing in our dataframe, the `procs`, our variables, what our `y` is, and how we want to split our data. `fastai` v2 is built on a Pipeline structure where first we dictate what we want to do, then we call the databunch (the high-level API is not done yet so we have nothing similar to directly DataBunching an object)

# In[ ]:


to = TabularPandas(train, procs, cat_vars, cont_vars, y_names=dep_var, type_y=Float, splits=splits)
to_test = TabularPandas(test, procs, cat_vars, cont_vars, y_names=dep_var, type_y=Float)


# If we look at what `to` actually is, we can see what looks to be a bunch of batches of our data aligned into a dataframe that can easily be read!

# In[ ]:


to


# We can then also easily look at our training and validation datasets by calling `.train` or `.valid`

# In[ ]:


to.train


# From here we can create our DataBunch object one of two ways. We can either directly do a `dbch = to.databunch()`, *or* we can take it one step further and apply custom works to some dataloaders. First let's look at the basic version

# In[ ]:


dbch = to.databunch()
dbch.valid_dl.show_batch()


# Now let's try doing this the second method. We can increase our batch size since the validation set is much smaller than our training dataset. We can also specify a few options with our training dataset too. To do this, we will need to create `TabDataLoaders` to, well, load the data!
# 
# We pass in a dataset, a batch size, our `num_workers`, along with if we want to shuffle our dataset and drop the last batch if it does not evenly split. You should always want to do this with the **training** dataset but not the validation. Defaultly they are both set to `False`

# In[ ]:


trn_dl = TabDataLoader(to.train, bs=64, num_workers=0)
val_dl = TabDataLoader(to.valid, bs=128, num_workers=0, shuffle=False, drop_last=False)


# In[ ]:


test_dl = TabDataLoader(to_test, bs=128, num_workers=0, shuffle=False, drop_last=False)


# Lastly we can create a `DataBunch` object by calling `DataBunch()` and passing in our two `DataLoaders`

# In[ ]:


dbunch = DataBunch(trn_dl, val_dl)
dbunch.valid_dl.show_batch()


# As you can see there are a *lot* of ways we can customize our DataBunch's now

# ## Training the Model

# First we need to create a `TabularModel` that needs an embedding matrix size, how many continuous variables to expect, the number of possible outputs (classes), and how big we want our layers. To pass in the embedding matrix sizes, we can use `get_emb_sz` onto a `TabularPandas` object

# First let's define our embedding size rule of thumb, along with our `get_emb_sz` function

# In[ ]:


def emb_sz_rule(n_cat): 
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))


# In[ ]:


def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz


# In[ ]:


def get_emb_sz(to, sz_dict=None):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    return [_one_emb_sz(to.procs.classes, n, sz_dict) for n in to.cat_names]


# Now we pass in our `TabularPandas` object, `to`

# In[ ]:


emb_szs = get_emb_sz(to); print(emb_szs)


# The last piece of the puzzle we need is our basic `TabularModel`

# In[ ]:


class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0., y_range=None, use_bn=True, bn_final=False):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        _layers = [BnDropLin(sizes[i], sizes[i+1], bn=use_bn and i!=0, p=p, act=a)
                       for i,(p,a) in enumerate(zip([0.]+ps,actns))]
        if bn_final: _layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*_layers)
    
    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


# If you noticed, most of what changed with the v2 API is focused on the dataloading / DataBunch creation. The rest of this Kernel sould look very familiar to fastai users

# In[ ]:


model = TabularModel(emb_szs, len(to.cont_names), 1, [1000,500]); model


# Now we can define our optimization function and create our `Learner`

# In[ ]:


opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dbunch, model, MSELossFlat(), opt_func=opt_func)


# In[ ]:


learn.fit_one_cycle(30)


# To get our predictions we just simply pass in that test dataloader from earlier

# In[ ]:


to_test = TabularPandas(test, procs, cat_vars, cont_vars, y_names=dep_var, type_y=Float)
test_dl = TabDataLoader(to_test, bs=128, shuffle=False, drop_last=False)


# In[ ]:


preds, _ = learn.get_preds(dl=test_dl) 
preds = np.expm1(preds.numpy())


# In[ ]:


submission = pd.DataFrame(columns=['row_id', 'meter_reading'])


# In[ ]:


test.head()


# Just for example. In reality for this competition you would want to use: `submission['row_id'] = test['row_id']`

# In[ ]:


submission['row_id'] = test['building_id']


# In[ ]:


submission['meter_reading'] = preds


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv')


# Hope this helps you get started! :)
# 
# - muellerzr
