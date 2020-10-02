#!/usr/bin/env python
# coding: utf-8

# # IEEE-CIS Fraud Detection Time Series Heatmaps
# 
# This notebook shows counts of transactions over time in a 2D heatmap, as a simple exploration of the time series structure of the train/test sets.
# 
# One row of pixels in each image is 1 day, with about 183 rows in each image, derived from the training set, and later on, the test set.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import gc, os, sys, re, time
import matplotlib.pyplot as plt
import cv2
from IPython.display import Image, display
from tqdm import tqdm_notebook as tqdm


# In[ ]:


DTYPE = {
    'TransactionID': 'int32',
    'isFraud': 'int8',
    'TransactionDT': 'int32',
    'TransactionAmt': 'float32',
    'ProductCD': 'category',
    'card1': 'int16',
    'card2': 'float32',
    'card3': 'float32',
    'card4': 'category',
    'card5': 'float32',
    'card6': 'category',
    'addr1': 'float32',
    'addr2': 'float32',
    'dist1': 'float32',
    'dist2': 'float32',
    'P_emaildomain': 'category',
    'R_emaildomain': 'category',
    'C1': 'float32',
    'C2': 'float32',
    'C3': 'float32',
    'C4': 'float32',
    'C5': 'float32',
    'C6': 'float32',
    'C7': 'float32',
    'C8': 'float32',
    'C9': 'float32',
    'C10': 'float32',
    'C11': 'float32',
    'C12': 'float32',
    'C13': 'float32',
    'C14': 'float32',
    'D1': 'float32',
    'D2': 'float32',
    'D3': 'float32',
    'D4': 'float32',
    'D5': 'float32',
    'D6': 'float32',
    'D7': 'float32',
    'D8': 'float32',
    'D9': 'float32',
    'D10': 'float32',
    'D11': 'float32',
    'D12': 'float32',
    'D13': 'float32',
    'D14': 'float32',
    'D15': 'float32',
    'M1': 'category',
    'M2': 'category',
    'M3': 'category',
    'M4': 'category',
    'M5': 'category',
    'M6': 'category',
    'M7': 'category',
    'M8': 'category',
    'M9': 'category',
}

IN_DIR = '../input'
TARGET = 'isFraud'
BASE_COLS = list(DTYPE.keys())
PLOTS_TRAIN_BASE = 'train'
PLOTS_TRAIN_V = 'train_v'
PLOTS_TEST = 'test'
V_COLS = [ f'V{i}' for i in range(1, 340) ]
V_DTYPE = {v:'float32' for v in V_COLS}
DTYPE.update(V_DTYPE)
TRAIN_USE = list(DTYPE.keys())
TEST_USE = [c for c in TRAIN_USE if c != TARGET]


# In[ ]:


train = pd.read_csv(f'{IN_DIR}/train_transaction.csv', usecols=TRAIN_USE, dtype=DTYPE)
train.shape


# TransactionDT is in seconds, with 15811131 maximum.

# In[ ]:


train.TransactionDT.max() 


# In[ ]:


train.TransactionDT.max() / 86400


# 183 days, put them on one row each in a heatmap, with 480 columns

# In[ ]:


86400 / 480


# So each pixel will represent 180 seconds of a day. All transactions in each 3 minute block will be counted.

# In[ ]:


WIDTH = 480
HEIGHT = 183
IMG_SIZE = WIDTH * HEIGHT
SECONDS_PER_PIXEL = 180
DAY_MARKER = 10


# In[ ]:


def make_plot(source_df, querystr, verbose=False):
    df = source_df.query(querystr, engine='python')
    if verbose:
        print(querystr, df.shape[0], 'transactions')
    ts = (df.TransactionDT // SECONDS_PER_PIXEL).values
    c = np.zeros(IMG_SIZE, dtype=int)
    np.add.at(c, ts, 1)
    return c.reshape((HEIGHT, WIDTH))

def normalize(c):
    return ((c/c.max()) * 255)

def make_and_save(source_df, png_file, querystr):
    p = make_plot(source_df, querystr)
    # NOTE: log1p() of counts to stretch the contrast
    cv2.imwrite(png_file, normalize(np.log1p(p)))


# `TransactionID>0` is a simple way to say 'all the training set' - it is true for all rows.

# In[ ]:


make_and_save(train, 'all_transactions.png', 'TransactionID>0')


# Take a preview look at what we made...

# In[ ]:


display(Image('all_transactions.png'))


# Looks good, now do it for all values that appear 5000 times or more, for every single column.

# In[ ]:


def save_all(df, cols, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for col in tqdm(cols):
        vc = df[col].value_counts(dropna=False)
        for value, count in vc.items():
            if count < 5000:
                continue
            #print(col, value, count)
            tag = f'{col}_{value}_{count}'
            png = f'{base_dir}/{tag}.png'
            if type(value) is float and np.isnan(value):
                make_and_save(df, png, f'{col}.isnull()')
            else:
                make_and_save(df, png, f'{col}=="{value}"')


# In[ ]:


save_all(train, BASE_COLS, PLOTS_TRAIN_BASE)


# Over 300 plots :)

# In[ ]:


get_ipython().system('ls -1 $PLOTS_TRAIN_BASE | wc -l')


# Now the V columns too:

# In[ ]:


save_all(train, V_COLS, PLOTS_TRAIN_V)


# In[ ]:


get_ipython().system('ls -1 $PLOTS_TRAIN_V | wc -l')


# # Matplotlib Display
# 
# Show some plots selected for interesting features...

# In[ ]:


xlabels = [f'{h}am' for h in range(12)] +           [f'{h}pm' for h in range(12)]
xlabels[12] = '12pm'


# In[ ]:


ylabels = [f'day{i}' for i in range(0, HEIGHT, DAY_MARKER)]


# In[ ]:


plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["image.cmap"] = 'afmhot'


# In[1]:


def show_plot(source_df, querystr):
    fig, ax = plt.subplots(figsize=(18, 6))
    p = make_plot(source_df, querystr, verbose=True)
    c = ax.pcolormesh(p)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, WIDTH, 20), False)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(0, HEIGHT, DAY_MARKER))
    ax.set_yticklabels(ylabels)
    ax.set_title(querystr)
    cbar = fig.colorbar(c, ax=ax)
    return plt.tight_layout()


# This is all transactions in the training set - I'm using the raw date values from the training data so the values are on the time axis are possibly the wrong timezone - night time appears clearly visible but about 3 hours late (depending on your lifestyle ;)
# 
# Also a fat peak around days 20-30, and a thin (1 day) peak at about day 92.
# 
# Looking closely (right click &rarr; *View Image* helps) there is some weekly seasonality - a darker line in the mornings - presumably Sunday.

# In[ ]:


show_plot(train, 'TransactionID>0')


# Fraud does not seem to dip so much overnight... (Note: max value is 8 - no three-minute block has more than 8 fraudulent transactions.)

# In[ ]:


show_plot(train, 'isFraud==1')


# Now some covariate shifts...

# In[ ]:


show_plot(train, 'D4.isnull()')


# In[ ]:


show_plot(train, 'D15.isnull()')


# card1 value 7919 seems periodic, at about 30 days, with some double peaks too...

# In[ ]:


show_plot(train, 'card1==7919')


# seems to be correlated with card2 value 194

# In[ ]:


show_plot(train, 'card2==194')


# card2 is not missing at random, but missing in streaks:

# In[ ]:


show_plot(train, 'card2.isnull()')


# card5 stops being equal to 202 at about day 80

# In[ ]:


show_plot(train, 'card5==202')


# but value 126 starts appearing more about that time... perhaps they have a similar meaning?

# In[ ]:


show_plot(train, 'card5==126')


# A bit hard to see, but gmail.com as R_emaildomain has a drop (dark band) around day 100.

# In[ ]:


show_plot(train, 'R_emaildomain=="gmail.com"')


# A bit easier to see here in D6...

# In[ ]:


show_plot(train, 'D6==0')


# The "D" columns refer to time, as stated by organizers. D9 being 0.75 means 6pm-7pm for example:

# In[ ]:


show_plot(train, 'D9==0.75')


# Value "S" for ProductCD only appears late - the natural choice for a validation era:

# In[ ]:


show_plot(train, 'ProductCD=="S"')


# More complicated expressions are possible, you could extract a decision tree path and plug it in as query string, a simple (uninsightful?) example (depth 2) is:

# In[ ]:


show_plot(train, '(P_emaildomain.isnull()) and (TransactionAmt<=30)')


# # Test Set

# In[ ]:


test = pd.read_csv(f'{IN_DIR}/test_transaction.csv', usecols=TEST_USE, dtype=DTYPE)
test.shape


# In[ ]:


test.TransactionDT.min()


# In[ ]:


test.TransactionDT.max()


# Test starts at day 213:

# In[ ]:


test.TransactionDT.min() / 86400


# In[ ]:


test.TransactionDT.max() / 86400


# To make test look like train, subtract 213 days, then reuse the above code. Note `day0 .. day183` in the plots now refers to test set days.

# In[ ]:


test['TransactionDT'] -= 213 * 86400


# In[ ]:


test.TransactionDT.max() / 86400


# Generate test set plots for offline use.

# In[ ]:


save_all(test, test.columns, PLOTS_TEST)


# In[ ]:


get_ipython().system('ls -1 $PLOTS_TEST | wc -l')


# ## Test Set Plots
# 
# The leaderboard page says **This leaderboard is calculated with approximately 20% of the test data**. [This great discussion topic][1] says the public/private split is by time. So, the public LB will be day 0 - day 37, all after is the private LB...
# 
#  [1]: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101040
# 

# Overall shape looks similar, perhaps night time drifts later towards the end?

# In[ ]:


show_plot(test, 'TransactionID>0')


# Date columns appear to mean the same thing, no daylight savings shift...

# In[ ]:


show_plot(test, 'D9==0.75')


# D15 continues it's erratic behaviour - and is different between public/private periods - though it is only null for 12069 rows (of ~500k), so perhaps it's ok to ignore this. Or just fillna(0) or fillna(1).

# In[ ]:


show_plot(test, 'D15.isnull()')


# card1 value 7919 seems familiar, similar to train...

# In[ ]:


show_plot(test, 'card1==7919')


# Above we saw that value "S" for ProductCD only appears late in the train set - but here in the test set it is more uniform, though around the same number of transactions.

# In[ ]:


show_plot(test, 'ProductCD=="S"')


# However - value "R" appears quite late in the *test* set - in the private LB zone. This is similar to how "S" appeared only late in the *train* set.

# In[ ]:


show_plot(test, 'ProductCD=="R"')


# Similarly, "H" for ProductCD ramps up in the test set.

# In[ ]:


show_plot(test, 'ProductCD=="H"')


# # Clean Up
# 
# Compress some of the generated pngs - Kaggle does not allow more than 500 output files.

# In[ ]:


get_ipython().system('7z a -bd -mmt4 -sdel $PLOTS_TRAIN_V.7z $PLOTS_TRAIN_V >>compress_7z.log')


# In[ ]:


get_ipython().system('7z a -bd -mmt4 -sdel $PLOTS_TEST.7z $PLOTS_TEST >>compress_7z.log')


# ## Notes
# 
# Much more is possible here, for example, count all transactions, then count subsets like
#  - `ProductCD==X`
#  
# and
#  - `ProductCD==X and isFraud==1` 
#  
# then apply Bayes rule to get **p(isFraud | ProductCD==X)** and color cells in accordingly, e.g. p(isFraud) as the red channel in RGB.
# 
# Also `np.add.at(c, ts, 1)` can be changed to `np.add.at(c, ts, df.isFraud)` to accumulate values instead of simply count transctions.
# 
# ____
# 
# *to be continued...*  **(HOWEVER: feel free to fork this notebook and try it out yourself :)**
