#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis: an end-to-end example
# 
# 1. How to work with the filesystem
# 2. How to work with CSV files
# 3. How to do exploratory data analysis (EDA)
# 4. How to display images in a grid
# 5. How to create a Kaggle submission file
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/1vUeDkORVcA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# ## 1. How to work with the filesystem & directories

# In[ ]:


# You can use shell commands with "!"
get_ipython().system('ls ../input')


# In[ ]:


# Pipe output to do basic analysis
get_ipython().system('ls ../input/train/ | wc -l')
get_ipython().system('ls ../input/train/ | head')


# In[ ]:


get_ipython().system('ls ../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_*.png')


# In[ ]:


# Better approach: use pathlib
from pathlib import Path

DATA_DIR = Path('../input')
TRAIN_DIR = DATA_DIR/'train'
TEST_DIR = DATA_DIR/'test'


# In[ ]:


list(set([str(fn).split('/')[-1].split('_')[0] for fn in TEST_DIR.iterdir()]))[:10]


# In[ ]:


# Use the full power of Python
test_ids = list(set([str(fn).split('/')[-1].split('_')[0]  for fn in TEST_DIR.iterdir()]))
print('Test IDs:', len(test_ids))
test_ids[:10]


# In[ ]:


# You can even create directories
SUB_DIR = Path('files/submissions')
SUB_DIR.mkdir(parents=True, exist_ok=True)
get_ipython().system('ls files')


# Learn more here: https://docs.python.org/3/library/pathlib.html

# ## 2. How to work with CSV files

# In[ ]:


# You could always use shell commands
LABELS_CSV = DATA_DIR/'train.csv'
get_ipython().system('head {LABELS_CSV}')


# In[ ]:


# Enter pandas
import pandas as pd

train_df = pd.read_csv(LABELS_CSV, index_col='Id')
train_df.head(10)


# In[ ]:


# You can look at a random sample
train_df.sample(10)


# In[ ]:


# Or get basic information about the data
train_df.info()


# In[ ]:


# Use Python to your advantage
train_df['Target'] = train_df['Target'].str.split(' ').map(lambda x: list(map(int, x)))
train_df.head(10)


# ## 3. How to do Exploratory Data Analysis (EDA)
# 
# Pandas dataframe is a great starting point for doing EDA. It provides many utilities for plotting graphs right out of the box.

# In[ ]:


label_names = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", 
               "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", 
               "Golgi apparatus", "Peroxisomes", "Endosomes","Lysosomes", 
               "Intermediate filaments", "Actin filaments", "Focal adhesion sites", 
               "Microtubules", "Microtubule ends", "Cytokinetic bridge", "Mitotic spindle", 
               "Microtubule organizing center", "Centrosome", "Lipid droplets", 
               "Plasma membrane", "Cell junctions", "Mitochondria", "Aggresome",   
               "Cytosol", "Cytoplasmic bodies", "Rods & rings"]


# In[ ]:


import numpy as np

def get_label_freqs(targets, label_names, ascending=None):
    n_classes = len(label_names)
    freqs = np.array([0] * n_classes)
    for lst in targets:
        for c in range(n_classes):
            freqs[c] += c in lst
    data = {
        'name': label_names, 
        'frequency': freqs, 
        'percent': (10000 * freqs / len(targets)).astype(int) / 100.,
    }
    cols = ['name', 'frequency', 'percent']
    df = pd.DataFrame(data, columns=cols)
    if ascending is not None:
        df = df.sort_values(by='frequency', ascending=ascending)
    return df


# In[ ]:


# Create a frequency table
train_freqs = get_label_freqs(train_df.Target, label_names, ascending=False)
train_freqs


# Clearly, there is a huge imbalance between the classes, and **15 of the 28 classes have less than 900 samples (~ 3% of the data)**, and 9 classes have fewer than 330 samples (~1% of the data). Any model which always predicts 0 or 'not present' for these classes is already 97% accurate.
# 
# So, it's going to be really difficult to train a model that can detect the less frequently occuring classes. This may lead to a recall of 0, which will lead to and F1 score of 0 for these classes, thus putting a ceiling of 0.465 on the evaluation metric. In fact, we might need to train a separate model for these classes.

# In[ ]:


# Visualize the frequency table using a chart
train_freqs.plot(x='name', y='frequency', kind='bar', title='Name vs. Frequency');


# In[ ]:


# Use logarithmic axis for easier interpretation
train_freqs.plot(x='name', y='frequency', kind='bar', logy=True, title='Name vs. log(Frequency)');


# ## 4. How to display an image, or show multiple images in a grid?

# In[ ]:


train_sample = "ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0_red.png"


# In[ ]:


from imageio import imread
import matplotlib.pyplot as plt

# Look at one channel/filter
img0 = imread(str(TRAIN_DIR/train_sample))
print(img0.shape)
plt.imshow(img0)
plt.title(train_sample[0]);


# In[ ]:


# Use a color map for grayscale images
plt.imshow(img0, cmap="Reds");


# In[ ]:


# For RGB images, it "just works"
get_ipython().system('curl https://www.what-dog.net/Images/faces2/scroll001.jpg -o sample.jpg')

img = imread('sample.jpg')
plt.imshow(img);


# In[ ]:


get_ipython().system('ls {TRAIN_DIR}/ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0_*.png')


# In[ ]:


CHANNELS = ['green', 'red', 'blue', 'yellow']

# Load images for multiple channels
def load_image(image_id, channels=CHANNELS, img_dir=TRAIN_DIR):
    image = np.zeros(shape=(len(channels),512,512))
    for i, ch in enumerate(channels):
        image[i,:,:] = imread(str(img_dir/f'{image_id}_{ch}.png'))
    return image


# In[ ]:


# Plot multiple images in a grid
def show_image_filters(image, title, figsize=(16,5)):
    fig, subax = plt.subplots(1, 4, figsize=figsize)
    # Green channel
    subax[0].imshow(image[0], cmap="Greens")
    subax[0].set_title(title)
    # Red channel
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("Microtubules")
    # Blue channel
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("Nucleus")
    # Orange channel
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("Endoplasmatic reticulum")
    return subax


# In[ ]:


# Use the traning data to show appropriate labels
def get_labels(image_id):
    labels = [label_names[x] for x in train_df.loc[image_id]['Target']]
    return ', '.join(labels)


# In[ ]:


# Look at a sample grid
img_id = 'ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0'
img, title = load_image(img_id), get_labels(img_id)
show_image_filters(img, title);
print(img.shape)


# In[ ]:


# Combine with pandas to view a random sample
for img_id in train_df.sample(3).index:
    print(img_id)
    img, title = load_image(img_id), get_labels(img_id)
    show_image_filters(img, title)


# ## 5. How to generate a submission file?

# In[ ]:


# Let's define a sophisticated and highly accurate model
def model(inputs):
    return np.random.randn(len(inputs), len(label_names))


# In[ ]:


# Generate some predictions (logits)
preds = model(test_ids)
print(preds.shape)
print(preds)


# In[ ]:


# Convert them into probabilities
def sigmoid(x):
    return np.reciprocal(np.exp(-x) + 1) 

probs = sigmoid(preds)
probs


# In[ ]:


# Convert probabilities into labels
def make_labels(y, thres=0.5):
    return ' '.join([str(i) for i, p in enumerate(y) if p > thres])

make_labels(probs[0])


# In[ ]:


# Create a pandas dataframe
labels = list(map(make_labels, probs))
sub_df = pd.DataFrame({ 'Id': test_ids, 'Predicted': labels}, columns=['Id', 'Predicted'])
sub_df.head(10)


# In[ ]:


# Export it to a file and make sure it looks okay
sub_fname = SUB_DIR/'basic.csv'
sub_df.to_csv(sub_fname, index=None)

get_ipython().system('head {sub_fname}')


# In[ ]:


# Use FileLink to download the file
from IPython.display import FileLink

FileLink(sub_fname)


# The last but **MOST IMPORTANT** step is to take all of the above code (once it works as expected), and wrap it into a function (or two)

# In[ ]:


def make_sub(fname):
    preds = model(test_ids)
    probs = sigmoid(preds)
    labels = list(map(make_labels, probs))
    sub_df = pd.DataFrame({ 'Id': test_ids, 'Predicted': labels}, columns=['Id', 'Predicted'])
    fpath = SUB_DIR/fname
    sub_df.to_csv(fpath, index=None)
    get_ipython().system('head {fpath}')
    return FileLink(fpath)


# In[ ]:


make_sub('best_submission.csv')


# Now you can generate test predictions with a single line of code!

# # Save and commit
# Finally, we you save and commit out work using Jovian, so that anyone (including you), can reproduce it later with a single command, on any machine.

# In[ ]:


get_ipython().system('pip install jovian --upgrade -q')


# In[ ]:


import jovian


# In[ ]:


jovian.commit()


# In[ ]:




