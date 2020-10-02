#!/usr/bin/env python
# coding: utf-8

# In[ ]:



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 8427,
      "digest": "sha256:32a1a53fdb28035db0bdb568c202ca7f13195427d01962a8f62e25bc9d6622eb"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 51081712,
         "digest": "sha256:ab85f11dd933cfda19130a6b6986c591e8e70d8a2ac8a73dddd8722c7ec8eab1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 825,
         "digest": "sha256:2922171b2513070bd982f9030107e0a9b22fa141ce22fe185cea0b22dc34dc32"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 442,
         "digest": "sha256:03dd2bb85c8ab012f80afe6dee177f3c05e20bdfc059beeb2c03ad6ca14189f2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 676,
         "digest": "sha256:4a9039a61c36f62d62cb71a8087c71fc6046b0918617a8ae13f20fa48060983f"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 163,
         "digest": "sha256:82e5f1ef816256c961c51a8deae0ff5c038e7a4e92f0b52bf1d60340f21d9502"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 833526336,
         "digest": "sha256:67eed9560bab0b1bf847ffbc73f384a2841d3680a37a3681c9ac8cd89e38e46d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 568,
         "digest": "sha256:ce8abd2c7b1cd94ca3fc88a34ea7c6d451b197b28bc02c5f6cf9714b95996ecd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 343945402,
         "digest": "sha256:11bc6f91b846cb9c1efddd0e132773099bc29e577eca085742566e5c258e8321"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3029994,
         "digest": "sha256:91433bc796c21fe3d7b513c5aab421b10c784bb22712ec1b68698b0910b8116e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 57935,
         "digest": "sha256:1e5c91351872e093c596d34ad80f63f8a539a57b18fba87a7ab708aa79983ed1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 81865183,
         "digest": "sha256:fb8c8966667d0589428e3f40598742586a49b4e484339cdac3196c2c22317394"
      }
   ]
}
l


# In[ ]:


#!/bin/bash
set -e
 
usage() {
cat << EOF
Usage: $0 [LABEL]
Push a newly-built image with the given LABEL to gcr.io and DockerHub.
See CHANGELOG.md file for LABEL naming convention.
EOF
}
 
set -x
 
SOURCE_IMAGE='kaggle/python-tensorflow-whl'
TARGET_IMAGE='gcr.io/kaggle-images/python-tensorflow-whl'
 
while :; do
    case "$1" in 
        -h|--help)
            usage
            exit
            (";")
        -?*)
            usage
            printf 'ERROR: Unknown option: %s\n' "$1" >&2
            exit
            (";")
        *)            
            break
    esac
 
    shift
done
 
LABEL=$1
 
if [[ -z "$LABEL" ]]; then
    echo 'You must provide a label for the image'
    exit 1
fi
 
readonly SOURCE_IMAGE
readonly TARGET_IMAGE
readonly LABEL
 
set -x
docker tag "$SOURCE_IMAGE:latest" "$TARGET_IMAGE:$LABEL"
gcloud docker -- push "$TARGET_IMAGE:$LABEL"
 


# In[ ]:



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 14966,
      "digest": "sha256:e517b3e3aa88208daffb56b528c4c8699db05eae75ec488ad76de0eb89fcfb05"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 50382957,
         "digest": "sha256:7e2b2a5af8f65687add6d864d5841067e23bd435eb1a051be6fe1ea2384946b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 222909892,
         "digest": "sha256:59c89b5f9b0c6d94c77d4c3a42986d420aaa7575ac65fcd2c3f5968b3726abfc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 195204532,
         "digest": "sha256:4017849f9f85133e68a4125e9679775f8e46a17dcdb8c2a52bbe72d0198f5e68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1522,
         "digest": "sha256:c8b29d62979a416da925e526364a332b13f8d5f43804ae98964de2a60d47c17a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 717,
         "digest": "sha256:12004028a6a740ac35e69f489093b860968cc37b9668f65b1e2f61fd4c4ad25c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247,
         "digest": "sha256:3f09b9a53dfb03fd34e35d43694c2d38656f7431efce0e6647c47efb5f7b3137"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 408,
         "digest": "sha256:03ed58116b0cb733cc552dc89ef5ea122b6c5cf39ec467f6ad671dc0ba35db0c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 331594702,
         "digest": "sha256:7844554d9ef75bb3f1d224e166ed12561e78add339448c52a8e5679943b229f1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 112943172,
         "digest": "sha256:f24d2d75ce9a9227d1ad5cbf6078b83e411a5e4f3e3376f7624e6d7073773c2e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 425,
         "digest": "sha256:b89ff65d69ce89fe9d05fe3acf9f89046a19eaed148e80a6e167b93e6dc26423"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 5476,
         "digest": "sha256:d7a15e9b63f265b3f895e4c9f02533d105d9b277e411b93e81bb98972018d11a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1950,
         "digest": "sha256:fc930f8fc7540d4c91453d7626df54a41228b7004396f6a061071c76b6d34f99"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2471113264,
         "digest": "sha256:c441d8c511008ca9d8acac88564913e46cefceb4b33fd278b0b7665d16b9f675"
      }
   ]
}


# In[ ]:


#!/bin/bash
set -e
 
usage() {
cat << EOF
Usage: $0 [OPTIONS]
Compare a given Docker image package versions against the prod image.
 
Options:
    -g, --gpu       Compare GPU images.
    -t, --target    The image to diff against the prod image.
                    Default is the locally built image.
EOF
}
 
 
BASE_IMAGE_TAG='gcr.io/kaggle-images/python:latest'
TARGET_IMAGE_TAG='kaggle/python-build'
TARGET_IMAGE_TAG_OVERRIDE=''
 
while :; do
    case "$1" in 
        -h|--help)
            usage
            exit
            (";")
        -g|--gpu)
            BASE_IMAGE_TAG='gcr.io/kaggle-private-byod/python:latest'
            TARGET_IMAGE_TAG='kaggle/python-gpu-build'
            (";")
        -t|--target)
            if [[ -z "$2" ]]; then
                usage
                printf 'ERROR: No IMAGE specified after the %s flag.\n' "$1" >&2
                exit
            fi
            TARGET_IMAGE_TAG_OVERRIDE="$2"
            shift # skip the flag value
            (";")
        -?*)
            usage
            printf 'ERROR: Unknown option: %s\n' "$1" >&2
            exit
            (";")
        *)            
            break
    esac
 
    shift
done
 
if [[ -n "$TARGET_IMAGE_TAG_OVERRIDE" ]]; then
    TARGET_IMAGE_TAG="$TARGET_IMAGE_TAG_OVERRIDE"
fi
 
readonly BASE_IMAGE_TAG
readonly TARGET_IMAGE_TAG
 
echo "Base: $BASE_IMAGE_TAG"
echo "Target: $TARGET_IMAGE_TAG"
 
docker pull "$BASE_IMAGE_TAG"
 
CMDS=('dpkg-query --show -f "${Package}==${Version}\n"' 'pip freeze')
for cmd in "${CMDS[@]}"; do
    echo "== Comparing $cmd =="
    diff --suppress-common-lines --side-by-side         <(docker run --rm "$BASE_IMAGE_TAG" $cmd)         <(docker run --rm "$TARGET_IMAGE_TAG" $cmd)         && echo 'No diff' || true
done
 


# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There are 5 csv files in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: /kaggle/input/predict.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# predict.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/predict.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'predict.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 6, 15)


# ### Let's check 2nd file: /kaggle/input/prepared_test.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# prepared_test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/prepared_test.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'prepared_test.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df2.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df2, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df2, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df2, 20, 10)


# ### Let's check 3rd file: /kaggle/input/prepared_train.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# prepared_train.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/prepared_train.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'prepared_train.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df3.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df3, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df3, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df3, 20, 10)


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
