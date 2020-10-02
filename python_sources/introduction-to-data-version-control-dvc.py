#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Data Version Control
# 
# 
# 
# In this kernel we are planning to introduce about Data Version Control which one of best open source tools available in the market for [Machine Learning
# Models and Dataset Versioning](https://dvc.org/doc/use-cases/data-and-model-files-versioning) and [other amazing features](https://dvc.org/features).
# 
# ![dvc.png](attachment:dvc.png)
# 
# <br>
# <br>

# We are going to play with an actual Machine Learning scenario. It explores the NLP problem of predicting tags for a given StackOverflow
# question. For example, we want one classifier which can predict a post that is about the Python language by tagging it python.
# 
# This kernel has been made adopting [DVC get-Started tutorial](https://dvc.org/doc/get-started/agenda) and full credit goes to DVC team for making that.
# A github repo for get-starter tutorial can be found [here](https://github.com/iterative/example-get-started).

# ## Installing DVC
# 
# Installing DVC is very easy. There are mainly three recommended ways:
# - pip
# - OS-specific package managers
# - HomeBrew(for apple users)
# 
# 
# We are going to install with `pip- Python package manger`. For other installation
# methods checkout [here](https://dvc.org/doc/get-started/install)

# In[ ]:


import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


# Installing DVC
get_ipython().system(' pip install dvc ')


# In[ ]:


# Checking out DVC installation
get_ipython().system(' dvc -h')


# ## Initialising NLP Project

# In[ ]:


get_ipython().system(' mkdir get-started && cd get-started')


# In[ ]:


from pathlib import Path
import os

a = Path.cwd() / "get-started"
os.chdir(a)


# In[ ]:


# Initialising git in our folder
get_ipython().system(' git init')


# In[ ]:


# Run DVC initialization in a repository directory to create the DVC meta files and directories
get_ipython().system(' dvc init')


# In[ ]:


# configuring git for user account
get_ipython().system(' git config --global user.name "kuranbenoy" #Replace with your github username')
get_ipython().system(' git config --global user.email "kurian.bkk@gmail.com" #Replace with your email id')
# commit the initialised git files
get_ipython().system(' git commit -m "initialize DVC"')


# ## Configuring DVC remotes
# 
# 
# A DVC remote is used to share your ML models and datasets with others. The various types of remotes DVC currently supports is:
# https://dvc.org/doc/get-started/configure
# - `local` - Local directory
# - `s3` - Amazon Simple Storage Service
# - `gs` - Google Cloud Storage
# - `azure` - Azure Blob Storage
# - `ssh` - Secure Shell
# - `hdfs` - The Hadoop Distributed File System
# - `http` - Support for HTTP and HTTPS protocolbucks
# 
# > Note we are using remote as a local directory as storage. **It's usually recommended to use Cloud storage services as DVC remote.**
# 
# [More information](https://dvc.org/doc/get-started/configure)

# In[ ]:


get_ipython().system(' dvc remote add -d myremote /tmp/dvc-storage')


# In[ ]:


get_ipython().system(' git commit .dvc/config -m "initialize DVC local remote"')


# ## Downloading files
# 
# 

# In[ ]:


# Download the data
get_ipython().system(' mkdir data/')
get_ipython().system('  dvc get https://github.com/iterative/dataset-registry         get-started/data.xml -o data/data.xml')


# In[ ]:


# add file(directory) to DVC
get_ipython().system(' dvc add data/data.xml')


# In[ ]:


# add DVC files to git and update gitignore
get_ipython().system(' git add data/.gitignore data/data.xml.dvc')
get_ipython().system(' git commit -m "add source data to DVC"')


# [more information](https://dvc.org/doc/get-started/add-files)

# In[ ]:


#  push them from your repository to the default remote storage*:
get_ipython().system(' dvc push')


# ## Retrieving Data
# 
# Now since we pushed our data, we are going to do the opposite of push ie `pull` similar to git analogy.
# An easy way to test it is by removing currently downloaded data.

# In[ ]:


get_ipython().system(' rm -f data/data.xml')


# In[ ]:


# Now your data returns back to repositary
get_ipython().system(' dvc pull')


# In[ ]:


# incase just to retrieve single dataset or file
get_ipython().system(' dvc pull data/data.xml.dvc')


# ## Conncting with code
# 
# For providing full Machine Learning reproducibility. It is important to connect code with Datasets which are being reproducible by
# using commands like `dvc add/push/pull`.
# 
# 

# In[ ]:


# run these commands to get the sample code:
get_ipython().system(' wget wget https://code.dvc.org/get-started/code.zip')
get_ipython().system(' unzip code.zip')
get_ipython().system(' rm -f code.zip')


# Having installed the `src/prepare.py` script in your repo, the following command
# transforms it into a reproducible
# [stage](https://dvc.org/doc/user-guide/dvc-files-and-directories) for the ML pipeline we're
# building (described in detail [in the documentation](https://dvc.org/doc/get-started/example-pipeline)).

# Stages are run using dvc run [command] and options among which we use:
# 
# - d for dependency: specify an input file
# - o for output: specify an output file ignored by git and tracked by dvc
# - M for metric: specify an output file tracked by git
# - f for file: specify the name of the dvc file.
# - command: a bash command, mostly a python script invocation

# In[ ]:


# Create a pipeline to create  folder data/prepared with files train.tsv and test.tsv
get_ipython().system(' dvc run -f prepare.dvc           -d src/prepare.py -d data/data.xml           -o data/prepared           python src/prepare.py data/data.xml')


# In[ ]:


get_ipython().system('  git add data/.gitignore prepare.dvc')
get_ipython().system('  git commit -m "add data preparation stage"')


# In[ ]:


get_ipython().system(' dvc push')


# ## Pipeline
# 
# Using `dvc run` multiple times, and specifying outputs of a command (stage) as dependencies in another one, we can describe a sequence of commands that gets to a desired result.
# This is what we call a data pipeline or computational graph.
# 
# 

# In[ ]:


# Lets create a second stage (after prepare.dvc, created in the previous chapter) to perform feature extraction
get_ipython().system(' dvc run -f featurize.dvc           -d src/featurization.py -d data/prepared/           -o data/features            python src/featurization.py data/prepared data/features')


# In[ ]:


# A third stage for training the model
get_ipython().system(' dvc run -f train.dvc           -d src/train.py -d data/features           -o model.pkl           python src/train.py data/features model.pkl')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'git add data/.gitignore .gitignore featurize.dvc train.dvc\ngit commit -m "add featurization and train steps to the pipeline"\ndvc push')


# ## Pipelines Visualisation

# In[ ]:


get_ipython().system(' dvc pipeline show --ascii train.dvc ')


# ## Metrics
# 
# The last stage we would like to add to our pipeline is its the evaluation. Data science is a metric-driven R&D-like process and `dvc metrics` along with DVC metric 
# files provide a framework to capture and compare experiments performance.
# 

# `evaluate.py` calculates AUC value using the test data set. It reads features from the `features/test.pkl` file and produces a DVC metric file - `auc.metric`. It is a special DVC output file type, in this case it's just a plain text file with a single number inside.

# In[ ]:


get_ipython().system(' dvc run -f evaluate.dvc           -d src/evaluate.py -d model.pkl -d data/features           -M auc.metric           python src/evaluate.py model.pkl                  data/features auc.metric')


# > Please, refer to the [dvc metrics](https://dvc.org/doc/commands-reference/metrics) command documentation to see more available options and details.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'git add evaluate.dvc auc.metric\ngit commit -m "add evaluation step to the pipeline"')


# In[ ]:


get_ipython().system(' dvc push')


# In[ ]:


# Tag as a checkpoint to cpmpare further experiments
get_ipython().system(' git tag -a "baseline-experiment" -m "baseline"')


# ## Experiments
# 
# Data science process is inherently iterative and R&D like - data scientist may try many different approaches, different hyper-parameter values and "fail" 
# many times before the required level of a metric is achieved.

# We are modifying our feature extraction of our files. Inorder to use `bigrams`. We are increasing no of features and n_gram_range in our file `src/featurization.py`.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'src/featurization.py', "import os\nimport sys\nimport errno\nimport pandas as pd\nimport numpy as np\nimport scipy.sparse as sparse\n\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.feature_extraction.text import TfidfTransformer\n\ntry:\n    import cPickle as pickle\nexcept ImportError:\n    import pickle\n\nnp.set_printoptions(suppress=True)\n\nif len(sys.argv) != 3 and len(sys.argv) != 5:\n    sys.stderr.write('Arguments error. Usage:\\n')\n    sys.stderr.write('\\tpython featurization.py data-dir-path features-dir-path\\n')\n    sys.exit(1)\n\ntrain_input = os.path.join(sys.argv[1], 'train.tsv')\ntest_input = os.path.join(sys.argv[1], 'test.tsv')\ntrain_output = os.path.join(sys.argv[2], 'train.pkl')\ntest_output = os.path.join(sys.argv[2], 'test.pkl')\n\ntry:\n    reload(sys)\n    sys.setdefaultencoding('utf-8')\nexcept NameError:\n    pass\n\n\ndef mkdir_p(path):\n    try:\n        os.makedirs(path)\n    except OSError as exc:  # Python >2.5\n        if exc.errno == errno.EEXIST and os.path.isdir(path):\n            pass\n        else:\n            raise\n\n\ndef get_df(data):\n    df = pd.read_csv(\n        data,\n        encoding='utf-8',\n        header=None,\n        delimiter='\\t',\n        names=['id', 'label', 'text']\n    )\n    sys.stderr.write('The input data frame {} size is {}\\n'.format(data, df.shape))\n    return df\n\n\ndef save_matrix(df, matrix, output):\n    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T\n    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T\n\n    result = sparse.hstack([id_matrix, label_matrix, matrix], format='csr')\n\n    msg = 'The output matrix {} size is {} and data type is {}\\n'\n    sys.stderr.write(msg.format(output, result.shape, result.dtype))\n\n    with open(output, 'wb') as fd:\n        pickle.dump(result, fd, pickle.HIGHEST_PROTOCOL)\n    pass\n\n\nmkdir_p(sys.argv[2])\n\n# Generate train feature matrix\ndf_train = get_df(train_input)\ntrain_words = np.array(df_train.text.str.lower().values.astype('U'))\n\nbag_of_words = CountVectorizer(stop_words='english',\n                               max_features=5000,\n                              ngram_range=(1, 2),)\nbag_of_words.fit(train_words)\ntrain_words_binary_matrix = bag_of_words.transform(train_words)\ntfidf = TfidfTransformer(smooth_idf=False)\ntfidf.fit(train_words_binary_matrix)\ntrain_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)\n\nsave_matrix(df_train, train_words_tfidf_matrix, train_output)\n\n# Generate test feature matrix\ndf_test = get_df(test_input)\ntest_words = np.array(df_test.text.str.lower().values.astype('U'))\ntest_words_binary_matrix = bag_of_words.transform(test_words)\ntest_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)\n\nsave_matrix(df_test, test_words_tfidf_matrix, test_output)\n")


# ## Reproduce
# 
# We described our first pipeline. Basically, we created a number of DVC-file. Each file describes a single stage we need to run (a pipeline) towards a final result.
# Each depends on some data (either source data files or some intermediate results from another DVC-file file) and code files.

# In[ ]:


# Using DVC Repro here 
get_ipython().system(' dvc repro train.dvc')


# In[ ]:


get_ipython().system(' git commit -a -m "bigram model"')


# In[ ]:


get_ipython().system(' git checkout baseline-experiment')
get_ipython().system(' dvc checkout')


# ## Compare Expermiments
# 
# DVC makes it easy to iterate on your project using Git commits with tags or Git branches. It provides a way to try different ideas, keep track of them, 
# switch back and forth. To find the best performing experiment or track the progress, a special metric output type is supported in 
# DVC (described in one of the previous steps).

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'git checkout master\ndvc checkout\ndvc repro evaluate.dvc')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'git commit -a -m "evaluate bigram model"\ngit tag -a "bigram-experiment" -m "bigrams"')


# In[ ]:


get_ipython().system(' dvc metrics show -T')


# ## Get older Data files
# 
# The answer is the `dvc checkout` command, and we already touched briefly the process of switching between different data versions in the Experiments step of this get started guide.

# In[ ]:


get_ipython().system(' git checkout baseline-experiment train.dvc')
get_ipython().system(' dvc checkout train.dvc')


# In[ ]:


get_ipython().system(' git checkout baseline-experiment')
get_ipython().system(' dvc checkout')


# ## fin.

# References
# ----------
# 
# - https://dvc.org/doc/get-started
# - https://medium.com/qonto-engineering/using-dvc-to-create-an-efficient-version-control-system-for-data-projects-96efd94355fe
