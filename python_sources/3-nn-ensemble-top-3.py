#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# This is my solution that I am glad to share.This solution got me to the top 3 % in the private leaderboard..so I hope you benefit from it.. My solution is an ensemble of 3 Neural Network Models 2 CNNs with nearly same architecture and different hyperparameters and 1 RNN..then I blended the 3 predictions using a simple linear regressor

# ## Import dependencies

# In[ ]:


get_ipython().run_cell_magic('time', '', "import pandas as pd\nimport numpy as np\nfrom nltk.corpus import stopwords\nfrom keras.preprocessing.text import Tokenizer\nfrom keras.layers import GRU,Conv1D,MaxPooling1D,Input,Dense,Dropout\nfrom sklearn.metrics import mean_squared_error\nimport datetime\ntime_start=datetime.datetime.now()\nfrom tensorflow import set_random_seed\nimport tensorflow as tf\nsession_conf = tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=1)\nfrom keras import backend\nbackend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))\nfrom collections import Counter\nimport os\nos.environ['PYTHONHASHSEED'] = '123'\nnp.random.seed(123)\nimport random\nnp.random.seed(123)\nrandom.seed(123)\nset_random_seed(123)")


# # Credit and Big Thanks
# This kernel https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755 has helped me a lot specially about the feature engineering,NN architecture and ensembling..Long story short..it was such an inspiration so big thanks for such kernel

# ## Reading train and test data

# In[ ]:


get_ipython().run_cell_magic('time', '', "np.random.seed(123)\ntrain = pd.read_csv('../input/train.tsv',sep='\\t')\ntest = pd.read_csv('../input/test.tsv',sep='\\t')\ntrain=train[(train.price>=3)]\n#train=train[train.price<2000]\nfull_df=pd.concat([train,test],ignore_index=True,axis=0)")


# ## Preprocess Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "all_brands = set(full_df['brand_name'].values)\nfull_df.brand_name.fillna('missing',inplace=True)\n# get to finding!\npremissing = len(full_df.loc[full_df.brand_name == 'missing'])\ndef brandfinder(line):\n    brand = line[0]\n    name = line[1]\n    namesplit = name.split(' ')\n    if brand == 'missing':\n        for x in namesplit:\n            if x in all_brands:\n                return name\n    if name in all_brands:\n        return name\n    return brand\nfull_df['brand_name'] = full_df[['brand_name','name']].apply(brandfinder, axis = 1)\nfound = premissing-len(full_df.loc[full_df['brand_name'] == 'missing'])\nprint(found)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def preprocess_data(df):\n    df.item_description.replace('No description yet',value='missing',inplace=True)\n    df.item_description.fillna('missing',inplace=True)\n    #df.brand_name.fillna('missing',inplace=True)\n    df.category_name.fillna('missing/missing/missing',inplace=True)\n    df['main_category']=df.category_name.apply(lambda x:x.split('/')[0])\n    df['sub_category1']=df.category_name.apply(lambda x:x.split('/')[1])\n    df['sub_category2']=df.category_name.apply(lambda x:x.split('/')[2])\npreprocess_data(full_df)")


# ## Tokenizer

# ## Notes
# * I tried to preprocess text and remove stopwords using regex since removing them through NLTK was too time consuming and time is an important factor in such competition.
# * Also stemming and lemmatizing text didn't work because of the same reason and I am not sure if it would help the models. 
# * Another surprising thing to me that the models performance were better when initializing embeddings from scratch other tha using pretrained vectors as Word2Vec/Glove/FastText 
# * I concatenated the words from train and test when fitting keras tokenizer and chose the maximum number of words removing only the words that occured once
# * I used the description length and name length as features

# In[ ]:


get_ipython().run_cell_magic('time', '', "print('preprocessing Text')\n#convert to lowercase\n#full_df.item_description=full_df.item_description.apply(lambda x:x.lower())\n#full_df.name=full_df.name.apply(lambda x:x.lower())\n#full_df.item_description.replace('\\W+',' ',regex=True,inplace=True)\n#full_df.item_description.replace('\\s+',' ',inplace=True,regex=True)\n#full_df.name.replace('\\W+',' ',regex=True,inplace=True)\n#full_df.name.replace('\\s+',' ',inplace=True,regex=True)\n#full_df.item_description.replace(r'\\bt\\sshirt\\b','t-shirt',inplace=True,regex=True)\n#full_df.name.replace(r'\\bt\\sshirt\\b',' t-shirt',inplace=True,regex=True)\n#full_df.item_description.replace('\\sgb','gb',inplace=True,regex=True)\n#full_df.name.replace('\\sgb','gb',inplace=True,regex=True)\n#full_df.item_description.replace('[rm]',' ',inplace=True)\n#full_df.item_description.replace('rm',' ',inplace=True)\n#full_df.item_description.replace('.*\\d+.*',' ',regex=True,inplace=True)\n#full_df.name.replace('.*\\d+.*',' ',regex=True,inplace=True)\n#full_df.item_description.replace('\\s+',' ',inplace=True,regex=True)\n\n#stopwords=stopwords.words('english')\n#pat = r'\\b(?:{})\\b'.format('|'.join(stopwords))\n#full_df.item_description.replace(pat,' ',inplace=True,regex=True)\n#full_df.name.replace(pat,' ',inplace=True,regex=True)\nraw_text=np.hstack([full_df.item_description.str.lower(),full_df.name.str.lower()])\nprint('Fitting Tokenzier')\ntokenizer=Tokenizer()\ntokenizer.fit_on_texts(raw_text)\nword_index=tokenizer.word_index\nword_counts=tokenizer.word_counts\nc=Counter(sorted(word_counts.values()))\ntokenizer.num_words=len(word_index)-c[1]\nNUM_WORDS=tokenizer.num_words\n#tokenizer_cat=Tokenizer()\n#tokenizer_cat.fit_on_texts(category)\nprint('transforming to sequences')\nn_train=train.shape[0]\nfull_df['seq_name']=tokenizer.texts_to_sequences(full_df.name.str.lower())\nfull_df['seq_desc']=tokenizer.texts_to_sequences(full_df.item_description.str.lower())\n#full_df['category_seq']=tokenizer_cat.texts_to_sequences(full_df.category_nameen\nfull_df['name_len']=full_df.seq_name.apply(lambda x: len(x))\nfull_df['desc_len']=full_df.seq_desc.apply(lambda x:len(x))\n\ndel tokenizer\ndel word_index\ndel c\ndel word_counts")


# ## Label Encoding

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.preprocessing import LabelEncoder\nprint("Handling categorical variables")\nle=LabelEncoder()\nfull_df[\'category_name\']=le.fit_transform(full_df.category_name)\nfull_df[\'brand_name\'] = le.fit_transform(full_df.brand_name)\nfull_df[\'main_category\']=le.fit_transform(full_df.main_category)\nfull_df[\'sub_category1\']=le.fit_transform(full_df.sub_category1)\nfull_df[\'sub_category2\']=le.fit_transform(full_df.sub_category2)\ndel le')


# ## train test split

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import train_test_split\ntrain=full_df[:n_train]\ntest=full_df[n_train:]\ndtrain,dvalid=train_test_split(train,random_state=123,test_size=0.01)')


# ## Embeddings MaxValues

# In[ ]:


get_ipython().run_cell_magic('time', '', 'MAX_NAME_SEQ = 10\nMAX_ITEM_DESC_SEQ = 75\n#MAX_TEXT = np.max([np.max(train.seq_name.max())\n#                   , np.max(test.seq_name.max())\n#                  , np.max(train.seq_desc.max())\n#                  , np.max(test.seq_desc.max())])+2\n#MAX_TEXT=NUM_WORDS+1\nif NUM_WORDS is None:\n    MAX_TEXT=len(word_index)+1\nelse:\n    MAX_TEXT=NUM_WORDS+1\nMAX_CATEGORY_MAIN = np.max([full_df.main_category.max()])+1\nMAX_CATEGORY_SUB1 = np.max([full_df.sub_category1.max()])+1\nMAX_CATEGORY_SUB2 = np.max([full_df.sub_category2.max()])+1\nMAX_BRAND = np.max([full_df.brand_name.max()])+1\nMAX_CONDITION = np.max([full_df.item_condition_id.max()])+1\nMAX_DESC_LEN = np.max([full_df.desc_len.max()])+1\nMAX_NAME_LEN = np.max([full_df.name_len.max()])+1')


# ## Keras Data Notes
# 
# * As you will notice there are two item condition variables called 'item_condition' and 'item_condition2'.One of them was one hot encoded as it helped the CNN models

# ## Get Keras data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from keras.preprocessing.sequence import pad_sequences\nfrom keras.utils import to_categorical\n\ndef get_keras_data(dataset):\n    X = {\n        \'name\': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),\n        \'item_desc\': pad_sequences(dataset.seq_desc, maxlen=MAX_ITEM_DESC_SEQ),\n        \'brand_name\': np.array(dataset[["brand_name"]]),\n        \'item_condition\': to_categorical(dataset[["item_condition_id"]]),\n        \'num_vars\': np.array(dataset[["shipping"]]),\n        \'desc_len\': np.array(dataset[["desc_len"]]),\n        \'name_len\': np.array(dataset[["name_len"]]),\n        \'main_category\':np.array(dataset[["main_category"]]),\n        \'sub_category1\': np.array(dataset[["sub_category1"]]),\n        \'sub_category2\':np.array(dataset[["sub_category2"]]),\n        \'category_name\':np.array(dataset[["category_name"]]),\n        \'item_condition2\':np.array(dataset[["item_condition_id"]]),\n    }\n    return X\n\nX_train = get_keras_data(dtrain)\ny_train=np.log1p(dtrain.price)\nX_valid = get_keras_data(dvalid)\ny_valid = np.log1p(dvalid.price)\nX_test = get_keras_data(test)\ntest_id=test[\'test_id\'].astype(\'int32\')\ndel dtrain\ndel dvalid\ndel test\ndel train\ndel full_df')


# ## NN Models
# Surprisingly the 2 CNN models outperformed the RNN model in this specific dataset.Another important factor that helped me improve the score was the hyperparameter tuning specially the kernel initialization for embeddings and Fully connected layers.Batch Normalization for description length and name length improved the score for CNN.
# 
# 
# **using dropout didn't improve the score so I set it to zero

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from keras.layers import Embedding,GlobalMaxPooling1D,concatenate,Flatten,SpatialDropout1D,Bidirectional,MaxPooling1D,LSTM,BatchNormalization,GlobalAveragePooling1D\nfrom keras.utils import to_categorical\nfrom keras.models import Model\nfrom keras.optimizers import Adam,SGD,Adamax\nfrom keras import regularizers\nfrom keras.initializers import Constant\ndef get_cnn_model(lr=0.001,decay=0):\n    dr_r=0\n    #Inputs\n    name=Input(shape=[X_train[\'name\'].shape[1]],name=\'name\')\n    category_name=Input(shape=[1],name=\'category_name\')\n    desc=Input(shape=[X_train[\'item_desc\'].shape[1]],name=\'item_desc\')\n    brand=Input(shape=[1],name=\'brand_name\')\n    item_condition=Input(shape=[X_train[\'item_condition\'].shape[1]],name=\'item_condition\')\n    shipping=Input(shape=[X_train[\'num_vars\'].shape[1]],name=\'num_vars\')\n    desc_len=Input(shape=[1],name=\'desc_len\')\n    name_len=Input(shape=[1],name=\'name_len\')\n    main_category=Input(shape=[1],name=\'main_category\')\n    sub_category1=Input(shape=[1],name=\'sub_category1\')\n    sub_category2=Input(shape=[1],name=\'sub_category2\')\n    #Embeddings\n    emb_name=Embedding(MAX_TEXT,50,embeddings_initializer=\'glorot_uniform\')(name)\n    emb_desc=Embedding(MAX_TEXT,50,embeddings_initializer=\'glorot_uniform\')(desc)\n    emb_brand=Embedding(MAX_BRAND,10,embeddings_initializer=\'glorot_uniform\')(brand)\n    emb_desc_len=Embedding(MAX_DESC_LEN,5,embeddings_initializer=\'glorot_uniform\')(desc_len)\n    emb_name_len=Embedding(MAX_NAME_LEN,5,embeddings_initializer=\'glorot_uniform\')(name_len)\n    emb_main_category=Embedding(MAX_CATEGORY_MAIN,10,embeddings_initializer=\'glorot_uniform\')(main_category)\n    emb_sub_category1=Embedding(MAX_CATEGORY_SUB1,10,embeddings_initializer=\'glorot_uniform\')(sub_category1)\n    emb_sub_category2=Embedding(MAX_CATEGORY_SUB2,10,embeddings_initializer=\'glorot_uniform\')(sub_category2)\n    emb_item_condition =Embedding(MAX_CONDITION, 5,embeddings_initializer=\'glorot_uniform\')(item_condition)\n    #emb_category_name=BatchNormalization()(category_name)#Embedding(MAX_CATEGORY_TEXT,20)(category_name)\n    #CNN\n    conv_name=Conv1D(filters=32,kernel_size=3,activation=None,kernel_initializer=\'glorot_uniform\')(emb_name)\n    #conv_name=Conv1D(filters=16,kernel_size=3,activation=\'relu\',kernel_initializer=\'glorot_uniform\')(conv_name)\n    \n    max_pooling_name=GlobalMaxPooling1D()(conv_name)\n    conv_desc=Conv1D(filters=64,kernel_size=3,activation=None,kernel_initializer=\'glorot_uniform\')(emb_desc)\n    #conv_desc=Conv1D(filters=32,kernel_size=3,activation=\'relu\',kernel_initializer=\'glorot_uniform\')(conv_desc)\n    max_pooling_desc=GlobalMaxPooling1D()(conv_desc)\n    #conv_category_name=Conv1D(filters=8,kernel_size=3,activation=\'relu\')(emb_category_name)\n    #max_pooling_category_name=GlobalMaxPooling1D()(conv_category_name)\n    #conv_sub_category1=Conv1D(filters=8,kernel_size=3,activation=\'relu\')(emb_sub_category1)\n    #max_pooling_sub_category1=GlobalMaxPooling1D()(conv_sub_category1)\n    #conv_sub_category2=Conv1D(filters=8,kernel_size=3,activation=\'relu\')(emb_sub_category2)\n    #max_pooling_sub_category2=GlobalMaxPooling1D()(conv_sub_category2)\n    \n    #Fully Connected Neural Netwo)rks\n    main_l=concatenate([\n        max_pooling_name,\n        max_pooling_desc,\n        Flatten()(emb_main_category),\n        Flatten()(emb_sub_category1),\n        Flatten()(emb_sub_category2),\n        Flatten()(emb_name_len),\n        Flatten()(emb_desc_len),\n        Flatten()(emb_brand),\n        item_condition,\n        shipping,\n    #    b_name_len2,\n    #    b_desc_len2\n    ])\n    #main_l=BatchNormalization()(main_l)\n    main_l=Dropout(dr_r)(Dense(256,activation=\'relu\',kernel_initializer=\'he_uniform\')(main_l))\n    #main_l=BatchNormalization()(main_l)\n    main_l=Dropout(dr_r)(Dense(128,activation=\'relu\',kernel_initializer=\'he_uniform\')(main_l))\n    #main_l=BatchNormalization()(main_l)\n    #main_l=Dropout(dr_r)(Dense(64,activation=\'relu\')(main_l))\n\n    output=Dense(1,kernel_initializer=\'he_uniform\')(main_l)\n    \n    cnn_model=Model(inputs=[name,desc,brand,item_condition,shipping,desc_len,name_len,category_name,main_category,sub_category1,sub_category2],outputs=output)\n    optimizer=Adam(lr=lr,decay=decay)\n    cnn_model.compile(loss="mse", optimizer=optimizer)\n    \n    return cnn_model\ndef new_rnn_model(lr=0.001, decay=0.0):\n    dr_r=0\n    name = Input(shape=[X_train["name"].shape[1]], name="name")\n    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")\n    brand_name = Input(shape=[1], name="brand_name")\n    item_condition = Input(shape=[1], name="item_condition2")\n    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")\n    desc_len = Input(shape=[1], name="desc_len")\n    name_len = Input(shape=[1], name="name_len")\n    subcat_0 = Input(shape=[1], name="main_category")\n    subcat_1 = Input(shape=[1], name="sub_category1")\n    subcat_2 = Input(shape=[1], name="sub_category2")\n    #new inputs\n    brand_enc=Input(shape=[1],name=\'brand_name_enc\')\n    item_condition_enc=Input(shape=[1],name=\'item_condition_enc\')\n    shipping_enc=Input(shape=[1],name=\'shipping_enc\')\n    main_category_enc=Input(shape=[1],name=\'main_category_enc\')\n    sub_category1_enc=Input(shape=[1],name=\'sub_category1_enc\')\n    sub_category2_enc=Input(shape=[1],name=\'sub_category2_enc\')\n    # Embeddings layers (adjust outputs to help model)\n    emb_name = Embedding(MAX_TEXT, 50,name=\'emb_name\',embeddings_initializer=\'glorot_uniform\')(name)\n    emb_item_desc = Embedding(MAX_TEXT, 50,name=\'emb_item_desc\',embeddings_initializer=\'glorot_uniform\')(item_desc)\n    emb_brand_name = Embedding(MAX_BRAND, 10,name=\'emb_brand_name\',embeddings_initializer=\'glorot_uniform\')(brand_name)\n    emb_item_condition = Embedding(MAX_CONDITION,5,name=\'emb_item_condition\',embeddings_initializer=\'glorot_uniform\')(item_condition)\n    emb_desc_len = Embedding(MAX_DESC_LEN, 5,name=\'emb_desc_len\',embeddings_initializer=\'glorot_uniform\')(desc_len)\n    emb_name_len = Embedding(MAX_NAME_LEN, 5,name=\'emb_name_len\',embeddings_initializer=\'glorot_uniform\')(name_len)\n    emb_subcat_0 = Embedding(MAX_CATEGORY_MAIN, 10,name=\'emb_sub_cat_0\',embeddings_initializer=\'glorot_uniform\')(subcat_0)\n    emb_subcat_1 = Embedding(MAX_CATEGORY_SUB1, 10,name=\'emb_sub_cat_1\',embeddings_initializer=\'glorot_uniform\')(subcat_1)\n    emb_subcat_2 = Embedding(MAX_CATEGORY_SUB2, 10,name=\'emb_sub_cat_2\',embeddings_initializer=\'glorot_uniform\')(subcat_2)\n    \n\n    # rnn layers (GRUs are faster than LSTMs and speed is important here)\n    #conv_name=Conv1D(filters=32,kernel_size=3,activation=None,kernel_initializer=\'glorot_uniform\')(emb_name)\n    #max_pooling_name=MaxPooling1D(pool_size=3)(conv_name)\n    #conv_desc=Conv1D(filters=32,kernel_size=3,activation=None,kernel_initializer=\'glorot_uniform\')(emb_item_desc)\n    #max_pooling_desc=MaxPooling1D(pool_size=3)(conv_desc)\n    rnn_layer1 = GRU(16,kernel_initializer=\'glorot_uniform\') (emb_item_desc)\n    #rnn_layer1 = GRU(16,kernel_initializer=\'glorot_uniform\')(rnn_layer1)\n    rnn_layer2 = GRU(8,kernel_initializer=\'glorot_uniform\') (emb_name)\n    #rnn_layer2 = GRU(8,kernel_initializer=\'glorot_uniform\') (rnn_layer2)\n    # main layers\n    main_l = concatenate([\n        Flatten() (emb_brand_name)\n        , Flatten()(emb_item_condition)\n        , Flatten()(emb_desc_len)\n        , Flatten()(emb_name_len)\n        , Flatten() (emb_subcat_0)\n        , Flatten() (emb_subcat_1)\n        , Flatten() (emb_subcat_2)\n        , rnn_layer1\n        , rnn_layer2\n        , num_vars,\n    ])\n    \n    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)\n    main_l = Dropout(dr_r)(Dense(512,kernel_initializer=\'he_uniform\',activation=\'relu\') (main_l))\n    main_l = Dropout(dr_r)(Dense(256,kernel_initializer=\'he_uniform\',activation=\'relu\') (main_l))\n    main_l = Dropout(dr_r)(Dense(128,kernel_initializer=\'he_uniform\',activation=\'relu\') (main_l))\n    main_l = Dropout(dr_r)(Dense(64,kernel_initializer=\'he_uniform\',activation=\'relu\') (main_l))\n    #main_l = Dropout(dr_r)(Dense(32,kernel_initializer=\'he_uniform\',activation=\'relu\') (main_l))\n    # the output layer.\n    output = Dense(1,activation="linear") (main_l)\n    \n    model = Model([name, item_desc, brand_name , item_condition, \n                   num_vars, desc_len, name_len, subcat_0, subcat_1, \n                   subcat_2], output)\n\n    optimizer = Adam(lr=lr, decay=decay)\n    \n    # (mean squared error loss function works as well as custom functions)  \n    model.compile(loss = \'mse\', optimizer = optimizer)\n\n    return model\ndef get_cnn_model2(lr=0.001,decay=0):\n    dr_r=0\n    #Inputs\n    name=Input(shape=[X_train[\'name\'].shape[1]],name=\'name\')\n    category_name=Input(shape=[1],name=\'category_name\')\n    desc=Input(shape=[X_train[\'item_desc\'].shape[1]],name=\'item_desc\')\n    brand=Input(shape=[1],name=\'brand_name\')\n    item_condition=Input(shape=[X_train[\'item_condition\'].shape[1]],name=\'item_condition\')\n    shipping=Input(shape=[X_train[\'num_vars\'].shape[1]],name=\'num_vars\')\n    desc_len=Input(shape=[1],name=\'desc_len\')\n    name_len=Input(shape=[1],name=\'name_len\')\n    main_category=Input(shape=[1],name=\'main_category\')\n    sub_category1=Input(shape=[1],name=\'sub_category1\')\n    sub_category2=Input(shape=[1],name=\'sub_category2\')\n    #Embeddings\n    emb_name=Embedding(MAX_TEXT,50,embeddings_initializer=\'glorot_uniform\')(name)\n    emb_desc=Embedding(MAX_TEXT,50,embeddings_initializer=\'glorot_uniform\')(desc)\n    emb_brand=Embedding(MAX_BRAND,10,embeddings_initializer=\'glorot_uniform\')(brand)\n    emb_desc_len=BatchNormalization()(desc_len)\n    emb_name_len=BatchNormalization()(name_len)\n    emb_main_category=Embedding(MAX_CATEGORY_MAIN,10,embeddings_initializer=\'glorot_uniform\')(main_category)\n    emb_sub_category1=Embedding(MAX_CATEGORY_SUB1,10,embeddings_initializer=\'glorot_uniform\')(sub_category1)\n    emb_sub_category2=Embedding(MAX_CATEGORY_SUB2,10,embeddings_initializer=\'glorot_uniform\')(sub_category2)\n    emb_item_condition = Embedding(MAX_CONDITION, 5,embeddings_initializer=\'glorot_uniform\')(item_condition)\n    #emb_category_name=Embedding(MAX_CATEGORY_TEXT,20)(category_name)\n    #CNN\n    conv_name=Conv1D(filters=32,kernel_size=3,activation=None,kernel_initializer=\'glorot_uniform\')(emb_name)\n    #conv_name=SpatialDropout1D(0.5)(conv_name)\n    #conv_name=SpatialDropout1D(dr_r)(conv_name)\n    max_pooling_name=GlobalMaxPooling1D()(conv_name)\n    conv_desc=Conv1D(filters=64,kernel_size=3,activation=None,kernel_initializer=\'glorot_uniform\')(emb_desc)\n    max_pooling_desc=GlobalMaxPooling1D()(conv_desc)\n    #conv_category_name=Conv1D(filters=8,kernel_size=3,activation=\'relu\')(emb_category_name)\n    #max_pooling_category_name=GlobalMaxPooling1D()(conv_category_name)\n    #conv_sub_category1=Conv1D(filters=8,kernel_size=3,activation=\'relu\')(emb_sub_category1)\n    #max_pooling_sub_category1=GlobalMaxPooling1D()(conv_sub_category1)\n    #conv_sub_category2=Conv1D(filters=8,kernel_size=3,activation=\'relu\')(emb_sub_category2)\n    #max_pooling_sub_category2=GlobalMaxPooling1D()(conv_sub_category2)\n    #Fully Connected Neural Networks\n    main_l=concatenate([\n        max_pooling_name,\n        max_pooling_desc,\n       # Flatten()(emb_category_name),\n        Flatten()(emb_main_category),\n        Flatten()(emb_sub_category1),\n        Flatten()(emb_sub_category2),\n        emb_name_len,\n        emb_desc_len,\n        Flatten()(emb_brand),\n        item_condition,\n        shipping\n    ])\n    main_l=Dropout(dr_r)(Dense(256,activation=\'relu\',kernel_initializer=\'he_uniform\')(main_l))\n    main_l=Dropout(dr_r)(Dense(128,activation=\'relu\',kernel_initializer=\'he_uniform\')(main_l))\n    #main_l=Dropout(dr_r)(Dense(64,activation=\'relu\')(main_l))\n\n    output=Dense(1,kernel_initializer=\'he_normal\')(main_l)\n    \n    cnn_model=Model(inputs=[name,desc,brand,item_condition,shipping,desc_len,name_len,main_category,sub_category1,sub_category2,category_name],outputs=output)\n    optimizer=Adam(lr=lr,decay=decay)\n    cnn_model.compile(loss="mse", optimizer=optimizer)\n    \n    return cnn_model\n\n    \n    ')


# In[ ]:


get_ipython().run_cell_magic('time', '', "BATCH_SIZE = 512*3\nepochs = 2\nnp.random.seed(123)\nexp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1\nsteps = int(len(X_train['name']) / BATCH_SIZE) * epochs\nlr_init, lr_fin = 0.005, 0.0001\nlr_decay = exp_decay(lr_init, lr_fin, steps)\n\n\n\nmodel= get_cnn_model(lr_init,lr_decay)\n#model.summary()\nmodel.fit(X_train,y_train, epochs=epochs, batch_size=BATCH_SIZE\n          , validation_data=(X_valid,y_valid)\n          , verbose=1)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "np.random.seed(123)\nBATCH_SIZE = 512*3\nepochs = 2\nlr_init, lr_fin = 0.004, 0.001\nsteps = int(len(X_train['name']) / BATCH_SIZE) * epochs\nexp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1\nlr_decay = exp_decay(lr_init, lr_fin, steps)\nmodel2= new_rnn_model(lr_init,lr_decay)\nmodel2.fit(X_train,y_train, epochs=epochs, batch_size=BATCH_SIZE\n          , validation_data=(X_valid,y_valid)\n          , verbose=1)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "np.random.seed(123)\nBATCH_SIZE = 512*3\nepochs = 2\nlr_init, lr_fin = 0.005, 0.001\nsteps = int(len(X_train['name']) / BATCH_SIZE) * epochs\nexp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1\nlr_decay = exp_decay(lr_init, lr_fin, steps)\nmodel3= get_cnn_model2(lr_init,lr_decay)\nmodel3.fit(X_train,y_train, epochs=epochs, batch_size=BATCH_SIZE\n          , validation_data=(X_valid,y_valid)\n          , verbose=1)")


# #Ensembling
# Here where we combined the 3 models and used Linear Regression for the best score..Ridge and RidgeCV were tried but didn't differ in the score so I chose the simple Linear Regression.. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LinearRegression,Ridge,RidgeCV\ncnn_predict=model.predict(X_valid,batch_size=BATCH_SIZE)\nrnn_predict=model2.predict(X_valid,batch_size=BATCH_SIZE)\ncnn_predict2=model3.predict(X_valid,batch_size=BATCH_SIZE)\nX_meta=np.hstack([cnn_predict,rnn_predict,cnn_predict2])\nlm=LinearRegression()\nlm.fit(X_meta,y_valid)\nnew_pred=lm.predict(X_meta)\nrmsle=np.sqrt(mean_squared_error(y_valid,new_pred))\nprint ('loss:'+str(rmsle))")


# In[ ]:


#%%time
#np.random.seed(123)
#cnn_predict=model.predict(X_valid,batch_size=BATCH_SIZE)
#rnn_predict=model3.predict(X_valid,batch_size=BATCH_SIZE)
#min_loss=2**23
#best_alpha=0
#for i in range(100):
#    alpha=i*0.01
#    predictions=alpha*cnn_predict+(1-alpha)*rnn_predict
#    loss=np.sqrt(mean_squared_error(y_valid,predictions))
    #print('loss:'+str(loss))
#    if(min_loss>loss):
        #print('min_loss:'+str(min_loss))
#        best_alpha=alpha
#        min_loss=loss
#print('best alpha:'+ str(best_alpha))
#print('best loss:'+str(min_loss))


# ## Predicting the test data
# Last and note least it's time to predict the test data 

# In[ ]:


get_ipython().run_cell_magic('time', '', "predictions=lm.coef_[0]*model.predict(X_test,batch_size=BATCH_SIZE)+lm.coef_[1]*model2.predict(X_test,batch_size=BATCH_SIZE)+lm.coef_[2]*model3.predict(X_test,batch_size=BATCH_SIZE)+lm.intercept_\npredictions=np.exp(predictions)-1\nsubmission=pd.DataFrame(columns=['test_id','price'])\nsubmission['test_id']=test_id\nsubmission['price']=predictions\nsubmission.to_csv('EnsembleSubmission.csv',index=False)")


# In[ ]:


print('program ended  took '+str(datetime.datetime.now()-time_start)+' minutes')


# # Conclusion
# So that was my solution I hope that you benefited from it and any feedback about the NN architectures or any questions you wanna ask about the code I would be more than happy to answer it

# In[ ]:




