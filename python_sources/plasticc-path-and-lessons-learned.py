#!/usr/bin/env python
# coding: utf-8

# ## Kernels and discussions used in this kernel
# - [Oliver's kernel](https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data)
# - [Alexander Firsov's kernel](https://www.kaggle.com/alexfir/fast-test-set-reading)
# - [Iprapas' kernel](https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135)
# - [Chia-Ta Tsai's kernel](https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss)
# - [Lving's kernel](https://www.kaggle.com/qianchao/smote-with-imbalance-data)
# - [Scirpus' class 99 method](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/72104)
# - [My something different kernel](https://www.kaggle.com/jimpsull/something-different)
# - [My Smote the training set kernel](https://www.kaggle.com/jimpsull/smote-the-training-sets)

# ## Broad summary of my path
# - Submitted just to proove we could submit where everything was as probable as it's share of the train set and class 99 was 0.01 (3.386)
# - First submission based on our [custom features](https://www.kaggle.com/jimpsull/train-and-submit).  Some of the data resides in deleted kernels. (1.958)
# - Copied  [Iprapas' kernel](https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135).
# - [Applied Smote](https://www.kaggle.com/jimpsull/smoteappliedeachstepwithoutweighting) to Iprapas kernel (1.111)
# - [Applied smote ](https://www.kaggle.com/jimpsull/forked-dart-w-ideas-from-kernels-added-smote)to  [Chia-Ta Tsai's kernel](https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss) (1.052)
# - Applied  [Scirpus' class 99 method](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/72104) to [Smote Dart With 99](https://www.kaggle.com/jimpsull/improveclass99fordartsmoteset?scriptVersionId=7622634) (1.039)
# - Added [our custom features](https://www.kaggle.com/jimpsull/somethingdifferent-testsetedition) to [SmoteDart99](https://www.kaggle.com/jimpsull/featuremergingkernelwithaggcustom?scriptVersionId=7767390) (1.030)
# - Did [some parameter tuning](https://www.kaggle.com/jimpsull/parametergridsearch) to arrive at [best parameters](https://www.kaggle.com/jimpsull/ourpathtowherewearenewparams) (1.010)
# - [Blended some of models](https://www.kaggle.com/jimpsull/blendmodels?scriptVersionId=8311774) (0.996)
# - [Eliminated near zero values](https://www.kaggle.com/jimpsull/adjustpredictiondataframe?scriptVersionId=8354235) (0.992)
# - Created a [decent neural net model](https://www.kaggle.com/jimpsull/mergeneural512newparams) by processing [extra-galactic](https://www.kaggle.com/jimpsull/egneural512newfeats) and [intra-galactic](https://www.kaggle.com/jimpsull/igneural1024newfeats) separately and by **[log-normalizing our data together](https://www.kaggle.com/jimpsull/lognormalizefeaturesetstogetherwextrafeats)** (better than 1.056)
# - [Blended our neural net model with our LGBM model](https://www.kaggle.com/jimpsull/fiftyfivefourtyfiveblend?scriptVersionId=8460185) (0.952)
# - [Blended](https://www.kaggle.com/jimpsull/latestneuralfiftyfiftyblendwithmoredecisiveneural) our [More Decisive Neural](https://www.kaggle.com/jimpsull/neuralmoredecisive) with our LGBM model (0.939)
# - [Blended](https://www.kaggle.com/jimpsull/fortyfortytwentyblend) our [More Decisive](https://www.kaggle.com/jimpsull/multiclasssvmmoredecisive) SVM with our Decisive Neural and our LGBM.  Note that SVM had to be [trained ](https://www.kaggle.com/jimpsull/svmmulticlassnewdataless)separately, [split up](https://www.kaggle.com/jimpsull/svmclf4top), and then [merged back together](https://www.kaggle.com/jimpsull/mergebottommiddleandtopclf0). (0.942, generalizable)
# 
# 

# ## Lessons Learned and interesting things that didn't work
# - The [newTestizeTrain method](https://www.kaggle.com/jimpsull/closethegapbetweencvandlb) showed promise but never delievered (1.076 LB)
# - [Integrated SMOTE with neural networks ](https://www.kaggle.com/jimpsull/egneuraltt512wsmote)but something wasn't quite right
# - [Tried using predictions from three models as features ](https://www.kaggle.com/jimpsull/lgbfourmoreregfastmax)(target leakage).  I tried several variations on this theme.  Some lessons have to be learned more than once.

# ## The code is in the linked kernels.  I have made efforts to make dependencies public
# - but I had quite a spiderweb of kernels by the end of the contest
# - if you want something shared just ask and I'll be happy to make it public

# In[ ]:


doSomething=False

