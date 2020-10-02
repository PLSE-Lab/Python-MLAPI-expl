#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'MathJax.Hub.Config({\n    TeX: { equationNumbers: { autoNumber: "AMS" } }\n});')


# # Introduction - 2nd Edition
# The original purpose of this notebook is to understand [Matthew Correlation Coefficient (MCC)](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient), especially in the context of VSB competition (If you are a VSB competitor wanting to visit the old note book, please see Section Introduction -- 1st Edition, below).
# 
# However, after the competition ended, I seemed to be able to understand MCC in a more general sense. I was able to derive the MCC formula into an intuitive form which will benefit all other future usages. Therefore, I decide to revise this notebook, and hence here's 2nd Edition part.

# ## Demythifying MCC
# 
# ![basic definitions](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/330px-Precisionrecall.svg.png)
# 
# ### Goal : to show that MCC can be reformulated as
# \begin{equation*}
# \text{MCC} = (Pos\_Precision + Neg\_Precision - 1) * PosNegRatio
# \end{equation*}
# 
# where $Pos\_Precision = \frac{\mathit{TP}}{\mathit{TP}+\mathit{FP}}$, $Neg\_Precision = \frac{\mathit{TN}}{\mathit{TN}+\mathit{FN}}$, standard definitions of [precisions](https://en.m.wikipedia.org/wiki/F1_score) for positive prediction and negative prediction, respectively (see picture above). The $PosNegRatio$ is defined as 
# 
# $$PosNegRatio = \sqrt{\frac{(PosPredictionCount)(NegPredictionCount)}{(PosLabelCount)(NegLabelCount)}}$$
# 
# where $PosPredictionCount = (TP+FP)$ means **total number of positive predictions** in the test set, and the other three variables can be interpreted in the same sense.
# 
# ## Interpretation : 
# Above, you can see that MCC score consists of 3 main variables, which we would like to maximize simultaneously.
# Note that, before seeing the simplified MCC version, we may have intuitively thought that **by increasing both positive precision and negative precision, MCC score should increase**, and perhaps the score is an average of the two precisions. **This intutiion is heuristically acceptable in many cases**, but not all, as explained in the next paragraph.
# 
# This intuition does not work especially in the VSB situation where we have such a imbalance distribution of positive (4%) and negative examples (96%). If a good MCC were just somehow be acheived by good precisions, we may do this by guess only one (very confident) positive example, and get 100% positive precision. Even though, we may predict all the rest as negative, due to the highly ratio of negative example, we still get around 96% of negative precision. In this case, we will have very high two precisions, but still a bad MCC score. The explanation lies in the third term, $PosNegRatio$.
# 
# Here, $PosNegRatio$ plays an important role. This ratio forces us to make the number of positive and negative predictions **roughly the same as the ground-truth number of labeled examples**. So in the above case, we cheated by predict only 1 positive example, where there are around 4% of 11592 (public test data) $\approx 464$, so we shall get a very bad score in the factor of $\frac{1}{\sqrt{464}}$.
# 
# To conclude, the correct intuition is to maximize the two precisions, and also $PosNegRatio$ together!
# 
# ## Deriving the intuitive formula
# We shall start by its standard definition.
# 
# \begin{equation*}
# \text{MCC} = \frac{ \mathit{TP} \times \mathit{TN} - \mathit{FP} \times \mathit{FN} } {\sqrt{ (\mathit{TP} + \mathit{FP}) ( \mathit{TP} + \mathit{FN} ) ( \mathit{TN} + \mathit{FP} ) ( \mathit{TN} + \mathit{FN} ) } }
# \end{equation*}
# 

# First note that $(TP+FN)(TN+FP)$ is constant with respect to the given test set; they are in fact $(PosLabelCount)(NegLabelCount)$. Therefore, to simplify the formula, we define 
# 
# $$\kappa = (PosLabelCount)(NegLabelCount)$$.
# 
# We also define 
# 
# $$ \alpha = (PosPredictionCount)(NegPredictionCount) = (TP+FP)(TN+FN)$$
# 
# Hence, now
# 
# \begin{equation*}
# \text{MCC} = \frac{1} { \sqrt{\kappa\alpha } } \times (\mathit{TP} \times \mathit{TN} - \mathit{FP} \times \mathit{FN})
# \end{equation*}
# 
# Next, we will focus on the second term of [RHS](https://en.m.wikipedia.org/wiki/Sides_of_an_equation). Observe that by definitions,
# 
# $$FP \times FN = (PosPredictionCount - TP)\times(NegPredictionCount - TN) $$
# 
# If we plug this into the MCC equation, we will have $(TP \times TN) - (FP \times FN)$ equal to:
# 
# \begin{equation*}
# (TP \times NegPredictionCount) + (TN \times PosPredictionCount) - \alpha
# \end{equation*}
# 
# Then, by the defition of **precision** (see the picture above), the first term is
# 
# \begin{equation*}
# TP \times NegPredictionCount = Pos\_Precision \times \overbrace{PosPredictionCount \times NegPredictionCount}^{\alpha}
# \end{equation*}
# 
# Similarly,
# \begin{equation*}
# TN \times PosPredictionCount = Neg\_Precision \times \alpha
# \end{equation*}
# 
# Therefore, we finally have :
# 
# \begin{equation*}
# (TP \times TN) - (FP \times FN) = \alpha \times (Pos\_Precision + Neg\_Precision - 1)
# \end{equation*}
# 
# And hence, 
# 
# \begin{equation*}
# \text{MCC} = \sqrt{\frac{\alpha}{\kappa}} \times (Pos\_Precision + Neg\_Precision - 1)
# \end{equation*}
# 
# By defining $PosNegRatio = \sqrt{\frac{\alpha}{\kappa}}$, we finish our proof.

# In[ ]:





# # Introduction - 1st Edition
# 
# Greeting everyone. I think almost everyone here agrees that this *VSB Power Line Fault Detection* is not an easy competition. The performance of the best public kernel is so difficult to beat and improve. 
# 
# In my opinion, one of the main reason is that the MCC formula 
# 
# \begin{equation*}
# \text{MCC} = \frac{ \mathit{TP} \times \mathit{TN} - \mathit{FP} \times \mathit{FN} } {\sqrt{ (\mathit{TP} + \mathit{FP}) ( \mathit{TP} + \mathit{FN} ) ( \mathit{TN} + \mathit{FP} ) ( \mathit{TN} + \mathit{FN} ) } }
# \end{equation*}
# 
# is quite complex and not so intuitive compared to usual metrics like **accuracy, F1**, etc. 
# 
# Eventhough this section on wikipedia https://en.wikipedia.org/wiki/Matthews_correlation_coefficient#Advantages_of_MCC_over_accuracy_and_F1_score gives us some explanation why MCC is nice :
# unlike the  **'accuracy' metric**, MCC adjust score well in the class imbalance situations, and unlike **F1** which concentrates only on positive-class prediction, MCC also force our performance to do well on both postive and negative prediction.
# 
# However, due to its complexity, we will have a hard time to gain insight from our current MCC score, e.g. if we score 0.694, 
# 
# - how many positive/negative examples that we are correctly classified? 
# 
# - Does our algorithm have a good precision?
# 
# - If we are able to correct one more positive / negative example, how much will MCC increase?
# 
# - To get 0.750, how many more data must I classify correctly?
# 
# Even though we may not answer the above questions perfectly, in this problem, we may have a way to approximate them.
# 
# ## Goal of the kernel
# To demythify some aspects of MCC, and understand impact of 'one more correct' positive and negative example to your current MCC score, so that you can better design a stretegic improvement on performance. Note that this is not a programming based kernel. But there is a small code in the end for illustration on MCC calculation.

# # Where shall we start?
# 
# To achieve our goal on MCC understanding, please note that if are able to guess the four values, *TP, FP, TN and FN*, we will understand the model performance on the test data much better, e.g.
# 
# ## How many positive/negative examples that we are correctly classified? 
# 
# **Answer** For positive-class and negative-class accuracy (or **recall**) calculate $\frac{TP}{TP+FN}$, and $\frac{TN}{TN+FP}$ respectively.
# 
# ## Does our algorithm have a good precision?
# 
# **Answer** For positive-class and negative-class **precision** calculate $\frac{TP}{TP+FP}$, and $\frac{TN}{TN+FN}$ respectively.
# 
# ## If we are able to correct one more positive / negative example, how much will MCC increase?
# 
# **Answer** For one more correctly classified positive example, calculate the following formula to see how much your score change, and you will know the impact of *one-more correct example*:
# \begin{equation*}
# \text{MCC} = \frac{ \mathit{(TP+1)} \times \mathit{TN} - \mathit{FP} \times \mathit{(FN-1)} } {\sqrt{ (\mathit{TP+1} + \mathit{FP}) ( \mathit{TP} + \mathit{FN} ) ( \mathit{TN} + \mathit{FP} ) ( \mathit{TN} + \mathit{FN-1} ) } }
# \end{equation*}
# 
# In the case that we group the three phase as one example, and always predict the thress phases together, the formula will be:
# 
# \begin{equation*}
# \text{MCC} = \frac{ \mathit{(TP+3)} \times \mathit{TN} - \mathit{FP} \times \mathit{(FN-3)} } {\sqrt{ (\mathit{TP+3} + \mathit{FP}) ( \mathit{TP} + \mathit{FN} ) ( \mathit{TN} + \mathit{FP} ) ( \mathit{TN} + \mathit{FN-3} ) } }
# \end{equation*}
# 
# And do similar calculation for one more correctly classified negative example.
# 
# If we understand more on this metric, we may able to devise the new strategy, for example,
# 
# *if you know that your model is quite precise on positive examples, but still not cover enough of them (low recall), and by MCC simulation, correctly classify one more positive will increase more score than correctly classify one more negative,  you may try to train your model to 'cover' more positive examples [e.g. by re-balancing the class]  *
# 
# ### So our goal is to make a hueristic just to **guess** these four values *TP, FP, TN and FN* !

# ## How can we calculate these four values?
# According to basic algebra, if we could get four equations, we can solve for all four variables (regarding public test data which is 57% of the total test data). And actually, we know of two trivially :
# 
# $$TP + TN + FP + FN = \text{Number of your test data} = 20337 * 0.57 = 11592 $$, and
# 
# $$TP + FP = \text{Number of your predicted positive examples} =673*0.57 \approx 384 $$
# 
# Note that in the second equation, I put the predicted number of positive examples of my models which is 673 (you can check your number easily at the summation of 1 in your submission.csv)
# 
# Further, note also that other similar relations such as $$ TN + FN = \text{Number of your predicted negative examples}  $$ cannot be used, since it can be derived from the above two equations. In other words, we need two more *linear independent* equations.
# 
# And where we need some hacks. Firstly, according to the post by Putalay and Sergey ( https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/82868 ), we can roughly probe **the ratio of the positive test data**, i.e. 
# 
# $$\frac{TP+FN}{TP+FP+FN+TN } = \frac{TP+FN}{11592}$$
# 
# Here, according to Putalay's method, I test with the test data number 17511 and get the score 0.06. That's mean this example is indeed positive (faulty signal), and its implies that the ratio of positive data is around 2.3%. So our third equations is : 
# 
# $$\frac{TP+FN}{11592} = 0.023 \Rightarrow TP+FN = 0.023*11592 \approx 267  $$
# 
# So far so good, now we have 3 linerly independent equations, we need just one more equation in order to decode for the four values. In fact, the fourth equation is the MCC equation itself where you can put the MCC you get from submission : 
# 
# \begin{equation*} \label{MCC}
# \frac{ \mathit{TP} \times \mathit{TN} - \mathit{FP} \times \mathit{FN} } {\sqrt{ (\mathit{TP} + \mathit{FP}) ( \mathit{TP} + \mathit{FN} ) ( \mathit{TN} + \mathit{FP} ) ( \mathit{TN} + \mathit{FN} ) } } = 0.694
# \end{equation*}
# 
# Theoretically, it's done. We can use all 4 linearly independent equations to solve for all the values of *TP, FP, TN and FN*. However, since the last equation (MCC) is non-linear, so it can be difficult in practice to solve it together with the first 3 equations. To simplify our process, we have to make some assumptions which may not 100% correct, but is sensible. Our assumption is based on the fact that we have abundant negative examples, and our model has no problem to accurately predict the negative ones at all. Hence, most of the time, our model will have $FN \approx 0$ (which I found out to be quite correct, see verification code below). By using this assumption together with all information we have from the 3 equations we get:
# 
# \begin{equation*} 
# \frac{ \mathit{TP} \times \mathit{(11592-384)} - \mathit{FP} \times \mathit{0} } {\sqrt{ (384) ( 267 ) ( 11592-384 ) ( 11592-267 ) } } = 0.694
# \end{equation*}
# 
# Hence, we can now solve for *TP*, and can then in turn solve for other values!
# 

# In[ ]:


N = 11592

TP = 0.694*np.sqrt((384)*(267)*(N-384)*(N-267))/(N-384)
TP = int(TP)
print(TP)


# Great!! We have *TP* around 223! And by using this number, we can better approximate *FN* (instead of 0) by using the fact that $TP+FN = 267$. In turn, we also have a better guess on *TN* as well.

# In[ ]:


FP = int(384- TP)
FN = 267-223
TN = int(N-TP-FN-FP)

print(FP,TN,FN)
print('sanity check : total number of examples = TP+FP+TN+FN = ',TP+FP+FN+TN)


# Before continue, note that the $FN \approx 0$ assumption can be verified easily by writing the following kears callback metrics :

# In[ ]:


def neg_precision(y_true, y_pred):

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tn = K.sum(y_neg * y_pred_neg)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = tn
    denominator = (tn + fn)

    return numerator / (denominator + 1e-15)


# and let it print out together with MCC metric when compiling your keras model:
# ```python
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation,neg_precision])
# ```
# and, provide that our model fits the data well enough, we should see this metric almost always print out around 0.95++.

# # And now it's done
# Now we can answer the question we posted in the beginnning easily. First of all, let us check whether our four numbers will produce MCC score similar to the original one (0.694)

# In[ ]:


tp,fp,tn,fn = TP,FP,TN,FN

def mcc(tp,fp,tn,fn):
    return (tp*tn - fp*fn)/((tp+fp)*(tp+fn)*(fn+tn)*(fp+tn))**0.5

print(mcc(tp,fp,tn,fn))


# We get 0.688! I think this is good enough and so the four number that we get provide quite accurate guess!!
# 
# Now, if you want to know how much score will be increase/decrease if you change your prediction by one data point:

# In[ ]:


#correctly from 0 to 1
print(mcc(tp+1,fp,tn,fn-1))

#correctly from 1 to 0
print(mcc(tp,fp-1,tn+1,fn))

#incorrectly from 0 to 1
print(mcc(tp,fp+1,tn-1,fn))

#incorrectly from 1 to 0
print(mcc(tp-1,fp,tn,fn+1))


# So now you can see the MCC effect of classify one more correct/incorrect test data! Here, correctly guess one more positive and negative data will give you around 0.0022 and 0.0009, respectively. Note that if you group all the 3 phases as one guess, this effect has to be multiplied by 3.

# In[ ]:


#correctly from 0 to 1
print(mcc(tp+1,fp,tn,fn-1) - mcc(tp,fp,tn,fn)) 

#correctly from 1 to 0
print(mcc(tp,fp-1,tn+1,fn) - mcc(tp,fp,tn,fn))

#incorrectly from 0 to 1
print(mcc(tp,fp+1,tn-1,fn) - mcc(tp,fp,tn,fn))

#incorrectly from 1 to 0
print(mcc(tp-1,fp,tn,fn+1) - mcc(tp,fp,tn,fn))


#  ### To get 0.750, how many more data must I classify correctly?
#  We can answer this question by just play around our mcc calculation

# In[ ]:


print(mcc(tp+28,fp,tn,fn-28))
print(mcc(tp,fp-60,tn+60,fn))
print(mcc(tp+14,fp-30,tn+30,fn-14))


# So you need to correct more 28 positive data, or around 9 if you group 3 phases togehter. Alternatively, you can correct more 60 negative data or around 20 for 3-phase answering. Just 20 more correct, from ten thousand examples, sound easy, right? I find it very difficult ;)
# 
# Okay. That's all for now and hope that you can find this kernel helpful in some way ;) 

# In[ ]:


print(mcc(tp+6,fp-12,tn+12,fn-6))
print(mcc(tp+12,fp-12,tn+12,fn-12))

