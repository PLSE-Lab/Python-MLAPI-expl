#!/usr/bin/env python
# coding: utf-8

# **Boost Your Score Through Leaderboard Probing**
# 
# The code in this notebook can polish your submission file, so that you get a guaranteed imporvement in your leadboard ranking.  Based on the feedback from graders, we share this leaderboard probing idea.
# 
# To boost your score:
# 
#     lbp=LeaderBoardProbing()
#     lbp.mean_scale('YourSubmission.csv')
# 
# You will see an improvement in the new submission file YourSubmission_mean.csv, then take the public score you obtained and do another boost:
# 
#     lbp.variance_scale('YourSubmission_mean.csv')
# 
# You will see another improvement with YourSubmission_variance_mean.csv.
# 
# We will explain this in more detail with two examples:
# 
# * Take predictions based on a previous-month model, turn its score from 1.167778 into 1.038940.
# * Take predictions based on random noise, turn its score from 3.473406 into 1.199848.
# 
# Sounds magic, read on ...
# 
# First, load the implementation code in the next cell.

# In[ ]:


# coding: utf-8
import numpy as np
import pandas as pd
import os

class LeaderBoardProbing:

    def __init__(self):
        if os.path.exists('new_test.csv.gz'):
            self.test  = pd.read_csv('new_test.csv.gz')
        else:
            self.test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
            sales=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
            # some routine data cleaning code
            #sales=sales[(sales.item_price<100000) & (sales.item_cnt_day<1001)]
            shop_id_map={0:57, 1:58, 10:11}
            sales['shop_id']=sales['shop_id'].apply(lambda x: shop_id_map.get(x, x))
            self.test['shop_id']=self.test['shop_id'].apply(lambda x: shop_id_map.get(x, x))

            pairs={ (a, b) for a, b in zip(sales.shop_id, sales.item_id) }
            items={ a for a in sales.item_id }
            self.test['date_block_num']=34
            self.test['test_group']=[ 2 if (a,b) in pairs else (1 if b in items else 0) for a,b in zip(self.test.shop_id, self.test.item_id)]
            self.test.sort_values('ID', inplace=True)
            self.test.to_csv('new_test.csv.gz', index=False)

        self.test['item_cnt_month']=0.0
        self.n=len(self.test)
        self.n0=sum(self.test.test_group==0)
        self.n1=sum(self.test.test_group==1)
        self.n2=sum(self.test.test_group==2)

    def probe_mean(self):
        """Generate 4 LeaderBoardProbing files, set target to 0 for all three test groups,
        then set target to 1 for only one test group at a time.
        Manually submit the files to obtain public leaderboard scores.
        Then feed the scores to estimate_mean() to obtain mean values for all groups
        and store those means in group_mean()
        """
        os.makedirs('leak', exist_ok=True)
        self.save(self.test, 'leak/Probe000.csv')

        tmp=self.test.copy()
        tmp.loc[tmp.test_group==2, 'item_cnt_month']=1.0
        self.save(tmp, 'leak/Probe001.csv')

        tmp=self.test.copy()
        tmp.loc[tmp.test_group==1, 'item_cnt_month']=1.0
        self.save(tmp, 'leak/Probe010.csv')

        tmp=self.test.copy()
        tmp.loc[tmp.test_group==0, 'item_cnt_month']=1.0
        self.save(tmp, 'leak/Probe100.csv')

    def estimate_mean(self, rmse000, rmse100, rmse010, rmse001):
        """Obtain public scores for Probe000, Probe100, Probe010, Probe001
        Public,Private
        Probe000,1.250111,1.236582
        Probe100,1.23528,1.221182
        Probe010,1.38637,1.373707
        Probe001,1.29326,1.279869
        """

        def calc(rmse000, n, rmse_i, ni):
            u=(1-(rmse_i**2-rmse000**2)*n/ni)/2
            return u

        u0=calc(rmse000, self.n, rmse100, self.n0)
        u1=calc(rmse000, self.n, rmse010, self.n1)
        u2=calc(rmse000, self.n, rmse001, self.n2)
        u=(self.n0*u0+self.n1*u1+self.n2*u2)/self.n
        return(u0, u1, u2, u)

    def true_means(self):
        # computed by leader board probing
        # u0, u1, u2 and overall mean u
        # Kaggle public scores and Coursera scores slightly differ
        # Kaggle scores
        #return [0.7590957299173547, 0.060230457160248385, 0.39458181098366407, 0.2839717256500001]
        # use Coursera scores here
        return [0.758939742420249, 0.0601995732152425, 0.3945593622881204, 0.28393632703149974]

    def mean_scale(self, filename):
        """Compare the mean of each test group to their true public leaderboard means
        shift the prediction so that the means match
        filename: your submission csv file name
        """
        df=pd.read_csv(filename)
        df.sort_values('ID', ascending=True, inplace=True)
        mask0=self.test.test_group==0
        mask1=self.test.test_group==1
        mask2=self.test.test_group==2
        U=self.true_means()
        print("Group0: predict mean=", df[ mask0 ].item_cnt_month.mean(), "true mean=", U[0])
        print("Group1: predict mean=", df[ mask1 ].item_cnt_month.mean(), "true mean=", U[1])
        print("Group2: predict mean=", df[ mask2 ].item_cnt_month.mean(), "true mean=", U[2])
        change=999
        previous=df.item_cnt_month.values.copy()
        i=1
        while change>1e-6:
            df.loc[mask0, 'item_cnt_month']+=U[0]-df[ mask0 ].item_cnt_month.mean()
            df.loc[mask1, 'item_cnt_month']+=U[1]-df[ mask1 ].item_cnt_month.mean()
            df.loc[mask2, 'item_cnt_month']+=U[2]-df[ mask2 ].item_cnt_month.mean()
            df['item_cnt_month']=df['item_cnt_month'].clip(0,20)
            change=np.sum(np.abs(df.item_cnt_month.values - previous))
            previous=df.item_cnt_month.values.copy()
            print(">loop", i, "change:", change)
            i+=1
        self.save(df, filename.replace('.csv', '_mean.csv'))

    def variance_scale(self, filename, rmse, rmse000=1.250111):
        """
        filename: your submission csv file name
        rmse: your public leaderboard score
        """
        df=pd.read_csv(filename)
        df.sort_values('ID', ascending=True, inplace=True)
        n=df.shape[0]
        u=self.true_means()[-1]
        Yp=df.item_cnt_month.values
        YpYp=np.sum(Yp*Yp)
        YYp=n*(rmse000**2-rmse**2)/2+YpYp/2
        lambda_ = (YYp-u*u*n)/(YpYp-u*u*n)
        print(">>>>>multipler lambda=", lambda_)
        df['item_cnt_month']=(Yp-u)*lambda_+u
        filename2=filename.replace('.csv', '_lambda.csv')
        self.save(df, filename2)
        self.mean_scale(filename2)

    def save(self, df, filename):
        """Produce LeaderBoardProbing file based on dataframe"""
        df = df[['ID','item_cnt_month']].copy()
        df.sort_values(['ID'], ascending=True, inplace=True)
        df['item_cnt_month']=df['item_cnt_month'].apply(lambda x: "%.5f" % x)
        if np.isnan(df.item_cnt_month.isnull().sum()):
            print("ERROR>>>>> There should be no nan entry in the LeaderBoardProbing file!")
        print("Save LeaderBoardProbing to file:", filename)
        df.to_csv(filename, index=False)

    def flip_signs(self, filename):
        """
        Produce LeaderBoardProbing file, flip the sign of prediction for each of the three groups
        filename: your submission csv file name
        output:
            three new submission files with suffix _mpp.csv, _pmp.csv, _ppm.csv
            notation in the notebook
            m: minus, p: plus
            mpp is -++, pmp is +-+, ppm is ++-

        You need to submit these three files to obtain
            rmse_mpp, rmse_pmp, rmse_ppm
        Then you call
            LeaderBoardProbing.variance_scale_v2(filename, rmse_mpp, rmse_pmp, rmse_ppm, rmse)
            Note: rmse is the original rmse score obtained by your filename
        """
        df=pd.read_csv(filename)
        df.sort_values(['ID'], ascending=True, inplace=True)
        mask0=self.test.test_group==0
        mask1=self.test.test_group==1
        mask2=self.test.test_group==2
        tmp=df.copy()
        tmp.loc[mask0, 'item_cnt_month']=-tmp[ mask0 ].item_cnt_month
        self.save(tmp, filename.replace('.csv', '_mpp.csv'))
        tmp=df.copy()
        tmp.loc[mask1, 'item_cnt_month']=-tmp[ mask1 ].item_cnt_month
        self.save(tmp, filename.replace('.csv', '_pmp.csv'))
        tmp=df.copy()
        tmp.loc[mask2, 'item_cnt_month']=-tmp[ mask2 ].item_cnt_month
        self.save(tmp, filename.replace('.csv', '_ppm.csv'))

    def variance_scale_v2(self, filename, rmse_mpp, rmse_pmp, rmse_ppm, rmse):
        """
        filename: your submission csv file name
        You must use LeaderBoardProbing.flip_signs(filename)
            to generate three additional submission files, obtain their public scores
            and feed those scores as parameters
        Scores: rmse-++, rmse+-+, rmse++-, rmse+++

        output:
            New scaled submission file
        """
        df=pd.read_csv(filename)
        df.sort_values(['ID'], ascending=True, inplace=True)
        mask0=self.test.test_group==0
        mask1=self.test.test_group==1
        mask2=self.test.test_group==2
        n=len(df)
        n0=sum(mask0)
        n1=sum(mask1)
        n2=sum(mask2)
        YYp0=n/4*(rmse_mpp**2-rmse**2)
        YYp1=n/4*(rmse_pmp**2-rmse**2)
        YYp2=n/4*(rmse_ppm**2-rmse**2)
        U=self.true_means()
        Yp0=df.loc[mask0, 'item_cnt_month'].values
        Yp1=df.loc[mask1, 'item_cnt_month'].values
        Yp2=df.loc[mask2, 'item_cnt_month'].values
        lambda0=(YYp0-U[0]**2*n0)/(np.sum(Yp0*Yp0)-U[0]**2*n0)
        lambda1=(YYp1-U[1]**2*n1)/(np.sum(Yp1*Yp1)-U[1]**2*n1)
        lambda2=(YYp2-U[2]**2*n2)/(np.sum(Yp2*Yp2)-U[2]**2*n2)
        print("Labmda: ", lambda0, lambda1, lambda2)
        df.loc[mask0, 'item_cnt_month']=U[0]+lambda0*(df[ mask0 ].item_cnt_month-U[0])
        df.loc[mask1, 'item_cnt_month']=U[1]+lambda1*(df[ mask1 ].item_cnt_month-U[1])
        df.loc[mask2, 'item_cnt_month']=U[2]+lambda2*(df[ mask2 ].item_cnt_month-U[2])
        df['item_cnt_month']=df['item_cnt_month'].clip(0,20)
        fn=filename.replace('.csv', '_labmdaV2.csv')
        self.save(df, fn)
        self.mean_scale(fn)


# Our first model is to simply use item_cnt_month data from 2015-Oct for the prediction, entries without training data are set to 0.  This is a popular basedline model seem in the forum.

# In[ ]:


import shutil
shutil.copy("../input/salescompetitionoctmodel/submit_oct.csv", "submit_oct.csv")
# your submission file is now at submit_oct.csv


# If you grade this file, you get 3/10 Coursera score with:
# 
# Your public and private LB scores are: 1.167778 and 1.172726.
# 
# Let us first use mean scaling, this strategy does not require any probing, therefore, you should always apply this step to your submission.

# In[ ]:


lbp = LeaderBoardProbing()
lbp.mean_scale('submit_oct.csv')


# Now submit the new file **submit_oct_mean.csv**:
# 
# Your public and private LB scores are: 1.118256 and 1.123108
# 
# Now we can take the public score 1.118256 and use it to do variance scaling.  We never use private scores, they are shown here just to convince you private scores improve throughout the process as well.

# In[ ]:


lbp.variance_scale('submit_oct_mean.csv', 1.118256)


# This new file **submit_oct_mean_lambda_mean.csv**:
# 
# Your public and private LB scores are: 1.038965 and 1.040923.
# 
# The Coursera score is now 5/10. You can improve it further, but first we need to create three submission files based on the mean-scaled file and obtain their scores manually (you are going to probe the public leaderboard three times):

# In[ ]:


lbp.flip_signs('submit_oct_mean.csv')


# You should obtain the following scores, notice, private scores are not needed, we only rely on the public scores.
# 
# #Submission,Public,Private
# #submit_oct_mean_mpp.csv, Your public and private LB scores are: 1.189455 and 1.194442.
# #submit_oct_mean_pmp.csv, Your public and private LB scores are: 1.121041 and 1.125736.
# #submit_oct_mean_ppm.csv, Your public and private LB scores are: 2.002616 and 1.958823.
# 
# Now use the previous score 1.118256 (**submit_oct_mean.csv**), together with the above three scores to create a final submission:
# 

# In[ ]:


lbp.variance_scale_v2('submit_oct_mean.csv', 1.189455, 1.121041, 2.002616, 1.118256)


# The final score on **submit_oct_mean_labmdaV2_mean.csv** is:
#     
# Your public and private LB scores are: 1.038940 and 1.040874.
# 
# So we have improve the score from 1.167778 to 1.038940![](http://)

# Let's work on another example to show you all the tweaks one more time.  We create a random submission first.

# In[ ]:


df=lbp.test.copy()
np.random.seed(42)
df['item_cnt_month']=np.clip(np.random.randn(len(df))*4+1, 0, 20)
lbp.save(df, 'submit_random.csv')
# Submit and we get
# Your public and private LB scores are: 3.473406 and 3.465503.


# In[ ]:


# You should always run mean_scale, as it requires not probing.
# I actually always send my prediction through mean_scale and submit the processed file
lbp.mean_scale('submit_random.csv')
# Your score for submit_random_mean.csv
# Your public and private LB scores are: 1.545094 and 1.525234.


# In[ ]:


lbp.variance_scale('submit_random_mean.csv', 1.545094)
# Your score for submit_random_mean_lambda_mean.csv
# Your public and private LB scores are: 1.200340 and 1.185505.

# If you are willing to do three more probes
lbp.flip_signs('submit_random_mean.csv')
# Your scores are
# submit_random_mean_mpp.csv
# Your public and private LB scores are: 1.604641 and 1.583784.
# submit_random_mean_pmp.csv
# Your public and private LB scores are: 1.547342 and 1.527319.
# submit_random_mean_ppm.csv
# Your public and private LB scores are: 1.645131 and 1.624193
lbp.variance_scale_v2('submit_random_mean.csv', 1.604641, 1.547342, 1.645131, 1.545094)
# submit_random_mean_labmdaV2_mean.csv
# Your public and private LB scores are: 1.199848 and 1.184957.


# You can ignore the theoretical part below, if you do not care about the mechanism behind the probing.  Enjoy an improved ranking.
# 
# **How Scaling Works Behind the Scene?**
# 
# There are 214,200 samples in the test set, including 5100 unique item_ids and 42 unique shop_ids.  The test set is made of the full product of 5100x42.
# 
# Looking up the test (shop_id, item_id) pairs in the training data, we discovered there are three groups of test samples:
# 
# **Group 2:** (shop_id, item_id) has historical sales data (i.e., in the training set). Sales of these samples could be predicted based on their historical time-series data. This group has 111404 samples (52%)
# 
# **Group 1:** not in Group 2 but (item_id) has historical sales data in other shop_id.  Sales of these samples could be predicted mostly by sale data for the same item_id in other shops.  This group has 87550 samples (41%).
# 
# **Group 0:** item_id has no historical data.  The only information available for prediction is through sales in the same item_category_id and shop_id.  This group has 15246 records (7%).
# 
# Let us first probe the mean of each group.
# 
# Let $j$ be the test group id, $j = 0, 1, 2$.
# There are $n_0, n_1, n_2$ samples within each group, respectively.
# There are $N$ total samples, where $N = n_0 + n_1 + n_2$.
# We create 4 submission files, Probe000.csv sets all predictions to 0.  Probe100.csv sets all predictions to 0 except 1 for test group 0.  Similarly, Probe010.csv and Probe001.csv has the prediction 1 for test group 1 and test group 2, respectively.
# 
# The RMSE score for the 4 submission are $rmse_{000}, rmse_{101}, rmse_{010}, rmse_{001}$.
# 
# The true target for sample $i$ is $y_i$.
# 
# We have:
# 
# $$N \cdot rmse_{000}^{2} = \sum_{j=0, i=1}^{n_0}{{y_{ji}}^2}+\sum_{j=1, i=1}^{n_1}{{y_{ji}}^2}+\sum_{j=2, i=1}^{n_2}{{y_{ji}}^2}$$
# 
# $$N \cdot rmse_{100}^{2} = \sum_{j=0, i=1}^{n_0}{{(y_{ji} - 1)}^2}+\sum_{j=1, i=1}^{n_1}{{y_{ji}}^2}+\sum_{j=2, i=1}^{n_2}{{y_{ji}}^2}$$
# 
# $$N \cdot rmse_{010}^{2} = \sum_{j=0, i=1}^{n_0}{{y_{ji}}^2}+\sum_{j=1, i=1}^{n_1}{{(y_{ji}-1)}^2}+\sum_{j=2, i=1}^{n_2}{{y_{ji}}^2}$$
# 
# $$N \cdot rmse_{001}^{2} = \sum_{j=0, i=1}^{n_0}{{y_{ji}}^2}+\sum_{j=1, i=1}^{n_1}{{y_{ji}}^2}+\sum_{j=2, i=1}^{n_2}{{(y_{ji}-1)}^2}$$
# 
# Therefore,
# 
# $$N \cdot (rmse_{100}^2 - rmse_{000}^2) = n_0 - 2 \sum_{j=0, i=1}^{n_0}{y_{ji}}$$
# 
# $$N \cdot (rmse_{010}^2 - rmse_{000}^2) = n_1 - 2 \sum_{j=1, i=1}^{n_1}{y_{ji}}$$
# 
# $$N \cdot (rmse_{001}^2 - rmse_{000}^2) = n_2 - 2 \sum_{j=2, i=1}^{n_2}{y_{ji}}$$
# 
# 
# i.e., the means $\mu_j$ for each test group and the overall mean $\mu$ are:
# 
# $$\mu_{0} = \frac{1}{2} (1 - \frac{N}{n_0} \cdot (rmse_{100}^2 - rmse_{000}^2))$$
# 
# $$\mu_{1} = \frac{1}{2} (1 - \frac{N}{n_1} \cdot (rmse_{010}^2 - rmse_{000}^2))$$
# 
# $$\mu_{2} = \frac{1}{2} (1 - \frac{N}{n_2} \cdot (rmse_{001}^2 - rmse_{000}^2))$$
# 
# $$\mu = \frac{1}{N}(n_0 \mu_0 + n_1 \mu_1 + n_2 \mu_2)$$
# 
# Therefore, we know the means for each test group:
# 
# $\mu_0=0.758939742420249$, $\mu_1=0.0601995732152425$, $\mu_2=0.3945593622881204$, and $\mu=0.28393632703149974$.

# In[ ]:


# Let's first generate submission files to obtain rmse000, rmse100, rmse010, rmse001
lbp.probe_mean()
# submit and obtain public leaderboard scores, then use the next line to obtain the group means
lbp.estimate_mean(1.250111, 1.23528, 1.38637, 1.29326)


# **Mean Scaling**
# 
# After our model predits the targets for the three test groups, we check the mean prediction for each group and compare them to $\mu_{0}, \mu_{1}, \mu_{2}$, respectively.  We can then add a constant to our predictions so that their means are shifted to better match the leaderboard means.
# 
# The proof for this mean scaling is the following:
# 
# If $y_i$ and $\hat{y}_i$ are the true value and the predicted value for sample $i$.  We would like to add a constant $c$ to all predictions to reduce the $rmse$ score, i.e.:
# 
# $$\frac{\partial}{\partial c} \sum_{i=1}^{n}{{\left(y_i - (\hat{y}_i + c)\right)}^2} = 0$$
# 
# It is not hard to derive that:
# 
# $$c = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$
# 
# The get the best results, each of the three test groups are mean scaled independently. Mean scaling is guaranteed to reduce RMSE. We then use clip(0,20) to further improve the score.  Notice clip(0,20) will change the means, therefore, we can repeatively apply the mean scaling routine until predictions converge.  This logic is implemented in **LeaderBoardProbing.mean_scale()**.
# 
# *You should always apply this routine to your predictions, as it does not cost you any presubmission and the corrected prediction is always better than your original file!*

# **Variance Scaling Version 1**
# 
# Our prediction has now been mean corrected, so that it matches the true value $\mu$.  The submission of this **mean.csv** file gives us the public leaderboard $rmse$ score.  This score allows us to multiple the residues of our prediction by a constant $\lambda$ to further reduce the score, i.e., we aim for a $\lambda$ by:
# 
# $$\frac{\partial}{\partial \lambda} \sum_i^N {{\left(y_i - \lambda (\hat{y}_i-\mu) - \mu \right)}^2} = 0 $$,
# 
# We can derive:
# 
# $$\lambda = \frac{\sum_i {(y_i -\mu)(\hat{y}_i - \mu)}}{\sum_i {(\hat{y}_i - \mu)}^2} = \frac{\sum_i {(y_i \hat{y}_i -\mu^2)} }{\sum_i {(\hat{y}_i^2 - \mu^2)}}$$
# 
# The only term unknown is $\sum_i {y_i \hat{y}_i}$.  From previous probings, we have $rmse$ score and $rmse_{000}$:
# 
# $$ N \cdot {rmse}_{000}^2 = \sum_i y_i^2$$
# 
# $$N \cdot {rmse}^2 = \sum_i {(y_i-\hat{y}_i)^2} = \sum_i y_i^2 + \sum_i \hat{y}_i^2 - 2 \sum_i {y_i \hat{y}_i}$$,
# 
# therefore, 
# 
# $$\sum_i {y_i \hat{y}_i} = \frac{1}{2} N \cdot (rmse_{000}^2 - rmse^2) + \frac{1}{2}\sum_i \hat{y}_i^2$$.
# 
# Thus, by obtaining the $rmse$ score for our **mean.csv** submission, we can calculate $\lambda$ and obtain a new **mean_lambda_mean.csv** file.  After we apply $\lambda$ scaling, we can clip(0, 20), then call mean_scale() to polish the answer further.  This whole logic is implemented in **LeaderBoardProbing.variance_scale()**.

# **Variance Scaling Version 2**
# 
# We first need to call **LeaderBoardProbling.flip_signs()** to create three submissions and obtain their RMSE scores.  For example, flip_signs() turns **mean.csv** into **mean_mpp.csv**, **mean_pmp.csv**, and **mean_ppm.csv**.  The suffix "p" stands for plus $+$ and "m" stands for $-$ in the formula below.
# 
# What flip_signs() does is to flip the signs of our predictions, once for each test group, i.e., turn $\hat{y}_i$ into $-\hat{y}_i$, but only one group at a time.  We already have $rmse$, we just need to obtain $rmse_{-++}, rmse_{+-+}, rmse_{++-}$ through three manual submissions.
# 
# Since
# 
# $$N \cdot rmse^2 = \sum_{j=0,i=1}^{n_0} {(y_i - \hat{y}_i)^2} + \sum_{j=1,i=1}^{n_1} {(y_i - \hat{y}_i)^2} +\sum_{j=2,i=1}^{n_2} {(y_i - \hat{y}_i)^2}$$
# 
# $$N \cdot rmse_{-++}^2 = \sum_{j=0,i=1}^{n_0} {(y_i + \hat{y}_i)^2} + \sum_{j=1,i=1}^{n_1} {(y_i - \hat{y}_i)^2} +\sum_{j=2,i=1}^{n_2} {(y_i - \hat{y}_i)^2}$$
# 
# $$N \cdot rmse_{+-+}^2 = \sum_{j=0,i=1}^{n_0} {(y_i - \hat{y}_i)^2} + \sum_{j=1,i=1}^{n_1} {(y_i + \hat{y}_i)^2} +\sum_{j=2,i=1}^{n_2} {(y_i - \hat{y}_i)^2}$$
# 
# $$N \cdot rmse_{++-}^2 = \sum_{j=0,i=1}^{n_0} {(y_i - \hat{y}_i)^2} + \sum_{j=1,i=1}^{n_1} {(y_i - \hat{y}_i)^2} +\sum_{j=2,i=1}^{n_2} {(y_i + \hat{y}_i)^2}$$
# 
# From the above 4 equations, we can calculate:
# 
# $$\sum_{j=0,i=1}^{n_0} { y_i \hat{y}_i } = \frac{N}{4} (rmse_{-++}^2 - rmse_{+++}^2)$$ 
# 
# $$\sum_{j=1,i=1}^{n_0} { y_i \hat{y}_i } = \frac{N}{4} (rmse_{+-+}^2 - rmse_{+++}^2)$$ 
# 
# $$\sum_{j=2,i=1}^{n_0} { y_i \hat{y}_i } = \frac{N}{4} (rmse_{++-}^2 - rmse_{+++}^2)$$ 
# 
# Similar to version 1, we can now calculate multiplers for each test group independently:
# 
# $$\lambda_0 = \frac{\sum_{j=0,i=1}^{n_0} {(y_i \hat{y}_i -\mu_0^2)} }{\sum_{j=0,i=1}^{n_0} {(\hat{y}_i^2 - \mu_0^2)}}$$
# 
# $$\lambda_1 = \frac{\sum_{j=1,i=1}^{n_1} {(y_i \hat{y}_i -\mu_1^2)} }{\sum_{j=1,i=1}^{n_1} {(\hat{y}_i^2 - \mu_1^2)}}$$
# 
# $$\lambda_2 = \frac{\sum_{j=2,i=1}^{n_2} {(y_i \hat{y}_i -\mu_2^2)} }{\sum_{j=2,i=1}^{n_2} {(\hat{y}_i^2 - \mu_2^2)}}$$
# 
# After multiplation, we do clip(0,20) followed by mean scaling. The logic is implemented in **LeaderBoardProbing.variance_scale_v2()**.
# 
# Theoretically, we can repeat these processes to obtain further improvement.  In practice, the score hardly changes and there is no need to repeat this process.  Also, version 2 typically is only slightly better than version 1.  So you can simply use version 1 and be done with it, only use version 2 for your final submission.
# 

# **Notes**
# 
# I enjoy the theoretical aspects of machine learning and occassionally blog on related topics (http://randomsciencystuff.blogspot.com/2017/05/notes-on-machine-learning.html).  Although data leaking and leaderboard probing are useless in real-life machine learning projects, they are a fun topic I learned from this course.  If you get other data leak ideas and would like to share, I will be very interested in learning.
# 

# In[ ]:




