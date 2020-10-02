#!/usr/bin/env python
# coding: utf-8

# Hi all, the following is code I wrote for my Fastai model in order to track the Spearman Rho statistic. Since this is the eventual score we are comparing it against, it only makes sense for us to track it to get a general sense of how we are performing. 
# 
# Simply add this to your learner similar to the following example:
# 
# ```
# learner = Learner(databunch, 
#                   custom_transformer_model, 
#                   metrics=[SpearmanRho()])
# ```
# 
# Hope this helps!

# In[ ]:


from scipy.stats import spearmanr 
from fastai.text import *

class SpearmanRho(Callback):
    def on_epoch_begin(self, **kwargs):
        # Reset
        self.spearman_rho = 0
        self.spearman_scores = []
        # Calculate the output scores 
        self.targets = None
        self.outputs = None
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.targets is None:
            self.targets = last_target
        else:
            self.targets = torch.cat((self.targets, last_target))
        if self.outputs is None:
            self.outputs = last_output
        else:
            self.outputs = torch.cat((self.outputs, last_output))
    
    def on_epoch_end(self, last_output, last_target, last_metrics, **kwargs):
        for i in range(self.outputs.shape[1]):
            # Calculate spearman score across the differerent metrics
            spearman_score =            spearmanr(self.outputs.permute(1, 0)[i], 
                      self.targets.permute(1, 0)[i])
            self.spearman_scores.append(spearman_score[0]) 
        self.spearman_rho = np.nanmean(self.spearman_scores)
        nan_count = sum([np.isnan(x) for x in self.spearman_scores])/len(self.spearman_scores)
        if nan_count > 0:
            print("The number of nan in the spearman scores is: "+str(nan_count))
        return add_metrics(last_metrics, self.spearman_rho)

