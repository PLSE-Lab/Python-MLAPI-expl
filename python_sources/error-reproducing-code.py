#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os


# In[ ]:


import os
import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'apt-get install -y -q xarchiver || true\n\nSAVED_MODEL_PATH="../input/reproducing-error"\n\nNEW_MODEL_PATH="/kaggle/working"\nfor model_file in $(ls $SAVED_MODEL_PATH/*.pth)\ndo\n    just_filename=$(basename "${model_file%.*}")\n    if [[ ! -e "$NEW_MODEL_PATH/$just_filename.pth" ]]; then\n        echo "Copying $model_file to $NEW_MODEL_PATH"\n        cp $model_file /tmp\n        cd /tmp/\n        mkdir -p $just_filename\n        echo 3 > $just_filename/version\n        echo ""\n        zip -u  $just_filename.pth $just_filename/version\n        echo "Moving model file $just_filename.pth to $NEW_MODEL_PATH"\n        mv $just_filename.pth $NEW_MODEL_PATH/$just_filename.pth\n    fi\n    echo "(After) Contents of \'$just_filename/version\' in the model file \'$model_file\'"\n    unzip -p $NEW_MODEL_PATH/$(basename $model_file) $just_filename/version\ndone')


# In[ ]:


from efficientnet_pytorch import EfficientNet 
import torch
import torch.nn as nn


class EfficientNetB0(nn.Module):
    def __init__(self, pretrained, out_dim):
        super(EfficientNetB0, self).__init__()
        if pretrained == True:
            self.model = EfficientNet.from_name('efficientnet-b0')
        else:
            self.model = EfficientNet.from_pretrained(None)            

        in_features = self.model._fc.in_features
        self.l0 = nn.Linear(in_features, out_dim)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNetB0(pretrained=True, out_dim=5)
model.load_state_dict(torch.load('exp1.pth', map_location=device))
model.to(device)


# In[ ]:




