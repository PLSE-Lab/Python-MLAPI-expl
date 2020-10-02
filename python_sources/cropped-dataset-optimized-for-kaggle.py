#!/usr/bin/env python
# coding: utf-8

# Hi Kagglers, I decided to upload my version of the dataset which I made using the awesome work from @akensert which you can find [here](https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset).
# This dataset is optimized for reading directly from a Kaggle Notebook. As a matter of fact, each images comes in dimension `(n_crops * 256, 256, 3)`, so it is possible to retrieve the crops by doing a reshape to `(-1, 256, 256, 3)`. This method let me spare precious space and disk reading speed.
# 
# I'm succesfully using it on my Kaggle notebook to train a network and, with basic transformations, I'm able to fully use the GPU. You can find the dataset in the resources of this notebook or [here](https://www.kaggle.com/mawanda/akensert-transform-panda-tiles)
# 
# 
# A little demonstration follows.

# In[ ]:


from PIL import Image
import numpy as np

path_to_img = '../input/akensert-transform-panda-tiles/akensert_little/0005f7aaab2800f6170c399693a96917.jpeg'

imgs = np.array(Image.open(path_to_img)).reshape(-1, 256, 256, 3)

imgs.shape


# Now show a crop:

# In[ ]:


Image.fromarray(imgs[0])


# If you find this dataset helpful pleas upvote! And don't forget to upvote also @akensert work which I mentioned above.

# In[ ]:




