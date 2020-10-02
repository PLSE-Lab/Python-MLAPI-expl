#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/drjerk1/oneshot-deepfake')
get_ipython().system('rm -rf oneshot-deepfake/demos')


# In[ ]:


get_ipython().system('cp -r ../input/one-shot-deepfake-weights/* oneshot-deepfake/weights/')


# In[ ]:


get_ipython().run_line_magic('cd', 'oneshot-deepfake')


# In[ ]:


get_ipython().system('ls -alh')


# In[ ]:


get_ipython().system('ls -alh weights')


# In[ ]:


get_ipython().system('mkdir images')


# In[ ]:


get_ipython().system("wget 'https://upload.wikimedia.org/wikipedia/commons/0/0f/A._Schwarzenegger.jpg' -O images/arnold.jpg")


# In[ ]:


get_ipython().system("wget 'https://upload.wikimedia.org/wikipedia/commons/a/a0/Reuni%C3%A3o_com_o_ator_norte-americano_Keanu_Reeves_%28cropped%29.jpg?download' -O images/reeves.jpg")


# In[ ]:


get_ipython().system('python process_image.py --input-image images/arnold.jpg --output-image images/result.jpg --source-image images/reeves.jpg')


# In[ ]:


from matplotlib import pyplot as plt
import imageio
fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(20, 10))
ax[0, 0].imshow(imageio.imread("images/arnold.jpg"))
ax[0, 1].imshow(imageio.imread("images/reeves.jpg"))
ax[0, 2].imshow(imageio.imread("images/result.jpg"))
plt.show()


# In[ ]:




