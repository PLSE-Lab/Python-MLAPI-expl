#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# StyleGAN Image Generator for Tinder Generation, based on NVIDIA
# 
# Make sure your GPU is turned on during your runtime

# In[ ]:


#specify your Dropbox token here
token= ''
#choose your trained model:
url = 'https://drive.google.com/uc?id=1dBkmtBHJzEmqgVakTR8GVL_qd5N1lN1-' #Flickr karras2019stylegan-ffhq-1024x1024.pkl
# url= 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' #celebrities


# In[ ]:


get_ipython().system('ls /kaggle/working')
get_ipython().system('rm -rf /kaggle/working/stylegan')
get_ipython().system('git clone https://github.com/NVlabs/stylegan.git')
  


# In[ ]:


cd stylegan/


# In[ ]:


import os
import pickle
import numpy as np
import PIL.Image
from IPython.display import display, Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()
      # Pick latent vector.
    for i in range(1,100):
      
      rnd = np.random.RandomState()
      latents = rnd.randn(1, Gs.input_shape[1])

      # Generate image.
      fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
      images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

      # Save image.
      os.makedirs(config.result_dir, exist_ok=True)
      png_filename = os.path.join(config.result_dir, 'example' +str(i)+'.png')
      PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
      #display(Image(filename=png_filename,width=300))


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:


get_ipython().system('ls /kaggle/working/stylegan/results')
get_ipython().system('apt-get install zip')
get_ipython().system('rm /kaggle/working/stylegan/results.zip')
get_ipython().system('zip -r /kaggle/working/stylegan/results.zip /kaggle/working/stylegan/results/*.png')


# In[ ]:


get_ipython().system(' find / -name "results.zip"')


# In[ ]:


get_ipython().system('pip install dropbox')
import pathlib
import dropbox
import re
import time

now = str(time.time())

# the source file
folder = pathlib.Path("/kaggle/working/stylegan")    # located in this folder
filename = "results.zip"         # file name
filepath = folder / filename  # path object, defining the file

# target location in Dropbox
target = "/Temp/"              # the target folder
targetfile = target + now + "_" + filename   # the target path and file name

# Create a dropbox object using an API v2 key
d = dropbox.Dropbox(token)

# open the file and upload it
with filepath.open("rb") as f:
   # upload gives you metadata about the file
   # we want to overwite any previous version of the file
   meta = d.files_upload(f.read(), targetfile, mode=dropbox.files.WriteMode("overwrite"))

# create a shared link
link = d.sharing_create_shared_link(targetfile)

# url which can be shared
url = link.url

# link which directly downloads by replacing ?dl=0 with ?dl=1
dl_url = re.sub(r"\?dl\=0", "?dl=1", url)


# In[ ]:


print("Now get your zip file with pictures!")
print (dl_url)

