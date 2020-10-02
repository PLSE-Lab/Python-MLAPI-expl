#!/usr/bin/env python
# coding: utf-8

# # Intro
# The goal of this kernel is to export an entire python package into a single string.  
# This may be usefull for kernels without internet.  
# If you're just looking to use the [string](#resultstring) then [use this kernel](https://www.kaggle.com/maxisoft/install-packed-package)
# 
# ## disclaimer
# In case you use this in a competition, you should double **check** that it is **allowed by the rules**
# 
# ## notes
# - you still have to install all the required package's dependencies on the target kernel first. In some case they're already preinstalled on kaggle. For other cases, you may be able to use this very technique on the missing dependencies.
# - it may not works for large package or package binds with uninstalled native libraries
# 
# Here, we'll use [gpytorch](https://gpytorch.ai/) package as exemple for this process

# ## download package
# lets download the source of gpytorch v0.3.5 from github

# In[ ]:


get_ipython().system(' wget https://github.com/cornellius-gp/gpytorch/archive/v0.3.5.zip')


# extract it

# In[ ]:


get_ipython().system(' unzip v0.3.5.zip')


# ## Cleanup
# In order to reduce our final archive's size, we are about to remove non required files (like tests, docs, ...).  
# Please note it's *optional* and that the following list is very **specific** to gpytorch, other packages may not work if we delete the same files

# In[ ]:


from pathlib import Path
import shutil

non_required_files = (
    ".conda",
    ".github",
    "docs",
    "examples",
    "test",
    "environment.yml",
    "LICENSE",
    "pyproject.toml",
    "readthedocs.yml",
)

src_folder = Path("gpytorch-0.3.5")
for fname in non_required_files:
    target = src_folder / fname
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


# list remaining files

# In[ ]:


get_ipython().system(' ls -lh {src_folder}')


# # Pack sources and encode to string
# now we can compress the sources to a *tar.xz* file and encode it with *base85* to be shared / imported as a python bytes

# In[ ]:


import tarfile
import base64
import io


def pack_folder(target):
    buff = io.BytesIO()
    with tarfile.open(fileobj=buff, mode='w:xz') as tar:
        tar.add(target, arcname='.', recursive=True)

    buff.seek(0)
    return base64.b85encode(buff.getbuffer()) 


data = pack_folder(src_folder)
# data variable contains the encoded bytes


#print(data)


# display the ~ bytes size

# In[ ]:


print(len(data))


# notice that it's a relatively large string

# # The result string
# copy paste the following cell's **result** into another notebook (without internet access for instance) in order to import gpytorch source code

# In[ ]:


from IPython.core.display import HTML
import html

data_id = str(hash(data)).zfill(8)[-8:]
escaped = html.escape(str(data))
js = """
function copy_cb(hash) {
  var inp = document.getElementById("i" + hash)
  inp.select()
  inp.setSelectionRange(0, inp.value.length + 1)
  document.execCommand('copy')
}"""

display(HTML(f"""
<script type="text/javascript">
{js}
</script>
<div name="resultstring" id="resultstring">
    <input id="i{data_id}" type="text" value="{escaped}"/>
    <button onclick="copy_cb({data_id})">Copy to clipboard</button>
</div>"""))


# In[ ]:


# cleanup
get_ipython().system(' rm -r gpytorch-0.3.5')
get_ipython().system(' rm v0.3.5.zip')


# This is the end of this notebook.
# 
# Take a look at [install-packed-package notebook](https://www.kaggle.com/maxisoft/install-packed-package) to see how to install the package
