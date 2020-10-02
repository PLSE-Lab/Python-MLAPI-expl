#!/usr/bin/env python
# coding: utf-8

# This notebook shows a simple way to classify all JPG images based on their quality (75, 90, 95). <br />
# This notebook used imagemagick app to determine the images quality. <br />
# In summary, among 75000 training images, we found <br />
# 23,415 images for 75q, <br />
# 23,451 images for 90q, <br />
# 23,442 images for 95q, and <br />
# 4,693 images for unidentified quality. <br />
# <br />
# I hope this notebook might be useful for anyone who want to take the image quality in account.
# 

# In[ ]:


get_ipython().system('apt -y install imagemagick')


# In[ ]:


base_path = '/kaggle/input/alaska2-image-steganalysis/'


# In[ ]:


import os
import numpy as np
import subprocess
from numpy import savetxt

c75 = [] 
c90 = [] 
c95 = [] 
cn = [] 

for id in range(75001): 
    id = '{:05d}'.format(id) + '.jpg'
    cover_path = os.path.join(base_path, 'Cover', id) 
    output = os.popen("identify -format '%Q' "+cover_path).read()
    #print(output)

    if output != '':
        output = int(output)

    if output == 75:
        #c75 = c75+1
        c75.append(id)
    elif output == 90:
        #c90 = c90+1
        c90.append(id)
    elif output == 95:
        #c95 = c95+1
        c95.append(id)
    else:
        #cn = cn+1
        cn.append(id)

print("Total 75 q = ",len(c75))
print("Total 90 q = ",len(c90))
print("Total 95 q = ",len(c95))
print("Total cannot detect = ",len(cn))

savetxt('c75.csv', c75, fmt='%s') 
savetxt('c90.csv', c90, fmt='%s') 
savetxt('c95.csv', c95, fmt='%s') 
savetxt('cn.csv', cn, fmt='%s')


# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

q_data = ('Q95', 'Q90', 'Q75', 'Not detect Q')
y_pos = np.arange(len(objects))
total = [len(c95),len(c90),len(c75),len(cn)]

plt.bar(y_pos, total, align='center', alpha=0.5)
plt.xticks(y_pos, q_data)
plt.ylabel('Total images')
plt.title('No. of JPG quality')

for i, v in enumerate(total):
    plt.text(i, v + 200, str(v))

plt.show()

