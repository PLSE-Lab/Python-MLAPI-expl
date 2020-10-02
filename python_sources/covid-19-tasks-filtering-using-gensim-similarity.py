#!/usr/bin/env python
# coding: utf-8

# # 1. Problem Statement

# ## We are Working on Unsupervised learning peoblem . Developing text and data mining tools that can help the medical community develop answers to high priority scientific questions. The CORD-19 dataset represents the most extensive machine-readable coronavirus literature collection available for data mining to date. That will allow us to apply text and data mining approaches to find answers to the below tasks:
# 1. What is known about transmission, incubation, and environmental stability?
# 1. What do we know about COVID-19 risk factors?
# 1. What do we know about virus genetics, origin, and evolution?
# 1. What do we know about vaccines and therapeutics?
# 1. What do we know about non-pharmaceutical interventions?
# 1. What do we know about diagnostics and surveillance?
# 1. What has been published about medical care?
# 1. What has been published about information sharing and inter-sectoral collaboration?
# 1. What has been published about ethical and social science considerations?
# 

# # 2. Problem Solving
# ## We Will Follow the below Flow Charts  in our Problem Solving:
# ## 1st Flow chart for Unsupervised Machine Learning algorithm that we will apply to Articles Words to group them in Clusters depending on theier similarities , and reaching to the optimum number of Clusters. 
# ![image.png](attachment:image.png)
# 
# ## 2nd Flow Chart showing how to filter the all Articles that we have around (29,323) Article , and Filtering them according to our Tasks(9 Tasks) , by applying NLP Techniques as shown Below:
# 
# 
# 

# ![image.png](attachment:image.png)

# ## 2.1 Fetching Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 3.  Data Cleaning and Visualization

# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top Ten authors.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top ten Journals.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top Ten license.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top Ten publish time.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top sources.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# ## 4.2 Data Visualization and analysis (Text Preprocessing)

# ### We Will follow the below chart for our CombinedData and our tasks for text preprocessing and data visualization:
# #### 1-Articles , which we have collected from Json Files.
# #### 2-Tasks , Which we will Combine together as we will see below
# ![image.png](attachment:image.png)

# ### 4.2.2.1   Word Cloud for Words in Texts

# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/text_corpus.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# ### We have more than 77,590,638 Words in our Articles 

# ### 4.2.2.2 Word Cloud for Words in Titles

# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Titles_Words.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# ### 4.2.2.3 Word Cloud for Words in abstract

# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/abstract_words.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# ## K-Means Model

# ![image.png](attachment:image.png)

# 

# ## 6. Model Evaluation 

# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/elbow_plot.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((1000,1000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


# ## Conclusion
# 1. -This scatter plot is generated from Articles text , each article text is a feature. 
# 1. -usinig features vector TfidfVectorizer. 
# 1. -Dimensionality Reduction using PCA.
# 1. -generating clustering using k-Means where k=15 (the best value as elbow plot).
# 1. -Topic Modeling is done on each cluster to get the keywords per cluster. 

# ## 4.2.3 Creating Bag of Words 
# ## 4.2.4 Creating TF-IDF
# ## 4.2.5 Creating Cosine Similarity
# 

# # Tasks Filtering (We will show the first 100 most coorelated  Articles for each task)
# # See how is the close coorelation between tasks and related articles.

#  # The First Task: What is known about transmission, incubation, and environmental stability?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks1 = pd.read_csv('/kaggle/input/summarydata/Task1.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks1_table=HTML(Tasks1.to_html(escape=False,index=False))
display(Tasks1_table)


# # The Second Task : What do we know about COVID-19 risk factors?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks2 = pd.read_csv('/kaggle/input/summarydata/Task2.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks2_table=HTML(Tasks2.to_html(escape=False,index=False))
display(Tasks2_table)


# # The Third Task : What do we know about virus genetics, origin, and evolution?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks3 = pd.read_csv('/kaggle/input/summarydata/Task3.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks3_table=HTML(Tasks3.to_html(escape=False,index=False))
display(Tasks3_table)


# # The Fourth Task : What do we know about vaccines and therapeutics?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks4 = pd.read_csv('/kaggle/input/summarydata/Task4.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks4_table=HTML(Tasks4.to_html(escape=False,index=False))
display(Tasks4_table)


# # The Fifth Task : What do we know about non-pharmaceutical interventions?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks5 = pd.read_csv('/kaggle/input/summarydata/Task5.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks5_table=HTML(Tasks5.to_html(escape=False,index=False))
display(Tasks5_table)


# # The Sixth Task : What do we know about diagnostics and surveillance?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks6 = pd.read_csv('/kaggle/input/summarydata/Task6.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks6_table=HTML(Tasks6.to_html(escape=False,index=False))
display(Tasks6_table)


# # The Seventh Task : What has been published about medical care?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks7 = pd.read_csv('/kaggle/input/summarydata/Task7.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks7_table=HTML(Tasks7.to_html(escape=False,index=False))
display(Tasks7_table)


# # The Eighth Task : What has been published about information sharing and inter-sectoral collaboration?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks8 = pd.read_csv('/kaggle/input/summarydata/Task8.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks8_table=HTML(Tasks8.to_html(escape=False,index=False))
display(Tasks8_table)


# # The Nineth Task : What has been published about ethical and social science considerations?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks9 = pd.read_csv('/kaggle/input/summarydata/Task9.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks9_table=HTML(Tasks9.to_html(escape=False,index=False))
display(Tasks9_table)

