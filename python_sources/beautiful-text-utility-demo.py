#!/usr/bin/env python
# coding: utf-8

# ## BeautifulText Demo
# 
# This kernel walks you through the sample process for priting markdown to highlight important sections of your notebook output.

# ### Imports

# In[ ]:


from beautifultext import BeautifulText


# ### Color

# In[ ]:


blue_text = BeautifulText(color='blue')
teal_text = BeautifulText(color='#008080')


# In[ ]:


blue_text.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# In[ ]:


teal_text.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# ### Font

# In[ ]:


comic = BeautifulText(font_family='Comic Sans MS', color='green')
comic.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# In[ ]:


bold_text = BeautifulText(font_weight='bold')
bold_text.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# In[ ]:


small_text = BeautifulText(font_size=10, color='RGB(205, 92, 92)')
small_text.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# In[ ]:


italic_text = BeautifulText(font_style='italic', color='#00FFFF', background_color='black')
italic_text.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# In[ ]:


shadow_text = BeautifulText(text_shadow='3px 3px 3px', color='orange')
shadow_text.printbeautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")


# ### Example: Color channging output

# In[ ]:


import numpy as np

fl = np.random.randint(2)

if fl:
    result = BeautifulText(color='green')
else:
    result = BeautifulText(color='red')
    
result.printbeautiful("The color of this text changes based on the results. Pretty cool, isn't it?" )

