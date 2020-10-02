#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# first pass analysis


# In[ ]:


import os
print (os.listdir("../input"))


# In[ ]:


get_ipython().system('printf "Total unique rows in csv file \'stage_1_train_labels.csv\': ";          grep -v "patientId,x,y,width,height,Target" ../input/stage_1_train_labels.csv | cut -d "," -f 1 | sort | uniq | wc -l')


# In[ ]:


get_ipython().system('printf "First 10 rows, including header, in stage_1_train_labels.csv: \\n\\n";          head -10 ../input/stage_1_train_labels.csv')


# In[ ]:


get_ipython().system('printf "Total unique rows in csv file \'stage_1_detailed_class_info.csv\': ";          grep -v "patientId,class" ../input/stage_1_detailed_class_info.csv | cut -d "," -f 1 | sort | uniq | wc -l')


# In[ ]:


get_ipython().system('printf "First 10 rows in stage_1_detailed_class_info.csv:\\n\\n";          head -10 ../input/stage_1_detailed_class_info.csv')


# In[ ]:


get_ipython().system('printf "High-level totals on unique examples provided: \\n\\n"')
get_ipython().system('printf "Normal examples: ";grep ",Normal" ../input/stage_1_detailed_class_info.csv | cut -d "," -f 1 | sort | uniq | wc -l')
get_ipython().system('printf "Pneumonia examples: "; grep ",Lung\\sOpacity" ../input/stage_1_detailed_class_info.csv | cut -d "," -f 1 | sort | uniq | wc -l')
get_ipython().system('printf "Other abnormal examples: "; grep ",No\\sLung\\sOpacity\\s\\/\\sNot\\sNormal" ../input/stage_1_detailed_class_info.csv | cut -d "," -f 1 | sort | uniq | wc -l')
get_ipython().system('printf "Total examples:: "; grep -v "patientId,class" ../input/stage_1_detailed_class_info.csv | cut -d "," -f 1 | sort | uniq | wc -l')


# In[ ]:


# sanity check from images
# number of training examples
print ("Training examples provided: {}".format(len(os.listdir("../input/stage_1_train_images"))))


# In[ ]:


# number of test cases
print ("Test cases to be predicted: {}".format(len(os.listdir("../input/stage_1_test_images"))))

