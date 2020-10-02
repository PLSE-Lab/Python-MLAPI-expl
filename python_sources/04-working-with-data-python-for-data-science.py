#!/usr/bin/env python
# coding: utf-8

# # 04 Working with Data

# - Reading files with open
# - Writing files with open
# - Loading data with pandas
# - Working with and Saving data with pandas

# In[ ]:


# Check current working dirctory
get_ipython().system('pwd')


# In[ ]:


get_ipython().system('mkdir data')


# In[ ]:


get_ipython().system('wget -O /kaggle/working/data/Example1.txt https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt')


# In[ ]:


example1_path = '/kaggle/working/data/Example1.txt'


# In[ ]:


file1 = open(example1_path, "r")
print(f'file1 object = {file1}')
print(f'Type of file1 object = {type(file1)}')


# In[ ]:


file1.name


# In[ ]:


file1.mode


# In[ ]:


file1.close()


# In[ ]:


file1.closed


# ## with statement

# In[ ]:


with open(example1_path,'r') as file1:
    file_contents = file1.read()
    print(f'file_contents \n{file_contents}')
print(file1.closed)    
print(f'file_contents \n{file_contents}')


# ### Read Multiple lines

# In[ ]:


with open(example1_path,'r') as file1:
    file_contents = file1.readlines()
    print(f'file_contents \n{file_contents}')
print(file1.closed)    
print(f'file_contents \n{file_contents}')


# In[ ]:


with open(example1_path,'r') as file1:
    file_contents = file1.readline()
    print(f'file_contents \n{file_contents}')
print(file1.closed)    
print(f'file_contents \n{file_contents}')


# In[ ]:


with open(example1_path,'r') as file1:
    for i,line in enumerate(file1):
        print(f'Line {i+1} contains {line}')


# # Writing files with open

# In[ ]:


example2_path ='/kaggle/working/data/example2.txt'


# In[ ]:


with open(example2_path,'w') as file2:
    file2.write('This is line A')


# In[ ]:


with open(example2_path,'r') as file2:
        print(file2.read())


# In[ ]:


with open(example2_path,'w') as file2:
    file2.write("This is line A\n")
    file2.write("This is line B\n")
    file2.write("This is line C\n")


# In[ ]:


lines = ["This is line D\n",
         "This is line E\n",
         "This is line F\n"]
lines


# In[ ]:


with open(example2_path,'a') as file2:
    for line in lines:
        file2.write(line)


# In[ ]:


with open(example2_path,'r') as file2:
    print(file2.read())


# In[ ]:


example3_path ='/kaggle/working/data/Example3.txt'


# In[ ]:


with open(example2_path,'r') as readfile:
    with open(example3_path,'w') as writefile:
        for line in readfile:
            writefile.write(line)


# In[ ]:


with open(example3_path,'r') as testfile:    
    print(testfile.read())


# In[ ]:


testfile.name


# # Loading data with pandas

# In[ ]:


import pandas as pd


# ###  Read data from CSV file online

# In[ ]:


csv_url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Datasets/TopSellingAlbums.csv'


# In[ ]:


df = pd.read_csv(csv_url)


# In[ ]:


df.head()


# ###  Read data from CSV file on disk

# ### Download CSV file to local disk

# In[ ]:


get_ipython().system('wget -O ./data/TopSellingAlbums.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%204/Datasets/TopSellingAlbums.csv')


# In[ ]:


csv_path = '/kaggle/working/data/TopSellingAlbums.csv'


# In[ ]:


df = pd.read_csv(csv_path)


# In[ ]:


df.head()


# In[ ]:


df.iloc[0,0]


# In[ ]:


df.loc[0,'Artist']


# In[ ]:


df.loc[1,'Artist']


# In[ ]:


df.iloc[0:2,0:3]


# In[ ]:


df.loc[0:2,'Artist':'Released']


# ## Working with and Saving data with pandas

# In[ ]:


df['Released']


# In[ ]:


len(df['Released'])


# In[ ]:


df['Released'].unique()


# In[ ]:


len(df['Released'].unique())


# In[ ]:


df['Released']>=1980


# In[ ]:


new_songs = df[df['Released']>=1980]
new_songs


# In[ ]:


new_songs.to_csv('/kaggle/working/data/new_songs.csv')


# In[ ]:


with open('/kaggle/working/data/new_songs.csv','r') as songsfile:    
    print(songsfile.read())

