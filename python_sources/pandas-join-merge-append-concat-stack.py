#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# Append
# 
# > * df_a.append(df_b)  
#   
# * Stacks vertically  
# * Append rows of **df_b** at the end of **df_a**  
# 
# Concat
# 
# > * pd.concat([df_a, df_b, df_c], axis=1)
#   
# * Stack two or more horizontally or vertically  
# * Can be inner or outer joins on index  
# 
# Join
# 
# > * df.join(other, on=None, how='left')
# 
# * Join with one or more other dataframe  
# * Either on index or on a key column
# * Can be inner, outer, left or right join
# 
# Merge
# 
# > * pd.merge(left, right, how='inner', on=None)
# 
# * Merge dataframes with databse style join
# * Either on index or on a key column

# # append

# > * df_a.append(df_b)  
#   
# * Stacks vertically  
# * Append rows of **df_b** at the end of **df_a**  

# In[ ]:


# df_a.append?


# In[ ]:


df_a = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
print(df_a, '\n')

df_b = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
print(df_b, '\n')

print(df_a.append(df_b, ignore_index=True))


# # concat

# > * pd.concat([df_a, df_b, df_c], axis=1)
#   
# * Stack two or more horizontally or vertically  
# * Can be inner or outer joins on index  

# In[ ]:


# pd.concat?


# In[ ]:


df_a = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
print(df_a, '\n')

df_b = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
print(df_b, '\n')

df_c = pd.DataFrame([[9, 10], [11, 12]], columns=list('AB'))
print(df_c, '\n')

print(pd.concat([df_a, df_b, df_c], axis=0), '\n') # default

print(pd.concat([df_a, df_b, df_c], axis=1))


# # join

# > * df.join(other, on=None, how='left')
# 
# * Join with one or more other dataframe  
# * Either on index or on a key column 
# * (if it's a column, then it need to converted to a index using set_index())
# * Can be inner, outer, left or right join

# In[ ]:


# df.join?


# In[ ]:


df_a = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 2, 6]})
print(df_a, '\n')

df_b = pd.DataFrame({'A': [1, 2, 3, 6], 'C': ['a', 'b', 'c', 'd']})
print(df_b, '\n')

df_c = pd.DataFrame({'A': [1, 4, 5, 6], 'D': ['xg', 'dt', 'qh', 'yw']})
print(df_b, '\n')

print(df_a.set_index('A').join([df_b.set_index('A'), df_c.set_index('A')], 
                               how='outer'))


# # merge

# > * pd.merge(left, right, how='inner', on=None)
# 
# * Merge dataframes with database style join
# * Either on index or on a key column

# In[ ]:


# pd.merge?


# In[ ]:


df_a = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 2, 6]})
print(df_a, '\n')

df_b = pd.DataFrame({'A': [2, 3, 1, 6], 'C': ['a', 'b', 'c', 'd']})
print(df_b, '\n')

print(pd.merge(df_a, df_b), '\n') # automatically finds the common colum

print(pd.merge(df_a, df_b, how='outer'))

