#!/usr/bin/env python
# coding: utf-8

# # Unicode Visualization of the Bengali Alphabet
# 
# This notebook attempts to extend upon this to visualize the Bengali Alphabet, and builds on my previous [Bengali AI Dataset - EDA Grapheme Combinations](https://www.kaggle.com/jamesmcguigan/bengali-ai-dataset-eda-grapheme-combinations/)
# 
# Unicode itself is encoded as a multibyte string, using a lower level of base_graphemes than root/vowel/consonant diacritics. Some Benglai Graphemes have multiple renderings for the same root/vowel/consonant combination, which is implemented in unicode by allowing duplicate base_graphemes within the encoding. 
# 
# * This potentually opens up another datasource for investigation, which is to explore the full range of diacritic combinations within the unicode specification. The paper [Fonts-2-Handwriting: A Seed-Augment-Train framework for universal digit classification](https://arxiv.org/pdf/1905.08633.pdf) also makes the suggestion that it may be possible to generate synethetic data for handwriting recognition by rendering each of the unicode graphemes using various Bengali fonts

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from IPython.display import Markdown, HTML
from collections import Counter
from itertools import chain
from functools import reduce
# from src.jupyter import grid_df_display, combination_matrix

pd.set_option('display.max_columns',   500)
pd.set_option('display.max_colwidth',   -1)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


# Source: https://github.com/JamesMcGuigan/kaggle-digit-recognizer/blob/master/src/utils/confusion_matrix.py
from typing import Union

import pandas as pd
from pandas.io.formats.style import Styler


def combination_matrix(dataset: pd.DataFrame, x: str, y: str, z: str,
                       format=None, unique=True) -> Union[pd.DataFrame, Styler]:
    """
    Returns a combination matrix, showing all valid combinations between three DataFrame columns.
    Sort of like a heatmap, but returning lists of (optionally) unique values

    :param dataset: The dataframe to create a combination_matrx from
    :param x: column name to use for the X axis
    :param y: column name to use for the Y axis
    :param z: column name to use for the Z axis (values that appear in the cells)
    :param format: '', ', '-', ', '\n'    = format value lists as "".join() string
                    str, bool, int, float = cast value lists
    :param unique:  whether to return only unique values or not - eg: combination_matrix(unique=False).applymap(sum)
    :return: returns nothing
    """
    unique_y = sorted(dataset[y].unique())
    combinations = pd.DataFrame({
        n: dataset.where(lambda df: df[y] == n)
            .groupby(x)[z]
            .pipe(lambda df: df.unique() if unique else df )
            .apply(list)
            .apply(sorted)
        for n in unique_y
    }).T

    if isinstance(format, str):
        combinations = combinations.applymap(
            lambda cell: f"{format}".join([str(value) for value in list(cell) ])
            if isinstance(cell, list) else cell
        )
    if format == str:   combinations = combinations.applymap(lambda cell: str(cell)      if isinstance(cell, list) and len(cell) > 0 else ''     )
    if format == bool:  combinations = combinations.applymap(lambda cell: True           if isinstance(cell, list) and len(cell) > 0 else False  )
    if format == int:   combinations = combinations.applymap(lambda cell: int(cell[0])   if isinstance(cell, list) and len(cell)     else ''     )
    if format == float: combinations = combinations.applymap(lambda cell: float(cell[0]) if isinstance(cell, list) and len(cell)     else ''     )

    combinations.index.rename(y, inplace=True)
    combinations.fillna('', inplace=True)
    if format == '\n':
        return combinations.style.set_properties(**{'white-space': 'pre-wrap'})  # needed for display
    else:
        return combinations  # Allows for subsequent .applymap()


# # Statistics

# We can decode the graphemes into their constituant base_graphemes using `list()` 

# In[ ]:


dataset = pd.read_csv('../input/bengaliai-cv19/train.csv'); 
dataset['base_graphemes'] = dataset['grapheme'].apply(list)
dataset.head()


# There are 62 base_graphemes in the unicode encoding, with a median of 4 and maximum of 8 symbols required to encode each grapheme. 
# 
# We can also see the percentage frequency for each root diacritic.

# In[ ]:


base_diacritics_unique = sorted(set(chain(*dataset['base_graphemes'].values)))
base_diacritics_stats  = {
    "mean":   round( dataset['base_graphemes'].apply(len).mean(), 2),
    "median": np.median( dataset['base_graphemes'].apply(len) ),
    "min":    dataset['base_graphemes'].apply(len).min(),
    "max":    dataset['base_graphemes'].apply(len).max(),
    "std":    dataset['base_graphemes'].apply(len).std(),    
    "unique": len( set(chain(*dataset['base_graphemes'].values))),
    "count":  len(list(chain(*dataset['base_graphemes'].values))),
    "mean_duplicated_bases":  dataset['base_graphemes'].apply(lambda value: (len(value) - len(set(value)))).mean(),
    "max_duplicated_bases":   dataset['base_graphemes'].apply(lambda value: (len(value) - len(set(value)))).max(),    
    "count_duplicated_bases": dataset['base_graphemes'].apply(lambda value: (len(value) - len(set(value))) != 0).sum(),        
}
base_diacritics_counter = dict( 
    sum(dataset['base_graphemes'].apply(Counter), Counter()).most_common()
)

display( pd.DataFrame([base_diacritics_counter]) / base_diacritics_stats['count'] )
display( " ".join(base_diacritics_unique) )
display( base_diacritics_stats )


# # Base Graphemes
# 
# As there are fewer base_graphemes than root_graphemes in the Bengali alphabet. Thus we can perform a set analyis to determine how the grapheme_roots are themselves decomposed into a lower level of root diacritic combinations.

# In[ ]:


base_diacritic_sets = {
    key: dataset.groupby(key)['base_graphemes']
                .apply(lambda group: reduce(lambda a,b: set(a) & set(b), group)) 
                .apply(sorted)     
    for key in [ 'vowel_diacritic', 'consonant_diacritic', 'grapheme_root' ]
}
display(
    pd.DataFrame(base_diacritic_sets)
        .applymap(lambda x: x if x is not np.nan else set())        
        .applymap(lambda group: "\n".join(group))
        .T
        .style.set_properties(**{'white-space': 'pre-wrap'})
)


# If we extend the dataset with the information, we can display a combination matrix in pure Bengali

# In[ ]:


for key in [ 'vowel_diacritic', 'consonant_diacritic', 'grapheme_root' ]:
    base_key = key.split('_')[0] + '_base'
    zfill = 3 if key == 'grapheme_root' else 2
    dataset[base_key] = (
        dataset[key]
            .apply(lambda value: [ str(value).zfill(zfill)] + sorted(base_diacritic_sets[key][value]))
            .apply(lambda value: " ".join(value))            
            .fillna('')
    )
# Make numeric strings sortable
dataset.head()


# # Visualization of Grapheme Combinations in Bengali Alphabet
# 
# # Vowel/Consonant Combinations Table

# In[ ]:


combination_matrix(dataset, x='consonant_base', y='vowel_base', z='grapheme', format=' ')


# # Base/Vowel Combinations Table

# In[ ]:


combination_matrix(dataset, x='grapheme_base', y='vowel_base', z='grapheme', format='\n')


# # Base/Consonant Combinations Table

# In[ ]:


combination_matrix(dataset, x='grapheme_base', y='consonant_base', z='grapheme', format='\n')


# # Full Combination Table

# In[ ]:


combination_matrix(dataset, x=['vowel_base','consonant_base'], y='grapheme_base', z='grapheme', format=' ').T


# In[ ]:


combination_matrix(dataset, x=['vowel_base','consonant_base'], y='grapheme_base', z='grapheme', format=' ')


# In[ ]:




