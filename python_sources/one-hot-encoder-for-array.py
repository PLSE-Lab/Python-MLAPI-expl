from typing import List
import numpy as np
import pandas as pd

def one_hot_array_values(series: pd.Series, sep=',', fillna_value='None', with_brackets=False) -> pd.DataFrame:
    """
    Function converts pandas.Series with array values into DataFrame with one hot encoded values.
    
    @param series: pandas.Series with array values
    @param sep: separator. Default value is `,`
    @param fillna_value: value that used for replace np.nan values in the `series`. Default value is `None`
    @param with_brackets: indicates that array contains brackets. If so first and last symbol on each values will be removed
    @return: new pandas.DataFrame with one hot encoded values
    """
    # Function works fine only with str and lists series
    if series.dtype != object:
        return pd.DataFrame(series)
    
    series = series.replace(r'^\s*$', np.nan, regex=True)
    series = series.fillna(fillna_value)
    
    result = pd.DataFrame()
    
    # Can't find proper way to check type of the series (list or str)
    if type(series[0]) == list:
        values = sum(series.values, [])
        result[series.name] = series
    else:
        series = series.str[1:-1] if with_brackets else series
        values = [arr.strip() for sublist in series.str.split(sep).values for arr in sublist]
        result[series.name] = series.apply(lambda x: [y.strip() for y in x.split(sep)])
        
    unique_values: List[str] =  sorted(set(values))
    for unique_value in unique_values:
        result[unique_value] = result[series.name].apply(lambda x: 1 if unique_value in x else 0)
    
    return result.drop(series.name, axis='columns')


df = pd.DataFrame({'tags': ['tag1, tag2', 'tag2, tag3', 'tag1, tag4']})
df2 = one_hot_array_values(df['tags'])
df3 = pd.DataFrame({
    'tag1': [1, 0, 1], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0], 
    'tag4': [0, 0, 1]
})
assert df3.equals(df2), 'pandas.Series with all filled values with default separator'


df = pd.DataFrame({'tags': ['tag1, tag2', 'tag2, tag3', None]})
df2 = one_hot_array_values(df['tags'], fillna_value='None')
df3 = pd.DataFrame({ 
    'None': [0, 0, 1],
    'tag1': [1, 0, 0], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0]
})
assert df3.equals(df2), 'pandas.Series with None value'

df = pd.DataFrame({'tags': ['tag1; tag2', 'tag2; tag3', 'tag1; tag4']})
df2 = one_hot_array_values(df['tags'], fillna_value='None', sep=';')
df3 = pd.DataFrame({
    'tag1': [1, 0, 1], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0], 
    'tag4': [0, 0, 1]
})
assert df3.equals(df2), 'pandas.Series with all filled values with custom separator'


df = pd.DataFrame({'tags': [1, 2, 3]})
df2 = one_hot_array_values(df['tags'], fillna_value='None', sep=';')
assert df.equals(df2), 'pandas.Series with all invalid values (int instead of string)'


df = pd.DataFrame({'tags': ['tag1; tag2', 'tag2; tag3', 'tag1, tag4']})
df2 = one_hot_array_values(df['tags'], sep=';')
df3 = pd.DataFrame({
    'tag1': [1, 0, 0], 
    'tag1, tag4': [0, 0, 1], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0]
})
assert df3.equals(df2), 'pandas.Series with different separators'

df = pd.DataFrame({'tags': ['tag1; tag2', 'tag2; tag3', '']})
df2 = one_hot_array_values(df['tags'], fillna_value='None', sep=';')
df3 = pd.DataFrame({ 
    'None': [0, 0, 1],
    'tag1': [1, 0, 0], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0]
})
assert df3.equals(df2), 'pandas.Series with empty value'


df = pd.DataFrame({'tags': ['[tag1, tag2]', '[tag2, tag3]', '[tag1, tag4]']})
df2 = one_hot_array_values(df['tags'], with_brackets=True)
df3 = pd.DataFrame({
    'tag1': [1, 0, 1], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0], 
    'tag4': [0, 0, 1]
})
assert df3.equals(df2), 'pandas.Series with brackets'


df = pd.DataFrame({'tags': [['tag1', 'tag2'], ['tag2', 'tag3'], ['tag1', 'tag4']]})
df2 = one_hot_array_values(df['tags'], with_brackets=True)
df3 = pd.DataFrame({
    'tag1': [1, 0, 1], 
    'tag2': [1, 1, 0], 
    'tag3': [0, 1, 0], 
    'tag4': [0, 0, 1]
})
assert df3.equals(df2), 'pandas.Series with lists'

df = pd.DataFrame({'tags': [[1, 2], [2, 3], [3, 5]]})
df2 = one_hot_array_values(df['tags'], with_brackets=True)
df3 = pd.DataFrame({1: [1, 0, 0], 2: [1, 1, 0], 3: [0, 1, 1], 5: [0, 0, 1]})
assert df3.equals(df2), 'pandas.Series with int lists'