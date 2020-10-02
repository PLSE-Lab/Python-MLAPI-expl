#!/usr/bin/env python
# coding: utf-8

# <h2> Welcome to MARK5826 Week 2!</h2>
# 
# <h3> Thanks for taking this first time ever offered course! I hope you'll love it! </h3>
# 
# Lecturer In Charge: Junbum Kwon;Teaching Assistant: Daniel Han-Chen & James Lin
# 
# In week 1, we want you to be exposed to the basics of programming, and the basics of Python.
# 
# <h2>AIM</h2>
# 
# This week (week 2), we are focusing on data analysis.
# 
# You will be using library functions.
# 
# <h2>ASSIGNMENT 1 (week 3 to 5) 7 marks</h2>
# 
# Next week night (Friday Week 3), you will receive a Kaggle link to **Assignment 1**. It's worth 7 marks, and super easy :).
# 
# The stuff you learnt in week 1 and today & next week will be enough to complete Assignment 1 entirely.
# 
# It's due in 2 weeks time on Saturday week 5 10pm. We need you to submit a URL / LINK to your assignment notebook on Kaggle.
# 
# IT MUST BE **PUBLIC**. Also, download the Kaggle file, and upload it to Moodle by 10pm Saturday week 5.
# 
# <h2> Suggestions from Week 1 </h2>
# 
# 1. **|Slides|** We will upload the slides through email / moodle before the lectures!
# 2. **|Applications|** Week 1 was intro. Week 2+ will be all industry applications.
# 3. **|Lab Delivery|**  Speaking will be slower now (and content has been cut back a lot. We will explain more --> more quality over quantity).
# 4. **|Too fast|** We will go slowly on the hard parts of the class. In fact, we will go through how to do Lab Questions as a class.
# 5. **|Heavy Content|** Sorry about the overwhelming content! I'm sure from week 2 onwards, the code you'll see should look familiar.
# 6. **|Slow Computers|** Some people's computers are slow. We have decided to optimise the code below (removing superfluous code)
# 7. **|Heavy Content|** Lab Organisation has been improved, with one streamlined marking system.
# 8. **|Python Documentation|** We will now have a Google Doc style documentation system. It is colloborative, however, we will endeavour to fill it in ourselves.
# 9. **|Lab Questions Weavement During Class|** At the start of the lab, we will directly read the Lab Q first. Then, you can see how the Lab Q can be answered.
# 
# <h2>Week Topics</h2>
# 
# (You can click the links below)
# 
# [SKIP BELOW CODE TO CONTENT](#Content)
# <hr>
# 1.[Reading Data](#Content)
# 
# 2.[Kickstarter Data](#Kickstarter)
# 
# 3.[Analysis of Time Data](#Quick)
# 
# 4.[Processing and Using Dates](#Year)
# 
# 5.[Lab Questions](#Lab)
# 

# [<h1>CLICK to SKIP BELOW CODE TO CONTENT</h1>](#Content)

# In[ ]:


import pandas as pd, numpy as np, os, gc, matplotlib.pyplot as plt, seaborn as sb, re, warnings, calendar, sys
from numpy import arange
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore'); np.set_printoptions(suppress=True); pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', lambda x: '%.3f' % x); pd.options.display.max_rows = 15
global directory; directory = '../input'

def files(): return os.listdir(directory)

def read_clean(data):
    data.columns = [str(x.lower().strip().replace(' ','_')) for x in data.columns]
    seen = {}; columns = []; i = 0
    for i,x in enumerate(data.columns):
        if x in seen: columns.append(x+'_{}'.format(i))
        else: columns.append(x)
        seen[x] = None
        
    for x in data.columns[data.count()/len(data) < 0.0001]: del data[x];
    gc.collect();
    try: data = data.replace({'':np.nan,' ':np.nan});
    except: pass;
    
    if len(data) < 10000: l = len(data);
    else: l = 10000;
    sample = data.sample(l);size = len(sample);
    
    for x in sample.columns:
        ints = pd.to_numeric(sample[x], downcast = 'integer', errors = 'coerce')
        if ints.count()/size > 0.97:
            minimum = ints.min()
            if minimum > 0: data[x] = pd.to_numeric(data[x], downcast = 'unsigned', errors = 'coerce')
            else: data[x] = pd.to_numeric(data[x], downcast = 'integer', errors = 'coerce')
        else:
            floats = pd.to_numeric(sample[x], downcast = 'float', errors = 'coerce')
            if floats.count()/size > 0.97: data[x] = pd.to_numeric(data[x], downcast = 'float', errors = 'coerce')
            else:
                dates = pd.to_datetime(sample[x], errors = 'coerce')
                if dates.count()/size > 0.97: data[x] = pd.to_datetime(data[x], errors = 'coerce')
    return data.reset_index(drop = True)

def read(x):
    '''Kaggle Reading in CSV files.
    Just type read('file.csv'), and you'll get back a Table.'''
    
    file = '{}/{}'.format(directory,x)
    try:     data = pd.read_csv(file)
    except:  data = pd.read_csv(file, encoding = 'latin-1')
    return read_clean(data)

def tally(column, minimum = 0, top = None, graph = False, percent = False, multiple = False, lowercase = False, min_count = 1):
    '''Provides a tally count of all values in a COLUMN.
        1. minimum  =  (>0)          Least count of item to show.
        2. top      =  (-1,>0)       Only show top N objects
        3. graph    =  (False,True)  Show bar graph of results
        4. percent  =  (False,>0)    Instead of showing counts, show percentages of total count
        
       multiple = False/True.
       If True, counts and tallies objects in list of lists (Count Vectorizer)
       
       lowercase = True / False.
       If True, lowers all text firsrt. So A == a
       
       min_count >= 1
       If a column sum for tag has less than min_count, discard whole column
    '''
    if multiple == False:
        counts = column.value_counts().astype('uint')
        counts = counts[counts >= minimum][:top]
        counts = pd.DataFrame(counts).reset_index()
        counts.columns = [column.name, 'tally']
        if percent: 
            counts['tally'] /= counts['tally'].sum()/100
            counts['tally'] = counts['tally']
        if graph:
            C = counts[::-1]
            C.plot.barh(x = column.name, y = 'tally', legend = False); plt.show();
        return counts
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        column = column.fillna('<NAN>')
        if type(column.iloc[0]) != list: column = column.apply(lambda x: [x])
        counter = CountVectorizer(lowercase = lowercase, tokenizer = lambda x: x, dtype = np.uint32, min_df = min_count)
        counter.fit(column)
        counts = pd.DataFrame(counter.transform(column).toarray())
        counts.columns = [column.name+'_('+str(x)+')' for x in counter.get_feature_names()]
        return counts
    
    
def describe(data):
    '''Provides an overview of your data
        1. dtype    =  Column type
        2. missing% =  % of the column that is missing
        3. nunique  =  Number of unique values in column
        4. top3     =  Top 3 most occuring items
        5. min      =  Minimum value. If not a number column, then empty
        6. mean     =  Average value. If not a number column, then empty
        7. median   =  Middle value. So sort all numbers, and get middle. If not a number column, then empty
        8. max      =  Maximum value. If not a number column, then empty
        9. sample   =  Random 2 elements
        10. name    =  Column Name
    '''
    dtypes = dtype(data)
    length = len(data)
    missing = ((length - data.count())/length*100)
    
    N = [];    most3 = []
    for dt,col in zip(dtypes,data.columns):
        if dt != 'datetime':
            U = data[col].value_counts()
            N.append(len(U))
            if U.values[0] > 1: most3.append(U.index[:3].tolist())
            else: most3.append([]);
        else: N.append(0); most3.append([]);
            
    df = pd.concat([dtypes, missing], 1)
    df.columns = ['dtype','missing%']
    df['nunique'] = N; df['top3'] = most3
    
    numbers = list(data.columns[df['dtype'].isin(('uint','int','float'))])
    df['min'] = data.min()
    df['mean'] = data[numbers].mean()
    df['median'] = data[numbers].median()
    df['max'] = data.max()
    df['sample'] = data.apply(lambda x : x.sample(2).values.tolist())
    df['name'] = list(data.columns)
    return df.sort_values(['missing%', 'nunique', 'dtype'], ascending = [False, False, True]).reset_index(drop = True)


def Checker(x):
    if type(x) is pd.DataFrame: return 0
    elif type(x) is pd.Series: return 1
    else: return -1

def columns(data): return list(data.columns)
def rows(data): return list(data.index)
def index(data): return list(data.index)
def head(data, n = 5): return data.head(n)
def sample(data, n = 10): return data.sample(n)

def dtype(data):
    what = Checker(data)
    if what == 0:
        dtypes = data.dtypes.astype('str')
        dtypes = dtypes.str.split(r'\d').str[0]
    else:
        dtypes = str(data.dtypes)
        dtypes = re.split(r'\d', dtypes)[0]
    return dtypes

def mean(data):
    what = Checker(data)
    _dt = ('uint','int','float')
    if what == 0:
        dtypes = dtype(data)
        numbers = data.columns[dtypes.isin(_dt)]
        return data[numbers].mean()
    elif what == 1:
        dtypes = dtype(data)
        if dtypes in _dt: return data.mean()
        else: return np.nan
    else:
        try:     return np.nanmean(data)
        except:  return np.nan

        
def log(data):
    what = Checker(data)
    _dt = ('uint','int','float')
    if what == 0:
        dtypes = dtype(data)
        numbers = data.columns[dtypes.isin(_dt)]
        x = np.log(data[numbers])
        x[np.isinf(x)] = np.nan
        return pd.Series(x)
    elif what == 1:
        dtypes = dtype(data)
        if dtypes in _dt:
            x = np.log(data)
            x[np.isinf(x)] = np.nan
            return x
        else: return np.nan
    else:
        try:
            x = np.log(data)
            x[np.isinf(x)] = np.nan
            return x
        except:  return np.nan
        
def median(data):
    what = Checker(data)
    _dt = ('uint','int','float')
    if what == 0:
        dtypes = dtype(data)
        numbers = data.columns[dtypes.isin(_dt)]
        return data[numbers].median()
    elif what == 1:
        dtypes = dtype(data)
        if dtypes in _dt: return data.median()
        else: return np.nan
    else:
        try:     return np.nanmedian(data)
        except:  return np.nan
        
def minimum(data):
    what = Checker(data)
    if what == 0:      return data.min()
    elif what == 1:    return data.min()
    else:              return np.min(data)
        
def maximum(data):
    what = Checker(data)
    if what == 0:      return data.max()
    elif what == 1:    return data.max()
    else:              return np.max(data)
    
def missing(data):
    what = Checker(data)
    if what >= 0:      return pd.isnull(data)
    else:              return np.isnan(data)
    
def count(data):
    what = Checker(data)
    if what >= 0:      return data.count()
    else:              return len(data)
    
def nunique(data):
    what = Checker(data)
    if what >= 0:      return data.nunique()
    else:              return len(np.unique(data))
    
def unique(data):
    if type(data) is pd.DataFrame:
        uniques = []
        for x in data.columns:
            uniques.append(data[x].unique())
        df = pd.Series(uniques)
        df.index = data.columns
        return df
    elif type(data) is pd.Series: return data.unique()
    else:              return np.unique(data)
        
def time_number(date): return hours(date)+minutes(date)/60+seconds(date)/60**2
def hours_minutes(date): return hours(date)+minutes(date)/60
def hours(date): return date.dt.hour
def minutes(date): return date.dt.minute
def seconds(date): return date.dt.second
def month(date): return date.dt.month
def year(date): return date.dt.year
def day(date): return date.dt.day
def weekday(date): return date.dt.weekday
def leap_year(date): return year(date).apply(calendar.isleap)
def date_number(date): return year(date)+month(date)/12+day(date)/(365+leap_year(date)*1)
def year_month(date): return year(date)+month(date)/12

def hcat(*columns):
    cols = []
    for c in columns:
        if c is None: continue;
        if type(c) in (list, tuple): 
            for i in c:
                if type(i) not in (pd.DataFrame, pd.Series): cols.append(pd.Series(i))
                else: cols.append(i)
        elif type(c) not in (pd.DataFrame, pd.Series): cols.append(pd.Series(c))
        else: cols.append(c)
    return pd.concat(cols, 1)

def vcat(*columns):
    cols = []
    for c in columns:
        if c is None: continue;
        if type(c) in (list, tuple): 
            for i in c:
                if type(i) not in (pd.DataFrame, pd.Series): cols.append(pd.Series(i))
                else: cols.append(i)
        elif type(c) not in (pd.DataFrame, pd.Series): cols.append(pd.Series(c))
        else: cols.append(c)
    return pd.concat(cols, 0)

    
def tabulate(*columns, method = 'count'):
    '''Splits columns into chunks, and counts the occurences in each group.
        Remember - tabulate works on the LAST column passed.
        Options:
            1. count            = Pure Count in group
            2. count_percent    = Percentage of Count in group
            3. mean             = Mean in group
            4. median           = Median in group
            5. max              = Max in group
            6. min              = Min in group
            7. sum_percent      = Percentage of Sum in group
        Eg:
            Apple | 1
            ---------
            Orange| 3
            ---------
            Apple | 2
            ---------
        Becomes:
            Apple | 1 | 1
            -------------
                  | 2 | 1
            -------------
            Orange| 3 | 1
        
        NOTE --------
            method can be a list of multiple options.
    '''
    if type(method) in (list, tuple):
        xs = []
        for x in method:
            g = tabulate(*columns, method = x)
            xs.append(g)
        xs = hcat(xs)
        xs = xs.T.drop_duplicates().T
        return read_clean(xs)        
    else:
        def percent(series):
            counts = series.count()
            return counts.sum()

        data = hcat(*columns)
        columns = data.columns.tolist()

        if method in ('count', 'count_percent'):
            groups = data.groupby(data.columns.tolist()).apply(lambda x: x[data.columns[-1]].count())

            if method == 'count_percent':
                groups = groups.reset_index()
                groups.columns = list(groups.columns[:-1])+['Group_Count']
                right = data.groupby(columns[:-1]).count().reset_index()
                right.columns = list(right.columns[:-1])+['Group_Sum']

                groups = pd.merge(left = groups, right = right, left_on = columns[:-1], right_on = columns[:-1])
                groups['Percent%'] = groups['Group_Count']/groups['Group_Sum']*100
                groups = groups[columns+['Percent%']]
                return groups

        elif method == 'mean': groups = data.groupby(data.columns.tolist()[:-1]).apply(lambda x: x[data.columns[-1]].mean())
        elif method == 'median': groups = data.groupby(data.columns.tolist()[:-1]).apply(lambda x: x[data.columns[-1]].median())
        elif method == 'max': groups = data.groupby(data.columns.tolist()[:-1]).apply(lambda x: x[data.columns[-1]].max())
        elif method == 'min': groups = data.groupby(data.columns.tolist()[:-1]).apply(lambda x: x[data.columns[-1]].min())
        elif method == 'sum_percent':
            groups = data.groupby(data.columns.tolist()[:-1]).apply(lambda x: x[data.columns[-1]].sum()).reset_index()
            groups.columns = list(groups.columns[:-1])+['Group_Count']
            right = data.groupby(columns[:-1]).sum().reset_index()
            right.columns = list(right.columns[:-1])+['Group_Sum']

            groups = pd.merge(left = groups, right = right, left_on = columns[:-1], right_on = columns[:-1])
            groups['Sum%'] = groups['Group_Count']/groups['Group_Sum']*100
            groups = groups[cols+['Sum%']]
            return groups
        else:
            print('Method does not exist. Please choose count, count_percent, mean, median, max, min, sum_percent.'); return None;
        #except: print('Method = {}'.format(method)+' cannot work on Object, Non-Numerical data. Choose count.'); return None;

        groups = pd.DataFrame(groups)
        groups.columns = [method]
        groups.reset_index(inplace = True)
        return groups


def keep(data, what, inplace = False):
    '''Keeps data in a column if it's wanted.
    Everything else is filled with NANs'''
    if type(what) not in (list,tuple,np.array,np.ndarray): what = [what]
    need = data.isin(what)
    if inplace: 
        df = data
        df.loc[~need] = np.nan
    else: 
        df = data.copy()
        df.loc[~need] = np.nan
        return df

def remove(data, what, inplace = False):
    '''Deletes data in a column if it's not wanted.
    Everything else is filled with NANs'''
    if type(what) not in (list,tuple): what = [what]
    need = data.isin(what)
    if inplace: 
        df = data
        df.loc[need] = np.nan
    else: 
        df = data.copy()
        df.loc[need] = np.nan
        return df
    
    
def ternary(data, condition, true, false = np.nan, inplace = False):
    '''C style ternary operator on column.
    Condition executes on column, and if true, is filled with some value.
    If false, then replaced with other value. Default false is NAN.'''
    try:
        execute = 'data {}'.format(condition)
        series = eval(execute)
        try: series = series.map({True:true, False:false})
        except: series = series.replace({True:true, False:false})
        return series
    except: print('Ternary accepts conditions where strings must be enclosed.\nSo == USD not allowed. == "USD" allowed.'); return False;

    
def locate(data, column):
    '''Use ternary to get result and then filter with notnull'''
    if dtype(column) == 'bool': return data.loc[column]
    return data.loc[column.notnull()]
    
def query(data, column = None, condition = None):
    '''Querying data based on conditions'''
    if type(column) is str:
        cond = f'data["{column}"]{condition}'
    else:
        cond = f'column{condition}'
    return data.loc[eval(cond)]
        
def keep_top(x, n = 5):
    '''Keeps top n (after tallying) in a column'''
    df = keep(x, tally(x)[x.name][:n].values)
    return df

def keep_bot(x, n = 5):
    '''Keeps bottom n (after tallying) in a column'''
    df = keep(x, tally(x)[x.name][:-n].values)
    return df

def cut(x, bins = 5, method = 'range'):
    '''Cut continuous column into parts.
        Method options:
            1. range
            2. quantile (number of quantile cuts)'''
    if method == 'range': return pd.cut(x, bins = bins, duplicates = 'drop')
    else: return pd.qcut(x, q = bins, duplicates = 'drop')
    
    
def plot(x, y = None, colour = None, column = None, data = None, size = 5, top = 10, wrap = 4, 
         subset = 5000, method = 'mean', quantile = True, bins = 10,
         style = 'lineplot', logx = False, logy = False, logc = False, power = 1):
    '''Plotting function using seaborn and matplotlib
        Options:
        x, y, colour, column, subset, style, method
        
        Plot styles:
            1. boxplot
            2. barplot
            3. tallyplot (counting number of appearances)
            4. violinplot (boxplot just fancier)
            5. lineplot (mean line plot)
            6. histogram
            7. scatterplot (X, Y must be numeric --> dates will be converted)
            8. bivariate (X, Y must be numeric --> dates will be converted)
            9. heatmap (X, Y will be converted into categorical automatically --> bins)
            10. regplot (X, Y must be numeric --> dates will be converted)
    '''
    if type(x) in (np.array,np.ndarray): x = pd.Series(x); x.name = 'x';
    if type(y) in (np.array,np.ndarray): y = pd.Series(y); y.name = 'y';
    if type(column) in (np.array,np.ndarray): column = pd.Series(column); column.name = 'column';
    if type(colour) in (np.array,np.ndarray): colour = pd.Series(colour); colour.name = 'colour';
        
    if type(x) == pd.Series: 
        data = pd.DataFrame(x); x = x.name
        if type(x) is not str:
            data.columns = [str(x)]
            x = str(x)
    if method == 'mean': estimator = np.nanmean
    elif method == 'median': estimator = np.nanmedian
    elif method == 'min': estimator = np.min
    elif method == 'max': estimator = np.max
    else: print('Wrong method. Allowed = mean, median, min, max'); return False;
    #----------------------------------------------------------
    sb.set(rc={'figure.figsize':(size*1.75,size)})
    dtypes = {'x':None,'y':None,'c':None,'col':None}
    names = {'x':None,'y':None,'c':None,'col':None}
    xlim = None
    #----------------------------------------------------------
    if data is not None:
        if type(x) is str: x = data[x];
        if type(y) is str: y = data[y]; 
        if type(colour) is str: colour = data[colour]; 
        if type(column) is str: column = data[column]; 
    if type(x) is str: print('Please specify data.'); return False;
    #----------------------------------------------------------
    if x is not None:
        dtypes['x'] = dtype(x); names['x'] = x.name
        if dtypes['x'] == 'object': x = keep_top(x, n = top)
        elif dtypes['x'] == 'datetime': x = date_number(x)
        if logx and dtype(x) != 'object': x = log(x)
    if y is not None: 
        dtypes['y'] = dtype(y); names['y'] = y.name
        if dtypes['y'] == 'object': y = keep_top(y, n = top)
        elif dtypes['y'] == 'datetime': y = date_number(y)
        if logy and dtype(y) != 'object': y = log(y)
    if colour is not None:
        dtypes['c'] = dtype(colour); names['c'] = colour.name
        if dtypes['c'] == 'object': colour = keep_top(colour, n = top)
        elif dtypes['c'] == 'datetime': colour = date_number(colour)
        if logc and dtype(colour) != 'object': colour = log(colour)
    if column is not None:
        dtypes['col'] = dtype(column); names['col'] = column.name
        if dtypes['col'] == 'object': column = keep_top(column, n = top)
        elif dtypes['col'] == 'datetime': column = date_number(column)
    #----------------------------------------------------------
    df = hcat(x, y, colour, column)
    if subset > len(df): subset = len(df)
    df = sample(df, subset)
    #----------------------------------------------------------
    if column is not None:
        if dtype(df[names['col']]) not in ('object', 'uint',' int') and nunique(df[names['col']]) > top: 
            if quantile: df[names['col']] = cut(df[names['col']], bins = bins, method = 'quantile')
            else: df[names['col']] = cut(df[names['col']], bins = bins, method = 'range')
    
    try: df.sort_values(names['y'], inplace = True);
    except: pass;
    #----------------------------------------------------------
    replace = {'boxplot':'box', 'barplot':'bar', 'tallyplot':'count', 'violinplot':'violin', 
               'lineplot': 'point', 'histogram':'lv'}
    
    if style == 'histogram' and y is None:
        plot = sb.distplot(df[names['x']].loc[df[names['x']].notnull()], bins = bins)
    elif style == 'lineplot' and y is None:
        plot = plt.plot(df[names['x']]);
        plt.show(); return;
    elif style == 'barplot' and y is None:
        plot = df.sort_values(names['x']).plot.bar();
        plt.show(); return;
    elif style in replace.keys():
        if dtype(df[names['x']]) not in ('object', 'uint',' int') and nunique(df[names['x']]) > top: 
            if quantile: df[names['x']] = cut(df[names['x']], bins = bins, method = 'quantile')
            else: df[names['x']] = cut(df[names['x']], bins = bins, method = 'range')
        
        if names['col'] is not None:
            plot = sb.factorplot(x = names['x'], y = names['y'], hue = names['c'], data = df, kind = replace[style], col = names['col'],
                             n_boot = 1, size = size, estimator = estimator, col_wrap = wrap)
        else:
            plot = sb.factorplot(x = names['x'], y = names['y'], hue = names['c'], data = df, kind = replace[style], col = names['col'],
                             n_boot = 1, size = size, estimator = estimator)
            
        for ax in plot.axes.flatten(): 
            for tick in ax.get_xticklabels(): 
                tick.set(rotation=90)
    
    elif style == 'heatmap':
        if dtype(df[names['x']]) != 'object'and nunique(df[names['x']]) > top:
            if quantile: df[names['x']] = cut(df[names['x']], bins = bins, method = 'quantile')
            else: df[names['x']] = cut(df[names['x']], bins = bins, method = 'range')
                
        if dtype(df[names['y']]) != 'object'and nunique(df[names['y']]) > top:
            if quantile: df[names['y']] = cut(df[names['y']], bins = bins, method = 'quantile')
            else: df[names['y']] = cut(df[names['y']], bins = bins, method = 'range')     

        df = tabulate(df[names['x']], df[names['y']]).pivot(index = names['x'], columns = names['y'], values = 'count')
        plot = sb.heatmap(df, cmap="YlGnBu")

        
    elif dtype(df[names['x']]) == 'object' or dtype(df[names['y']]) == 'object':
            print('{} can only take X = number and Y = number.'.format(style)); return False;
        
    elif style  in ('regplot', 'scatterplot'):
        if column is None: col_wrap = None
        else: col_wrap = wrap
        if style == 'regplot': reg = True
        else: reg = False
        
        plot = sb.lmplot(x = names['x'], y = names['y'], hue = names['c'], data = df, col = names['col'],
                             n_boot = 2, size = size, ci = None, scatter_kws={"s": 50,'alpha':0.5},
                        col_wrap = col_wrap, truncate = True, fit_reg = reg, order = power)
        plot.set_xticklabels(rotation=90)
        
    elif style == 'bivariate':
        plot = sb.jointplot(x = names['x'], y = names['y'], data = df, dropna = True, size = size, kind = 'reg',
                           scatter_kws={"s": 50,'alpha':0.5}, space = 0)
    plt.show()


# <a id='Content'></a>
# <h1> 1. Reading Data </h1>

# **Reading CSV data in**
# 
# First, we want to know what files are in the database.
# 
# Remember to execute commands in a cell, press CTRL + ENTER

# In[ ]:


files()


# We will be using the second file.
# 
# Read it in using read(...)

# In[ ]:


data = read('ks-projects-201801.csv')


# **Showing what's inside the data**
# 
# Use the head(..., N) command to show the first N rows

# In[ ]:


head(data)


# In[ ]:


head(data, 5)


# In Python, getting 1 column is used by indexing [] and ["..."]

# In[ ]:


head(  data['id']  )


# In[ ]:


head(  data['currency']  )


# Let us see a description or summary of the data
# 
# (It's a bit slow since theres a lot of calculations to be done - so give it some time)
# 
# Describe gives you information about the:
# 
# 1. Data Type (float? 1.002, - decimals), (int? integer 1,2,3), (object? string)
# 2. Missing% (how much of the data is NAN values - missing)
# 3. Nunique (how many unique values are in the data - [1,1,1,2,2,3] has 3 unique values == [1,2,3]
# 4. Top3 (the top 3 values which are the most occuring)
# 5. Min (minimum value in the data)
# 6. Mean (average value in the data)
# 7. Median (the middle value in the data)
# 8. Max (the maximum value in the data)
# 9. Sample (an example of what the data looks like)

# In[ ]:


describe(data)


# In[ ]:


head(data)


# <a id='Kickstarter'></a>
# <h1> 2. Kickstarter Data </h1>
# 
# **Aim for Kickstarter data**
# 
# Our aim for the Kickstarter data is to see how the STATE (failed, successful) is correlated with the other variables.
# 
# This means, first we need to check the STATE column.
# 
# Note - all datasets in real life are very corrupt. In fact, theres a statistic which says that over 80% of a data scientists' job is just data cleaning and data manipulation.
# 
# First, let just inpsect the STATE column.
# 
# Since it's over 300,000 rows, it's infeasible to check all rows.
# 
# Let's use the TALLY command. This counts how many occurences of each unique value is in the data.
# 
# If you want more information on a function, type HELP

# In[ ]:


help(tally)


# In[ ]:


head(data, 1)


# In[ ]:


tally(  data['state']  )


# First, let us see how the GOAL relates to PERCENTAGE SUCCESS.
# 
# Using TABULATE, we can summarise a dataset's information into a table. The goal is to find the AVERAGE GOAL per STATE.
# 
# Let us check what TABULATE does after the code below usin HELP

# In[ ]:


tabulate(data['state'], data['goal'], method = 'mean')


# In[ ]:


help(tabulate)


# In[ ]:


goal = tabulate(data['state'], data['goal'], method = 'mean')


# In[ ]:


plot(x = 'state', y = 'mean', data = goal)


# In[ ]:


plot(x = 'state', y = 'mean', data = goal, style = 'barplot')


# Clearly, a SUCCESSFUL campaign has a very very low GOAL.
# 
# This essentially means more expensive projects are more likely to fail.
# 
# You can see SUSPENDED has a very high GOAL. This is also not good, and is indicative of failure.

# Let's check the country vs success rate.
# 
# Country is "country". Let us first inspect the column for data issues using the tally command

# In[ ]:


head(data, 1)


# In[ ]:


tally(data['country'])


# Now, we use TABULATE again to summarise the country and success rate.
# 
# We want to know the success rate per country, so we use the PERCENT mode in TABULATE

# In[ ]:


tabulate(data['country'], data['state'], method = 'count_percent')


# But, we only want the SUCCESSFUL %.
# 
# So, use QUERY, and filter == "successful" out.

# In[ ]:


country = tabulate(data['country'], data['state'], method = 'count_percent')


# In[ ]:


query(data = country, column = 'state', condition = '=="successful"')


# In[ ]:


country_success = query(data = country, column = 'state', condition = '=="successful"')


# Choose to plot ALL countries, so I placed top = 30. You can do top = 10000 ur choice, but the X Axis will be longer IF the data has 10000 rows

# In[ ]:


plot(data = country_success, x = 'country', y = 'Percent%', top = 30, style = 'barplot')


# Clearly, US wins, with over 35% success rate. Next is Great Britain 35% ish. Worst is IT, JP (Japan and Italy?) at around 15-17%

# Now, let's check the actual currency which might affect success rates. I expect it to have a similar trend with COUNTRY vs STATE
# 
# First, inspect CURRENCY using TALLY again

# In[ ]:


head(data, 1)


# In[ ]:


tally(data['currency'])


# I don't see any errors, so directly lets use TABULATE.

# In[ ]:


tabulate(data['currency'], data['state'], method = 'count_percent')


# Likewise, choose only STATE == "successful"

# In[ ]:


currency = tabulate(data['currency'], data['state'], method = 'count_percent')

query(data = currency, column = 'state', condition = '=="successful"')


# <a id='Quick'></a>
# <h1> 3. Analysis of Time Data </h1>
# 
# Now, let's analyse **launched** using **YEAR**. This'll convert the dates to keep only YEARS.

# In[ ]:


head(   year(data['launched'])  , 10)


# You can see we only got the years.
# 
# Now, do the same, check how much MAIN_CATEGORY has changed.
# 
# Which was most popular in 2014? 2015?

# In[ ]:


launched = tabulate(year(data['launched']), data['main_category'], method = 'count_percent')

head(launched)


# Also it's a bit strange --> why is there 1970?
# 
# Let's remove it using **QUERY** and **YEAR**

# In[ ]:


year(data['launched'])


# In[ ]:


data = query(data, column = year(data['launched']), condition = '>1970')


# Use the COLOUR option to plot multiple lines instead of 1

# In[ ]:


plot(data = launched, x = 'main_category', y = 'Percent%', colour = 'launched', style = 'barplot')


# You can also instead of plotting multiple lines, split it into multiple plots using COLUMN instead of COLOUR

# In[ ]:


plot(data = launched, x = 'main_category', y = 'Percent%', column = 'launched', style = 'lineplot')


# Finally, let's check what happens to the SUCCESS RATE over time. This time, let's not just use YEARS, but MONTH + YEARS.
# 
# Use the year_month command.
# 
# So, 2012 July (7th month) will be == 2012+7/12.
# 
# <a id='Year'></a>
# <h1> 4. Processing and Using Dates </h1>

# In[ ]:


head(    year_month(data['launched'])  )


# In[ ]:


months = tabulate(year_month(data['launched']), data['state'], method = 'count_percent')

months_percent = query(data = months, column = 'state', condition = '=="successful"')

head(months_percent)


# In[ ]:


plot(data = months_percent, x = 'launched', y = 'Percent%', style = 'scatterplot')


# Clearly, % Success have been decreasing over the years!
# 
# Let's draw a REGRESSION LINE to forecast what might happen! Use regplot in STYLE

# In[ ]:


plot(data = months_percent, x = 'launched', y = 'Percent%', style = 'regplot')


# We can see a clearer trend if we draw a CUBIC graph (degree 3)

# In[ ]:


plot(data = months_percent, x = 'launched', y = 'Percent%', style = 'regplot', power = 3)


# But WHY??? Is the number of projects correlated?
# 
# So over the years, would more projects = overall low success rate?

# <a id='Lab'></a>
# <h1> 5. Lab Questions </h1>
# 
# <img src="https://previews.123rf.com/images/christianchan/christianchan1503/christianchan150300425/37144675--now-it-s-your-turn-note-pinned-on-cork-.jpg" style="width: 300px;"/>

# (1) In this exercise, you need to:
# 
# 1. Figure out WHY success rate has been decreasing. Our hypothesis is that theres MORE projects, hence the decrease.
# 2. Your job is to verfiy this.
# 4. Use TABULATE to get the COUNT of the YEAR_MONTH to see how many projects are launched in that month
# 5. PLOT a REGPLOT and compare this to the SUCCESS per year_month.
# 
# 6. Explain to the tutor what's your insight.
# 
# It should look like this:
# <img src="https://drive.google.com/uc?id=1eDVCGTcGJ8JN3aUqixDSrk8WP7CilSse" style="width: 300px;"/>

# In Python, comments can be placed inside code (text that is not executed). You need to place the # symbol.

# In[ ]:


#Your code goes here


# (2) Now, we want to know per COUNTRY and per MAIN_CATEGORY, what is the SUCCESS PERCENTAGE.
# 
# 1. Use TABULATE for country, main_category and state using COUNT_PERCENT method.
# 2. QUERY only successful.
# 3. BAR PLOT using COLUMN for each country, and show X = main_category, Y = Percent
# 
# The Plot should look like:
# <img src="https://drive.google.com/uc?id=1BXyt9196iCClmJ-Eh8Dm-Vdnunnly_br" style="width: 500px;"/>

# In[ ]:


# Your code goes here


# Once you are done, and satisfied with your work, let the tutor mark you.
# 
# Note, if you can't get it, it's fine. Marks are awarded for trying the questions out. We don't mind if the output is wrong.
# 
# <h2>REMINDER ABOUT ASSIGNMENT 1 AGAIN</h2>
# 
# Next week night (Friday Week 3), you will receive a Kaggle link to **Assignment 1**. It's worth 7 marks, and super easy :).
# 
# The stuff you learnt in week 1 and today & next week will be enough to complete Assignment 1 entirely.
# 
# It's due in 2 weeks time on Saturday week 5 10pm. We need you to submit a URL / LINK to your assignment notebook on Kaggle.
# 
# IT MUST BE **PUBLIC**. Also, download the Kaggle file, and upload it to Moodle by 10pm Saturday week 5.
