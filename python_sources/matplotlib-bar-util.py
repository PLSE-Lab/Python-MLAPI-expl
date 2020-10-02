# %% [code]
import matplotlib.pyplot as plt
import seaborn as sns
import locale
locale.setlocale(locale.LC_ALL, '')

def groupby_operation(dataframe, groupbycolumn, operation_column, operation, 
                      size=None, total=None, title=None, height=10, width=5, index=None):
    fig = plt.figure()
    fig.set_size_inches(width, height)

    ax1 = plt.subplot(1,1,1)

    if operation == 'sum':
        serie = dataframe.groupby(groupbycolumn)[operation_column].sum().sort_values(ascending=True).astype(float)
    elif operation == 'count':
        serie = dataframe.groupby(groupbycolumn)[operation_column].count().sort_values(ascending=True).astype(float)
    else:
        serie = dataframe.groupby(groupbycolumn)[operation_column].count().sort_values(ascending=True).astype(float)
        
    if not total:
        total = serie.sum()
        
    corte = ''
        
    if size and len(serie) > size:
        serie = serie.sort_values(ascending=False)
        serie = serie[:size]
        serie = serie.sort_values(ascending=True)
        corte = ' ({} maiores)'.format(size)
    
    if not title:
        if operation_column:
            column = operation_column
        else:
            column = serie.name
        title = "Soma de {} agrupado por {}{}".format(operation_column, groupbycolumn, corte)
   
    ax1.barh(serie.index, serie, align='center', color='c', ecolor='black')
    percentage = serie/total*100
    number_distance = serie.max()*0.005
    
    for i, v in enumerate(serie):
        pct = locale.format_string('%.2f', percentage[i], True)
        v_str = locale.format_string('%.2f', v, True)
        ax1.text(v+number_distance , i-0.2, '{0} ({1}%)'.format(v_str, pct), color='k')
    ax1.set(title=title,
           xlabel='',
           ylabel='')
    sns.despine(left=True, bottom=True)

    plt.show()

def show_value_counts(serie, column_desc=None, grain='Registers', 
                      size=None, total=None, title=None, height=10, width=5, index=None):
    fig = plt.figure()
    fig.set_size_inches(width, height)

    ax1 = plt.subplot(1,1,1)

    serie = serie.value_counts().sort_values(ascending=True)

    if not total:
        total = serie.sum()
    
    corte = ''
    
    if (index):
        serie = serie.rename(index)
    
    if serie.index.dtype != 'object':
        if serie.index.dtype == 'float64':
            serie.index = serie.index.map(int)
        serie.index = serie.index.map(str)
    serie.index = serie.index.map(str)
    
    if size and len(serie) > size:
        serie = serie.sort_values(ascending=False)
        serie = serie[:size]
        serie = serie.sort_values(ascending=True)
        corte = ' ({} mais frequentes)'.format(size)
    
    if not title:
        if column_desc:
            column = column_desc
        else:
            column = serie.name
        title = "Nº de {} por {}{}".format(grain, column, corte)
   
    ax1.barh(serie.index, serie, align='center', color='c', ecolor='black')
    percentage = serie/total*100
    number_distance = serie.max()*0.005
    
    for i, v in enumerate(serie):
        pct = locale.format_string('%.2f', percentage[i], True)
        ax1.text(v+number_distance , i-0.2, '{0:,} ({1}%)'.format(v, pct), color='k')
    ax1.set(title=title,
           xlabel='',
           ylabel='')
    sns.despine(left=True, bottom=True)

    plt.show()