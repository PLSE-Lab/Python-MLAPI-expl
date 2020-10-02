#!/usr/bin/env python
# coding: utf-8

# # Differences between personality traits in groups
# 
# <img src="https://cdn.mos.cms.futurecdn.net/DFJEngWvSkQKpQMVF5Nw5R-650-80.jpg" width="1000px">
# 
# ## Abstract 
# 
# In this notebook, I will try to confirm a known statement in psychology: That there is more variability between members of a given group, than between the median case of each different group.
# 
# ## Why ?
# 
# This has some deep implications, like the fundamental description of a society is not a competition for power, between homogeneous groups where every member of a group shares the same personlity traits. 
# 
# Rather than that, the society would be a cooperation between people, that happens to share enough common characteristics in specific dimensions, to call such a set of common features, a group.
# 
# ## How
# 
# * First, I will do some exploratory analysis, and a list of pair groups to be compared. 
# * For each pair of groups, I will calculate its average personality traits, the intra-group distance.
# * Also, I'll calculate the distance between the centroids of each group, the inter-group distance.
# * Finally, I will analyze the personality traits differences for each group.
# 
# 
# 
# ## Groups to be compared: 
# 
# * Female vs Male
# * Male vs Other
# * Female vs Other
# * European vs North African
# * United States vs India
# * Left handed vs Right handed
# * Millenials vs. Boomers
# 
# ## Bonus: How do we age ?
#  
#  We will explore how the personality traits changes over time.
#  

# As a reminder, the [Big Five personality traits model](https://en.wikipedia.org/wiki/Big_Five_personality_traits) identifies five factors:
# 
# 1. openness to experience (inventive/curious vs. consistent/cautious)
# 2. conscientiousness      (efficient/organized vs. extravagant/careless)
# 3. extraversion           (outgoing/energetic vs. solitary/reserved)
# 4. agreeableness          (friendly/compassionate vs. challenging/callous)
# 5. neuroticism            (sensitive/nervous vs. resilient/confident)
# 

# In[ ]:



from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial import distance
import math
import statistics
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


def get_data(nRowsRead=None):
    
    

    df = pd.read_csv('/kaggle/input/ocean-five-factor-personality-test-responses/data.csv', delimiter='\\t', nrows = nRowsRead)

    reverse_values_dict = { 5:1 , 4:2 ,3:3 , 2:4, 1:5}
    # Some answers has reverse values
    for answer_num in ['E2','E4','E6','E8','E10',
                            'N2','N4',
                            'A1','A3','A5','A7',
                            'C2','C4','C6','C8','C10',
                            'O2','O4','E6'
                            ]:
        df[answer_num]=df.replace({answer_num: reverse_values_dict})
        
    """Load and format data, returning a df"""
    df.dataframeName = '"Big Five" personality traits scores'
    df['score_O'] =  (df['O1'] + df['O2'] + df['O3'] + df['O4'] + df['O5'] + df['O6'] + df['O7'] + df['O8' ] + df['O9'] + df['O10']) / 10
    df['score_C'] =  (df['C1'] + df['C2'] + df['C3'] + df['C4'] + df['C5'] + df['C6'] + df['C7'] + df['C8' ] + df['C9'] + df['C10']) / 10    
    df['score_E'] =  (df['E1'] + df['E2'] + df['E3'] + df['E4'] + df['E5'] + df['E6'] + df['E7'] + df['E8' ] + df['E9'] + df['E10']) / 10
    df['score_A'] =  (df['A1'] + df['A2'] + df['A3'] + df['A4'] + df['A5'] + df['A6'] + df['A7'] + df['A8' ] + df['A9'] + df['A10']) / 10
    df['score_N'] =  (df['N1'] + df['N2'] + df['N3'] + df['N4'] + df['N5'] + df['N6'] + df['N7'] + df['N8' ] + df['N9'] + df['N10']) / 10           
    df = df[['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country', 'score_O','score_C', 'score_E' ,'score_A' ,'score_N',  ]]
    return df

df = df1 = get_data()

nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns, {df.columns}')

df


# In[ ]:


"""
df=pd.read_csv('/kaggle/input/ocean-five-factor-personality-test-responses/data.csv', delimiter='\\t', nrows = 100)
df['score_O'] =  (df['O1'] + df['O2'] + df['O3'] + df['O4'] + df['O5'] + df['O6'] + df['O7'] + df['O8' ] + df['O9'] + df['O10']) / 10
df['score_C'] =  (df['C1'] + df['C2'] + df['C3'] + df['C4'] + df['C5'] + df['C6'] + df['C7'] + df['C8' ] + df['C9'] + df['C10']) / 10    
df['score_E'] =  (df['E1'] + df['E2'] + df['E3'] + df['E4'] + df['E5'] + df['E6'] + df['E7'] + df['E8' ] + df['E9'] + df['E10']) / 10
df['score_A'] =  (df['A1'] + df['A2'] + df['A3'] + df['A4'] + df['A5'] + df['A6'] + df['A7'] + df['A8' ] + df['A9'] + df['A10']) / 10
df['score_N'] =  (df['N1'] + df['N2'] + df['N3'] + df['N4'] + df['N5'] + df['N6'] + df['N7'] + df['N8' ] + df['N9'] + df['N10']) / 10           
df = df[['race', 'age', 'engnat', 'gender', 'hand', 'source', 'country', 'score_O','score_C', 'score_E' ,'score_A' ,'score_N',  ]]
df
"""


# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    


# # Exploratory analysis

# In[ ]:


plotPerColumnDistribution(df, 12, 4)


# In[ ]:


df.dataframeName = '"Big Five" personality traits scores'

plotCorrelationMatrix(df, 10)


# In[ ]:


df = get_data()
df['vector'] = df.apply(lambda x:  [x.score_O,x.score_C,x.score_E,x.score_A,x.score_N], axis=1)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt # plotting


# Display options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 3)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

def get_intent_centroid(array):
    centroid = np.zeros(len(array[0]))
    for vector in array:
        centroid = centroid + vector
    return centroid/len(array)


def enrich_vector_values(df):
    for i,letter in enumerate("OCEAN"):
        df[letter] =  float(df.vector.values[i])
    return df 

def group_vectors_by_criteria(df,criteria):
    centroids = df.groupby(criteria)['vector'].apply(lambda x: get_intent_centroid(x.tolist()))
    return pd.DataFrame(centroids.reset_index( ))


def highlight_max(data, color='lightblue'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

    



def calculate_distances(df,criteria,compare1,compare2,df_criteria):
    """ Calculate intra group and inter group distances"""
    results = pd.DataFrame(columns=["Group",'Distance type',"Description","Value"])
    array_1 = df_criteria[compare1:compare1].vector.values[0]
    array_2 = df_criteria[compare2:compare2].vector.values[0]
    d = distance.euclidean(array_1, array_2)
    for compare_iter,group_distance in zip( [compare1,compare2], [array_1,array_2])  :
        list_index    = df_criteria.index.values
        compare_index = np.where(list_index == compare_iter)[0][0]
        array_distances = []
        # We build the filter string for this category, depending the type of the category.
        cadena_busqueda = f""" {criteria} == {compare_index} """ if "int64" in str(df[criteria].dtypes) else f""" {criteria} == '{compare_iter}' """
        for vector_item in df.query(cadena_busqueda).vector.values:
            dst = distance.euclidean(vector_item,group_distance)
            array_distances.append(dst )
        results = results.append({"Group" : compare_iter ,
                                  'Distance type' : "Intra group distance",
                                  "Description" : "Average distance between each member of the gruop and the centroid of its group",
                                  "Value" : statistics.mean(array_distances) },ignore_index=True)
    results = results.append({"Group" : compare1 + " - "+ compare2  ,
                              'Distance type' : "Inter groups distance",
                              "Description" : "Distance between centroid of each groups",
                              "Value" : d },ignore_index=True)
    
    return results

def debug_calculate_distances():
    """ Just to debug the function"""
    df = get_data()
    df['vector'] = df.apply(lambda x:  [x.score_O,x.score_C,x.score_E,x.score_A,x.score_N], axis=1)
    criteria ="gender"
    compare1='Male'
    compare2='Female'
    df_criteria = group_vectors_by_criteria(df,criteria)
    df_criteria.index = ['missed', 'Male' , 'Female' ,'Other']

    #criteria          = 'country'
    #compare1          = 'US'
    #compare2          = 'IN'
    #df_criteria       = group_vectors_by_criteria(df,criteria)
    #df_criteria.index = df_criteria[criteria]
    #df_criteria

    return calculate_distances(df,criteria,compare1,compare2,df_criteria)


import seaborn as sns
from matplotlib import pyplot as plt

def compare_two_criteria(    df,compare1,compare2,criteria,color1,color2):
    """Return a df comparing two posible criterrias of a given group"""
    
    def draw_chart( compare1,compare2,criteria,color1,color2,list_index):
        df = get_data()
        df['vector'] = df.apply(lambda x:  [x.score_O,x.score_C,x.score_E,x.score_A,x.score_N], axis=1)

        df_criteria       = group_vectors_by_criteria(df,criteria)
        if criteria == 'country' : 
            df_criteria       = df_criteria[(df_criteria['country'] ==  compare1 ) | (df_criteria['country'] == compare2)]
            df_criteria.index = list_index
        elif criteria == 'age' :
            bins = [0, 24, 39, 56, 74, np.inf]
            names = ['<24 y.o.', 'millenials (born 1981-1996)', '39-56 y.o.', 'baby boomers (born 1946 1964)', '74+ y.o.']
            df['age'] = pd.cut(df['age'], bins, labels=names)
            df_criteria.index = df_criteria[criteria]
        else: 
            df_criteria.index = list_index

            
        #df_criteria.index = ['missed', 'Male' , 'Female' ,'Other']
        #color1="tab:blue"
        #color2="tab:pink"

        list_index      = df_criteria.index.values
        if criteria != 'age' :
            compare_index1 = np.where(list_index == compare1)[0][0]
            compare_index2 = np.where(list_index == compare2)[0][0]

        cadena_busqueda = f""" {criteria} == {compare_index1} or {criteria} == {compare_index2} """ if "int64" in str(df[criteria].dtypes) else f"""  {criteria} == '{compare1}' or {criteria} == '{compare2}'  """
        cadena_busqueda 
        df = df.query(cadena_busqueda)
        
        if criteria != 'age' :
            df.loc[df[criteria] == compare_index1,criteria] = compare1
            df.loc[df[criteria] == compare_index2,criteria] = compare2


        dict_column_names = { "score_O" : "openness to experience (inventive/curious vs. consistent/cautious)",
                          "score_C" :  "conscientiousness (efficient/organized vs. extravagant/careless)" ,
                          "score_E" :  "extraversion (outgoing/energetic vs. solitary/reserved)" ,
                          "score_A" :  "agreeableness (friendly/compassionate vs. challenging/callous)" ,
                          "score_N" : "neuroticism (sensitive/nervous vs. resilient/confident)" }

        df_char = pd.DataFrame(columns = ["trait" ,"value", "criteria"])
        arr_char = []
        for temp_dict in  df[[c for c in df.columns if 'score' in c ]+[criteria]].to_dict("records"): 
            for k,v in temp_dict.items():
                if 'score' in k   :
                    arr_char.append({'trait': dict_column_names[k] , 'value' : v, 'criteria' : temp_dict[criteria] })
        df_char = pd.DataFrame(arr_char)
        fig = ax = None
        from matplotlib import pyplot

        try:
            sns.set_style("darkgrid")
            pal = {compare1 : color1, compare2: color2}
            a4_dims = (12, 6)
            fig, ax = pyplot.subplots(figsize=a4_dims)
            sns.boxplot(ax=ax,y="trait", x="value", hue="criteria", data=df_char,showfliers  = False,  palette=pal)
            ax.set_title(f"Personality traits for {compare1} vs. {compare2}")
            ax.legend(loc="upper left" )
        except Exception as e :
            print("error: " + str(e))
        return fig,ax

    array_1 = df[compare1:compare1].vector.values[0]
    array_2 = df[compare2:compare2].vector.values[0] 
    listOfStr = ["openness to experience (inventive/curious vs. consistent/cautious)",
                 "conscientiousness (efficient/organized vs. extravagant/careless)" ,
                 "extraversion (outgoing/energetic vs. solitary/reserved)" ,
                 "agreeableness (friendly/compassionate vs. challenging/callous)" ,
                 "neuroticism (sensitive/nervous vs. resilient/confident)" ]
    dictOfWords_1 = { j : array_1[i] for i,j in enumerate(listOfStr) }
    dictOfWords_2 = { j : array_2[i] for i,j in enumerate(listOfStr) }
    df_compare = pd.DataFrame([dictOfWords_1,dictOfWords_2], index = [compare1,compare2])    
    
    df_compare.loc['Difference'] = df_compare.loc[compare1] - df_compare.loc[compare2]
    fig, ax = draw_chart(    compare1,compare2,criteria,color1,color2,df.index)
    return df_compare

def debug_compare_two_criteria():
    criteria          = 'gender'
    compare1          = 'Male'
    compare2          = 'Female'
    df_criteria       = group_vectors_by_criteria(df,criteria)
    #df_criteria.index = ['missed', 'Male' , 'Female' ,'Other']
    color1="tab:blue"
    color2="tab:pink"
    compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2)
    compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2)
    plt.show()
    
#debug_calculate_distances()
#debug_compare_two_criteria()




# # Compare gender: Male vs Female
# 
# <img src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/93b32a4b-14fb-4c68-ab02-f49c62021f38/d59565x-24e6e3d9-d660-45a2-a01d-c9278b7dda7d.jpg/v1/fill/w_1000,h_600,q_75,strp/male_n_female_symbol_by_devrez_d59565x-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3siaGVpZ2h0IjoiPD02MDAiLCJwYXRoIjoiXC9mXC85M2IzMmE0Yi0xNGZiLTRjNjgtYWIwMi1mNDljNjIwMjFmMzhcL2Q1OTU2NXgtMjRlNmUzZDktZDY2MC00NWEyLWEwMWQtYzkyNzhiN2RkYTdkLmpwZyIsIndpZHRoIjoiPD0xMDAwIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmltYWdlLm9wZXJhdGlvbnMiXX0.4_MNzLsbjcMVMOd3bKmnctYs6MyRD_2NTLo9wHDlur4" width="750px">
# 
# 

# In[ ]:



df = get_data()
df['vector'] = df.apply(lambda x:  [x.score_O,x.score_C,x.score_E,x.score_A,x.score_N], axis=1)

criteria          = 'gender'
compare1          = 'Male'
compare2          = 'Female'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria.index = ['missed', 'Male' , 'Female' ,'Other']
color1="tab:blue"
color2="tab:pink"

calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare gender: Male vs Female. Remarks
# 
# * Regarding distances, we confirm that intra-groups distances are higher than inter-group distance.
# 
# * "Male" group has less inter-group distance than Female group. It is interesting that this distribution does not follow the [Male variability hypothesis](https://en.wikipedia.org/wiki/Variability_hypothesis) that happens in many species, not only humans.
# 
# * Women score higher in the agreeableness and neuroticism. Also, in organization.
# * Men score higher in inventive.
# 
# 

# # Compare gender: Male vs Other
# 
# <img src="https://ih1.redbubble.net/image.220105526.2106/sn,x1000-pad,1000x1000,f8f8f8.u3.jpg" width="750px">
# 
# 
# In the questionnaire source of this dataset, in the "Genre" section,  one of the possible answers was "Other".
# We do not know exactly what does it mean, other than "Other" is not "Male" neither "Female" groups, but we will compare it.
# 
# 

# In[ ]:


criteria          = 'gender'
compare1          = 'Male'
compare2          = 'Other'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria.index = ['missed', 'Male' , 'Female' ,'Other']
color1="tab:blue"
color2="tab:orange"

calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare gender: Male vs Others. Remarks
# 
# 
# * Regarding intra groups distance, the "Others" groups has a high variability in the response. It is hard to explain it, as I cannot correlate the responses marked as genre : "Other", to an specific group.
# 
# * It is interesting, that due to the categories "solitary/reserved" and "sensitive/nervous", there is a great distance (0.399) from this Other group to the Male group.
# 

# In[ ]:





# # Compare gender: Female vs Other
# 
# <img src="https://ih1.redbubble.net/image.220089765.1599/sn,x1000-pad,1000x1000,f8f8f8.u1.jpg" width="750px">
# 
# 
# 

# In[ ]:


criteria          = 'gender'
compare1          = 'Female'
compare2          = 'Other'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria.index = ['missed', 'Male' , 'Female' ,'Other']
color1="tab:pink"
color2="tab:orange"

calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare gender: Female vs Others. Remarks
# 
# * As in the previous comparision, I cannot correlate the responses marked as genre : "Other" to a particular group.
# 
# * It is even more interesting, that this "Other" group, is not close to the "Female" group neither Male group. It seems a category on its own.
# 
# * The "Other" group has a high score neuroticism, the emotional stability trait. Also it has a great intra-group distance, that corresponds with a higher variability. 
# 

# # Compare hand: Left handed vs Right handed
# 
# <img src="https://www.frontiersin.org/files/Articles/124457/frym-2014-00013-HTML-r3/image_m/main.jpg" width="750px">
# 

# In[ ]:


criteria          = 'hand'
compare1          = 'Right handed'
compare2          = 'Left handed'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria.index = ['missed', 'Right handed' , 'Left handed' ,'Both']

color1="tab:blue"
color2="tab:red"
calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare hand: Left handed vs Right handed. Remarks
# 
# * The inter group distance is low (0.061) that suggest that there is no really a big difference between left and right handed groups

# # Compare race: European vs North African
# 
# <img src="https://www.debatingeurope.eu/wp-content/uploads/2013/06/africa-eu.png" width="750px">
# 
# 

# In[ ]:


"""
1=Mixed Race, 2=Arctic (Siberian, Eskimo), 3=Caucasian (European), 4=Caucasian (Indian), 5=Caucasian (Middle East), 6=Caucasian (North African, Other), 7=Indigenous Australian,
8=Native American, 9=North East Asian (Mongol, Tibetan, Korean Japanese, etc), 10=Pacific (Polynesian, Micronesian, etc), 11=South East Asian (Chinese, Thai, Malay, Filipino, etc), 12=West African, Bushmen, Ethiopian, 13=Other (0=missed)
"""
criteria          = 'race'
compare1          = 'Caucasian (European)'
compare2          = 'Caucasian (North African, Other)'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria.index = ['missed', 
                 'Mixed Race' ,
                 'Arctic (Siberian, Eskimo)' ,
                 'Caucasian (European)',
                 'Caucasian (Indian)',
                 'Caucasian (Middle East)',
                 'Caucasian (North African, Other)',
                 'Indigenous Australian',
                 'Native American',
                 'North East Asian (Mongol, Tibetan, Korean Japanese, etc)',
                 'Pacific (Polynesian, Micronesian, etc)',
                 'South East Asian (Chinese, Thai, Malay, Filipino, etc)',
                 'West African, Bushmen, Ethiopian',
                 'Other'
                ]
color1="#003399"
color2="tab:orange"

calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare race: European vs North African. Remarks
# 
# * There is really a big inter groups distance, where north africaners have higher scores on every dimension.

# In[ ]:


# Debug, we calculate distance for all countries, to know which countries has more differences.

def get_countries_distance():
    df_country = group_vectors_by_criteria(df,'country')
    df_country.index = df_country.country
    array_country = []
    for country1 in df_country.country.values :
        for country2 in df_country.country.values : 
            if country1 == country2 : continue
            array_1 = df_country[country1:country1].vector.values[0]
            array_2 = df_country[country2:country2].vector.values[0]
            d = distance.euclidean(array_1, array_2) 
            full_description = country1 + " - " + country2
            array_country.append({ "complete_name" : full_description , "distance" : d , "country1" : country1 , "country2" : country2})

    df_country_distance = pd.DataFrame(array_country )
    df_country_distance

    df_country_distance.sort_values('distance', ascending=False).head(50)
    df_country_distance[(df_country_distance['country1'] == 'US')].sort_values('distance', ascending=False).head(20)

#get_countries_distance()


# # Compare countries: United States vs India
# 
# <img src="https://tp.taxsutra.com/sites/tp.taxsutra.com/files//iStock-530159353.jpg" width="750px">
# 
# 

# In[ ]:


criteria          = 'country'
compare1          = 'US'
compare2          = 'IN'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria       = df_criteria[(df_criteria['country'] ==  compare1 ) | (df_criteria['country'] == compare2)]
df_criteria.index = df_criteria[criteria]
df_criteria

color1="#BF0A30"
color2="#FF8F1C"

calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare countries: United States vs India. Remarks
# 
# * As in the previous comparison, one of the groups scores higher en every measure.
# 
# * I found this a very interesting comparision. I wish I could have more data to draw futher insights. The variability in the intra-group distance for India group would need more investigation.

# # Compare age: Millenials vs Boomers
# 
# <img src="https://kaysharbor.com/wp-content/uploads/2015/09/millennials_featured-776x415.png" width="750px">
# 

# In[ ]:


# Categorize age
df = get_data()
df['vector'] = df.apply(lambda x:  [x.score_O,x.score_C,x.score_E,x.score_A,x.score_N], axis=1)
bins = [0, 24, 39, 56, 74, np.inf]
names = ['<24 y.o.', 'millenials (born 1981-1996)', '39-56 y.o.', 'baby boomers (born 1946 1964)', '74+ y.o.']
df['age'] = pd.cut(df['age'], bins, labels=names)

# Create comparison
compare1         = 'millenials (born 1981-1996)'
compare2         = 'baby boomers (born 1946 1964)'
criteria         = 'age'
df_criteria       = group_vectors_by_criteria(df,criteria)
df_criteria.index = df_criteria[criteria]
df_criteria

color1="tab:brown"
color2="tab:gray"



calculate_distances(df,criteria,compare1,compare2,df_criteria)


# In[ ]:


compare_two_criteria(df_criteria,compare1,compare2,criteria,color1,color2).style.apply(highlight_max,axis=0)


# # Compare age: Millenials vs Boomers. Remarks
# 
# 
# * It is hard to tell wether the differences observed are consequence of the many changes happened in the world during the last decades, or it is just a matter of age. Probably, both.

# # Bonus : How do we age ?
# 
# <img src="https://c4.wallpaperflare.com/wallpaper/506/550/531/cartoon-up-cartoon-boy-wallpaper-preview.jpg" width="750px">
# 
# 
# We explore how the median value for each traits changes, in each age interval.
# 
# Also, we split data for each genre, to find out if men and women are affected by aging in a similar manner.
# 
# 
# 
# 

# In[ ]:


def extract_age_summary(df,gender):
    
    """return a dataframe with the age, given a gender value as a parameter"""
    listOfStr = ["openness to experience (inventive/curious vs. consistent/cautious)",
                 "conscientiousness (efficient/organized vs. extravagant/careless)" ,
                 "extraversion (outgoing/energetic vs. solitary/reserved)" ,
                 "agreeableness (friendly/compassionate vs. challenging/callous)" ,
                 "neuroticism (sensitive/nervous vs. resilient/confident)" ]
    names = ['<13 y.o.', '13-22 y.o.', '22-40 y.o.', '40-60 y.o.', '60+ y.o.']

    array_dictOfWords = []
    df_gender = df.loc[df['gender'] == gender]
    df_age = group_vectors_by_criteria(df_gender,'age')
    df_age.index = names 
    df_age

    for row in df_age.iterrows():
        # Convert to df of aggregate data
        array = row[1][1]  # extract vector
        dictOfWords = { j : array[i] for i,j in enumerate(listOfStr) }
        dictOfWords['age'] = row[0] # extract age interval 
        array_dictOfWords.append(dictOfWords)
    df_age_summary = pd.DataFrame(array_dictOfWords) 
    return df_age_summary.set_index('age')

dict_ages = { 'Male' : 1 , "Female" : 2  }


def Draw_Age_Charts():
    param_array=[
                 {"df_gender" : 1, "gender" : "Male:\n"   , "color" :'tab:blue' },
                 {"df_gender" : 2, "gender" : "Female:\n" , "color" :'tab:pink' },
                ]   
    df = get_data()
    df['vector'] = df.apply(lambda x:  [x.score_O,x.score_C,x.score_E,x.score_A,x.score_N], axis=1)
    # Categorize age
    bins = [0, 13, 22, 40, 60, np.inf]
    names = ['<13 y.o.', '13-22 y.o.', '22-40 y.o.', '40-60 y.o.', '60+ y.o.']
    df['age'] = pd.cut(df['age'], bins, labels=names)
    for gender_dict in param_array :
        df_gender = extract_age_summary(df,gender_dict['df_gender']) 
        gender    = gender_dict['gender'] 
        color     = gender_dict['color'] 
        f, axes = plt.subplots(1 , 5 ,figsize=(18, 3), sharex=True)
        for i,ca in enumerate(df_gender.columns):
            ax = axes[i]
            plt.axes(ax)
            plt.ylim(2,5)
            plt.plot(df_gender[ca],color=color)
            ax.set_title(gender+ ca.replace("(","\n(").replace("vs.","\nvs."))
            plt.xticks(rotation=45)


# In[ ]:


# data for Females
df_ages = extract_age_summary(df,dict_ages["Female"])
df_ages.columns = [ "Female: " + c for c in df_ages.columns ]
df_ages.style.apply(highlight_max,axis=0)    


# In[ ]:


# data for Males
df_ages = extract_age_summary(df,dict_ages["Male"])
df_ages.columns = [ "Male: " + c for c in df_ages.columns ]
df_ages.style.apply(highlight_max,axis=0)    


# In[ ]:


Draw_Age_Charts()


# # How do we age ? - Remarks
# 
# - The "conscientiousness" trait decreases in each life phase, probably related to the physical status decay. 
# 
# - I really expected that the "friendly/compassionate" dimension, would shown an increase in the old age.  I suppose that means that the archetypal "lovely grandma" woman only is reached if you were equally friendly before aging
# 
# - Men and women tend to grow older in a similar manner, altough it is remarkable that the main difference between traits for men and women - neuroticism - disappears as age increments.
