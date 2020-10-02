#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession, Row


# In[ ]:


sc = SparkContext("local",'quiz-data-viz')


# In[ ]:


spark = SparkSession(sc)
data_quiz = sc.textFile('/kaggle/input/output1.csv').map(lambda line:line.split(','))


# In[ ]:


df_quiz = spark.read.csv('/kaggle/input/output1.csv',header=True)
df_quiz.head(3)


# First task that we need to do some research for the relation between the attributs : 
# <br>gender ------- radio, music, ambiance, emission</br>
# <br>age    ------- radio, music, ambiance, emission</br>
# <br>habits ------- radio, music, ambiance, emission</br>
# <br>color  ------- radio, music, ambiance, emission</br>

# In[ ]:


df_quiz.printSchema()


# In[ ]:


import ast
sum_male = df_quiz.rdd.filter(lambda x:x[6]=='Homme').count()
sum_female = df_quiz.rdd.filter(lambda x:x[6] == 'Femme').count()


# In[ ]:


def get_statistic_by(key_num):
    x_radio = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[11]))).filter(lambda x: len(x[1]) > 0)
    x_music = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[12]))).filter(lambda x: len(x[1]) > 0)
    x_ambiance = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[13]))).filter(lambda x: len(x[1]) > 0)
    x_emission = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[14]))).filter(lambda x: len(x[1]) > 0)
    # get flat
    x_radio_flat = x_radio.map(lambda x: [((x[0],y),1) for y in x[1]]).flatMap(lambda x: (t for t in x))
    x_music_flat = x_music.map(lambda x: [((x[0],y),1) for y in x[1]]).flatMap(lambda x: (t for t in x))
    x_ambiance_flat = x_ambiance.map(lambda x: [((x[0],y),1) for y in x[1].items()]).flatMap(lambda x: (t for t in x))
    x_emission_flat = x_emission.map(lambda x: [((x[0],y),1) for y in x[1]]).flatMap(lambda x: (t for t in x))
    # get statistics
    data_x_radio = x_radio_flat.reduceByKey(lambda a,b : (a+b)).collect()
    data_x_music = x_music_flat.reduceByKey(lambda a,b : (a+b)).collect()
    data_x_ambiance = x_ambiance_flat.reduceByKey(lambda a,b : (a+b)).collect()
    data_x_emission = x_emission_flat.reduceByKey(lambda a,b : (a+b)).collect()
    return data_x_radio, data_x_music, data_x_ambiance, data_x_emission


# In[ ]:


import matplotlib.pyplot as plt 
import numpy as np

def plot_data_gender(dataset, vr, name):
    data_male = list(filter(lambda x: x[0][0]=='Homme', dataset))
    data_female = list(filter(lambda x: x[0][0]=='Femme', dataset))
    dm = sorted(data_male ,key=lambda x:x[1], reverse=True)
    df = sorted(data_female ,key=lambda x:x[1], reverse=True)
    fig = plt.figure(figsize=(15,10), dpi=80, facecolor='w')
    ax1 = fig.add_subplot(311)
    ax1.bar(np.arange(len(dm)),[x[1] for x in dm], align='center')
    ax1.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(dm))))
    ax1.xaxis.set_major_formatter(plt.FixedFormatter([x[0][1] for x in dm]))
    plt.xticks(rotation=vr)
    plt.title("The "+name+" distribution for male")
    ax2 = fig.add_subplot(313)
    ax2.bar(np.arange(len(df)), [x[1] for x in df], align='center', color='r')
    ax2.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(df))))
    ax2.xaxis.set_major_formatter(plt.FixedFormatter([x[0][1] for x in df]))
    plt.xticks(rotation=vr)
    plt.title("The "+name+" distribution for female")
    plt.show()
    


# In[ ]:


import plotly.graph_objects as go
import plotly.express as px
data_gender_radio, data_gender_music, data_gender_ambiance, data_gender_emission = get_statistic_by(6)


# In[ ]:


def plot_data_by_gender(dataset, gtitle, xtitle):
    data_male = list(filter(lambda x: x[0][0]=='Homme', dataset))
    data_female = list(filter(lambda x: x[0][0]=='Femme', dataset))
    dm = sorted(data_male ,key=lambda x:x[1], reverse=True)
    df = sorted(data_female ,key=lambda x:x[1], reverse=True)
    fig = go.Figure(
        data=[
            go.Bar(name='Male', marker_color='#EB89B5', x=list(map(lambda x:x[0][1], dm[:12])), y=list(map(lambda x:x[1]/sum_male, dm[:12]))),
            go.Bar(name="Female", marker_color='#330C73', x=list(map(lambda x:x[0][1], df[:12])), y=list(map(lambda x:x[1]/sum_female, df[:12])))
        ], 
        layout=go.Layout(
            title=gtitle,
            xaxis=dict(
                title=xtitle,
                titlefont=dict(
                    family = 'Courier New, monospace',
                    size = 18,
                    color = '#7f7f7f'
                )
            )
        )
    )

    fig.update_layout(barmode='group')
    fig.show()
    return None


plot_data_by_gender(data_gender_radio, 'Comparison of the radio preference by gender', 'Radio')


# Radio preference character :
# <br>the selection of radio is limited. So the people don't have a variety of resources, they could be limited by the time.</br>
# <br></br>
# <br>Analyse of the graph comparison of the preference by gender :</br> 
# <br>Here, the question is always like that 'have the factor gender actually affected the preference ?'. Look at this graph, the results here we nominate first 12 most frequent selections for each gender are presented. From the comparison of the radio preference by gender, the male preference of radio is different from female except for the most frequent choose the radio RTL2. And we can get a clear observation of the difference of radio preference between male and female. </br><br></br>
# <br>Conclusion for gender influencing radio preference : To some degree, the radio preference is affected by gender. </br> 

# In[ ]:


plot_data_by_gender(data_gender_music, 'Comparison of the music preference by gender', 'music')


# Here the result is certain. Obviously, there is a variation in the selection of music between male and female.

# In[ ]:


dga = list(map(lambda x: ( ( (x[0][0], x[0][1][0]+':'+str(x[0][1][1]) ),x[1]) ), data_gender_ambiance))
plot_data_by_gender(dga, 'Comparison of the ambiance preference by gender', 'ambiance')


# From the comparison of the ambiance preference by gender, we must clarify the importance of two factors, gender and ambiance in the vehicle. And we assume that these conditions would be with family, alone, on the morning, on the evening. The number 1-3 represents that the ambiance is more energetic and dynamic. 
# 
# Firstly, we need to understand the affect of gender. For the same situation, we compare the distribution of preference tendency (the value of each column in the bar graph) and we can see the common part and different part. 
# 1. The common part : 'alone:3', 'evening:1', 'evening:2', 'evening:3', 'alone:2', 'morning:2'. These bars don't have so much many distinctions for the choose probability. To some aspect, these are common possible preferences.
# 2. The different part : 'family:1', 'morning:1', 'family:2', these selections will be more possible to be choosed by male. And 'morning:3', 'alone:1', 'family:3' these options will be prefered by female. 
# 
# Secondly we explore the more details from each condition:
# 1. 'alone' : we just extract the columns 'alone:1', 'alone:2' and 'alone:3'. According to the probability representation 'alone:1' - (0.13,0.17) ,  'alone:2' - (0.23,0.24) and 'alone:3' - (0.46,0.43), the most people like more enegetic ambiance when they are alone in the vehicle like commute.
# 2. 'family' : 'family:1' - (0.39,0.28), 'family:2' - (0.33,0.25), 'family:3' - (0.1, 0.29). For the male, they prefer an ambiance more peaceful. But for the female, it all depends on. 
# 3. 'morning' : 'morning:1' - (0.38,0.31), 'morning:2' - (0.19,0.21), 'morning:3' - (0.25,0.33). They prefer either to be quiet or to be energetic.
# 4. 'evening' : 'evening:1' - (0.29,0.34), 'evening:2' - (0.28,0.28), 'evening:3' - (0.26,0.24). It all depends on the other factors.
# 
# Then we get the information about how these two factors affect the style selection of audio. Generally, the difference between male and female does exist but it is not so grand. If we calculate the variance between male and famale, we will observe an ituitive indicator. Specifically, the situation with family is definitely different and the others are somehow consistent. 

# In[ ]:


plot_data_by_gender(data_gender_emission, 'Comparison of the emission preference by gender', 'emission')


# For first 10 most probable selections, these two tendencies are similar.

# **Conclusion for gender influence** :
# the ordering for whom getting influenced from the most effectively to the least is music, radio, ambiance, emission. 
#   

# ** Age influence**

# In[ ]:


sum_19to26 =  df_quiz.rdd.filter(lambda x:x[7]=='19-26').count()
sum_27to35 = df_quiz.rdd.filter(lambda x:x[7] == '27-35').count()
sum_36to50 = df_quiz.rdd.filter(lambda x:x[7] == '36-50').count()
sum_51to65 = df_quiz.rdd.filter(lambda x:x[7] == '51-65').count()
print(sum_19to26, sum_27to35, sum_36to50, sum_51to65)


# In[ ]:


data_age_radio, data_age_music, data_age_ambiance, data_age_emission = get_statistic_by(7)


# In[ ]:


def plot_data_by_age(dataset, gtitle, xtitle):
    data_19to26 = list(filter(lambda x: x[0][0]=='19-26', dataset))
    data_27to35 = list(filter(lambda x: x[0][0]=='27-35', dataset))
    data_36to50 = list(filter(lambda x: x[0][0]=='36-50', dataset))
    data_51to65 = list(filter(lambda x: x[0][0]=='51-65', dataset))
    
    d19 = sorted(data_19to26 ,key=lambda x:x[1], reverse=True)[:12]
    d27 = sorted(data_27to35 ,key=lambda x:x[1], reverse=True)[:12]
    d36 = sorted(data_36to50 ,key=lambda x:x[1], reverse=True)[:12]
    d51 = sorted(data_51to65 ,key=lambda x:x[1], reverse=True)[:12]
    
    def get_dict(l, somme):
        res = dict()
        for x in l:
            res.update({x[0][1]:x[1]/somme})
        return res
    
    dd19 = get_dict(d19, sum_19to26)
    dd27 = get_dict(d27, sum_27to35)
    dd36 = get_dict(d36, sum_36to50)
    dd51 = get_dict(d51, sum_51to65)
    
    keys = list(set(list(dd19.keys()) + list(dd27.keys()) + list(dd36.keys()) + list(dd51.keys())))
    
    def get_y(keys, dd):
        res = list()
        for k in keys:
            v = dd.get(k)
            if v:
                res.append(v)
            else:
                res.append(0)
        return res
    
    v19 = get_y(keys, dd19)
    v27 = get_y(keys, dd27)
    v36 = get_y(keys, dd36)
    v51 = get_y(keys, dd51)
    
    fig = go.Figure(
        data=[
            go.Bar(name='19-26', x=keys, y=v19),
            go.Bar(name="27-35", x=keys, y=v27),
            go.Bar(name="36-50", x=keys, y=v36),
            go.Bar(name="51-65", x=keys, y=v51),
            go.Scatter(name='19-26', x=keys, y=v19),
            go.Scatter(name="27-35", x=keys, y=v27),
            go.Scatter(name="36-50", x=keys, y=v36),
            go.Scatter(name="51-65", x=keys, y=v51),
        ], 
        layout=go.Layout(
            title=gtitle,
            xaxis=dict(
                title=xtitle,
                titlefont=dict(
                    family = 'Courier New, monospace',
                    size = 18,
                    color = '#7f7f7f'
                )
            )
        )
    )

    fig.update_layout(barmode='group')
    fig.show()
    return None

plot_data_by_age(data_age_radio, 'Comparison of the radio preference by age', 'Radio')


# What we've got is that each polyline has its own charactors :
# <br>19-26 : most people like NRJ and then Fun Radio.</br>
# <br>27-35 : most people like RTL2, the second is Virgin Radio.</br>
# <br>36-50 : the favorite radio is RTL2 and the second is France Inter.</br>
# <br>51-65 : the most popular is France Inter and France Info.</br>

# In[ ]:


plot_data_by_age(data_age_music, 'Comparison of the music preference by age', 'Music')


# It's obvious that each peak of polyline is at the same place for the music Rock and second is Pop.

# In[ ]:


daa = list(map(lambda x: ( ( (x[0][0], x[0][1][0]+':'+str(x[0][1][1]) ),x[1]) ), data_age_ambiance))
plot_data_by_age(daa, 'Comparison of the ambiance preference by age', 'Ambiance')


# These four polylines have the same peak 'alone:3'. That is to say, most people like more energetic when they are alone in vehicle.
# 
# And we analyse from the 4 conditions:
# 1. alone : 'alone:1' - (0.1,0.13,0.13,0.21), 'alone:2' - (0.27,0.22,0.26,0.21), 'alone:3' - (0.45,0.51,0.44,0.44). Most people like more energetic and active ambiance.
# 2. family : 'family:1' - (0.41,0.42,0.27,0.44), 'family:2' - (0.27,0.30,0.35,0.29), 'family:3' - (0.12,0.14,0.20,0.1). Except for the generation whose age is from 36 to 50, the generation from 19 to 65 years old has the same preference when they are with family. Among them, most like more quiet and peaceful atmosphere. 
# 3. morning : 'morning:1' - (0.17,0.29,0.39,0.48),'morning:2' - (0.30,0.22,0.21,0.14),'morning:3' - (0.32,0.35,0.23,0.23). Here, it's a demonstration of the influence of different generation. The 51-65 and 36-50 prefer to be quiet in the morning, The 19-26 have a tendency to be more dynamic in the morning. The 27-35 depend on other factors. 
# 4. evening : 'evening:1' - (0.13,0.28,0.30,0.40), 'evening:2' - (0.27,0.31,0.31,0.23), 'evening:3' - (0.4,0.28,0.21,0.22). Here, we can see a obvious graded distribution that is a typical and intuitive phenomenon. Older people like more quiet and younger people like more dynamic in the evening. 

# In[ ]:


plot_data_by_age(data_age_emission, 'Comparison of the emission preference by age', 'Emission')


# All peaks of all polylines are same, there is not any distinction from difference of age.
# 
# **Conclusion for age influence** :
# The ordering for whom getting affected from the most to the least : radio, ambiance, music, emission.

# **Leisure**

# In[ ]:


data_leisure_radio, data_leisure_music, data_leisure_ambiance, data_leisure_emission = get_statistic_by(8)


# In[ ]:


sum_s=  df_quiz.rdd.filter(lambda x:x[8]=='Sportif').count()
sum_t = df_quiz.rdd.filter(lambda x:x[8] == 'Touristique').count()
sum_c = df_quiz.rdd.filter(lambda x:x[8] == 'Culturel').count()
sum_i = df_quiz.rdd.filter(lambda x:x[8] == 'Informatique').count()
print(sum_s, sum_t, sum_c, sum_i)


# In[ ]:


def plot_data_by_leisure(dataset, gtitle, xtitle):
    data_s = list(filter(lambda x: x[0][0]=='Sportif', dataset))
    data_t = list(filter(lambda x: x[0][0]=='Touristique', dataset))
    data_c = list(filter(lambda x: x[0][0]=='Culturel', dataset))
    data_i = list(filter(lambda x: x[0][0]=='Informatique', dataset))
    
    ds = sorted(data_s ,key=lambda x:x[1], reverse=True)[:12]
    dt = sorted(data_t ,key=lambda x:x[1], reverse=True)[:12]
    dc = sorted(data_c ,key=lambda x:x[1], reverse=True)[:12]
    di = sorted(data_i ,key=lambda x:x[1], reverse=True)[:12]
    
    def get_dict(l, somme):
        res = dict()
        for x in l:
            res.update({x[0][1]:x[1]/somme})
        return res
    
    dds = get_dict(ds, sum_s)
    ddt = get_dict(dc, sum_c)
    ddc = get_dict(dt, sum_t)
    ddi = get_dict(di, sum_i)
    
    keys = list(set(list(dds.keys()) + list(ddt.keys()) + list(ddc.keys()) + list(ddi.keys())))
    
    def get_y(keys, dd):
        res = list()
        for k in keys:
            v = dd.get(k)
            if v:
                res.append(v)
            else:
                res.append(0)
        return res
    
    vs = get_y(keys, dds)
    vt = get_y(keys, ddt)
    vc = get_y(keys, ddc)
    vi = get_y(keys, ddi)
    
    fig = go.Figure(
        data=[
            go.Bar(name='Sportif', x=keys, y=vs),
            go.Bar(name="Touristique", x=keys, y=vt),
            go.Bar(name="Culturel", x=keys, y=vc),
            go.Bar(name="Informatique", x=keys, y=vi),
            go.Scatter(name='Sportif', x=keys, y=vs),
            go.Scatter(name="Touristique", x=keys, y=vt),
            go.Scatter(name="Culturel", x=keys, y=vc),
            go.Scatter(name="Informatique", x=keys, y=vi),
        ], 
        layout=go.Layout(
            title=gtitle,
            xaxis=dict(
                title=xtitle,
                titlefont=dict(
                    family = 'Courier New, monospace',
                    size = 18,
                    color = '#7f7f7f'
                )
            )
        )
    )

    fig.update_layout(barmode='group')
    fig.show()
    return None

plot_data_by_leisure(data_leisure_radio, 'Comparison of the radio preference by hobby', 'Radio')


# In[ ]:


plot_data_by_leisure(data_leisure_music, 'Comparison of the music preference by hobby', 'Music')


# In[ ]:


dla = list(map(lambda x: ( ( (x[0][0], x[0][1][0]+':'+str(x[0][1][1]) ),x[1]) ), data_leisure_ambiance))
plot_data_by_leisure(dla, 'Comparison of the ambiance preference by hobby', 'Ambiance')


# The most probable preferences are 'alone:3', 'family:1' and 'morning:1'. The most people like more energetic atmosphere when they are alone, and prefer to be peaceful with family and in the morning when they drive or maybe commute. 
# 
# Analyse from 4 aspects : 
# 1. alone : 'alone:1' - (0.13,0.25,0.14,0.18), 'alone:2' - (0.28,0.21,0.26,0.33), 'alone:3' - (0.52,0.47,0.51,0.46). Most people like more dynamic.
# 2. family : 'family:1' - (0.37,0.48,0.40,0.41), 'family:2' - (0.37,0.26,0.33,0.44), 'family:3' - (0.17,0.17,0.17,0.07). Most people prefer to be more quiet with family. 
# 3. morning : 'morning:1' - (0.40,0.46,0.37,0.44),'morning:2' - (0.26,0.18,0.19,0.26),'morning:3' - (0.27,0.28,0.35,0.28). Most people like more peaceful ambiance except for whose hobby is cultral. 
# 4. evening : 'evening:1' - (0.30,0.43,0.33,0.28), 'evening:2' - (0.35,0.20,0.30,0.0.41), 'evening:3' - (0.27,0.30,0.28,0.28). We can get the information like that people probablely like peaceful atmosphere whose hobby is touritic. And people whose hobby is informatic will prefer more or less dynamic. But others depend on other factors. 

# In[ ]:


plot_data_by_leisure(data_leisure_emission, 'Comparison of the emission preference by hobby', 'Emission')


# **Conclusion:**
# The ordering is radio, ambiance, music, emission.

# <h3>After qualitative analyse for these figures, 3 orderings about the affect degree : </h3>
# <br>Gender : music, radio, ambiance, emission</br>
# <br>Age :  radio, ambiance, music, emission</br>
# <br>Leisure : radio, ambiance, music, emission</br>

# <h5>Color distribution</h5>
# The question is that how the color affects the final choose. But the type of data in the column color is different from others. The column color get more information about the preference : 1 - the favorite, 2 - the second favorite etc. And the whole represent a character. 
# * Firstly, we need to look at the distribution of choose as the change of color. And then, the values in the columns Color become vectors. The objectif of the transformation is to enlage the difference between the favorite and the others. 

# In[ ]:


color_radio = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[11]))).filter(lambda x: len(x[0]) > 0)
color_music = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[12]))).filter(lambda x: len(x[0]) > 0)
color_ambiance = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[13]))).filter(lambda x: len(x[0]) > 0)
color_emission = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[14]))).filter(lambda x: len(x[0]) > 0)


# In[ ]:


lColor = ['red', 'blue', 'green', 'yellow', 'brown', 'black', 'white']
def color2vec(dColor):
    res = [0 for i in range(7)]
    for k, v in dColor.items():
        res[lColor.index(k)] = (8-int(v))*30/(int(v)+1)
    return np.array(res)

lRadio = list(set(color_radio.flatMap(lambda x: x[1]).collect()))
lMusic = list(set(color_music.flatMap(lambda x: x[1]).collect()))

data_color_radio = color_radio.map(lambda line: [(color2vec(line[0]), lRadio.index(e) ) for e in line[1] ]).flatMap(lambda l: (t for t in l))
data_color_music = color_music.map(lambda line: [(color2vec(line[0]), lMusic.index(e) ) for e in line[1] ]).flatMap(lambda l: (t for t in l))


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xy_vector_color_radio = pca.fit_transform(data_color_radio.map(lambda x:x[0]).collect())
xy_vector_color_music = pca.fit_transform(data_color_music.map(lambda x:x[0]).collect())

jColor = ['red', 'blue', 'green', 'yellow', 'brown', 'black', '#00ffae'] # replace white by #00ffae because withe is not visible
vector_color_radio_label = data_color_radio.map(lambda x: jColor[np.argmax(x[0])]).collect()
z_color_radio = data_color_radio.map(lambda x: x[1]).collect()
x_vector_color_radio = list(map(lambda x: x[0], xy_vector_color_radio))
y_vector_color_radio = list(map(lambda x: x[1], xy_vector_color_radio))

x_vector_color_music = list(map(lambda x: x[0], xy_vector_color_music))
y_vector_color_music = list(map(lambda x: x[1], xy_vector_color_music))
z_color_music = data_color_music.map(lambda x:x[1]).collect()
vector_color_music_label = data_color_music.map(lambda x: jColor[np.argmax(x[0])]).collect()


# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(rows=1,
                    cols=2, 
                    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                    subplot_titles=("Radio", "Music")
                   )
fig.add_trace(
    go.Scatter3d(
        x=x_vector_color_radio,
        y=y_vector_color_radio,
        z=z_color_radio,
        marker_color = vector_color_radio_label,
        opacity=0.8,
        mode='markers'
    ),
    row=1,
    col=1
)
fig.add_trace(
     go.Scatter3d(
        x=x_vector_color_music,
        y=y_vector_color_music,
        z=z_color_music,
        marker_color = vector_color_music_label,
         opacity=0.8,
         mode='markers'
    ),
    row=1, col=2
)
fig.update_layout(height=600, width=800, title_text="Comparison of different color preference")
fig.show()


# A vector on the plane x-y represents the color preference from the choose of sample (red, blue, green, yellow, brown, black, white). The z-axis represents the radio preference or music preference. In this graph, white is replaced by #00ffae. And the color of each point is the favorite of the user. As the result, we can easily identify the the green and blue parts. That is to say, the green group and the blue group are difficult to have somme subordinate favorite color. But from each one plane x-y with the change of z, we hardly get the one radio or one music is specially preferred by one color group. 

# In[ ]:




