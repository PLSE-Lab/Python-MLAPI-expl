#!/usr/bin/env python
# coding: utf-8

# About game sequels

# 
# 
#     import numpy as np
#     import pandas as pd;
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     plt.style.use('ggplot')
# 
# 
# # I have used python 2.7 to perform the operation
# 
# #Function to get sequels per platform
# 
#     def sequelset(Games):
#     a=Games.iloc[0][1]
#     b=[a]
#     c={a:1}
#     for i in range(1,Games.count()[0]):
#         if a in Games.iloc[i][2]:
#             if a not in b:
#                 b.append(a)
#                 c[a]=1
#             else:
#                 c[a] += 1
#             b.append(Games.iloc[i][3])
#         else:
#             a=Games.iloc[i][4]
# 
#     c = {key: value for key, value in c.items() if value > 1}
#     d = pd.DataFrame(c.items())
#     e = Games[Games['title'].isin(list(c.keys()))]
#     if len(d.columns)>1:
#         d.columns=['title','count']
#         e= pd.merge(e, d, on='title', sort=False)
#     if len(b)>1:
#         if b[0] not in b[1]:
#             del b[0]
#         return Games[Games['title'].isin(b)],e
#     return None
# 
# #read data and sort it by title so as to filter and find sequel
# 
#      df=pd.read_csv('ign.csv')
#      df=df.drop('url',1)
#      platform=df['platform'].unique().tolist()
#      df=df.sort_values(['title'])
# 
# 
# 
# #store as dictonary all the sequels per platform
# 
#     DataFrameDict = {elem : pd.DataFrame for elem in platform}
#     for key in DataFrameDict.keys():
#           DataFrameDict[key] = df[:][df.platform == key]
#           firstgames={elem : pd.DataFrame for elem in platform}
#           S={elem : pd.DataFrame for elem in platform}
#     for key in DataFrameDict.keys():
#           if sequelset(DataFrameDict[key]) is not None:
#                  Games,listgames =sequelset(DataFrameDict[key])
#                  Sortseq=Games.sort_values(['title','release_year','release_month'])
#                  S[key]= pd.DataFrame(Sortseq)
#                  firstgames[key]=listgames
#           else:
#                  del S[key]
#                  del firstgames[key]
# 
# 
#      result=pd.concat(list(S.values()))
#       total=result.groupby('platform')['platform'].count()
# 
# 
# 
# 
# #Plot to find the platform with most sequels
# 
#     total.plot(kind='bar')
# 
# 
# #Plot for platform with most sequel.Note that pc has a lot of sequels.Suprising to note Xbox 360 having more sequels than PS3
# 
# ![][1]
# 
# 
# 
# #Analyse by release_year with sequels
# 
#     year=result[['platform','release_year']].groupby(['platform','release_year']).size().unstack('platform')
#     year.plot(kind='bar',stacked=True,colormap='Paired')
#     plt.figure()
# 
# 
# # We find that PC has sequels throughout its history.No wonder it beats the other platform.Odd to note that Xbox had so many sequels in 2008,2009 and 2010.
# 
# ![enter image description here][2]
# 
# 
# [Source](https://postimg.org/image/v3jag9585/)
# 
# #find the avg score and make a boxplot to find variation
# 
#     avgscore=result[['platform','score']].fillna(0)
#     sns.boxplot(avgscore['score'],avgscore['platform'])
#     plt.figure()
# 
# 
# #It seems like most of the newer platforms seem to have a higher score compared to the older ones
# ![][3]
# 
# [Source](https://postimg.org/image/pdbfxtgsl/)
# 
# 
# 
# #check the sequel count.How many sequels do each game have
# 
#     result2=pd.concat(list(firstgames.values()))
#     platformcount=result2[['platform','count']]
#     platformcount=platformcount[platformcount['count']<10]
#     sns.countplot(hue='platform',x='count',data=platformcount)
#     plt.figure()
# 
# 
# 
# #Note that most sequels are just 2.That is most games have at the max two sequels.
# 
# ![][4]
# 
# 
# [Source](https://postimg.org/image/8ko1ekecn/)
# 
# #find the most popular genre to make sequels
# 
#     genre=result[['platform','genre','release_year']]
#     sns.countplot(y='genre',data=genre)
#     plt.show()
# 
# 
# #We see that most popular genre is action followed by shooter and then RPG
# 
# ![][5]
# 
# [Source](https://postimg.org/image/8ko1ekecn/)
# 
# 
# 
#   [1]: https://s16.postimg.org/jqnkt520l/figure_1.png
#   [2]: https://s10.postimg.org/vt22sm5rt/figure_2_1.png
#   [3]: https://s10.postimg.org/5ipebp1l5/figure_3_1.png
#   [4]: https://s15.postimg.org/jx0mwcn1n/figure_4_1.png
#   [5]: https://s14.postimg.org/99aih4gmp/figure_5_1.png

# #Source code
# 
# 
# 
#     import pandas as pd;
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     plt.style.use('ggplot')
#     import numpy as np
# 
# 
#     #Function to get sequels per platform
#     def sequelset(Games):
#            a=Games.iloc[0][2]
#            b=[a]
#            c={a:1}
#            for i in range(1,Games.count()[0]):
#                  if a in Games.iloc[i][2]:
#                     if a not in b:
#                        b.append(a)
#                        c[a]=1
#                     else:
#                        c[a] += 1
#                        b.append(Games.iloc[i][2])
#                   else:
#                        a=Games.iloc[i][2]
#           c = {key: value for key, value in c.items() if value > 1}
#           d = pd.DataFrame(c.items())
#           e = Games[Games['title'].isin(list(c.keys()))]
#            if len(d.columns)>1:
#                 d.columns=['title','count']
#                 e= pd.merge(e, d, on='title', sort=False)
#            if len(b)>1:
#              if b[0] not in b[1]:
#                 del b[0]
#                 return Games[Games['title'].isin(b)],e
#             return None
# 
# 
#      #read data and sort it by title so as to filter and find sequel
#      df=pd.read_csv('ign.csv')
#      df=df.drop('url',1)
#      platform=df['platform'].unique().tolist()
#      df=df.sort_values(['title'])
# 
#      #store as dictonary all the sequels per platform
#      DataFrameDict = {elem : pd.DataFrame for elem in platform}
#      for key in DataFrameDict.keys():
#      DataFrameDict[key] = df[:][df.platform == key]
#      firstgames={elem : pd.DataFrame for elem in platform}
#     S={elem : pd.DataFrame for elem in platform}
#     for key in DataFrameDict.keys():
#           if sequelset(DataFrameDict[key]) is not None:
#              Games,listgames =sequelset(DataFrameDict[key])
#              Sortseq=Games.sort_values(['title','release_year','release_month'])
#              S[key]= pd.DataFrame(Sortseq)
#              firstgames[key]=listgames
#         else:
#             del S[key]
#             del firstgames[key]
#      result=pd.concat(list(S.values()))
#      total=result.groupby('platform')['platform'].count()
# 
#     #Plot to find the platform with most sequels
#      total.plot(kind='bar')
#     #Analyse by release_year with sequels
#     year=result[['platform','release_year']].groupby(['platform','release_year']).size().unstack('platform')
#     year.plot(kind='bar',stacked=True,colormap='Paired')
#     plt.figure()
# 
#     #find the avg score and make a boxplot to find variation
#     avgscore=result[['platform','score']].fillna(0)
#     sns.boxplot(avgscore['score'],avgscore['platform'])
#     plt.figure()
# 
#     #check the sequel count.How many sequels do each game have
#     result2=pd.concat(list(firstgames.values()))
#     platformcount=result2[['platform','count']]
#     platformcount=platformcount[platformcount['count']<10]
#     sns.countplot(hue='platform',x='count',data=platformcount)
#     plt.figure()
# 
#     #find the most popular genre to make sequels
#     genre=result[['platform','genre','release_year']]
#     sns.countplot(y='genre',data=genre)
# 
#     plt.show()
