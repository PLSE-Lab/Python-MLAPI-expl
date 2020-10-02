#!/usr/bin/env python
# coding: utf-8

# The aim of this kernal is to discover what is the active research area in this institute by using basic language processing to count the number of each token appearing and identify the most frquent type of cancer. The analysis then go on to explore the impact factor of each publication to see which area of research is more impactful.

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import numpy as np \nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom nltk.corpus import stopwords#to filter out stop words to scale down the data \nfrom sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer \nfrom sklearn.linear_model import LogisticRegression\n#the last two are used for NLP \nfrom IPython.core.interactiveshell import InteractiveShell\nInteractiveShell.ast_node_interactivity = "all"\n#this is for ipython to display all the results of cell\nimport os\nprint(os.listdir("../input"))\n# Any results you write to the current directory are saved as output.')


# In[ ]:


raw = pd.read_csv('../input/publications-from-roswell-park-cancer-institute-beginning-2006.csv')
raw.columns #show what informations are available
raw.head()


# In[ ]:


get_ipython().run_cell_magic('capture', '', "#see what each column contain\nfor feature in list(raw.columns):\n    print(feature + str(raw[feature].unique()) + '\\n')\n'''\nyear: looks fine, can be used\ntype: can be used, there is a 967-9753 can take out\njournal name: could be useful\ntitle: essential for my purpose\narthor: maybe can use\njournal volume: definitely can drop\nissue number: also useless\nrange: could be used to calculate the length of the paper\nISSN: useless\npeer reviewed: important?\nImpact: the most important prediction here\n'''")


# In[ ]:


#first stage cleaning: dropping the stuff definitely irrelevant to analysis
DF = raw.drop(['ISSN', 'Journal Issue Number', 'Journal Volume'], axis = 1)
DF = DF[DF['Publication Type']!='967-9753']
DF.head()
DF['Publication Type'].unique()#no more 967-963


# In[ ]:


DF['Year Published'].value_counts().plot(kind='bar');
#see the number of journals each year
#2018 doesn't really count but we still see a gradual decrease


# In[ ]:


get_ipython().run_cell_magic('capture', '', "#from here, if the analysis has to do with impact score, use DF_impact, unless use DF as there are more data\n#some are non-rated so replace the string: 'Not Rated' with nan\nDF['Impact Factor'] = DF['Impact Factor'].replace(to_replace = 'Not Rated',value=np.nan)\n#ignore the data without rating\nDF_impact = DF[DF['Impact Factor'].notnull()]\n#covert the score/year from string to a number\nDF_impact['Impact Factor'] = DF_impact['Impact Factor'].astype('float64')\nDF_impact['Year Published'] = DF_impact['Year Published'].astype('int64')\n#ignore years from 2015 since impact score shouldn't be calculated before three years after prublication\nDF_impact = DF_impact[DF_impact['Year Published'].apply(lambda x: x <=2015)]")


# In[ ]:


#check the most impactful publication 
DF_impact.sort_values(by=['Impact Factor'], ascending=False).head(10)


# In[ ]:


#trying to combine all the titles from all the articles
combined_long =''
#adding them
for _ in DF['Publication Title']:
    combined_long+=_
    #combined_long+=' '.join(list(set(_.split(' '))))
#filtered=' '.join([words for words in combined_long.split(" ") if words not in uninformation_words])

#build a bag of words for each title with NLP toolbox
#second one is more robust and will be used from now on
count_vectorizer = CountVectorizer(lowercase=True)
tfidf_vectorizer = TfidfVectorizer(lowercase=True)
#build bag of words with the vectorizer
bag_of_words = count_vectorizer.fit_transform([combined_long])
bag_of_words2 = tfidf_vectorizer.fit_transform([combined_long])
#get name of the feature i.e. the key words
feature_names = count_vectorizer.get_feature_names()
feature_names2 = tfidf_vectorizer.get_feature_names()
#the words we don't want
customed_uselesswords=set(['cancer','study','cell','cells','analysis','tumor','risk','phase','human','group',
                           'advanced','expression','thearpy','treatment','patients','non','based','survival'
                          'small','gene','trial','results','novel'])
uninformation_words = customed_uselesswords|set(stopwords.words('english')) 
uninformation_words=list(uninformation_words&set(feature_names))
#convert our results for the bag of words into a data frame for the normal method
BoW=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)
BoW= BoW.transpose()
BoW.columns = BoW.columns.astype(str)
BoW.columns = ['counts']
#filter the less important ones(less frequent)
BoW = BoW.drop(uninformation_words).loc[BoW['counts']>200].sort_values(by=['counts'], ascending=False)
#same thing for the more robust method
BoW2=pd.DataFrame(bag_of_words2.toarray(), columns = feature_names2)
BoW2= BoW2.transpose()
BoW2.columns = BoW2.columns.astype(str)
BoW2.columns = ['frequency']
BoW2 = BoW2.drop(uninformation_words).loc[BoW2['frequency']>0.015].sort_values(by=['frequency'], ascending=False)    
BagWords=BoW2

BoW.plot.bar();
BoW2.plot.bar();

#the two happened to be the same 


# In[ ]:


BagWords.head(20)
#we can pick out the main cancer research area to invertigate: breast, prostate, ovarian, carcinoma, lung, leukemia,lymphoma,myeloid


# In[ ]:


get_ipython().run_cell_magic('capture', '', '#next stage will be to investigate how the key words of cancer has raised popularity \ngrouped_df = DF_impact.groupby([\'Year Published\'])\n\nfor key, item in grouped_df:\n    print(grouped_df.get_group(key), "\\n\\n")')


# In[ ]:


#define a class contain the methods for the type of cancer we're focusing on

class cancer():
    #initiate the cancer type such that we have a data frame that doesn't contain the type of cancer we interesested
    def __init__(self,cancer_type):
        self.cancer_type=cancer_type
        self.df = DF[DF['Publication Title'].apply(lambda x: self.cancer_type in x)]
        self.df_i = DF_impact[DF_impact['Publication Title'].apply(lambda x: self.cancer_type in x)]
    #define the method for the verious things we might be interested from the data
    def summary(self,h,t):
        def counts_year(t):
            self.f_y = self.df.groupby('Year Published')['Publication Title'].count()
            self.f_y.plot.bar(title=self.cancer_type, ax=ax[t], rot =0);
        def impact_year(t):
            self.i_y = self.df_i.groupby('Year Published')['Impact Factor'].agg('mean')
            self.i_y.plot.bar(title=self.cancer_type, ax=ax[t], rot =0);
        def average_impact():
            return self.df_i['Impact Factor'].mean() 
        def correlation():
            self.f_y = self.df_i.groupby('Year Published')['Publication Title'].count()
            self.i_y = self.df_i.groupby('Year Published')['Impact Factor'].agg('mean')
            return self.f_y.corr(self.i_y)
        def box(t):
            self.df_i[self.df_i['Year Published'] == int(t)+2006].boxplot(column=['Impact Factor'], ax=ax[t])
        def box_year():
            self.df_i.groupby('Year Published')['Impact Factor'].agg('mean').plot(kind='line', ax = ax);
            self.df_i.boxplot(column = ['Impact Factor'], by = 'Year Published', ax=ax,  rot=0 );
           
        if h=='f':#the publication counts by year
            counts_year(t)
        elif h=='i':#the average impact factor by year
            impact_year(t)
        elif h=='a':#find the average impact score for each type of cancer
            return average_impact()
        elif h=='c':#find the correlation between publication number and impact factor
            return correlation()
        elif h=='b':#the distribution of impact factor score by year
            box(t)
        elif h=='b_y':#the distribution of impact factor score by year
            box_year()


# In[ ]:


#the most frequently appeared type of cancer in this order
all_cancer = ['breast', 'prostate', 'ovarian', 'carcinoma', 'lung', 'leukemia', 'lymphoma', 'myeloid']
#define a function that tells us the cancer and its associate information
def find_summary(Cancer, s, t):
    _ = cancer(Cancer)
    return _.summary(s,t)


# In[ ]:


#find the average impact score according to year by cancer
fig, ax = plt.subplots(8,1,figsize=(20,20))
plt.subplots_adjust(hspace=1)
for i in range(8):
    find_summary(all_cancer[i],'i',i)


# In[ ]:


#find the publication counts according to year by cancer
fig, ax = plt.subplots(8,1,figsize=(20,20))
plt.subplots_adjust(hspace=1)
for i in range(8):
    find_summary(all_cancer[i],'f',i)


# In[ ]:


#find the distribution of impact score with boxplot for breast cancer from 2006 to 2015(left to right)
fig, ax = plt.subplots(1,10,figsize=(20,5))
for i in range(10):
    find_summary('breast','b',i)


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(20,20))
find_summary('breast','b_y',2)


# In[ ]:




