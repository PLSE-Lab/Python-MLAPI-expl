#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis for Financial Statements

# ![image.png](attachment:image.png)

# ref : 
# 1. https://github.com/ThilinaRajapakse/simpletransformers#minimal-start-for-multiclass-classification
# 2. https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
# 

# # Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud


# # Reading Data

# In[ ]:


df = pd.read_csv("../input/sentiment-analysis-for-financial-news/all-data.csv",encoding='ISO-8859-1')


# In[ ]:


df.head()


# In[ ]:


df = df.rename(columns={'neutral':'sentiment','According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'statement'})


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.drop_duplicates(subset=['statement'],keep='first',inplace=True)
df.info()


# In[ ]:


df.describe()


# # WordCloud

# In[ ]:


text = " ".join([x for x in df.statement])

wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


## for neutral

text = " ".join([x for x in df.statement[df.sentiment=='neutral']])

wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


## for positive

text = " ".join([x for x in df.statement[df.sentiment=='positive']])

wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


## for negative

text = " ".join([x for x in df.statement[df.sentiment=='negative']])

wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# # Count plot of sentiments

# In[ ]:


sns.countplot(df.sentiment)


# In[ ]:


df['sentiment'].value_counts()


# # Test and Train dataframes

# In[ ]:


train,eva = train_test_split(df,test_size = 0.2)


# # Building a model

# In[ ]:


get_ipython().system('pip install simpletransformers')


# In[ ]:



from simpletransformers.classification import ClassificationModel


# Create a TransformerModel
model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)


# In[ ]:


# 0,1,2 : positive,negative
def making_label(st):
    if(st=='positive'):
        return 0
    elif(st=='neutral'):
        return 2
    else:
        return 1
    
train['label'] = train['sentiment'].apply(making_label)
eva['label'] = eva['sentiment'].apply(making_label)
print(train.shape)


# In[ ]:


train_df = pd.DataFrame({
    'text': train['statement'][:1500].replace(r'\n', ' ', regex=True),
    'label': train['label'][:1500]
})

eval_df = pd.DataFrame({
    'text': eva['statement'][-400:].replace(r'\n', ' ', regex=True),
    'label': eva['label'][-400:]
})


# In[ ]:


model.train_model(train_df)


# In[ ]:


result, model_outputs, wrong_predictions = model.eval_model(eval_df)


# # Model Evaluation

# In[ ]:


result


# In[ ]:


model_outputs


# In[ ]:


lst = []
for arr in model_outputs:
    lst.append(np.argmax(arr))


# In[ ]:


true = eval_df['label'].tolist()
predicted = lst


# In[ ]:


import sklearn
mat = sklearn.metrics.confusion_matrix(true , predicted)
mat


# In[ ]:


df_cm = pd.DataFrame(mat, range(3), range(3))

sns.heatmap(df_cm, annot=True) 
plt.show()


# In[ ]:


sklearn.metrics.classification_report(true,predicted,target_names=['positive','neutral','negative'])


# In[ ]:


sklearn.metrics.accuracy_score(true,predicted)


# # Give your statement

# In[ ]:


def get_result(statement):
    result = model.predict([statement])
    pos = np.where(result[1][0] == np.amax(result[1][0]))
    pos = int(pos[0])
    sentiment_dict = {0:'positive',1:'negative',2:'neutral'}
    print(sentiment_dict[pos])
    return


# In[ ]:


## neutral statement
get_result("According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .")


# In[ ]:


## positive statement
get_result("According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales .")


# In[ ]:


## negative statement
get_result('Sales in Finland decreased by 2.0 % , and international sales decreased by 9.3 % in terms of euros , and by 15.1 % in terms of local currencies .')


# In[ ]:


statement = "Give your statement"
get_result(statement)


# In[ ]:




