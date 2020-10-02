#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


DATA_PATH = "/kaggle/input/jigsaw-comment-classification-notebook-one/"
os.listdir(DATA_PATH)


# In[ ]:


TRAIN_PATH = DATA_PATH + "notebook_one_train_data.csv"
VAL_PATH = DATA_PATH + "notebook_one_val_data.csv"
TEST_PATH = DATA_PATH + "notebook_one_test_data.csv"

train_data = pd.read_csv(TRAIN_PATH)
val_data = pd.read_csv(VAL_PATH)
test_data = pd.read_csv(TEST_PATH)


# In[ ]:


train_data.head()


# In[ ]:


val_data.head()


# In[ ]:


test_data.head()


# # Translation 
# 
# In the [first notebook of this notebook series](https://www.kaggle.com/sleebapaul/jigsaw-comment-classification-notebook-one), I've done some deductions based on the EDA. 
# 
# Main findings are, 
# 
# 1. The training data is hightly skewed. 90% of the data is non-toxic comments.
# 2. The non-english comments are only 1% of training data
# 3. In contrast, in testing and validation data, all the comments are non-english. 
# 4. A viable solution is to translate these sentences to English and train the model. 
# 5. On testing, first translate the sentence to English and do the prediction. 
# 6. In support of this decision, most of the NLP packages available are trained and tuned in English language data.
# 7. Thus processing the non-english sentences with these packages and arriving on a conclusion is meanless. 
# 
# 
# For detailed information, check out the [first notebook](https://www.kaggle.com/sleebapaul/jigsaw-comment-classification-notebook-one) mentioned above. 
# 
# 
# 
# # Translated Validation and Test Data
# 
# Translation of the validation and test data are already by [Yury Kashnitsky](https://www.kaggle.com/kashnitsky). The data is reused for further analysis. 

# In[ ]:


DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-test-translated/"
os.listdir(DATA_PATH)


# In[ ]:


TRANSLATED_VAL_PATH = DATA_PATH + "jigsaw_miltilingual_valid_translated.csv"
TRANSLATED_TEST_PATH = DATA_PATH + "jigsaw_miltilingual_test_translated.csv"

trans_val_data = pd.read_csv(TRANSLATED_VAL_PATH)
trans_test_data = pd.read_csv(TRANSLATED_TEST_PATH)


# In[ ]:


trans_val_data.head()


# In[ ]:


trans_test_data.head()


# In[ ]:


val_data["translated_comment"] = trans_val_data.translated
test_data["translated_comment"] = trans_test_data.translated


# # About training data
# 
# Now we have the validation and testing data translations, now let's dig deep into the non-english comments in training data. 

# In[ ]:


train_non_eng_sentences = train_data[(train_data.lang_code != 'en')]


# In[ ]:


print("Total number of comments which are Non-English: ",
      train_non_eng_sentences.shape[0])

print("Total number of languages other than English: ",
      len(train_non_eng_sentences.lang_code.unique()))

print("Average number of comments in Non-English Languages: ",
      train_non_eng_sentences.lang_code.value_counts().mean())


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), )
prob = train_non_eng_sentences.lang_code.value_counts(normalize = True)
threshold = .01
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
ax = sns.barplot(x=prob.index, y=prob.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
ax.set_xlabel("Languages")
ax.set_ylabel("Comment count")
ax.set_title('Non English language comment count in training data (>1% of total)', fontsize=14)
fig.show()


# ## Findings
# 
# 1. The graph plot the non-english languages which has a count greater than atleadt 1% of total count.
# 2. From 142 non english languages, 23 languages are eligible for that count. 
# 3. That means, most of the languages present are feeble in number (`other ~ 37%`). 
# 4. `un` category consists of those comments which couldn't be detected by `polyglot` package. 
# 5. More than 22% of all non-english comments comes under `un`. This category shoud be looked closely. 
# 6. Can we remove the rest of the languages other than English?
# 
#     - It can found out from the amount at which they are contributing to `toxic` comments.
#     - Because, it is known that the dataset is heavily skewed with 99% of `non-toxic` comments.
#     - So we can't lose the data, if these comments comes under `toxic` category. 

# In[ ]:


train_non_eng_toxic = train_data[(train_data.lang_code != 'en') & (train_data.label == 1)]


# In[ ]:


print("Total toxic comments available in the training dataset: ", train_data.label.value_counts()[1])
print("Total toxic comments that are non english: ", train_non_eng_toxic.shape[0])
print("Percetage contribution: {:.2f} %".format(100*(
    train_non_eng_toxic.shape[0]/train_data.label.value_counts()[1])))


# ## Findings 
# 
# 1. 2% of non english comments are `toxic`. 
# 2. Let's save this data and chuck the rest, since we've abundant data for `non-toxic` comments already. 
# 3. Translate these 446 comments to English and drop the rest.

# In[ ]:


get_ipython().system(' pip install googletrans')


# In[ ]:


from googletrans import Translator

translator = Translator()

def translate(sentence):
    return translator.translate(sentence).text


# ## Findings
# 
# 1. This is a very interesting finding that the examples with undetected language (`un`) are not really foreign.
# 2. Almost all of them are English language with punctuation issues, HTML, repeated text etc. 
# 3. The good news is, they are need not to be removed from the dataset. 

# In[ ]:


translated_train_sents = []

train_non_eng_toxic_comment_list = train_non_eng_toxic.comment_text.to_list()
print("Total length: ", len(train_non_eng_toxic_comment_list))

def clean_text_for_translate(comment):
    if type(comment) == str:
        x = "".join(x for x in comment if x.isprintable())        
        return x.replace("\n", " ")
    else:
        return ""

count = 0 

while count < len(train_non_eng_toxic_comment_list):
    sent = clean_text_for_translate(train_non_eng_toxic_comment_list[count])
    translated_train_sents.append(translate(sent))
    count += 1


# In[ ]:


train_data.loc[(train_data.lang_code != 'en') & (train_data.label == 1), 'comment_text']= translated_train_sents
train_data.loc[(train_data.lang_code != 'en') & (train_data.label == 1), 'lang_code'] = 'en'
train_data.loc[(train_data.lang_code != 'en') & (train_data.label == 1), 'lang_name'] = 'English'


# In[ ]:


train_data.drop(train_data[(train_data['lang_code'] != 'en') & (train_data['label'] == 0)].index, inplace = True)


# In[ ]:


train_data.shape


# # Cleaning the comments

# In[ ]:


get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install contractions')


# In[ ]:


from bs4 import BeautifulSoup
import re
import contractions
import unicodedata


def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    html_free = soup.get_text()
    return html_free

def remove_url(text):
    return re.sub(r'http\S', ' ', text)

def remove_digits_spec_chars(text):
    return re.sub(r'[^a-zA-Z]', " ", text)

def to_lower_case(text):
    return text.lower()

def remove_extra_spaces(text):
    return re.sub(r'\s\s+', " ", text)

def remove_next_line(text):
    return re.sub(r'[\n\r]+', " ", text)
    
def remove_non_ascii(comment):
    """
    Remove non-ASCII characters from list of tokenized words
    """
    ascii_string = unicodedata.normalize('NFKD', comment).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return ascii_string
    
def remove_between_square_brackets(comment):
    result = re.sub('\[[^]]*\]', '', comment)
    return result

def replace_contractions(comment):
    """
    Replace contractions in string of text
    """
    contraction_free = contractions.fix(comment)
    return contraction_free

def clean_comment(comment):
    comment = remove_non_ascii(comment)
    comment = remove_next_line(comment)
    comment = replace_contractions(comment)
    comment = remove_url(comment)
    comment = remove_html(comment)
    comment = remove_between_square_brackets(comment)
    comment = remove_digits_spec_chars(comment)
    comment = remove_extra_spaces(comment)
    comment = to_lower_case(comment)
    return comment.strip()


# In[ ]:


train_data["cleaned_text"] = train_data["comment_text"].apply(lambda row: clean_comment(row))
val_data["cleaned_text"] = val_data["translated_comment"].apply(lambda row: clean_comment(row))
test_data["cleaned_text"] = test_data["translated_comment"].apply(lambda row: clean_comment(row))


# # Add sentiment data 

# In[ ]:


get_ipython().system('pip install --upgrade vaderSentiment')


# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_pos_polarity(comment):
    return analyzer.polarity_scores(comment)

def get_comment_length(comment):
    try:
        return len(comment.split())
    except:
        return 0


# In[ ]:


sentiment = train_data["cleaned_text"].apply(lambda row: get_pos_polarity(row))
train_data = pd.concat([train_data,sentiment.apply(pd.Series)],1)
train_data.columns = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate', 'lang_code', 'lang_name', 'country', 'label', 'cleaned_text',
                      'neg_pol','neutral_pol', 'pos_pol', 'compound_pol']


# In[ ]:


sentiment = val_data["cleaned_text"].apply(lambda row: get_pos_polarity(row))
val_data = pd.concat([val_data,sentiment.apply(pd.Series)],1)
val_data.columns = ['id', 'comment_text', 'lang', 'toxic', 'lang_name', 'country',
                    'translated_comment', 'cleaned_text','neg_pol', 
                    'neutral_pol', 'pos_pol', 'compound_pol']


# In[ ]:


sentiment = test_data["cleaned_text"].apply(lambda row: get_pos_polarity(row))
test_data = pd.concat([test_data,sentiment.apply(pd.Series)],1)
test_data.columns = ['id', 'content', 'lang', 'lang_name', 'country', 'translated_comment',
                    'cleaned_text','neg_pol', 'neutral_pol', 'pos_pol', 'compound_pol']


# # Add comment length to data

# In[ ]:


train_data["comment_len"] = train_data["cleaned_text"].apply(lambda row: get_comment_length(row))
val_data["comment_len"] = val_data["cleaned_text"].apply(lambda row: get_comment_length(row))
test_data["comment_len"] = test_data["cleaned_text"].apply(lambda row: get_comment_length(row))


# # Comment length analysis

# In[ ]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), )
ax = sns.kdeplot(train_data.comment_len, shade=True, label = "Training")
ax = sns.kdeplot(val_data.comment_len, shade=True, label = "Validation")
ax = sns.kdeplot(test_data.comment_len, shade=True, label = "Testing")
ax.set_title('Density distribution of comment length over different datasets')
ax.set_xlabel("Comment length")
ax.set_ylabel("Density")
f.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
sns.kdeplot(train_data[train_data.label ==1].comment_len, shade=True, label = "Toxic Comment", ax=ax1)
sns.kdeplot(train_data[train_data.label ==0].comment_len, shade=True, label = "Non-Toxic Comment", ax=ax1)
ax1.set_title('Training data')
ax1.set_xlabel("Comment length")
ax1.set_ylabel("Density")

sns.kdeplot(val_data[val_data.toxic ==1].comment_len, shade=True, label = "Toxic Comment", ax=ax2)
sns.kdeplot(val_data[val_data.toxic ==0].comment_len, shade=True, label = "Non-Toxic Comment", ax=ax2)
ax2.set_title('Validation data')
ax2.set_xlabel("Comment length")
ax2.set_ylabel("Density")


f.suptitle("Density distribution of comment length over labels", fontsize = 22)
f.tight_layout()
f.subplots_adjust(top=0.8)
f.show()


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

# Training data 
var = 'comment_len'
tmp = pd.concat([train_data['label'], train_data[var]], axis=1)
sns.boxplot(x='label', y=var, data=tmp, fliersize=5, ax=ax1)
ax1.set_title('Training data')
ax1.set_xlabel("Labels")
ax1.set_ylabel("Comment Length")
ax1.set_xticklabels(["Non-Toxic", "Toxic"])

tmp = pd.concat([val_data['toxic'], val_data[var]], axis=1)
sns.boxplot(x='toxic', y=var, data=tmp, fliersize=5, ax=ax2)
ax2.set_title('Validation data')
ax2.set_xlabel("Labels")
ax2.set_ylabel("Comment Length")
ax2.set_xticklabels(["Non-Toxic", "Toxic"])

fig = sns.boxplot(y=var, data=test_data, fliersize=5, ax=ax3)
ax3.set_title('Test data')
ax3.set_ylabel("Comment Length")

f.suptitle("Comment count across datasets", fontsize = 22)
f.tight_layout()
f.subplots_adjust(top=0.8)
f.show()


# In[ ]:


Q1 = train_data.comment_len.quantile(0.25)
Q3 = train_data.comment_len.quantile(0.75)
IQR = Q3 - Q1

print("Total comment length outliers in training data: ", 
      ((train_data.comment_len < (Q1 - 1.5 * IQR)) | (train_data.comment_len > (Q3 + 1.5 * IQR))).sum())

print("Total comment length outliers in toxic comments of training data: ", 
      ((train_data.loc[train_data.label == 1]['comment_len'] < (Q1 - 1.5 * IQR))|(
          train_data.loc[train_data.label == 1]['comment_len'] > (Q3 + 1.5 * IQR))).sum())

print("Total comment length outliers in non-toxic comments of training data: ", 
      ((train_data.loc[train_data.label == 0]['comment_len'] < (Q1 - 1.5 * IQR))|(
          train_data.loc[train_data.label == 0]['comment_len'] > (Q3 + 1.5 * IQR))).sum())


print("-"*80)

Q1 = val_data.comment_len.quantile(0.25)
Q3 = val_data.comment_len.quantile(0.75)
IQR = Q3 - Q1

print("Total comment length outliers in validation data: ", 
      ((val_data.comment_len < (Q1 - 1.5 * IQR)) | (val_data.comment_len > (Q3 + 1.5 * IQR))).sum())

print("Total comment length outliers in toxic comments of validation data: ", 
      ((val_data.loc[val_data.toxic == 1]['comment_len'] < (Q1 - 1.5 * IQR))|(
          val_data.loc[val_data.toxic == 1]['comment_len'] > (Q3 + 1.5 * IQR))).sum())

print("Total comment length outliers in non-toxic comments of validation data: ", 
      ((val_data.loc[val_data.toxic == 0]['comment_len'] < (Q1 - 1.5 * IQR))|(
          val_data.loc[val_data.toxic == 0]['comment_len'] > (Q3 + 1.5 * IQR))).sum())


print("-"*80)

Q1 = test_data.comment_len.quantile(0.25)
Q3 = test_data.comment_len.quantile(0.75)
IQR = Q3 - Q1

print("Total comment length outliers in testing data: ", 
      ((test_data.comment_len < (Q1 - 1.5 * IQR)) | (test_data.comment_len > (Q3 + 1.5 * IQR))).sum())


# ## Findings
# 
# 1. Density distribution of comment length over datasets are almost similar. No significance change is observed. 
# 2. Having said that, training data is more right skewed than any other data. 
# 3. Density distributions of comment lengths over toxic/non-toxic comments, are also similar. Comment length is not significatly seperating the two labels. 
# 4. Coming to outliers,
#     - Validation set comment length ranges upto 350 and in test data it is 800.
#     - But in training data, this range is upto 2000+. 
#     - There are comments in training data which are significantly larger comparing to other datasets.
#     - Removing outliers might reduce the number of toxic comments as well, thus keeping them in the dataset. 

# # Sentiments 
# 
# 1. There are four components recorded for a sentence in the dataset
# 2. Positive, Negative, Neutral and compound. 
# 
# 
# ## Negative sentiment

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
sns.kdeplot(train_data[train_data.label ==1].neg_pol, shade=True, label = "Toxic Comment", ax=ax1)
sns.kdeplot(train_data[train_data.label ==0].neg_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)
ax1.set_title('Training data')
ax1.set_xlabel("Negative Polarity")
ax1.set_ylabel("Density")

sns.kdeplot(val_data[val_data.toxic ==1].neg_pol, shade=True, label = "Toxic Comment", ax=ax2)
sns.kdeplot(val_data[val_data.toxic ==0].neg_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)
ax2.set_title('Validation data')
ax2.set_xlabel("Negative Polarity")
ax2.set_ylabel("Density")


f.suptitle("Density distribution of Negative Polarity over labels", fontsize = 22)
f.tight_layout()
f.subplots_adjust(top=0.8)
f.show()


# # Positive sentiments

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
sns.kdeplot(train_data[train_data.label ==1].pos_pol, shade=True, label = "Toxic Comment", ax=ax1)
sns.kdeplot(train_data[train_data.label ==0].pos_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)
ax1.set_title('Training data')
ax1.set_xlabel("Positive Polarity")
ax1.set_ylabel("Density")

sns.kdeplot(val_data[val_data.toxic ==1].pos_pol, shade=True, label = "Toxic Comment", ax=ax2)
sns.kdeplot(val_data[val_data.toxic ==0].pos_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)
ax2.set_title('Validation data')
ax2.set_xlabel("Positive Polarity")
ax2.set_ylabel("Density")


f.suptitle("Density distribution of Positive Polarity over labels", fontsize = 22)
f.tight_layout()
f.subplots_adjust(top=0.8)
f.show()


# # Neutral polarity

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
sns.kdeplot(train_data[train_data.label ==1].neutral_pol, shade=True, label = "Toxic Comment", ax=ax1)
sns.kdeplot(train_data[train_data.label ==0].neutral_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)
ax1.set_title('Training data')
ax1.set_xlabel("Neutral Polarity")
ax1.set_ylabel("Density")

sns.kdeplot(val_data[val_data.toxic ==1].neutral_pol, shade=True, label = "Toxic Comment", ax=ax2)
sns.kdeplot(val_data[val_data.toxic ==0].neutral_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)
ax2.set_title('Validation data')
ax2.set_xlabel("Neutral Polarity")
ax2.set_ylabel("Density")


f.suptitle("Density distribution of Neutral Polarity over labels", fontsize = 22)
f.tight_layout()
f.subplots_adjust(top=0.8)
f.show()


# # Compound polarity

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
sns.kdeplot(train_data[train_data.label ==1].compound_pol, shade=True, label = "Toxic Comment", ax=ax1)
sns.kdeplot(train_data[train_data.label ==0].compound_pol, shade=True, label = "Non-Toxic Comment", ax=ax1)
ax1.set_title('Training data')
ax1.set_xlabel("Compound Polarity")
ax1.set_ylabel("Density")

sns.kdeplot(val_data[val_data.toxic ==1].compound_pol, shade=True, label = "Toxic Comment", ax=ax2)
sns.kdeplot(val_data[val_data.toxic ==0].compound_pol, shade=True, label = "Non-Toxic Comment", ax=ax2)
ax2.set_title('Validation data')
ax2.set_xlabel("Compound Polarity")
ax2.set_ylabel("Density")


f.suptitle("Density distribution of Compound Polarity over labels", fontsize = 22)
f.tight_layout()
f.subplots_adjust(top=0.8)
f.show()


# ## Findings
# 
# 1. As expected, negative polarity is significantly seperating the comments to toxic and non-toxic. 
# 2. For non-toxic comments, negative polarity is near to zero. 
# 3. Surprisingly, positive polarity is not adding much value to seperate the comments. The distribution is almost identical for both groups. 
# 4. This can be reasoned using the following plots.
#     - Both types of comments are giving small values (0.0 ~ 0.2) for positive polarities. 
#     - Both types of comments are giving high neutral polarity sentiment score. Comparitively, non-toxic comments 
#     get high neutral values. 
#     - Thus, the sentiment analyser is classifying most of the comments in the data as neutral comments.
#     - This can be confirmed in next plot of compound polarity.
#     - Signficant amount of data aligns between `-0.5 to 0.5` which is by definition, neutral. 
#     
# 5. Negative and Compound sentiments are contributing features to classify the comments. 

# # Saving the notebooks

# In[ ]:


train_data.to_csv("notebook_two_train_data.csv", index = False)
val_data.to_csv("notebook_two_val_data.csv", index = False)
test_data.to_csv("notebook_two_test_data.csv", index = False)

