#!/usr/bin/env python
# coding: utf-8

# 
# # **TASK - 3 Answers - NL Questions**
# 
# ![](https://sportslogohistory.com/wp-content/uploads/2018/09/georgia_tech_yellow_jackets_1991-pres-1.png)
# 
# **Executive Summary:** Unsupervised scientific literature understanding system that accepts natural language questions and returns specific answers from the CORD19 scientific paper corpus. The answers are wholly generated by the system from the publicatons cited below the answer.  There is also a link with the question pre-loaded to the CORD19 web-based corpus (QA) search.
# 
# **PROBLEM:** When a new virus is discovered and causes a pandemic, it is important for scientists to get information coming from all scientific sources that may help them combat the pandemic.  The challenege, however, is that the number of scientific papers created is large and the papers are published very rapidly, making it nearly impossible for scientists to digest and understand important data in this mass of data.
# 
# **SOLUTION:** Unsupervised scientific literature understanding system that accepts natural language quesitons and returns specific answers from the CORD19 scientific paper corpus.
# 
# **APPROACH:**
# - meta.csv and full text versions (if available) of the COVID-19 relevant documents were exported to a MySQL database 
# - the natural language questions for the task are contained in a list 
# - the natural language questions are stemmed and stop words removed for passing to the database query
# - the query is run on the MySQL datatbase using NLP mode in MySQL through a custom API
# - The full text results are parsed into sentences and then scored and ordered for relevance
# - raw_score = total count of the keywords in the sentence
# - final_score = (raw_score/len(sentence))*raw_score - if all terms are in a sentence it recieves +1
# - a csv file is returned to the notebook as a pandas dataframe
# - if the question calls for a specific answer, the quesiton_answer module is sent the first 5 sentences and the original question - shout out to Dave Mezzetti for suggesting this QA impelmentation from huggingface.
# - https://www.kaggle.com/c/tensorflow2-question-answering/discussion/123434
# - if the question calls for a summary, the first 50 sentences are sent to the prepare summary answer.
# - https://pypi.org/project/bert-extractive-summarizer/
# - The question,answers and table of relevant scientific papers are returned in HTML format.
# 
# **Pros:** The system provides very responsive and seemingly accurate answers to specific questions. 
# 
# **Cons:** The system currently indicates study design (right column above author) from a keyword density system that is a work in progress - it also references a list of study design by Dave Mezzetti - it compares these lists and if they match shows one study design number and if not, shows both. It is a work in progress
# 
# In addition, the system is currently being updated to mine the full text for a number followed by the word cases e.g. 2015 cases. If suchs a pattern is found, it shows the "number of cases" under the title link. It is a work in progress to find outcomes in patients etc.
# 
# Study Desing Codes
# 
# - 1 - Systematic Review
# - 2 - Experimental Study (Randomized)
# - 3 - Experimental Study (Non-Randomized)
# - 4 - Ecological Regression
# - 5 - Prospective Cohort
# - 6 - Time Series Analysis
# - 7 - Retrospective Cohort
# - 8 - Cross Sectional
# - 9 - Case Control
# - 10 - Case Study
# - 11 - Simulation
# - 0 - Unknown Design (Default for no match)

# In[ ]:


###################### LOAD PACKAGES ##########################
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from IPython.core.display import display, HTML
import torch
get_ipython().system('pip install -q transformers --upgrade')
from transformers import *
get_ipython().system('pip install bert-extractive-summarizer')
from summarizer import Summarizer
#from transformers import pipeline
import pandas as pd
modelqa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = Summarizer()
#https://colab.research.google.com/drive/1rN0CS0hoxeByoPZu6_zF-AFJijYLsPw3


# In[ ]:


def remove_stopwords(query,stopwords):
    qstr=''
    qstr=qstr.join(query)
    #remove punctuaiton
    qstr = "".join(c for c in qstr if c not in ('!','.',',','?','(',')','-'))
    text_tokens = word_tokenize(qstr)
    #remove stopwords
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    #stem words
    #tokens_without_sw=stem_words(tokens_without_sw)
    str1=''
    #str1=' '.join(word for word in tokens_without_sw)
    str1=' '.join(word for word in text_tokens)
    return str1

def stem_words(words):
    stemmer = PorterStemmer()
    singles=[]
    for w in words:
        singles.append(stemmer.stem(w))
    return singles

# query MySQL database returns most relevant paper sentences in dataframe
def get_search_table(query,keyword):
    query=query.replace(" ","+")
    urls=r"https://edocdiscovery.com/covid_19/covid_19_search_api_v2.php?search_string="+query+'&keyword='+keyword
    table = pd.read_csv(urls,encoding= 'unicode_escape')
    return table

# BERT pretrained question answering module
def answer_question(question,text,model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
    input_ids = tokenizer.encode(input_text)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    # show qeustion and text
    #tokenizer.decode(input_ids)
    return answer

def prepare_summary_answer(text,model):
    #model = pipeline(task="summarization")
    return model(text)


###################### MAIN PROGRAM ###########################

### NL questions
#'''
questions = [
'What do we know about the genome and  evolution of the virus?',
'Where is information or data about the genome shared?',
'Are diverse genome sample sets available?',
'How many COV 2 strains are circulating?',
'What agreement exist to share data or information?',
'What do we know about COV 2 and livestock?',
'What do we know about animal reservoirs?',
'What do we know about framers being infected?',
'What do we know about human wildlife interface or interaction and the infected?',
'What do we know about experiments for the host range?',
'What do we know about wild animal as reservoir or hosts and spillover?',
'What are the socioeconomic and behavioral risks for this spillover infection?',
'What are some sustainable risk reduction strategies to avoid spillover from animals to humans?'
]
#'''
### focus quesiton with single keyword
keyword=['genome','genome data','genome','strain','data sharing','livestock','reservoir','farmer','wildlife','host range','hosts','spillover','animal','spillover']
# use QA or summarize for the realted NL question?
a_type=['sum','qa','qa','qa','sum','sum','sum','sum','sum','sum','sum','sum','sum']

#test one question
#questions=['Can livestock such as cows, horses and sheep be infected?']
#keyword=['livestock']
#test one question type of answer
#a_type=['sum']

q=0

# loop through the list of questions
for question in questions:

    #remove punctuation, stop words and stem words from NL question
    search_words=remove_stopwords(question,stopwords)
    
    #clean up bad stems that do not render search results
    bad_stems=['phenotyp','deadli','contagi','recoveri','rout','viru', 'surfac','immun','respons','person','protect','includ','smoke','diabet']
    replace_with=['phenotype','dead','contagious','recovery','route','virus','surface','immune','response','personal','protective','include','smoking','diabetes']
    r=0
    for words in bad_stems:
        search_words=search_words.replace(words,replace_with[r])
        r=r+1
    # use to see stemmed query for troubleshooting
    #print (search_words)

    # get best sentences
    df_table=get_search_table(search_words,keyword[q])
    df_answers=df_table
    
    # if qa limit dataframe search rows to consider
    if a_type[q]=='qa':
        df_answers=df_table.head(5)
    
    # if sum expand dataframe search rows to consider
    if a_type[q]=='sum':
        df_answers=df_table.head(100)
    
    text=''
    
    for index, row in df_answers.iterrows():
        text=text+' '+row['excerpt']
        
    display(HTML('<h1>'+question+'</h1>'))
    
    #if qa use the question answering function
    if a_type[q]=='qa':
        answer=answer_question(question,text,modelqa)
        answer=answer.replace("#", "")
        answer=answer.replace(" . ", ".")
        display(HTML('<h4> Answer:</h4> '+ answer))
         
    #if sum use the summarizer function
    if a_type[q]=='sum':
        summary_answer=prepare_summary_answer(text,model)
        #summary_answer=summary_answer[0]['summary_text']
        display(HTML('<h4> Summarized Answer: </h4><i>'+summary_answer+'</i>'))
    
    #print (text)
    
    #limit the size of the df for the html table
    df_table=df_table.head(5)
    
    #convert df to html
    df_table=HTML(df_table.to_html(escape=False,index=False))
    
    display(HTML('<h5>results limited to 5 for ease of scanning</h5>'))
    # show the HTML table with responses
    display(df_table)
    
    # link to web based CORD search preloaded
    sstr=search_words.replace(" ","+")
    cord_link='<a href="http://edocdiscovery.com/covid_19/xscript_serp.php?search_string='+sstr+'">see more web-based results</a>'

    display(HTML(cord_link))
    
    q=q+1
    
print ('done')
