#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def update_word_count_presence_in_text(fileId,inputText):
    global key2DocumentCount,keyColumnNames
    for keyName in keyColumnNames :
        count = inputText.count(keyName)
        if (count > 0): 
            existingValue = key2DocumentCount.get_value(fileId,keyName)
            updatedValue = existingValue + count 
            key2DocumentCount.set_value(fileId,keyName,updatedValue) 


# In[ ]:


def fetch_eligible_document(inputKey): # Return a set of matching documents 
    global key2DocumentCount
    columnList = key2DocumentCount[inputKey].to_list()
    nonZeroIndexes = pd.Series(columnList).nonzero()
    dataList = nonZeroIndexes[0] 
    dataList = dataList + 1
    finalSetOfDocuments = set(dataList)
    return finalSetOfDocuments


# In[ ]:


def get_shortlisted_textId_using_sentence_encoder(questionId,individualDocumentId,contextSentencesListPerQuestion) : # returns list of text ids
    global fileList,embed
    
    textcount = 0
    finalBatchOfSentence = []
    textSet = {}
    finalShortListedTextIds = []
    
    #Open the specific document file
    filePath = fileList[individualDocumentId][1]
    with open(filePath,'r') as f:
        fileContent = json.load(f)
    body_text = fileContent['body_text']
    for textData in body_text :
        actualText = textData['text'].lower()
        sentencesInText = actualText.split('.')
        finalBatchOfSentence =  contextSentencesListPerQuestion + sentencesInText
        sentenceEmbedding = embed(finalBatchOfSentence)
        decision = should_batch_be_considered(sentenceEmbedding,len(contextSentencesListPerQuestion))
        if (decision) :
            finalShortListedTextIds.append(textcount)
        textcount = textcount + 1 
    return finalShortListedTextIds


# In[ ]:


def should_batch_be_considered(sentenceEmbedding , countOfRefSentences) :
    match_batching = False
    corelation = np.inner(sentenceEmbedding,sentenceEmbedding)
    related_entries = np.array(corelation[countOfRefSentences:,:countOfRefSentences])
    grade_A_match = len(np.where(related_entries[:] >= 0.4)[0])
    if (grade_A_match > 1) :
        match_batching = True
    return match_batching    


# In[ ]:


def prepare_result():
    global finalResultPerOpenQuestions,referenceContextSentences,key2OpenQuestions
    updatedFinalListOfData = []
    shortlistedDocumentPerQuestion = {}
    #Step 1 - Prepare the list of probable Documents for each of the Question Ids
    for questionId in key2OpenQuestions.keys():
        keysListPerQuestionId =  key2OpenQuestions[questionId]
        for key in keysListPerQuestionId:
            #Derive the documentId against the keys 
            shortlistedDocumentSet = {} 
            shortlistedDocumentSet =  fetch_eligible_document(key)
            updatedDocumentSet = {}
            if (questionId in shortlistedDocumentPerQuestion.keys() ):
                updatedDocumentSet = shortlistedDocumentPerQuestion[questionId]
                updatedDocumentSet.update(shortlistedDocumentSet)
            else :
                updatedDocumentSet = shortlistedDocumentSet
            shortlistedDocumentPerQuestion[questionId] = updatedDocumentSet
    #Step 2 - For each of the shorlisted document score the body_text against the reference sentences and prepare the finallist
    for questionId in  shortlistedDocumentPerQuestion.keys() :
        contextSentencesListPerQuestion = referenceContextSentences[questionId]
        documentcnt = 0
        print('Step2 Working on Question Id {}'.format(questionId))
        documentSet = shortlistedDocumentPerQuestion[questionId]
        totalNumberOfDocument = len(documentSet)
        for individualDocumentId in documentSet :
            documentcnt = documentcnt + 1 
            print('Step2 - Processing document {}/{}'.format(documentcnt,totalNumberOfDocument))
            shortlistTextIdsWithinDocument = []
            shortlistTextIdsWithinDocument = get_shortlisted_textId_using_sentence_encoder(questionId,individualDocumentId,contextSentencesListPerQuestion)            
            if (len(shortlistTextIdsWithinDocument) > 0 ) :
                recordToBeInserted = {}
                recordToBeInserted[individualDocumentId] = shortlistTextIdsWithinDocument
                #Update the Master finalResultPerOpenQuestions 
                if (questionId in finalResultPerOpenQuestions.keys()) : 
                    updatedFinalListOfData = finalResultPerOpenQuestions[questionId]
                    updatedFinalListOfData.append(recordToBeInserted)
                else : 
                     updatedFinalListOfData.append(recordToBeInserted)
                finalResultPerOpenQuestions[questionId] = updatedFinalListOfData
    print("finalResultPerOpenQuestions-->",finalResultPerOpenQuestions)
            


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from IPython.display import FileLink

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
finalResultPerOpenQuestions = {} # Format -->{1(OpenQuestionId): [{1(document id):[number within the body_text eg- 1,5,6]}]}
referenceContextSentences = {} 
fileList = {}
keyColumnNames = ['naproxen','clarithromycin','minocycline','antibody-dependent enhancement','animal models','therapeutic',
                  'antiviral agents','models','prioritize','distribute scarce','population','vaccine','standardize','prophylaxis',
                 'enhanced disease','corona','response']
initializedData = {'naproxen':0,'clarithromycin':0,'minocycline':0,'antibody-dependent enhancement':0,
                  'animal models':0,'therapeutic':0,
                  'antiviral agents':0,'models':0,'prioritize':0,'distribute scarce':0,
                  'population':0,'vaccine':0,'standardize':0,'prophylaxis':0,
                 'enhanced disease':0,'corona':0,'response':0}

cnt = 0 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
       # print(os.path.join(dirname, filename))
        if(filename.find('.json') >=0):
            cnt = cnt + 1 
            fileList[cnt] = [filename,os.path.join(dirname, filename)]

referenceContextSentences = {1:['Drugs experimented against COVID.','Clinical and bench trials of naproxen against COVID','Clinical and bench trials of clarithromycin against COVID','Clinical and bench trials of minocycline  against COVID-19'],
                             2: ['Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients','Methods evaluating potential effect of vaccine on recipient.'],
                            3: ['Exploration of use of best animal models and their predictive value for a human vaccine','Animal models and their value for a human vaccine'],
                            4: ['Capabilities to discover a therapeutic  for the disease.','Cinical effectiveness studies to discover therapeutics, to include antiviral agents.'],
                            5: ['Models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics',' Approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.'],
                            6: ['Efforts at a universal coronavirus vaccine','Preparing an universal coronavirus vaccine'],
                            7: ['developing animal models and standardize challenge studies for corona','developing models on animals','developing animal models and standardize challenge studies for covid19'],
                            8: ['develop prophylaxis clinical studies'],
                            9: ['Evaluate risk for enhanced disease after vaccination.','Post vaccination risks on new diseases'],
                            10: ['Evaluate vaccine immune response','Response to Corona Vaccine']}
key2OpenQuestions = {1: ['naproxen','clarithromycin','minocycline'], 
                     2: ['antibody-dependent enhancement'],
                   3: ['animal models'] , 4: ['therapeutic','antiviral agents'],
                   5: ['models','prioritize','distribute scarce'],
                   6: ['vaccine','corona'],7: ['standardize'] , 
                   8: ['prophylaxis'] , 9:['enhanced disease'] , 10: ['response','corona']}

#key2OpenQuestions = {1: ['naproxen','clarithromycin','minocycline']}

isDir = os.path.isdir('./model')
print("isDir->",isDir)
if (isDir != True):
   get_ipython().system('mkdir model')
   get_ipython().system('curl -L "https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed" | tar -zxvC ./model')

embed = hub.load("./model")    
key2DocumentCount = pd.DataFrame(initializedData,columns=keyColumnNames,index=fileList.keys())

#Step1 - Prepare the initial list of shorlisted Document 
to_be_executed = True 
if(to_be_executed) :
    for fileId in fileList.keys(): 
        fileData = {}
        fileName = fileList[fileId][0]
        filePath = fileList[fileId][1]
        with open(filePath) as f:
            fileData = json.load(f)
        body_text = fileData['body_text']
        for textData in body_text :
            actualText = textData['text'].lower()
            update_word_count_presence_in_text(fileId,actualText)
key2DocumentCount.head(1000)


# In[ ]:


#Step2 - Prepare the final list of shorlisted document 
prepare_result()

#Step3 - Prepare for final dump to a csv file Format - Question  ,DocumentName ,Author Name ,Relevent Contexts
prepare_final_result()


# In[ ]:


def prepare_final_result() :
    global finalResultPerOpenQuestions,fileList
    
    is_first_record_per_question = True
    is_first_record_per_document = True
    area_focus = " "
    documentName = " "
    documentTitle = " "
    relatedText =  " "
    
    resultantColumns = {'Area Under Focus','Document Name' , 'Title' , 'Related Context within document'}
    initialData = {}
    resultantDataFrame = pd.DataFrame(initialData,columns=resultantColumns)
    
    actual_opening_questions = {1: "Effectiveness of drugs being developed and tried to treat COVID-19 patients.",
                                  2: "Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.",
                                  3: "Exploration of use of best animal models and their predictive value for a human vaccine.",
                                  4: "Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",
                                  5: "Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up.",
                                  6: "Efforts targeted at a universal coronavirus vaccine.",
                                  7: "Efforts to develop animal models and standardize challenge studies.",
                                  8: "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers.",
                                  9: "Approaches to evaluate risk for enhanced disease after vaccination.",
                                  10: "Assays to evaluate vaccine immune response and process development for vaccines."}
    
    for questionId in finalResultPerOpenQuestions.keys():
        print("Preparing for Question Id",questionId)
        finalDocumentList = finalResultPerOpenQuestions[questionId]
        is_first_record_per_question = True 
        for documentData in finalDocumentList :
            is_first_record_per_document = True
            documentId = list(documentData.keys())[0]
            print("Preparing for documentId",documentId)
            documentTextList = documentData[documentId]
            filePath = fileList[documentId][1]
            with open(filePath ,'r') as f :
                 documentContent = json.load(f)
            documentBodyText = documentContent['body_text']
            for individualTextIds in documentTextList : 
                relatedTextWithinDocument = documentBodyText[individualTextIds]['text']
                if(is_first_record_per_question or is_first_record_per_document ) :
                    area_focus = actual_opening_questions[questionId]
                    documentName = fileList[documentId][0]
                    documentTitle = documentContent['metadata']['title']
                    relatedText =  relatedTextWithinDocument
                    is_first_record_per_question = False
                    is_first_record_per_document = False
                else :
                    area_focus = " "
                    documentName = " "
                    documentTitle = " "
                    relatedText =  relatedTextWithinDocument
                recordToInsert = {'Area Under Focus' : area_focus,'Document Name' : documentName, 
                                  'Title': documentTitle , 'Related Context within document' : relatedText}
                
                recordDataFrame = pd.DataFrame([recordToInsert])
                print("record to insert -->",recordDataFrame)
                resultantDataFrame = pd.concat([resultantDataFrame,recordDataFrame],ignore_index=True,sort=False)
        #print ("Final Result -->",resultantDataFrame.head(800))
        #Moving the data to csv 
        isDir = os.path.isdir('./result')
        print("isDirpresent->",isDir)
        if (isDir != True):
           get_ipython().system('mkdir result')
        resultantFileName = str(questionId) + "_" + "result.csv"
        resultantDataFrame.to_csv('./result/' + resultantFileName) 
        resultantDataFrame = pd.DataFrame(initialData,columns=resultantColumns)
    


# In[ ]:


get_ipython().system('ls -lrt ./result')


# In[ ]:


for i in range(1,11)
    fileName = "./result/" + str(i) + "_result.csv"
    FileLink(fileName)


# In[ ]:


get_ipython().system('ls -lrt ')

