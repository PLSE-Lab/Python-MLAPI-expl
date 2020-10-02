# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# import training and test data
train_data = pd.read_csv('/kaggle/input/train.csv')
train_data.head()
train_data.dropna(inplace=True)
test_data = pd.read_csv('/kaggle/input/test.csv')
test_data.head()


# we can't preprocess the text as selected text contains raw text
# lets develop a custom NER model with the help of spacy, Named entities will be the selected text

import spacy
from tqdm import tqdm
import plac
import random
import warnings
from pathlib import Path
from spacy.util import minibatch, compounding



# create a function to get the training data for our model

def get_training_data(sentiment):
    train_df = []
    for index, row in train_data.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_df.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_df    


def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = '/kaggle/working/model_pos'
    elif sentiment == 'negative':
        model_out_path = '/kaggle/working/model_neg'
    elif sentiment == 'neutral':
        model_out_path = '/kaggle/working/model_neu'
    return model_out_path


def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,   # dropout - make it harder to memorise data
                    losses=losses, 
                )
            
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')
    

def save_model(output_dir, nlp, new_model_name):
#    output_dir = f'/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models/'
    output_dir=get_model_out_path(sentiment)
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
sentiments=[ 'positive','negative','neutral']
# training a model for each sentiment
for sentiment in sentiments:
    train_df=get_training_data(sentiment)
    model_path = get_model_out_path(sentiment)
    train(train_df, model_path, n_iter=2, model=None)
    

TRAINED_MODELS_BASE_PATH = '/kaggle/working/' #path where models are saved


def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text  

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
    
    jaccard_score = 0
    for row in tqdm(train_data.itertuples(), total=train_data.shape[0]):
        text = row.text
        if row.sentiment == 'neutral':
            jaccard_score += jaccard(predict_entities(text, model_neu), row.selected_text)
#            count +=1
#            k=0
        
            
        elif row.sentiment == 'positive':
            jaccard_score += jaccard(predict_entities(text, model_pos), row.selected_text)
#            count +=1
#           k=0
            
        else:
            jaccard_score += jaccard(predict_entities(text, model_neg), row.selected_text) 
#            count +=1
#           k=0



#print(f'Average Jaccard Score is {jaccard_score/train_data.shape[0]}') 

def predict_on_test_df(text, sentiment):
    if sentiment == 'neutral':
        selected = predict_entities (text, model_neu)
    elif sentiment == 'positive':
        selected = predict_entities (text, model_pos)
    else :
        selected = predict_entities (text, model_neg)
        
    return(selected)

test_data['selected_text']= test_data.apply(lambda x: predict_on_test_df(x['text'], x['sentiment']), axis=1)
data_to_submit= test_data.drop(['text', 'sentiment'],axis=1)
data_to_submit.columns

data_to_submit.to_csv('submission.csv', index=False)