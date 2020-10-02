#!/usr/bin/env python
# coding: utf-8
import subprocess
subprocess.run('pip uninstall -y spacy',shell=True, check=True)
subprocess.run('pip install scispacy',shell=True, check=True)
subprocess.run('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz',shell=True, check=True)

import os

from concurrent.futures import ThreadPoolExecutor, as_completed
n_cpus = os.cpu_count()
print(f'Number of CPUs: {n_cpus}')
executor = ThreadPoolExecutor(max_workers=n_cpus)
"""
warning: while this is memory efficient script it takes a long time to run (20+ hours on small model)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import joblib
import ahocorasick

datapath = Path('/kaggle/input')

import sqlalchemy

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
engine = create_engine('sqlite:////kaggle/working/df_covid')

from sqlalchemy import Column, VARCHAR, TEXT,INTEGER
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class Processed_articles(Base):
    __tablename__ = 'processed_articles'
    article_id = Column(VARCHAR(40), primary_key = True)
    
class Sentences(Base):
    __tablename__='sentences_table'
    article_id=Column(VARCHAR(40), primary_key = True)
    paragraph_id=Column(INTEGER(), primary_key=True)
    sentence_id=Column(INTEGER(), primary_key=True)
    sentence_text=Column(TEXT())
    sentence_tokenised=Column(TEXT())
                         
class Entities_synonims(Base):
    __tablename__='entities_synonims'
    ent_id=Column(INTEGER(), primary_key=True)
    ent_text=Column(TEXT())
    concept_id= Column(VARCHAR(30))
    synonims=Column(TEXT())
    
Base.metadata.create_all(engine)

import scispacy
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
nlp = spacy.load("en_core_sci_lg")



# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
# NOTE: The resolve_abbreviations parameter is optional, and requires that
# the AbbreviationDetector pipe has already been added to the pipeline. Adding
# the AbbreviationDetector pipe and setting resolve_abbreviations to True means
# that linking will only be performed on the long form of abbreviations.
linker = UmlsEntityLinker(resolve_abbreviations=True)

nlp.add_pipe(linker)


# at this point script loads models and consumer 8GB RAM

processed_docs=set()
abbreviations = {}
A = ahocorasick.Automaton()



session = Session(engine)
for processed_doc in session.query(Processed_articles):
    processed_docs.add(processed_doc.article_id)


print(len(processed_docs))



# parse json document body into paragragraphs
def parse_json_body_text(json_filename, processed_docs=processed_docs):
        if json_filename.stem not in processed_docs:
            print("Processing ..", json_filename.stem)
            with open(json_filename) as json_data:
                data = json.load(json_data)
                paper_id=data['paper_id']
                if paper_id not in processed_docs:
                    for body_text in data['body_text']:
                        para = body_text['text']
                        yield para


# parse paragraph expanding abbreviations
def parse_paragraph(para, nlp=nlp, abbreviations=abbreviations):
    para = para.strip()
    doc=nlp(para)
    abbreviations.update({str(abrv): str(abrv._.long_form) for abrv in doc._.abbreviations})
    return doc


# expand abbreviations 
def expand_abbreviations(sentence,abbreviations=abbreviations):
    sent_tokens = []
    for token in nlp(sentence.text):
        if token.is_stop==False:
            token_text = token.text
            if token_text in abbreviations.keys():
                sent_tokens.append(abbreviations[token_text])
            else:
                sent_tokens.append(token_text)
    if len(sent_tokens)>0:
        return " ".join(sent_tokens)

# map entities to Unified Medical Language System - we map to external ontology
def map_to_synonims(sentence,session=session):
    """ input is sentence and return ent.text and linked 
    concept_id, canonical name and list of synonims capped at 5
    """
    linked_entities=[]
    for ent in sentence.ents:
        for umls_ent in ent._.umls_ents:
            linked_entity=linker.umls.cui_to_entity[umls_ent[0]]
            unique_syns = list(set(linked_entity.aliases[0:5]))
            session.add(Entities_synonims(ent_text=ent.text, concept_id=linked_entity.concept_id, canonical_name=linked_entity.canonical_name,synonims="|".join(unique_syns)))



#process document return sentences and entities 
def process_file(f,engine=engine):
    session = Session(engine)
    pid, sid = 0, 0
    article_id=f.stem
    print("Processing article_id ", article_id)
    if session.query(Processed_articles).filter(Processed_articles.article_id == article_id).count()>0:
        session.close()
        print("already processed ", article_id)
        return article_id
    for para in parse_json_body_text(f):
        doc = parse_paragraph(para)
        for sentence in doc.sents:
            sent_tokens=expand_abbreviations(sentence)
            session.add(Sentences(article_id=article_id, paragraph_id=pid,sentence_id=sid, sentence_text = sentence.text, sentence_tokenised=sent_tokens))
#             map_to_synonims(sentence, session)
            session.commit()
            sid+=1
        pid+= 1
    session.add(Processed_articles(article_id=article_id))
    session.commit()
    session.close()
    return article_id

# main submission loop 
processed=[]
json_filenames = datapath.glob('**/*.json')
for each_file in json_filenames:
    print("Submitting task")
    task=executor.submit(process_file,each_file,engine)
    processed.append(task)

print("Waiting for tasks to complete")
for each_task in as_completed(processed):
    print(task.result())
session.close()
