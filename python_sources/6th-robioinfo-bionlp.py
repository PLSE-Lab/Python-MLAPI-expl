#!/usr/bin/env python
# coding: utf-8

# GENIA EVENT 2011 dataset
# ==
# The GENIA event extraction (GENIA) task is a main task in [BioNLP Shared Task 2011](http://2011.bionlp-st.org/home/genia-event-extraction-genia) (BioNLP-ST '11). 
# 
# The data is composed of the original text, alongside which sit two annotation files: one for relevant entities, and one for interactions (or events) between these entities (or, sometimes, other events).
# The link between these triples of files is the file name: \*.txt for the text, \*.a1 for the entities, and \*.a2 for the events.
# 
# Entities are marked with an id starting with T, followed by a tab, followed by the entity type, space, begin character offset, space, end character offset, tab, the text marked-up as the entity (text sitting between the character offsets).
# 
# Events are marked with an id starting with E, followed by tab, the predicate of the event (event type, colon, predicate/entity ID), space, followed by all arguments of that event. Arguments can be both entities and other events.
# 
# There are also some other pieces of information which handle negation and speculation, but we will not focus on these today.
# 
# The format of \*.a1 is:
# > T1 \t Protein 0 10 \t my_protein
# 
# The format of \*.a2 is:
# > T2 \t Gene_expression 11 20 \t expresses
# >
# > E1 \t Gene_expression:T2 Theme:T1
# 
# Let's peek inside the files.

# In[ ]:


import codecs

with codecs.open("/kaggle/input/geniaevent2011/dev/PMC-1134658-00-TIAB.txt", "r", "utf-8") as fin:
    print(fin.readlines())


# In[ ]:


with codecs.open("/kaggle/input/geniaevent2011/dev/PMC-1134658-00-TIAB.a1", "r", "utf-8") as fin:
    print(fin.readlines())


# In[ ]:


with codecs.open("/kaggle/input/geniaevent2011/dev/PMC-1134658-00-TIAB.a2", "r", "utf-8") as fin:
    print(fin.readlines())


# We now know how the input data looks like, so we can extract the data inside the annotation files and shape it up in a way which is usable at later stages.
# 
# Particularly, we need the raw text, as well as the entity and event information, keeping the character offsets, their types, and the connections between them.
# 
# For the text, we need to store the actual raw text, without any modifications, so that the offsets of the annotations still make sense to the machine learning algorithms.
# 
# For the entities, we need to store the begin and end offsets and the type of the entity. Store these against the ID of the entity, as this is required by the events.
# 
# For the events, we need to store the predicate of the event, which is equivalent to an entity in its own right. The extra information that is required comprises the arguments of the event (i.e., theme, cause). These are referenced by ID to either entities or other events.

# In[ ]:


import os

def read_input_files(subset):
    texts = {}
    ents = {}
    eves = {}
    ent_labels = set()
    eve_labels = set()
    for dirname, _, filenames in os.walk('/kaggle/input/geniaevent2011/' + subset):
        for filename in filenames:
            fn = os.path.join(dirname, filename)
            with codecs.open(fn, 'r', 'utf8') as fin:
                # process original text files (.txt)
                if filename.endswith('.txt'):
                    texts[filename[:-4]] = "".join(fin.readlines())
                # process entity files (.a1)
                elif filename.endswith('.a1'):
                    if filename[:-3] not in ents:
                        ents[filename[:-3]] = {}
                    for l in fin.readlines():
                        if l.strip() != '':
                            tabs = l.split('\t')
                            spaces = tabs[1].split(' ')
                            ents[filename[:-3]][tabs[0]] = (int(spaces[1]), int(spaces[2]), spaces[0])
                            ent_labels.add(spaces[0])
                # process event files (.a2)
                elif filename.endswith('.a2'):
                    if filename[:-3] not in eves:
                        eves[filename[:-3]] = {}
                    if filename[:-3] not in ents:
                        ents[filename[:-3]] = {}
                    for l in fin.readlines():
                        if l.strip() != '':
                            l = l.strip()
                            if l[0] == "T":
                                # entity
                                tabs = l.split('\t')
                                spaces = tabs[1].split(' ')
                                ents[filename[:-3]][tabs[0]] = (int(spaces[1]), int(spaces[2]), spaces[0])
                                ent_labels.add(spaces[0])
                            elif l[0] == "E":
                                # event
                                tabs = l.split('\t')
                                eves[filename[:-3]][tabs[0]] = tabs[1].split(' ')
                    
    return texts, ents, eves, list(ent_labels), list(eve_labels)
# Any results you write to the current directory are saved as output.


# Let's give that a try!

# In[ ]:


texts, ents, eves, ent_labels, eve_labels = read_input_files('dev/')
# print(ent_labels)
# print(ents)


# Spacy requires the data in a slightly different shape, so we need to convert our dictionaries into something that spacy can digest.
# The input format for spacy is a list of tuples, in which the first element is the text, and the second is a dictionary, in which the key is "entities" and the value is a list of triples of start offset, end offset, and entity type.
# > [(text: str, {"entities": [(start: int, end: int, type: str)]})]

# In[ ]:


TRAIN_DATA = [(texts[k], {"entities": list(ents[k].values()) if k in ents else []}) for k in texts]
# print(TRAIN_DATA)


# Spacy
# ==
# Spacy is a framework created and used for NLP applications. It is quite fast (by comparison) and achieves decent scores across a multitude of NLP tasks, languages, and domains. It can be a bit of a learning curve to get to know how it operates.

# In[ ]:


import spacy
import en_core_web_sm
from spacy import displacy

nlp = en_core_web_sm.load()
texts = ["This is a text.", "And now Andy will process it with Spacy. Great things are happening in lovely Cluj Napoca."]
for doc in nlp.pipe(texts):
    print(list(doc.sents))
#     print([[(token.text, token.lemma_, token.pos_, token.tag_) for token in sent] for sent in doc.sents])
#     print(list(doc.noun_chunks))
#     print([(ent.text, ent.label_) for ent in doc.ents])
#     displacy.render(doc.sents, style="dep")
#     displacy.render(doc.sents, style="ent")


# You can find a neat [spacy cheatsheet](https://www.datacamp.com/community/blog/spacy-cheatsheet) for when you are not sure of how to do something.

# How Spacy works
# --
# Spacy uses DNNs to analyse and produce various types of annotation on input text.
# Words (more precisely, tokens) are represented as multi-dimensional vectors, trained with GloVe on Common Crawl.
# Each word also takes into account the words around it, so that instead of describing what a word means, you describe it by its context. E.g., a dog is not a mammal, but something that barks, is fluffy, is a pet, chases cats etc.
# 
# In the case of NER, Spacy learns transitions between the five states for each of the tokens: B, I, L, U, or O. BIL corresponds to **B**egin, **I**nside, **L**ast token of a multi-token entity span, an entity formed of a single **U**nique token, or **O**utside an entity span.

# Biomedical IE
# --
# Biomedical named entities, such as Proteins and Diseases, are not tipically included in general language NER models.
# We need to train an NER model to recognise these domain-specific entities.
# Spacy can (re-)train any existing or new component to do what is required, as long as it gets some data to learn from. 

# In[ ]:


from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# To train a new model, we only need the training data and the list of possible labels.
# Ensure that you will train only the NER and not other pipes in the pipeline.
# Minibatch your data to ensure a higher speed of processing. And making them random ensures that you will not get stuck in a local minima.

# In[ ]:


def train(train_data, labels):
    random.seed(0)
    # start off with a blank model, for the English language, and extract the NER pipe so it can be set up
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it
    else:
        ner = nlp.get_pipe("ner")

    # add new entity labels to the entity recogniser
    [ner.add_label(label) for label in labels] 
    n_iter = 10 # how many times should repeat the training procedure
    
    optimiser = nlp.begin_training()
    move_names = list(ner.move_names)
    print(move_names)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    with nlp.disable_pipes(*other_pipes):  # only train NER, ignore the other pipes
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimiser, drop=0.35, losses=losses)
            print("Losses", losses)
    return nlp


# Let's train it! Mind you, it will take a couple of minutes to train the 10 epochs on this size of training data.

# In[ ]:


model = train(TRAIN_DATA, ent_labels)


# To test the new model, we just need to apply the model object to some text, similar to previous times.

# In[ ]:


# test the trained model
def test(model, text):
    doc = model(text)
    print("Entities in '%s'" % text)
    for ent in doc.ents:
        print(ent.label_, ent.text)
    displacy.render(doc, style="ent")


# Let's test it!

# In[ ]:


test_text = "MTb induces NFAT5 gene expression via the MyD88-dependent signaling cascade."
test(model, test_text)


# Scispacy
# --
# 
# Training a model seems to be easy-peasy!
# 
# But to train a highly performant model, we need lots more data, lots more training epochs.
# Instead, we will use one of the off-the-shelf models developed in scispacy, a scientific version built on top of spacy.
# 
# Scispacy comes with a variety of models, differing in size and employed training dataset.
# These are not readily available in Kaggle, so they need to be downloaded and installed locally.
# You will need to activate your Internet connection (probably by confirming that you are a real person using a mobile phone).
# It might take several minutes for the packages to be downloaded and installed.

# In[ ]:


get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_sm-0.2.0.tar.gz')


# Scispacy comes with other tools that are useful in bio text mining: abbreviation detection and entity linking.
# 
# The abbreviation detector finds abbreviations in text and attempts to figure out their meaning in that context using heuristics.
# 
# The entity linking pipe attempts to find a unique identifier for each named entity in an established database, so that it can be put into a larger context of knowledge. In this case, the pipe links it to UMLS (Unified Medical Language System).

# In[ ]:


import scispacy
import spacy
import en_core_sci_sm
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.abbreviation import AbbreviationDetector


# We can load the new model, and extend the default pipeline with the extra two useful pipes: abbreviation detection and entity linking.

# In[ ]:


nlp = en_core_sci_sm.load()
# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp.add_pipe(abbreviation_pipe)
nlp.add_pipe(linker)
text = """
Myeloid-derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. 
They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC).
MTb induces NFAT5 gene expression via the MyD88-dependent signaling cascade.
"""
doc = nlp(text)

print(list(doc.sents))
print(doc.ents)


# Let's check out the abbreviations it can find.

# In[ ]:


for abrv in doc._.abbreviations:
    print(f"{abrv} \t ({abrv.start}, {abrv.end}) \t {abrv._.long_form}")


# Let's see the unique identifiers of each of the named entities it has uncovered.
# For some ambiguous entities, the pipe returns a list of possible matches. It is up to the user to select one of them for further analysis.

# In[ ]:


for entity in doc.ents:
    if len(entity._.umls_ents) > 0:
        print(entity, linker.umls.cui_to_entity[entity._.umls_ents[0][0]], "\n\n\n\n")
#     for umls_ent in entity._.umls_ents:
#         print(entity, linker.umls.cui_to_entity[umls_ent[0]], "\n\n\n\n")


# Let's look at the entities, marked up again, so that we realise how complex medical language is.
# Also, let's look at the syntactic dependencies between all the tokens.
# Finishing the bio text mining task requires identifying the arguments of each of the relevant events.
# This is usually done using machine learning over syntactic dependency paths between the predicate of the event and the other entities and/or predicates (events between events are very common in biomedicine!).

# In[ ]:


displacy.render(doc, style="ent")
displacy.render(doc, style="dep")


# Knowledge Graph
# ==
# Having extracted all the information we require from text, we need to combine the various facts gathered from a large collection of texts into one knowledge base. A knowledge graph is a very useful means of representing data, where vertices represent the entities, and edges represent events between them.
# 
# Whilst there are several frameworks that can store data as graphs, one of the most used and usable is Neo4j.
# Unfortunately, Neo4j is not readily supported in Kaggle. 
# 
# A very good example of Neo4j applied to biomedical information is [Hetionet](http://neo4j.het.io/). We will run some queries on their knowledge graph.
# 
# The query language for Neo4j is called Cypher, and is, to some extent, similar to SQL.
# 
# Explore the graph by selecting a few random nodes, possibly specifying the type or edges between the nodes. Remember to set a limit to the returned results, so that they return fast.

# Example queries
# --
# Return max 25 nodes. Double click on any node to expand all of its direct connections.
# > MATCH (n) RETURN n LIMIT 25
# 
# Return max 25 edges. Double click on any node to expand all of its direct connections.
# > MATCH path = ()-[r]-() RETURN path LIMIT 25
# 
# Note that all nodes and all edges have metadata associated to them. Most importantly, they mention the sources of that specific piece of information, either in the form of a DB id (e.g., DrugBank, Entrez, UMLS), or a publication id (pubmed).
# 
# Return max 25 Compounds. 
# > MATCH (n:Compound) RETURN n LIMIT 25
# 
# Return max 25 edges of type TREATS_CtD (Treats_Compound-treats-Disease) between a compound and any other node. It will most likely be a Disease, unless there is a problem in the data.
# > MATCH path = (n:Compound)-[:TREATS_CtD]-() RETURN path Limit 25
# 
# Return the Compound whose name is 'Bupropion'.
# > MATCH (n:Compound {name: "Bupropion"}) RETURN n 
# 
# Return max 100 paths in which Bupropion causesa side-effect also caused by another compound.
# > MATCH path = (:Compound {name: "Buproprion"})-[:CAUSES_CcSE]-(:SideEffect)-[:CAUSES_CcSE]-(:Compound ) RETURN [path] limit 100
# 
# Last one:
# > MATCH p0 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p1 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:BINDS_CbG]-(:Compound {identifier: "DB01273"})-[:TREATS_CtD]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p2 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "WP1603_r78574"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 1143})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p3 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "PC7_4468"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 1142})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p4 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "PC7_4468"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 8973})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p5 = (:Compound {identifier: "DB01156"})-[:CAUSES_CcSE]-(:SideEffect {identifier: "C0541798"})-[:CAUSES_CcSE]-(:Compound {identifier: "DB01273"})-[:TREATS_CtD]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p6 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "PC7_4469"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 1142})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p7 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "PC7_4469"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 8973})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p8 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "PC7_4470"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 1143})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > MATCH p9 = (:Compound {identifier: "DB01156"})-[:BINDS_CbG]-(:Gene {identifier: 1136})-[:PARTICIPATES_GpPW]-(:Pathway {identifier: "PC7_6571"})-[:PARTICIPATES_GpPW]-(:Gene {identifier: 1142})-[:ASSOCIATES_DaG]-(:Disease {identifier: "DOID:0050742"})
# > RETURN [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
