#!/usr/bin/env python
# coding: utf-8

# <img src='https://spacy.io/assets/img/pipeline.svg'><center><span style="font-size:10px;">Source: <a href="https://spacy.io/usage/processing-pipelines">spaCy Language Processing Pipelines</a></span></center>

# # Quick and Dirty - Entity Extraction

# <span style="color:gray;font-size:20px;">From idea to prototype in AI.</span>

# <em>If you've ever been around a startup or in the tech world for any significant amount of time, you've <strong>definitely</strong> encountered some, if not all of the following phrases: "agile software development", "prototyping", "feedback loop", "rapid iteration", etc.</em>
# 
# <em>This Silicon Valley techno-babble can be distilled down to one simple concept, which just so happens to be the mantra of many a successful entrepreneur: test out your idea as quickly as possible, and then make it better over time. Stated more verbosely, before you invest mind and money into creating a cutting-edge solution to a problem, it might benefit you to get a baseline performance for your task using off-the-shelf techniques. Once you establish the efficacy of a low-cost, easy approach, you can then put on your Elon Musk hat and drive towards #innovation and #disruption.</em> 
# 
# <em>A concrete example might help illustrate this point:</em>

# ## Introduction

# ### Entity Extraction

# Let's say our goal was to create a natural language system that effectively allowed someone to converse with an academic paper. This task could be step one of many towards the development of an automated scientific discovery tool. Society can thank us later. 
# 
# But where do we begin? Well, a part of the solution has to deal with [knowledge extraction](https://en.wikipedia.org/wiki/Knowledge_extraction). In order to create a conversational engine that understands scientific papers, we'll first need to develop an entity recognition module, and this, lucky for us, is the topic of our notebook! 
# 
# "What's an entity?" you ask? Excellent question. Take a look at the following sentence:

# > Dr. Abraham is the primary author of this paper, and a physician in the specialty of internal medicine.

# Now, it should be relatively straighforward for an English-speaking human to pick out the important concepts in this sentence:
# 
# > **[Dr. Abraham]** is the **[primary author]** of this **[paper]**, and a **[physician]** in the **[specialty]** of **[internal medicine]**.

# These words and/or phrases are categorized as "entities" because they represent salient ideas, nouns, and noun phrases in the real world. A subset of entities can be "named", in that they correspond to <strong><em>specific</em></strong> places, people, organizations, and so on. A [named entity](https://en.wikipedia.org/wiki/Named_entity) is to a regular entity, what "Dr. Abraham" is to a "physician". The good doctor is a real person and an instance of the "physician" class, and is therefore considered "named". Examples of named entities include "Google", "Neil DeGrasse Tyson", and "Tokyo", while regular, garden-variety entities can include the list just mentioned, as well as things like "dog", "newspaper", "task", etc.
# 
# Let's see if we can get a computer to run this kind of analysis to pull important concepts from sentences. 

# ### The Task

# For our conversational academic paper program, we won't be satisfied with simply capturing named entities, because we need to understand the relationships between general concepts as well as actual things, places, etc. Unfortunately, while most out-of-the-box text processing libraries have a moderately useful <strong>named entity recognizer</strong>, they have little to no support for a generalized <strong>entity recognizer</strong>. 
# 
# This is because of a subtle, yet important constraint. 
# 
# Entities, as we've discussed, correspond to a superset of named entities, which <strong><em>should</em></strong> make them easier to extract. Indeed, blindly pulling all entities from a text source is in fact simple, but it's sadly not all that useful. In order to justify this exercise, we'd need to develop an entity extraction approach that is restricted to, or is cognizant of, some particular domain, for example, neuroscience, psychology, computer science, economics, etc. This paradoxical complexity makes it nontrivial to create a generic, but useful, entity recognizer. Hence the lack of support in most open-source libraries that deal with natural language processing. 
# 
# To largely simplify our task then, we must generate a set of entities from a scientific paper, that is <strong><em>larger</em></strong> than a simple list of named entities, but <strong><em>smaller</em></strong> than the giant list of all entities, restricted to the domain of a particular paper in question. 
# 
# Yikes. Are you sweating a little? Because I am. 
# 
# Instead of reaching for some Ibuprofen and deep learning pills, let's make a prototype using a little ingenuity, simple open-source code, and a lot of heuristics. Hopefully, through this process, we'll also learn a bit about the text processing pipeline that brings understanding natural language into the realm of the possible. 

# Enought chit-chat. Let's get to it!

# ## Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# <em><strong>Fun fact</strong>: Curious about what 'autoreload' does? <a href="https://ipython.org/ipython-doc/3/config/extensions/autoreload.html">Check this out</a>.</em>

# In[ ]:


import pandas as pd
import spacy
from spacy.displacy.render import EntityRenderer
from IPython.core.display import display, HTML


# ## Utils and Prep

# Let's do some basic housekeeping before we start diving headfirst into entity extraction. We'll need to deal with visualization, load up a language model, and of course, examine/set-up our data source.
# 
# ### Show and Tell
# Our prototype will lean heavily on a popular natural langauge processing (NLP) library known as spaCy, which also has a wonderful set of classes and methods defined to help visualize parts of the NLP pipeline. Up top, where we've imported modules, you'll have noticed that we're pulling 'EntityRenderer' from spaCy's displacy module, as we'll be repurposing some of this code for our... um... purposes. In general, this is a good exercise if you ever want to get your hands dirty and really learn how certain classes work in your friendly neighborhood open-source projects. Nothing should ever be off-limits or a black box; always dissect and play with your code before you eat it.  
# 
# Wander on over to spaCy's [website](https://spacy.io/), and you'll quickly discover that they've put in some serious thought into making the user interface absolutely gorgeous. (While Matthew undeniably had some input on this, I'm going to make an intelligent assumption that the design ideas are probably Ines' [contribution](https://explosion.ai/about)). 
# 
# <em><strong>&lt;rant&gt;</strong> Why spend so much time discussing visualization? Well, one of my biggest pet peeves is this: even if you can create a product, if you don't put in the time to make it look beautiful, or delightful to use, then you don't care about packaging your ideas for export to an audience. And that makes me sad. Once you get something working, make it pretty. <strong>&lt;/rant&gt;</strong></em>

# In[ ]:


def custom_render(doc, df, column, options={}, page=False, minify=False, idx=0):
    """Overload the spaCy built-in rendering to allow custom part-of-speech (POS) tags.
    
    Keyword arguments:
    doc -- a spaCy nlp doc object
    df -- a pandas dataframe object
    column -- the name of of a column of interest in the dataframe
    options -- various options to feed into the spaCy renderer, including colors
    page -- rendering markup as full HTML page (default False)
    minify -- for compact HTML (default False)
    idx -- index for specific query or doc in dataframe (default 0)
    
    """
    renderer, converter = EntityRenderer, parse_custom_ents
    renderer = renderer(options=options)
    parsed = [converter(doc, df=df, idx=idx, column=column)]
    html = renderer.render(parsed, page=page, minify=minify).strip()  
    return display(HTML(html))

def parse_custom_ents(doc, df, idx, column):
    """Parse custom entity types that aren't in the original spaCy module.
    
    Keyword arguments:
    doc -- a spaCy nlp doc object
    df -- a pandas dataframe object
    idx -- index for specific query or doc in dataframe
    column -- the name of of a column of interest in the dataframe
    
    """
    if column in df.columns:
        entities = df[column][idx]
        ents = [{'start': ent[1], 'end': ent[2], 'label': ent[3]} 
                for ent in entities]
    else:
        ents = [{'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_}
            for ent in doc.ents]
    return {'text': doc.text, 'ents': ents, 'title': None}

def render_entities(idx, df, options={}, column='named_ents'):
    """A wrapper function to get text from a dataframe and render it visually in jupyter notebooks
    
    Keyword arguments:
    idx -- index for specific query or doc in dataframe (default 0)
    df -- a pandas dataframe object
    options -- various options to feed into the spaCy renderer, including colors
    column -- the name of of a column of interest in the dataframe (default 'named_ents')
    
    """
    text = df['text'][idx]
    custom_render(nlp(text), df=df, column=column, options=options, idx=idx)


# In[ ]:


# colors for additional part of speech tags we want to visualize
options = {
    'colors': {'COMPOUND': '#FE6BFE', 'PROPN': '#18CFE6', 'NOUN': '#18CFE6', 'NP': '#1EECA6', 'ENTITY': '#FF8800'}
}


# In[ ]:


pd.set_option('display.max_rows', 10) # edit how jupyter will render our pandas dataframes
pd.options.mode.chained_assignment = None # prevent warning about working on a copy of a dataframe


# ### Load Model

# spaCy's pre-built models are trained on different corpora of text, to capture parts-of-speech, extract named entities, and in general understand how to tokenize words into chunks that have meaning in a given language. 
# 
# We'll grab the 'en_core_web_lg' model by running the following command in the shell (Kaggle only has access to the 'en_core_web_sm' model, but running this outside of a Kernel, you'd want to use 'en_core_web_lg'). 

# In[ ]:


nlp = spacy.load('en_core_web_sm')


# <em><strong>Fun fact</strong>: We can run shell commands in a Jupyter notebook by using the bang operator. This is an example of a <a href="https://ipython.readthedocs.io/en/stable/interactive/magics.html">magic</a> command, of which we saw an example at the begnning with '%autoreload'.</em>

# ### Gather Data

# As our data source, we'll be using papers presented at the [Neural Information Processing Systems (NIPS)](https://nips.cc/) conference held in a different location around the world each year. NIPS is the premier conference for all things machine learning, and considering our goal with this notebook, is an apropos choice to source our data. We'll pull a conveniently packaged dataset from [Kaggle](https://www.kaggle.com/benhamner/nips-2015-papers/version/2/home), and then work with a subset of the papers to keep our prototyping as lean and fast as possible.

# In[ ]:


PATH = '../input/'


# In[ ]:


get_ipython().system('ls {PATH}')


# <em><strong>Fun fact</strong>: You can use python variables in shell commands by nesting them inside curly braces.</em>

# In[ ]:


file = './nips-2015-papers/Papers.csv'
df = pd.read_csv(f'{PATH}{file}')

mini_df = df[:10]
mini_df.index = pd.RangeIndex(len(mini_df.index))

# comment this out to run on full dataset
df = mini_df


# ### Game Plan

# Now that we're all ready to get started, let's come up with a general list of tasks to to guide our approach. 
# 
# <br>
# <ol>
#     <strong><li>Inspect and clean data</li></strong>
#     <strong><li>Extract named entities</li></strong>
#     <strong><li>Extract nouns</li></strong>
#     <strong><li>Combine named entities and nouns</li></strong>
#     <strong><li>Extract noun phrases</li></strong>
#     <strong><li>Extract compound noun phrases</li></strong>
#     <strong><li>Combine entities and compound noun phrases</li></strong>
#     <strong><li>Reduce entity count with heuristics</li></strong>
#     <strong><li>Celebrate with excessive fist-pumping</li></strong>
# </ol>
# 
# That doesn't look too bad now does it? Let's build ourselves a prototype entity extractor.

# ## Step 1: Inspect and clean data

# In[ ]:


display(df)


# In[ ]:


lower = lambda x: x.lower() # make everything lowercase


# In[ ]:


df = pd.DataFrame(df['Abstract'].apply(lower))
df.columns = ['text']
display(df)


# ### Analysis

# Initially, there was quite a bit of metadata associated with each entry, including a unique identifier, the type of paper presented at the conference, as well as the actual paper text. After pulling out just the abstracts, we've now ended up with with a clean, read-to-go dataframe, and are ready to begin extracting entities. 

# ## Step 2: Extract named entities

# In[ ]:


def extract_named_ents(text):
    """Extract named entities, and beginning, middle and end idx using spaCy's out-of-the-box model. 
    
    Keyword arguments:
    text -- the actual text source from which to extract entities
    
    """
    return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]

def add_named_ents(df):
    """Create new column in data frame with named entity tuple extracted.
    
    Keyword arguments:
    df -- a dataframe object
    
    """
    df['named_ents'] = df['text'].apply(extract_named_ents)    


# In[ ]:


add_named_ents(df)
display(df)


# In[ ]:


column = 'named_ents'
render_entities(9, df, options=options, column=column) # take a look at one of the abstracts


# ### Analysis

# A quick glance at some of the abstracts shows that while we are able to extract numeric entities, not much else comes through. Not great. But then again, this is exactly why simply extracting named entities is not enough. On the plus side, our intuition about built-in models and scientific text was spot on! The spaCy named entity recognizer just wasn't exposed to this category of corpora and was instead trained on [blogs, news, and comments](https://spacy.io/models/en#en_core_web_lg). Academic papers don't use the most common English words, so it isn't unreasonable to expect a generally trained model to fail when confronted with text in such a restricted domain.   
# 
# Look at a few more abstracts by changing the index parameter in our "render_entities" function to convince yourself of the following notion:
# 
# We need to widen our search. 

# ## Step 3: Extract all nouns

# In[ ]:


def extract_nouns(text):
    """Extract a few types of nouns, and beginning, middle and end idx using spaCy's POS (part of speech) tagger. 
    
    Keyword arguments:
    text -- the actual text source from which to extract entities
    
    """
    keep_pos = ['PROPN', 'NOUN']
    return [(tok.text, tok.idx, tok.idx+len(tok.text), tok.pos_) for tok in nlp(text) if tok.pos_ in keep_pos]

def add_nouns(df):
    """Create new column in data frame with nouns extracted.
    
    Keyword arguments:
    df -- a dataframe object
    
    """
    df['nouns'] = df['text'].apply(extract_nouns)


# In[ ]:


add_nouns(df)
display(df)


# In[ ]:


column = 'nouns'
render_entities(0, df, options=options, column=column)


# ### Analysis

# This is more colorful. But is it useful? It appears as if we are able to pull out a lot of concepts, but things like "rest", "popularity", and "data", aren't all that interesting (atleast in the first abstract). Our search is too wide at this point. 

# Good to know. Let's power through for now, and merge our lists of entities.  

# ## Step 4: Combine named entities and nouns

# In[ ]:


def extract_named_nouns(row_series):
    """Combine nouns and non-numerical entities. 
    
    Keyword arguments:
    row_series -- a Pandas Series object
    
    """
    ents = set()
    idxs = set()
    # remove duplicates and merge two lists together
    for noun_tuple in row_series['nouns']:
        for named_ents_tuple in row_series['named_ents']:
            if noun_tuple[1] == named_ents_tuple[1]: 
                idxs.add(noun_tuple[1])
                ents.add(named_ents_tuple)
        if noun_tuple[1] not in idxs:
            ents.add(noun_tuple)
    
    return sorted(list(ents), key=lambda x: x[1])

def add_named_nouns(df):
    """Create new column in data frame with nouns and named ents.
    
    Keyword arguments:
    df -- a dataframe object
    
    """
    df['named_nouns'] = df.apply(extract_named_nouns, axis=1)


# In[ ]:


add_named_nouns(df)
display(df)


# In[ ]:


column = 'named_nouns'
render_entities(1, df, options=options, column=column)


# ### Analysis

# In this step, we're just combining the named entities extracted using spaCy's built-in model with nouns identified by the part-of-speech or POS tagger. We're dropping any numeric entities for now because they are harder to deal with and don't really represent new concepts. You'll notice (if you look closely enough), that we are also ignoring any hyphenated entities. In spaCy's tokenizer, it is possible to prevent hyphenated words form being split apart, but we'll reserve this, along with other types of advanced fine-tuning or low-level editing to if and when we move beyond the prototype phase. 
# 
# So far, in the past few steps, we've deal with one-word entities. However, it's also entirely permissible for combinations of two or more words to represent a single concept. This means that in order for our prototype to successfully capture the most relevant concepts, we'll need to pull n-length phrases from our academic abstracts in addition to single word entities. 

# ## Step 5: Extract noun phrases

# ### A Chunky Pipeline

# Even mild exposure to computer science, or any of the various isoforms of engineering, will have introduced you to the idea of an abstraction, wherein low-level concepts are bundled into higher-order relationships. The <strong>noun phrase</strong> or <strong>chunk</strong> is an abstraction which consists of two or more words, and is the by-product of dependency parsing, POS tagging, and tokenization. spaCy's POS tagger is essentially a statistical model which learns to predict the tag (noun, verb, adjective, etc.) for a given word using examples of tagged-sentences. 
# 
# This supervised machine learning approach relies on tokens generated from splitting text into somewhat atomic units using a rule-based tokenizer (although there are some interesting [unsupervised models](https://github.com/google/sentencepiece) out there as well). Dependency parsing then uncovers relationships between these tagged tokens, allowing us to finally extract noun chunks or phrases of relevance. 
# 
# The full pipeline goes something like this: 
# 
# <strong>raw text</strong> &rarr; <strong>tokenization &rarr; </strong> <strong>POS tagging</strong> &rarr; <strong>dependency parsing</strong> &rarr; <strong>noun chunk extraction</strong>
# 
# Theoretically, one could swap out noun chunk extraction for named entity recognition, but that's the part of the pipeline we are attempting to modify for our own purposes, because we want n-length entities. Barring our custom intrusion, however, this is exactly how spaCy's built-in model works! If you don't believe me (which you shouldn't, since you're a scientist), scroll up to the very top of this notebook to convince yourself. 

# Neat huh? Need a visualization of tokenization, POS tagging, and dependency parsing to convince you of just how cool this is? 
# 
# Take a look:

# In[ ]:


text = "Dr. Abraham is the primary author of this paper, and a physician in the specialty of internal medicine."

spacy.displacy.render(nlp(text), jupyter=True) # generating raw-markup using spacy's built-in renderer


# Just gorgeous. Following our pipeline, let's use this dependency tree to tease out the noun phrases in our dummy sentence. We'll have to create a few functions to do the heavy lifting first (we can reuse these guys for our full dataset later), and then use a simple procedure to visualize our example.

# In[ ]:


def extract_noun_phrases(text):
    """Combine noun phrases. 
    
    Keyword arguments:
    text -- the actual text source from which to extract entities
    
    """
    return [(chunk.text, chunk.start_char, chunk.end_char, chunk.label_) for chunk in nlp(text).noun_chunks]

def add_noun_phrases(df):
    """Create new column in data frame with noun phrases.
    
    Keyword arguments:
    df -- a dataframe object
    
    """
    df['noun_phrases'] = df['text'].apply(extract_noun_phrases)


# In[ ]:


def visualize_noun_phrases(text):
    """Create a temporary dataframe to extract and visualize noun phrases. 
    
    Keyword arguments:
    text -- the actual text source from which to extract entities
    
    """
    df = pd.DataFrame([text]) 
    df.columns = ['text']
    add_noun_phrases(df)
    column = 'noun_phrases'
    render_entities(0, df, options=options, column=column)


# In[ ]:


visualize_noun_phrases(text)


# Compare this to what we'd originally set out to accomplish:
# 
# > **[Dr. Abraham]** is the **[primary author]** of this **[paper]**, and a **[physician]** in the **[specialty]** of **[internal medicine]**.

# I don't know about you, but everytime I see this work, I'm blown away by both the intricate complexity and beautiful simplicity of this process. Ignoring the prepositions, with one single move, we've done a damn-near perfect job of extracting the main ideas from this sentence. How amazing is that?! 
# 
# Hats off to spaCy, and the hordes of data scientists, machine learning engineers, and linguists that made this possible.

# ### Back to School

# Now, if we just use this approach and add together the single-word entities we extracted from our academic abstracts earlier, we should be getting close to a pretty awesome set of concepts! Let's capture some noun phrases and see what we get. 

# In[ ]:


add_noun_phrases(df)
display(df)


# In[ ]:


column = 'noun_phrases'
render_entities(0, df, options=options, column=column)


# ### Analysis

# Hmm... should've seen this coming. While we've now done a great job of extracting noun phrases from our abstracts, we're running into the same problem as before. Our funnel is too wide, and we're pulling uninteresting bigrams like "the simplicity", "the rest", and "this mechanism". These chunks are indeed noun phrases, but not domain-specific concepts. Not to mention, we still have to deal with those pesky prepositions (try saying that five times fast). 
# 
# Let's see if we can narrow our search and just get the most important phrases. 

# ## Step 6: Extract compound noun phrases

# In[ ]:


def extract_compounds(text):
    """Extract compound noun phrases with beginning and end idxs. 
    
    Keyword arguments:
    text -- the actual text source from which to extract entities
    
    """
    comp_idx = 0
    compound = []
    compound_nps = []
    tok_idx = 0
    for idx, tok in enumerate(nlp(text)):
        if tok.dep_ == 'compound':

            # capture hyphenated compounds
            children = ''.join([c.text for c in tok.children])
            if '-' in children:
                compound.append(''.join([children, tok.text]))
            else:
                compound.append(tok.text)

            # remember starting index of first child in compound or word
            try:
                tok_idx = [c for c in tok.children][0].idx
            except IndexError:
                if len(compound) == 1:
                    tok_idx = tok.idx
            comp_idx = tok.i

        # append the last word in a compound phrase
        if tok.i - comp_idx == 1:
            compound.append(tok.text)
            if len(compound) > 1: 
                compound = ' '.join(compound)
                compound_nps.append((compound, tok_idx, tok_idx+len(compound), 'COMPOUND'))

            # reset parameters
            tok_idx = 0 
            compound = []

    return compound_nps

def add_compounds(df):
    """Create new column in data frame with compound noun phrases.
    
    Keyword arguments:
    df -- a dataframe object
    
    """
    df['compounds'] = df['text'].apply(extract_compounds)


# In[ ]:


add_compounds(df)
display(df)


# In[ ]:


column = 'compounds'
render_entities(0, df, options=options, column=column)


# ### Analysis

# That's starting to look pretty good! By targetting words in the dependency tree that were tagged as belonging to a compound, we were able to drive the number of noun phrases down rather nicely. Next, we'll add these phrases to the list of entities we extracted from each abstract, to create a set which will include unigrams, bigrams, and more. Oh my!

# ## Step 7: Combine entities and compound noun phrases

# In[ ]:


def extract_comp_nouns(row_series, cols=[]):
    """Combine compound noun phrases and entities. 
    
    Keyword arguments:
    row_series -- a Pandas Series object
    
    """
    return {noun_tuple[0] for col in cols for noun_tuple in row_series[col]}

def add_comp_nouns(df, cols=[]):
    """Create new column in data frame with merged entities.
    
    Keyword arguments:
    df -- a dataframe object
    cols -- a list of column names that need to be merged
    
    """
    df['comp_nouns'] = df.apply(extract_comp_nouns, axis=1, cols=cols)


# In[ ]:


cols = ['nouns', 'compounds']
add_comp_nouns(df, cols=cols)
display(df)


# In[ ]:


# take a look at all the nouns again
column = 'named_nouns'
render_entities(0, df, options=options, column=column)


# In[ ]:


# take a look at all the compound noun phrases again
column = 'compounds'
render_entities(0, df, options=options, column=column)


# In[ ]:


# take a look at combined entities
df['comp_nouns'][0] 


# ### Analysis

# Now that we have all the entities grouped together, we can see how good we are doing. We've successfully captured single-word as well as n-grams, but there appear to be a lot of duplicates. Words that should've been included in a phrase were somehow split apart, most likely as a result of not properly dealing with hyphenation when we first tokenized our abstracts. 
# 
# Not to worry, this should be relatively easy to take care. We'll also apply a few other heuristics to clean up our list and remove the most common English words to further pare down the list of entities.  

# ## Step 8: Reduce entity count with heuristics

# In[ ]:


def drop_duplicate_np_splits(ents):
    """Drop any entities that are already captured by noun phrases. 
    
    Keyword arguments:
    ents -- a set of entities
    
    """
    drop_ents = set()
    for ent in ents:
        if len(ent.split(' ')) > 1:
            for e in ent.split(' '):
                if e in ents:
                    drop_ents.add(e)
    return ents - drop_ents

def drop_single_char_nps(ents):
    """Within an entity, drop single characters. 
    
    Keyword arguments:
    ents -- a set of entities
    
    """
    return {' '.join([e for e in ent.split(' ') if not len(e) == 1]) for ent in ents}

def drop_double_char(ents):
    """Drop any entities that are less than three characters. 
    
    Keyword arguments:
    ents -- a set of entities
    
    """
    drop_ents = {ent for ent in ents if len(ent) < 3}
    return ents - drop_ents

def keep_alpha(ents):
    """Keep only entities with alphabetical unicode characters, hyphens, and spaces. 
    
    Keyword arguments:
    ents -- a set of entities
    
    """
    keep_char = set('-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    drop_ents = {ent for ent in ents if not set(ent).issubset(keep_char)}
    return ents - drop_ents


# These last four functions will slice and dice the list of entities gathered from each abstract in various ways. In addition to this granular processing, we'll also want to remove words that are frequent in the English language, as a heuristic to naturally drop stop words and uncover the domain of each academic source. 
# 
# Why is this?
# 
# Well, in NLP, as in search engine optimization (SEO), the most common words in a given corpus are known as [stop words](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html). These unfortunate candidates are hunted down with extreme prejudice and removed from the population to improve search results, enhance semantic analysis, and in our case, help restrict the domain. This is because removing stop words automatically limits the vocabulary of a corpus to the words that are less frequent and therefore, more likely to exist in that abstract than anywhere else. 
# 
# You can, of course, argue that the most common words in a scientific paper might in fact be the most important concepts, but stop words are usually overwhelmingingly overrepresented in any corpus. This intuition however, is exactly why we aren't going to simply take the most common words in one specific abstract and remove them. Instead, we'll be targetting the most frequent words based on a large, general domain sample of the English language. 
# 
# The "freq_words.csv" file you might have noticed earlier in our data environment, is actually a list generated from a corpus with 10 billion words gathered by the good people at [Word Frequencey Data](https://www.wordfrequency.info/).
# 
# Let's take a look at the list and then remove these words from our set of entities. 

# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


filename = './top-freq-english-words/freq_words.csv'
freq_words_df = pd.read_csv(f'{PATH}{filename}')
display(freq_words_df)


# In[ ]:


freq_words = freq_words_df['Word'].iloc[1:]
display(freq_words)


# In[ ]:


def remove_freq_words(ents):
    """Drop any entities in the 5000 most common words in the English langauge. 
    
    Keyword arguments:
    ents -- a set of entities
    
    """
    freq_words = pd.read_csv('../input/top-freq-english-words/freq_words.csv')['Word'].iloc[1:]
    for word in freq_words:
        try:
            ents.remove(word)
        except KeyError:
            continue # ignore the stop word if it's not in the list of abstract entities
    return ents

def add_clean_ents(df, funcs=[]):
    """Create new column in data frame with cleaned entities.
    
    Keyword arguments:
    df -- a dataframe object
    funcs -- a list of heuristic functions to be applied to entities
    
    """
    col = 'clean_ents'
    df[col] = df['comp_nouns']
    for f in funcs:
        df[col] = df[col].apply(f)


# In[ ]:


funcs = [drop_duplicate_np_splits, drop_double_char, keep_alpha, drop_single_char_nps, remove_freq_words]
add_clean_ents(df, funcs)
display(df)


# In[ ]:


def visualize_entities(df, idx=0):
    """Visualize the entities for a given abstract in the dataframe. 
    
    Keyword arguments:
    df -- a dataframe object
    idx -- the index of interest for the dataframe (default 0)
    
    """
    # store entity start and end index for visualization in dummy df
    ents = []
    abstract = df['text'][idx]
    for ent in df['clean_ents'][idx]:
        i = abstract.find(ent) # locate the index of the entity in the abstract
        ents.append((ent, i, i+len(ent), 'ENTITY')) 
    ents.sort(key=lambda tup: tup[1])

    dummy_df = pd.DataFrame([abstract, ents]).T # transpose dataframe
    dummy_df.columns = ['text', 'clean_ents']
    column = 'clean_ents'
    render_entities(0, dummy_df, options=options, column=column)


# In[ ]:


visualize_entities(df, 0)


# ### Analysis

# That's a good looking list of concepts wouldn't you say? By removing stop words and fine-tuning our set, we were able to capture only the most important entities in this first abstract! Let's finish up with a quick recapitulation of our approach and some thoughts on what we can do going forward. 

# ## Step 9: Celebrate with excessive fist-pumping

# Well, at the risk of tooting our own horn, I feel rather confident saying that we've accomplished what we set out to do! We took an abstract from a scientific paper, combined named and regular entities, extracted compound noun phrases, and pared down the final list using heuristics and stop word domain restriction to generate a set of important concepts. 

# <img src='https://media.giphy.com/media/t3Mzdx0SA3Eis/giphy.gif'><center><span style="font-size:10px;">Source: <a href="https://giphy.com/gifs/excited-the-office-yes-t3Mzdx0SA3Eis">GIPHY</a></span></center>

# Keep in mind that this exercise wasn't to create the world's best entity extractor. It was to get a fast baseline for what we can do with limited knowledge about the domain, and limited use of deep learning superpowers. We've now ended up with a prototype that shows we can get relatively far using out-of-the-box methods, with minor scripting for customization. And the best part? Our approach didn't require any extensive compute or proprietary software! 
# 
# Going forward, we'd want to test our approach on larger data sets (perhaps full scientific papers), and create an easy-to-use API for visualization, as well as individual and batch processing of text sources. Improving the actual entity extraction itself might involve a <strong>language model trained on academic papers</strong> or the addition of <strong>other intelligent heuristics</strong>. At some point, we'd also want to link each entity to an external database with further information, so that our conversational academic paper program would be able to orient these concepts within a larger knowledge graph. 
# 
# At the end of all of this, we've built a fast entity extraction prototype that confidently moves us towards creating an engine to communicate with academic papers, which will (hopefully) set the foundation for an automated scientific discovery tool.
# 
# Great work! 

# In[ ]:




