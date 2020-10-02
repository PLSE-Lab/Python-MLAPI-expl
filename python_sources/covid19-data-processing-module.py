# This Python 3 module has a select set of functions used to process data within the covid-19 grand data challenge.

# install additional packages and download NLP model for processing.
import os
import subprocess
import sys
import inspect
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import errno
import json
import pickle
import string
import spacy
import unicodedata
import nltk.corpus
from pathlib import Path
from tqdm.notebook import tqdm

# script based install of packages using pip subprocess 
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install scispacy and spacy-langdetect
install("scispacy")
install("spacy-langdetect")
from scispacy.abbreviation import AbbreviationDetector
from spacy_langdetect import LanguageDetector

# download NLTK utilities 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
stop = stopwords.words('english')

# install spacy language model 
os.system("python -m spacy download en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

# extend nlp with Language detection functionality
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]


# The following Text Cleaning Functions have some generic function for text standardization,
# custom synmon replacment functions as well as bespoke regex functions to clean the 
# publication lists. These can be expanded further to improve the quality of the extracted 
# information.  


def replaceAccentedChar(s):
    """
    Basic function to replace accented char from input string

    :param s: input string
    :return: cleaned string
    """
    
    if s is None:
        return None
    else:
        return ''.join((c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c) != "Mn"))

    
def removePunctuationFromString(s):
    """
    Function to remove punctuationd from input string

    :param s: input string
    :return: cleaned string
    """
    
    if s is None:
        return None
    else:
        table = str.maketrans({key: None for key in string.punctuation})
        return s.translate(table)

    
def formatAbbreviations(abbreviations_by_):
    """
    Fhelper function used to format abbreviations from input string

    :param abbreviations_by_: input string
    :return: formatted abbreviations
    """
    
    for k in abbreviations_by_.keys():
        temp = abbreviations_by_[k]
        abbreviations = []
        
        for k2 in temp.keys():
            abbreviations.append(k2)
            abbreviations.extend(temp[k2])
            abbreviations_by_[k] = abbreviations
    
    return abbreviations_by_


def synonymFileLoader(synonym_file_path):
    """
    Helper function used to load a synonyms file

    :param synonym_file_path: full input file path & name
    :return: dict ofsynonyms
    """
    fp = open(synonym_file_path)
    
    synonyms_dict = dict()
    lines = [line.rstrip("\n") for line in fp]
    for line in lines:
        words = line.split()
        
        if words:
            if words[0] != "#":
                words = line.split(",")
                words = [w.strip() for w in words]
                temp = dict()
                temp[words[0]] = words[1:]
                synonyms_dict.update(temp)

    return synonyms_dict


def swapStringUsingDict(s, swap_dict):
    """
    Function used to replace element in a string with swap item in a dictionary

    :param s: input string
    :param swap_dict: item to swap out
    :return: cleaned string
    """
    
    replace_list = []
    s = " " + s + " "
    
    for k in sorted(swap_dict, key=len, reverse=True):
        delta_len = 0
        
        # use iter search and account for . wilcard
        k_hat = re.sub("\.", "\.", k)
        
        for m in re.finditer(k_hat.lower(), s.lower()):
            # loop over finding every single hit per loop 
            position = m.span()
            replace_count = "{" + str(len(replace_list)) + "}"
            s = s[:position[0] - delta_len] + replace_count + s[position[1]- delta_len:]
            
            delta_len += position[1] - position[0] - len(replace_count)
            replace_list.append(" " + swap_dict[k] + " ")
        
    if replace_list:
        s = s.format(*replace_list)
    
    return s.strip()


def swapStringUsingSynonymsList(s, swap_dict):
    """
    Function used to replace element in a string with swap item (synonyms list) in a dictionary

    :param s: input string
    :param swap_dict: synonym items to swap out
    :return: cleaned string
    """
    
    synonym_dict = {}
    for value in swap_dict.keys():
        [synonym_dict.update({k:value}) for k in swap_dict[value]]

    return swapStringUsingDict(s, synonym_dict)


def word_lemmatizer(s):
    """
    basic function used to envoke NLTK word lemmatizer

    :param s: input string
    :return: lemmatized string
    """
    
    s_lem = " ".join([WordNetLemmatizer().lemmatize(word) for word in s.split()])
    return s_lem


def clean(txt_in, swap_dict=None):
    """
    Function to apply set number of string cleaning operations.

    :param txt_in: text string to be processed
    :return: cleaned string, txt
    """
    
    # lower case
    txt = txt_in.lower()
    
    # replace accented char 
    txt = replaceAccentedChar(txt)
    
    # apply regex cleaning logic
    txt = re.sub(r'.\n+', '. ', txt)  # replace multiple newlines with period
    txt = re.sub(r'\n+', '', txt)  # replace multiple newlines with period
    txt = re.sub(r'\[\d+\]', ' ', txt)  # remove reference numbers
    txt = re.sub(' +', ' ', txt)
    txt = re.sub(',', ' ', txt)
    txt = re.sub(r'\([^()]*\)', '', txt)
    txt = re.sub(r'https?:\S+\sdoi', '', txt)
    txt = re.sub(r'biorxiv', '', txt)
    txt = re.sub(r'preprint', '', txt)
    txt = re.sub(r':', ' ', txt)
    
    # swap words with synonym
    if swap_dict:
        txt = swapStringUsingSynonymsList(txt, swap_dict)
    
    # use NLTK to remove stop words and apply Lemmatization
    txt = ' '.join([word for word in txt.split() if word not in (stop)])
    
    # use NLTK to apply Lemmatization
    txt = word_lemmatizer(txt)
    
    return txt

 
# The following Entity Extraction Functions are derived by the work of Christan Thorton 
# summaries in [this](https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81) medium post. 


def filter_spans(spans):
    """
    Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.
    
    :param spans (iterable): The spans to filter.
    :return (list): The filtered spans.
    """
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def refine_ent(ent, sent):
    """
    Filter any unwanted entity types

    :param ent: spacy entity object
    :param sent: spacy sentence object
    :return: entity, entity type
    """
    
    unwanted_tokens = (
        'PRON',  # pronouns
        'PART',  # particle
        'DET',  # determiner
        'SCONJ',  # subordinating conjunction
        'PUNCT',  # punctuation
        'SYM',  # symbol
        'X',  # other
    )
    ent_type = ent.ent_type_  # get entity type
    if ent_type == '':
        ent_type = 'NOUN_CHUNK'
        ent = ' '.join(str(t.text) for t in
                       nlp(str(ent)) if t.pos_
                       not in unwanted_tokens and t.is_stop == False)
    elif ent_type in ('ENTITY', 'NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
        t = ''
        for i in range(len(sent) - ent.i):
            if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                t += ' ' + str(ent.nbor(i))
            else:
                ent = t.strip()
                break
    return ent, ent_type


def get_sent_list(text):
    """
    Perform spacy processing and break document into sentences
    
    :param text: document text
    :return: list of sentences
    """
    
    text = clean(text)
    text = nlp(text)
    spans = list(text.ents) + list(text.noun_chunks)  # collect nodes
    spans = filter_spans(spans)
    with text.retokenize() as retokenizer:
        [retokenizer.merge(span) for span in spans]

    sentences = [sent for sent in text.sents]  # split text into sentences

    return sentences


def get_sent_entity_pairs(sent):
    """
    Extract entity pairs from sentences
    
    :param sent: spacy sentenace object
    :return: entity pair list of lists
    
    """
    ent_pairs = []
    dep = [token.dep_ for token in sent]
    try:
        if sum([dep.count(object) for object in OBJECTS]) == 1 \
                and sum([dep.count(object) for object in SUBJECTS]) == 1:
            for token in sent:
                if token.dep_ in ('obj', 'dobj'):  # identify object nodes
                    subject = [w for w in token.head.lefts if w.dep_
                               in ('subj', 'nsubj')]  # identify subject nodes
                    if subject:
                        subject = subject[0]
                        # identify relationship by root dependency
                        relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                        if relation:
                            relation = relation[0]
                            # add adposition or particle to relationship
                            if relation.nbor(1).pos_ in ('ADP', 'PART'):
                                relation = ' '.join((str(relation),
                                                     str(relation.nbor(1))))
                        else:
                            relation = 'unknown'
                        subject, subject_type = refine_ent(subject, sent)
                        token, object_type = refine_ent(token, sent)
                        ent_pairs.append([str(subject), str(relation), str(token),
                                          str(subject_type), str(object_type)])
    except:
        pass

    return ent_pairs


def entity_extract(sent):
    """
    Extract entity and entity type from sentence tokens
    
    :param sent: spacy sentence object
    :return: entity, entity type dictionary
    
    """
    ent_types = {}
    for token in sent:
        if token.ent_type_ is not '':
            ent_types[token.text] = token.ent_type_

    return ent_types


def get_entity_pairs(text):
    """
    Process text and get entity paris and entities within the text string.
    
    :param text: document
    :return: enitiy pairs datframe and dictionary of entities
    """
    
    sents = get_sent_list(text)
    pairs = list(map(get_sent_entity_pairs, sents))
    ents = list(map(entity_extract, sents))
    flatten_pairs = [item for pair in pairs for item in pair]
    pairs = pd.DataFrame(flatten_pairs, columns=['subject', 'relation', 
                                                 'object', 'subject_type',
                                                 'object_type'])
    entities = {}
    for l in ents:
        entities.update(l)

    return pairs, entities


# The following functions are utilities for common Text Analysis tasks

def get_text_language(text):
    """
    Process text and identify the language used, return language identified by spacy
    
    :param text: text string
    :return: language of text
    """
    
    doc = nlp(text)
    return doc._.language


# The below code is used to define the *document* class and its component parts.  


class author():
    """
        An object that captures the properties of an author
    """
    
    def __init__(self, input_dict=None):

        self.first = ""
        self.middle = []
        self.last = ""
        self.suffix = ""
        self.affiliation = {}
        self.email = ""
        
        # if input_dict is not None, cycle through keys and set the 
        # elements withtin class if they are included within the dict
        if input_dict:
            for key in input_dict.keys():
                if "first" in key:
                    self.first = input_dict[key]
                if "middle" in key:
                    self.middle = input_dict[key]
                if "last" in key:
                    self.last = input_dict[key]
                if "suffix" in key:
                    self.suffix = input_dict[key]
                if "affiliation" in key:
                    self.affiliation = input_dict[key]
                if "email" in key:
                    self.email = input_dict[key]    
    
    def print_items(self):

        # utility function to print each element within the author class
        print("first: " + str(self.first) +  
              ", middle: " + str(self.middle) + 
              ", last: " + str(self.last) + 
              ", suffix: " + str(self.suffix) +
              ", email: " + str(self.email) + 
              ", affiliation: " + json.dumps(self.affiliation, indent=4, sort_keys=True)
             )


class inline_ref_span():
    """
        A utility object used to capture the properties of a within line cross reference
    """
    
    def __init__(self, input_dict=None):
        
        self.start = 0
        self.end = 0
        self.text = ""
        self.ref_id = ""
        
        if input_dict:
            for key in input_dict.keys():
                if "start" in key:
                    self.start = input_dict[key]
                if "end" in key:
                    self.end = input_dict[key]
                if "text" in key:
                    self.text = input_dict[key]
                if "ref_id" in key:
                    self.ref_id = input_dict[key]
    
    def print_items(self):
        
        print("Text: " + str(self.text) + ", Start: " + 
              str(self.start) + ", End: " + str(self.end) + 
              ", Ref_id: " + str(self.ref_id))

    def step_index(self, n):
        
        self.start += n
        self.end += n
        
        
class text_block():
    """
        An object that captures the properties of a text block
    """
    
    def __init__(self, input_dict=None):
        
        self.text = ""
        self.cite_spans = []
        self.ref_spans = []
        self.eq_spans = []
        self.section = ""
        
        if input_dict:
            for key in input_dict.keys():
                if "text" in key:
                    self.text = input_dict[key]
                if "cite_spans" in key:
                    self.cite_spans = [inline_ref_span(c) for c in input_dict[key]]                
                if "ref_spans" in key:
                    self.ref_spans = [inline_ref_span(r) for r in input_dict[key]] 
                if "eq_spans" in key:
                    self.eq_spans = [inline_ref_span(e) for e in input_dict[key]]
                if "section" in key:
                    self.section = input_dict[key]
        
    def clean(self, swap_dict=None):
        
        # apply set cleaning logic from covid19_data_processing_module
        self.text = clean(self.text, swap_dict)
    
    def print_items(self):
        
        print("\ntext: " + str(self.text))
        print("\nsection: " + str(self.section))
        print("\ncite_spans: ")
        [c.print_items() for c in self.cite_spans]
        print("\nref_spans: ")
        [r.print_items() for r in self.ref_spans]
        print("\neq_spans: ")
        [e.print_items() for e in self.eq_spans]


def combine_text_block(text_block_list):
    """
        A utility function used to combine a list of text_blocks into a single item within a list
        
        :param text_block_list: input list of text_block objects
        :return: A text_block object with the combined contents of inputted list 
    """
    
    if text_block_list:
        
        combined_block = text_block_list[0]
        block_length = len(combined_block.text)
        
        for i in range(1,len(text_block_list)):
            combined_block.text += " " + text_block_list[i].text
            block_length += 1
            
            # update spans start & stop index
            [ref.step_index(block_length) for ref in text_block_list[i].cite_spans]
            [ref.step_index(block_length) for ref in text_block_list[i].ref_spans]
            [ref.step_index(block_length) for ref in text_block_list[i].eq_spans]
            
            # combine spans
            combined_block.cite_spans += text_block_list[i].cite_spans
            combined_block.ref_spans += text_block_list[i].ref_spans
            combined_block.eq_spans += text_block_list[i].eq_spans           
            combined_block.section += ", " + str(text_block_list[i].section)           
            
            block_length += len(text_block_list[i].text)
                       
        return [combined_block]
    else:
        return [text_block()]
      

class bib_item():
    """
        An object that captures the properties of a bibliography item
    """
    
    def __init__(self, ref_id=None, input_dict=None):
        
        self.ref_id: ""
        self.title: ""
        self.authors = []
        self.year = 0
        self.venue = ""
        self.volume = ""
        self.issn = ""
        self.pages = ""
        self.other_ids = {}
        
        # ref_if is the reference key used with the original document dict.
        if ref_id:
            self.ref_id = ref_id
            
            if input_dict:
                for key in input_dict.keys():
                    if "title" in key:
                        self.title = input_dict[key]
                    if "authors" in key:
                        self.authors = [author(a) for a in input_dict[key]]
                    if "year" in key:
                        self.year = input_dict[key]
                    if "venue" in key:
                        self.venue = input_dict[key]
                    if "volume" in key:
                        self.volume = input_dict[key]
                    if "issn" in key:
                        self.issn = input_dict[key]
                    if "pages" in key:
                        self.pages = input_dict[key]
                    if "other_ids" in key:
                        self.other_ids = input_dict[key]
    
    def print_items(self):
        
        print("\nBib Item:")
        print("ref_id: " + str(self.ref_id))
        print("title: " + str(self.title))
        print("Authors:")
        [a.print_items() for a in self.authors]
        print("year: " + str(self.year))
        print("venue: " + str(self.venue))
        print("issn: " + str(self.issn))
        print("pages: " + str(self.pages))
        print("other_ids: " + json.dumps(self.other_ids, indent=4, sort_keys=True))
        
        
class ref_entries():
    """
        An object that captures the properties of a cross references within the document
    """
    
    def __init__(self, ref_id=None, input_dict=None):
        
        self.ref_id = ""
        self.text = ""
        self.latex = None
        self.type = ""
        
        if ref_id:
            self.ref_id = ref_id
            
            if input_dict:
                for key in input_dict.keys():
                    if "text" in key:
                        self.text = input_dict[key]
                    if "latex" in key:
                        self.latex = input_dict[key]
                    if "type" in key:
                        self.type = input_dict[key]
    
    def print_items(self):
        
        print("ref_id: " + str(self.ref_id))
        print("text: " + str(self.text))
        print("latex: " + str(self.latex))
        print("type: " + str(self.type))
        
                    
class back_matter():
    """
        An object that captures the properties of back matter, such as the document appendix
    """
    
    def __init__(self, input_dict=None):
        
        self.text = ""
        self.cite_spans = []
        self.ref_spans = []
        self.section = ""
        
        if input_dict:
            for key in input_dict.keys():
                if "text" in key:
                    self.text = input_dict[key]
                if "cite_spans" in key:
                    self.cite_spans = [inline_ref_span(c) for c in input_dict[key]]                
                if "ref_spans" in key:
                    self.ref_spans = [inline_ref_span(r) for r in input_dict[key]] 
                if "section" in key:
                    self.section = input_dict[key]
    
    def print_items(self):
        
        print("text: " + str(self.text))
        print("cite_spans: ")
        [c.print_items() for c in self.cite_spans]
        print("ref_spans: ")
        [r.print_items() for r in self.ref_spans]        
        print("section:" + str(self.section))


class document():
    """
        The following Class Definition is a useful helper object to 
        store various different covid-19 data types.
        
        It uses among others:
            - author class
            - inline_ref_span class
            - text_block class
            - bib_item class
            - ref_entries class
            - back_matter class
    """
    
    def __init__(self, file_path=None):
        
        self.doc_filename = ""
        self.doc_language = {}
        self.paper_id = ""
        self.title = ""
        self.authors = []
        self.abstract = []
        self.text = []
        self.bib = []
        self.ref_entries = []
        self.back_matter = []
        self.tripples = {}
        self.key_phrases = {}
        self.entities = {}
        
        # load content from file on obj creation
        self.load_file(file_path)
    
    def _load_paper_id(self, data):
        
        # utility load function
        if "paper_id" in data.keys():
            self.paper_id = data['paper_id']
        
    def _load_title(self, data):
        
        if "metadata" in data.keys():
            if "title" in data['metadata'].keys():
                self.title = data['metadata']["title"]
    
    def _load_authors(self, data):
        
        if "metadata" in data.keys():
            if "authors" in data['metadata'].keys():
                self.authors = [author(a) for a in data['metadata']["authors"]]
                
    def _load_abstract(self, data):
        
        if "abstract" in data.keys():
            self.abstract = [text_block(a) for a in data["abstract"]]
    
    def _load_body_text(self, data):
        
        if "body_text" in data.keys():
            self.text = [text_block(t) for t in data["body_text"]]
    
    def _load_bib(self, data):
        
        if "bib_entries" in data.keys():
            self.bib = [bib_item(b, data["bib_entries"][b]) for b in data["bib_entries"].keys()]
    
    def _load_ref_entries(self, data):
        
        if "ref_entries" in data.keys():
            self.ref_entries = [ref_entries(r, data["ref_entries"][r]) for r in data["ref_entries"].keys()]
            
    def _load_back_matter(self, data):
        
        if "back_matter" in data.keys():
            self.back_matter = [back_matter(b) for b in data["back_matter"]]
        
    def load_file(self, file_path):
        
        # Read a CORD-19-research-challenge directory .json file, add content into 
        # corresponding document class elements 
        
        if file_path:
            
            with open(file_path) as file:
                data = json.load(file)
                
                # call inbuilt data loading functions
                self.doc_filename = file_path
                self._load_paper_id(data)
                self._load_title(data)
                self._load_authors(data)
                self._load_abstract(data)
                
                # for speed of processing, in this instance we have ommited loading 
                # the text boady as it unused within later notebooks. To load the 
                # text_body content, uncomment the below line of code:
                #self._load_body_text(data)
                
                self._load_bib(data)
                self._load_ref_entries(data)
                self._load_back_matter(data)
    
    def combine_data(self):
        
        # a utility function to combine class elements into a single 
        # data field
        self.data = {'doc_filename': self.doc_filename,
                     'doc_language': self.doc_language,
                     'paper_id': self.paper_id,
                     'title': self.title,
                     'authors':self.authors,
                     'abstract': self.abstract,
                     'text': self.text,
                     'bib_entries':self.bib,
                     'ref_entries': self.ref_entries,
                     'back_matter': self.back_matter,
                     'tripples': self.tripples,
                     'key_phrases': self.key_phrases,
                     'entities': self.entities}

    def extract_data(self):
        
        # a utility function to explode .data fiels into class elements. This 
        # effectively does the reverse document.combine_data()
        self.doc_filename = self.data['doc_filename']
        self.doc_language = self.data['doc_language']
        self.paper_id = self.data['paper_id']
        self.title = self.data['title']
        self.authors = self.data['authors']
        self.abstract = self.data['abstract']
        self.text = self.data['text']        
        self.bib = self.data['bib_entries']
        self.ref_entries = self.data['ref_entries']
        self.back_matter = self.data['back_matter']
        self.tripples = self.data['tripples']
        self.key_phrases = self.data['key_phrases']
        self.entities = self.data['entities']

    def save(self, dir):
        
        # a utility function save the contents of class into a file
        self.combine_data()
        
        # create the directory if needed
        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                    
        # open the file and write the data in string format using 
        # the json package 
        with open(dir, 'w') as json_file:
            json_file.write(json.dumps(self.data))

    def load_saved_data(self, dir):
        
        # a utility function to reload saved class contents contents
        with open(dir) as json_file:
            self.data = json.load(json_file)
        self.extract_data()
    
    def print_items(self):
         
        print("---- Document Content ----") 
        print("doc_filename: " + str(self.doc_filename))
        print("doc_language: " + str(self.doc_language))
        print("paper_id: " + str(self.paper_id))
        print("title: " + str(self.title))
        print("\nAuthors: ")
        [a.print_items() for a in self.authors]
        print("\nAbstract: ")
        [a.print_items() for a in self.abstract]
        print("\nText: ")
        [t.print_items() for t in self.text]
        print("\nBib_entries: ")
        [b.print_items() for b in self.bib]
        print("\nRef_entries: ")
        [r.print_items() for r in self.ref_entries]
        print("\nBack_matter: ")
        [b.print_items() for b in self.back_matter]
        
        print("\nTripples: ")
        print(json.dumps(self.tripples, indent=4, sort_keys=True))
        print("\nKey Phrases: ")
        print(json.dumps(self.key_phrases, indent=4, sort_keys=True))        
        print("\nEntities: ")
        print(json.dumps(self.entities, indent=4, sort_keys=True))

    def clean_text(self, swap_dict=None):
        
        # clean all blocks of text
        [t.clean(swap_dict) for t in self.text]
    
    def clean_abstract(self, swap_dict=None):
        
        [t.clean(swap_dict) for t in self.abstract]
    
    def combine_text(self):
        
        # this function takes all text blocks within document.text and combines them into a single text_block object
        self.text = combine_text_block(self.text)
    
    def combine_abstract(self):
        
        self.abstract = combine_text_block(self.abstract)   
    
    def set_title_tripples(self):
        
        # a utility function to extract pairwise entities from document.title
        title_tripples = {}
        pairs, entities = get_entity_pairs(self.title)
            
        #if any tripples found
        if pairs.shape[0]>0:
            title_tripples["title"] = pairs.to_json()     
            self.tripples.update(title_tripples)
        
    def set_abstract_tripples(self):
                
        abstract_tripples = {}
        for i in range(0, len(self.abstract)):
            #for every block in the abstract, extract entity tripples
            self.abstract[i].clean()                       
            pairs, entities = get_entity_pairs(self.abstract[i].text)
            
            #if any tripples found
            if pairs.shape[0]>0:
                abstract_tripples["abstract_" + str(i)] = pairs.to_json()
        
        if abstract_tripples:
            self.tripples.update(abstract_tripples)
        
    def set_text_tripples(self):
        
        text_tripples = {}
        for i in range(0, len(self.text)):
            
            self.text[i].clean()                       
            pairs, entities = get_entity_pairs(self.text[i].text)
            if pairs.shape[0]>0:
                text_tripples["text_" + str(i)] = pairs.to_json()
        
        if text_tripples:
            self.tripples.update(text_tripples)
        
    def set_ref_tripples(self):
        
        ref_tripples = {}
        for r in self.ref_entries:
            pairs, entities = get_entity_pairs(r.text)
            if pairs.shape[0]>0:
                ref_tripples["ref_" + r.ref_id] = pairs.to_json()
        
        if ref_tripples:
            self.tripples.update(ref_tripples)
        
    def set_doc_language(self):
        
        # a utility function to identify the language of the document using
        # the documents text, abstract, or title 
        if self.text:
            self.doc_language = get_text_language(self.text[0].text)
        elif self.abstract:
            self.doc_language = get_text_language(self.abstract[0].text)
        else:
            self.doc_language = get_text_language(self.title)