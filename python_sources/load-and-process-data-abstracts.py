#!/usr/bin/env python
# coding: utf-8

# ### Data Extraction and Enrichment
# This notebook loads and processes the publications to extract, clean and load the contents into a class *document* and enrich with entity pairs for further analysis.
# Entity extraction is enabled by the [SpaCy](https://spacy.io/) library and pre-trained NLP modules and text standardization using the [NLTK](https://www.nltk.org/) library.
# The medical text in the publications contains several abbreviations which can be difficult to analyse with standard NLP techniques and difficult to read and compare. We try to decode these with the [Sci-SpaCy](https://allenai.github.io/scispacy/) library.  
# 
# The document object contains the origional text data and select meta data and enriched features to aid in the generation of knowledge graphs for furture analyis. 
# 
# The processed data is exported as a list documents objects and saved as a compressed *.pickle* file, one for each directory of json files. This processing is time comsuming and is decoupled form analysis to speed up experimentation. 

# In[ ]:


from covid19_data_processing_module import *


# ## Run Code to Load and Process Data Abstracts

# In[ ]:


# recreate the schema from "json_schema.txt"
class author():
    
    def __init__(self, input_dict=None):
        
        self.first = ""
        self.middle = []
        self.last = ""
        self.suffix = ""
        self.affiliation = {}
        self.email = ""
        
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
        
        print("first: " + str(self.first) +  
              ", middle: " + str(self.middle) + 
              ", last: " + str(self.last) + 
              ", suffix: " + str(self.suffix) +
              ", email: " + str(self.email) + 
              ", affiliation: " + json.dumps(self.affiliation, indent=4, sort_keys=True)
             )


class inline_ref_span():
    
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

        
# The following Class Definition is a useful helper object to store various 
# different covid-19 data types.
class document():
    
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
        
        if file_path:
            
            with open(file_path) as file:
                data = json.load(file)
                
                # call inbuilt data loading functions
                self.doc_filename = file_path
                self._load_paper_id(data)
                self._load_title(data)
                self._load_authors(data)
                self._load_abstract(data)
                #self._load_body_text(data) - temp removal to save memory for later scripts
                self._load_bib(data)
                self._load_ref_entries(data)
                self._load_back_matter(data)
    
    def combine_data(self):
        
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
        
        self.combine_data()

        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(dir, 'w') as json_file:
            json_file.write(json.dumps(self.data))

    def load_saved_data(self, dir):
        
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
        # set the doc language based on the analysis of the first block within the abstract
        if self.text:
            self.doc_language = get_text_language(self.text[0].text)
        elif self.abstract:
            self.doc_language = get_text_language(self.abstract[0].text)
        else:
            self.doc_language = get_text_language(self.title)


# Apply extraction, cleaning and entity pair identifycation function to each of the publications in the corpus. 

# In[ ]:


def process_document(file_path):
    
    doc = None
    
    try:
        # create doc obj
        doc = document(file_path)
    except:
        print("Error Processing: " + file_path)
        return None
    
    # identify doc language
    try:
        doc.set_doc_language()
    except:
        print("Error set_doc_language: " + file_path)
        
    # process abstract and text
    try:
        doc.set_abstract_tripples()
    except:
        print("Error set_abstract_tripples: " + file_path)
    
    try: 
        doc.set_ref_tripples()
    except:
        print("Error set_ref_tripples: " + file_path)
    
    
    return doc


# Run a silgle file to test code

# In[ ]:


# dir_input_data = '/kaggle/input/CORD-19-research-challenge/'
#
# file_list = []
# for dirname, _, filenames in os.walk(dir_input_data):
#     
#     files = [names for names in filenames if '.json' in names]    
# 
#     files = files[0:10]
#     for file in files:
#         input_file = os.path.join(dirname, file)
#     
#         #process abstracts in file
#         print(input_file)
#         doc = process_document(input_file)
#         if doc:
#             print("    Title: " + doc.title)
#             print("    Language: " +  str(doc.doc_language["language"]))


# Process all files

# In[ ]:


dir_input_data = '/kaggle/input/CORD-19-research-challenge'

file_list = []
for dirname, _, filenames in os.walk(dir_input_data):
    
    data = []
    files = [names for names in filenames if '.json' in names]    
    
    for file in files:
        
        input_file = os.path.join(dirname, file)
        
        #process abstracts in file
        doc = process_document(input_file)
        if doc:
            data.append(doc)
    
    if data:
        
        # format input and output file paths
        dirname_split = dirname.split("/")
        filename = Path(dirname_split[-3] + "_" + dirname_split[-2] + "_" + dirname_split[-1]).with_suffix('.pickle')

        # Save File to output folder
        with open(filename,"wb") as f:
            pickle.dump(data, f)

