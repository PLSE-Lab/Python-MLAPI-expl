#!pip install Whoosh
# %% [code]
import os.path
import pandas as pd
import json

import pip

#install whoosh
def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])
install('Whoosh')

from whoosh.index import * # full-text indexing and searching
from whoosh.fields import *
from whoosh import qparser


class Indexer(object):
    def __init__(self, df_mdata, index_dir, corpus_dir):
        self.df_mdata = df_mdata
        self.index_dir = index_dir
        self.path_files = corpus_dir

        # Schema definition:
        # - id: type ID, unique, stored; cord_uid + "##abs" for abstract, and "##pmc-N" or "##pdf-N" for paragraphs in body text (Nth paragraph)
        # - path: type ID, stored; path to the JSON file (only for papers with full text)
        # - title: type TEXT processed by StemmingAnalyzer; not stored; title of the paper
        # - text: type TEXT processed by StemmingAnalyzer; not stored; content of the abstract section or the paragraph
        self.schema = Schema(id = ID(stored=True,unique=True),
                        path = ID(stored=True),
                        title = TEXT(analyzer=analysis.StemmingAnalyzer()),
                        text = TEXT(analyzer=analysis.StemmingAnalyzer())
                       )
        self.ix = None
        self.writer = None

    def load_index(self):
        self.ix = open_dir(self.index_dir)

    def create_index(self):
        # Create an index
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
        
        # Add papers to the index, iterating through each row in the metadata dataframe
        self.ix = create_in(self.index_dir, self.schema)
        self.writer = self.ix.writer()

        not_indexed = []
        indexed_sha = []

        for ind in self.df_mdata.index: 
            indexed = False

            # If paper has an abstract, index the abstract
            if not pd.isnull(self.df_mdata.loc[ind,'abstract']):
                if pd.isnull(self.df_mdata.loc[ind,'title']):
                    self.df_mdata.at[ind,'title'] = ""
                # Add document to the index
                self.writer.add_document(id=ind+"##abs", title=self.df_mdata['title'][ind], text=self.df_mdata['abstract'][ind])
                indexed = True

            # If paper has PMC or PDF full text, access its JSON file and index each paragraph separately
            # First check if paper has PMC xml
            if self.df_mdata['has_pmc_xml_parse'][ind] == True:
                if pd.isnull(self.df_mdata.loc[ind,'title']):
                    self.df_mdata.at[ind,'title'] = ""

                # Find JSON file: path specified in 'full_text_file', file name specidfied in 'pmcid'
                path_json = self.path_files + '/' + self.df_mdata['full_text_file'][ind] + '/' + self.df_mdata['full_text_file'][ind] + '/pmc_json/' + self.df_mdata['pmcid'][ind] + '.xml.json'
                with open(path_json, 'r') as j:
                    jsondata = json.load(j)

                    ## Iterate through paragraphs of body_text
                    for p, paragraph in enumerate(jsondata['body_text']):  
                        # Add document to the index
                        self.writer.add_document(id=ind+"##pmc-" + str(p), path = path_json, title=self.df_mdata['title'][ind], 
                                                 text=paragraph['text'])
                        indexed = True

            # As current paper does not have PMC, check if it has JSON PDF
            elif self.df_mdata['has_pdf_parse'][ind] == True:
                if pd.isnull(self.df_mdata.loc[ind,'title']):
                    self.df_mdata.at[ind,'title'] = ""

                # Find JSON file: path specified in 'full_text_file', file name specidfied in 'sha'
                # There could be more than one reference in 'sha' separated by ;
                shas = self.df_mdata['sha'][ind].split(';')
                for sha in shas:
                    sha = sha.strip()
                    # Check if paper with this sha has been indexed already
                    if sha not in indexed_sha:
                        indexed_sha.append(sha)
                        path_json = self.path_files + '/' + self.df_mdata['full_text_file'][ind] + '/' + self.df_mdata['full_text_file'][ind] + '/pdf_json/' + sha + '.json'
                        with open(path_json, 'r') as j:
                            jsondata = json.load(j)

                            ## iterate through paragraphs of body_text
                            for p, paragraph in enumerate(jsondata['body_text']):  
                                # Add document to the index
                                self.writer.add_document(id=ind+"##pdf-" + str(p), path = path_json, title=self.df_mdata['title'][ind], 
                                                         text=paragraph['text'])
                                indexed = True

            if not indexed:
                not_indexed.append(ind)
                
        # Save the added documents
        self.writer.commit()
        print("Index successfully created")

        # Sanity check
        print("Number of documents (abstracts and paragraphs of papers) in the index: ", self.ix.doc_count())
        print("Number of papers not indexed (because they don't have neither the abstract nor full text): ", len(not_indexed))


    def retrieve_documents(self, qstring, topn=20):
        # Open the searcher for reading the index. The default BM25F algorithm will be used for scoring
        with self.ix.searcher() as searcher:
            searcher = self.ix.searcher()
            # TODO: ZE EREMUTAN BILAKETA?? OR edo AND?
            q = qparser.QueryParser("text", self.ix.schema, group=qparser.OrGroup).parse(qstring)

            # Search using the query q, and get the topn documents, sorted with the highest-scoring documents first
            results = searcher.search(q, limit=topn)
            #print("Query: ", q)
            #print("Number of retrieved documents: ", len(results))

        # results is a list of dictionaries where each dictionary is the stored fields of the document (id, heading).
        # 'title' and text' are not stored

        # Create dataframe columns for id, sha, date, title, heading, text and score
        ids = []
        dates = []
        titles = []
        journals = []
        texts = []
        scores = []
        for hit in results:
            # Add id to a dataframe column
            ids.append(hit['id'])

            # As year, title and text are not stored in the index, they are not returned in results object.
            # They have to be extracted from metadata
            # Get paper id and type of section (abstract, full text paragraph)
            pid, sect = hit['id'].split("##")  # id examples: 'vho70jcx##sect1', a5x5ga60##abs

            # Add year to a dataframe column
            if pd.isnull(self.df_mdata.loc[pid, 'publish_time']):
                dates.append("")
            else:
                dates.append(self.df_mdata['publish_time'][pid])

            # Add journal to a dataframe column
            if pd.isnull(self.df_mdata.loc[pid, 'journal']):
                journals.append("")
            else:
                journals.append(self.df_mdata['journal'][pid])

            # Add title to a dataframe column (with link to the doi, if exists)
            if pd.isnull(self.df_mdata.loc[pid, 'url']):
                titles.append(self.df_mdata['title'][pid])
            else:
                titles.append(
                    "<a target=blank href=\"" + self.df_mdata['url'][pid] + "\">" + self.df_mdata['title'][pid] + "</a>")

            # Add text to a dataframe column
            if sect == 'abs': # get text of the abstract (reading from metadata)
                texts.append(self.df_mdata['abstract'][pid])
            else: # get text of the paragraph (reading from a JSON file)
                # get pmc or pdf, and the number of paragraph in body full text
                json_type,nsect = sect.split("-") # sect examples: 'pmc-1', 'pdf-5'
    
                # path of the JSON file whether text has been extracted from PMC or PDF
                #if json_type == 'pmc':
                #    path_json = path_dataset + df_metadata['full_text_file'][pid] + '/' + df_metadata['full_text_file'][pid] + '/pmc_json/' + df_metadata['pmcid'][pid] + '.xml.json'    
                #else: 
                #    path_json = path_dataset + df_metadata['full_text_file'][pid] + '/' + df_metadata['full_text_file'][pid] + '/pdf_json/' + df_metadata['sha'][pid] + '.json'
                with open(hit['path'], 'r') as j:
                    jsondata = json.load(j)
                    texts.append(jsondata['body_text'][int(nsect)]['text'])

            # Add score to a dataframe column
            scores.append(hit.score)

        # Create a dataframe of results with the columns
        df_results = pd.DataFrame()
        df_results['id'] = ids
        df_results['date'] = dates
        df_results['journal'] = journals
        df_results['title'] = titles
        df_results['text'] = texts
        df_results['score'] = scores

        return df_results


if __name__ == '__main__':
    
    mdata_f = '/kaggle/input/CORD-19-research-challenge/metadata.csv'
    corpus_d = '/kaggle/input/CORD-19-research-challenge/'
    index_d = '/kaggle/working/index'

    # Select interesting fields from metadata file
    fields = ['cord_uid','title', 'authors', 'publish_time', 'abstract', 'journal','url', 
              'has_pdf_parse', 'has_pmc_xml_parse', 'pmcid', 'full_text_file', 'sha']

    # Extract selected fields from metadata file into dataframe
    df_mdata = pd.read_csv(mdata_f, skipinitialspace=True, index_col='cord_uid', usecols=fields)

    # WARNING: cord_uid is described as unique, but c4u0gxp5 is repeated. So I remove one of this
    df_mdata = df_mdata.loc[~df_mdata.index.duplicated(keep='first')]
    df_mdata['publish_time'] = pd.to_datetime(df_mdata['publish_time'], errors="coerce")
    df_mdata['publish_year'] = df_mdata['publish_time'].dt.year
    df_mdata = df_mdata[df_mdata['abstract'].notna()]
    df_mdata = df_mdata[df_mdata['authors'].notna()]
    # df_mdata = df_mdata[df_mdata['sha'].notna()]
    df_mdata['authors'] = df_mdata['authors'].apply(lambda row: str(row).split('; '))

    # DEBUGGING: get smaller dataset
    df_small = df_mdata.head(5)
    
    indexer = Indexer(df_small, index_d, corpus_d)
    print('creating index...')
    indexer.create_index()
    print(' done.')
    #indexer.load_index()

    qstring = "What is known about transmission, incubation, and environmental stability?"
    print('retrieve query:', qstring)
    df_result = indexer.retrieve_documents(qstring, topn=2)
    print(df_result)
