# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:35:08 2020

@author: shubotian
"""

# Loading packages
import os
import csv
import json
from tqdm import tqdm
from datetime import datetime
import html
from IPython.display import HTML
from itertools import combinations
from spacy.lang.en import English

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

from utils import *

from absl import logging

#logging.get_absl_handler().use_absl_log_file('absl_logging', 'befreeout')
#logging.set_verbosity(logging.INFO)


# pack the annotated papers into a search class
class SearchPapers(object):
    # initialization
    def __init__(self, data_path, json_path, mapping_pnid, index_year,
                 index_title, index_abstract, word_counts, index_table, paper_tables,
                 entity_lists, entity_nodes, entity_relations,
                 index_sents, mapping_sents, sentences):
        # set path 
        self.data_path = data_path
        self.json_path = json_path
        self.csv_path = 'csv'
        self.apikey = "your-pubmed-api-key"
        
        # load mapping file (6M)
        self.corduid2nid = json.load(open(f'{self.data_path}/{mapping_pnid}', 'r', encoding = 'utf-8'))
        self.nid2corduid = {v:k for k, v in self.corduid2nid.items()}
        self.all_papers = set([int(i) for i in self.corduid2nid.values()])
        # covid19_papers = json.load(open(f'{self.data_path}/{covid19_tag}', 'r', encoding = 'utf-8'))
        # self.covid19_papers = set(covid19_papers['covid19'])
        self.sentid2nid = json.load(open(f'{self.data_path}/{mapping_sents}', 'r', encoding = 'utf-8'))
        self.nid2sentid = {v:k for k, v in self.sentid2nid.items()}
        self.all_sents = set([int(i) for i in self.sentid2nid.values()])
        # self.covid19_sents = set([int(j) for i, j in self.sentid2nid.items() if i.split('|')[0] in self.covid19_papers])
        
        # load index files (320M)
        self.index_year = json.load(open(f'{self.data_path}/{index_year}', 'r', encoding = 'utf-8'))
        self.index_title = json.load(open(f'{self.data_path}/{index_title}', 'r', encoding = 'utf-8'))
        self.index_abstract = json.load(open(f'{self.data_path}/{index_abstract}', 'r', encoding = 'utf-8'))
        self.word_counts = json.load(open(f'{self.data_path}/{word_counts}', 'r', encoding = 'utf-8'))
        self.index_sents = json.load(open(f'{self.data_path}/{index_sents}', 'r', encoding = 'utf-8'))
        self.paper_tables = json.load(open(f'{self.data_path}/{paper_tables}', 'r', encoding = 'utf-8'))
        self.index_table = json.load(open(f'{self.data_path}/{index_table}', 'r', encoding = 'utf-8'))
        self.all_tables = set([f'{k}|{tid}' for w in self.index_table.values() for k, v in w.items() for tid in v])

        # load entity nodes, relations and sentences (760M)
        self.entity_lists = json.load(open(f'{self.data_path}/{entity_lists}', 'r', encoding = 'utf-8'))
        self.entity_nodes = json.load(open(f'{self.data_path}/{entity_nodes}', 'r', encoding = 'utf-8'))
        self.entity_relations = json.load(open(f'{self.data_path}/{entity_relations}', 'r', encoding = 'utf-8'))
        self.sentences = json.load(open(f'{self.data_path}/{sentences}', 'r', encoding = 'utf-8'))
        #embeddings = 
        #model = 


    def get_paper_id(self, paper_id):
        if paper_id.isdigit():
            return self.nid2corduid[int(paper_id)]
        return self.corduid2nid[paper_id]
        

    def get_paper(self, paper_id):
        if paper_id.isdigit():
            file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(paper_id)]}.json', 'r', encoding = 'utf-8'))
        else:
            file = json.load(open(f'{self.json_path}/{paper_id}.json', 'r', encoding = 'utf-8'))
        
        return file
        
        
    # search function
    def search_papers(self, user_query, section = 'tiabs', publish_year = None):
        """
        user_query: string of phrases, return list of paper ids for papers containing
                    anyone of phrases split by ','. Can have prefix follow by ':', e.g.
                    "age: young, middle, old" will search any one of "age young", "age middle",
                    "age old"
                    covid19 = False
        """
        # process query phrases
        if len(user_query.split(':', 1)) > 1:
            prefix = user_query.split(':', 1)[0].strip()
            phrases = user_query.split(':', 1)[1]
            phrases = [f'{prefix} {phrase.strip()}' for phrase in phrases.split(',')]
        else:
            phrases = [phrase.strip() for phrase in user_query.split(',')]
        
        # process searching
        search_return = {}
        for phrase in phrases:
            phrase_result = self.all_papers
            # if covid19:
            #     phrase_result = phrase_result.intersection(self.covid19_papers)
            if publish_year != None:
                if ',' in publish_year:
                    years = publish_year.split(',')
                    result = set()
                    for year in years:
                        result = result.union(set(self.index_year.get(str(year.strip()), [])))
                    phrase_result = phrase_result.intersection(result)
                elif '-' in publish_year:
                    years = publish_year.split('-')
                    result = set()
                    for year in range(int(years[0].strip()), int(years[1].strip())+1):
                        result = result.union(set(self.index_year.get(str(year), [])))
                    phrase_result = phrase_result.intersection(result)
                else:
                    result = set(self.index_year.get(str(publish_year.strip()), []))
                    phrase_result = phrase_result.intersection(result)
            
            doc = nlp(phrase.strip())
            for token in doc:
                #print(token)
                if token.is_stop or token.is_punct or token.is_digit: continue
                if section == 'ti':
                    result = set(self.index_title.get(token.lemma_.lower(), []))
                    phrase_result = phrase_result.intersection(result)
                elif section == 'abs':
                    result = set(self.index_abstract.get(token.lemma_.lower(), []))
                    phrase_result = phrase_result.intersection(result)
                elif section == 'table':
                    result = list(self.index_table.get(token.lemma_.lower(), {}).keys())
                    result = set([int(i) for i in result])
                    phrase_result = phrase_result.intersection(result)
                elif section == 'tiabs':
                    result_t = set(self.index_title.get(token.lemma_.lower(), []))
                    result_a = set(self.index_abstract.get(token.lemma_.lower(), []))
                    phrase_result = phrase_result.intersection(result_t.union(result_a))
                else:
                    result_t = set(self.index_title.get(token.lemma_.lower(), []))
                    result_a = set(self.index_abstract.get(token.lemma_.lower(), []))
                    result = list(self.index_table.get(token.lemma_.lower(), {}).keys())
                    result = set([int(i) for i in result])
                    phrase_result = phrase_result.intersection(result_t.union(result_a).union(result))
            
            # calculate word count for ranking
            if phrase_result:
                phrase_count = {}
                for paper in phrase_result:
                    for token in doc:
                        if token.is_stop or token.is_punct or token.is_digit: continue
                        phrase_count[paper] = phrase_count.get(paper, 0) + self.word_counts[str(paper)][token.lemma_.lower()]
    
                #search_result = search_result.union(phrase_result)
                for k, v in phrase_count.items():
                    search_return[k] = search_return.get(k, 0) + v
        
        # rank search return
        if search_return:
            search_return = sorted(search_return.items(), key = lambda x: x[1], reverse = True)
            search_return = [i[0] for i in search_return]
        else:
            search_return = []
        
        return search_return


    # paper title and abstract display function
    def display_papers(self, search_return):
        """
        search_return: a list of paper ids returned from search
        """
        for pid in search_return:
            file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(pid)]}.json', 'r', encoding = 'utf-8'))
            file['title']['ents'] = combine_entities(file['title'])
            file['abstract']['ents'] = combine_entities(file['abstract'])
            display_title(file)
            if file['abstract']['ents'] != []:
                displacy.render(get_section_doc(file['abstract'], self.entity_lists), style="ent")
            elif file['abstract']['text'] != '':
                print(file['abstract']['text'])
            else:
                print('\n')


    # save papers to csv
    def save_papers(self, search_return, file_name):
        with open(f"{self.csv_path}/{file_name}_{datetime.now().timestamp()}.csv", 'w', encoding = 'utf-8') as fcsv:
            csv_writer = csv.writer(fcsv)
            csv_writer.writerow(['Date', 'Study', 'Study Link', 'Journal', 'Study Type',
                                 'Chemicals', 'Sample Size', 'Severity', 'General Outcome',
                                 'Primary Endpoints', 'Clinical Improvement', 'Added On',
                                 'Abstract', 'DOI', 'CORD_UID', 'PMC_Link'])
            papers = {}
            for pid in search_return:
                file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(pid)]}.json', 'r', encoding = 'utf-8'))
                date = file['publish_time']
                study = file['title']['text']
                # get title chems
                chem_title = set()
                if ('ents' in file['title']) and (file['title']['ents'] != []):
                    for ent in file['title']['ents']:
                        if ent[3] == 'Chemical':
                            chem_title.add(ent[2].lower())
                chem_title = list(chem_title)
                study_link = file['url']
                journal = file['journal']
                abstract = file['abstract']['text']
                # get abstr chems
                chem_abs = set()
                if ('ents' in file['abstract']) and (file['abstract']['ents'] != []):
                    for ent in file['abstract']['ents']:
                        if ent[3] == 'Chemical':
                            chem_abs.add(ent[2].lower())
                chem_abs = list(chem_abs)
                chems = f"{', '.join(chem_title)}|{', '.join(chem_abs)}"
                # get sample size
                re_search = re.search(r'(?<=[\s,.;:!])([1-9]+[\d,]*)\s?.{,40}(patients|cases|records|studies)', abstract, flags = re.I)
                if re_search and re_search[1]:
                    sample_size = int(re_search[1].replace(',', ''))
                else:
                    sample_size = '-'
                # get gen outcome from abstr
                doc = nlp(abstract)
                if len(list(doc.sents)) >= 2:
                    gen_outcome = f"{list(doc.sents)[-2]} {list(doc.sents)[-1]}"
                else:
                    gen_outcome = abstract
                doi = file['doi']
                cord_uid = file['cord_uid']
                pmc_link = file['pmcid']
                if pmc_link != '':
                    pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_link}/"
                pmid = file['pmid']
                study_type = 'Other'
                if pmid != '':
                    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                    soup = BeautifulSoup(get_page(url), 'xml')
                    if soup.find('PublicationTypeList') != None:
                        study_type = ', '.join(list(soup.find('PublicationTypeList').stripped_strings))
                        study_type = 'Review' if 'review' in study_type.lower() else 'Other'
                papers[pid] = [date, study, study_link, journal, study_type, chems, sample_size, '',
                               gen_outcome, '', '', '', abstract, doi, cord_uid, pmc_link]
            rows = sorted(papers.values(), key=lambda x:x[4]+'#'+x[0], reverse=True)
            csv_writer.writerows(rows)
        
    
    # search sentences function
    def search_sentences(self, user_query):
        """
        user_query: string of phrases, return list of paper ids for papers containing
                    anyone of phrases split by ','. Can have prefix follow by ':', e.g.
                    "age: young, middle, old" will search any one of "age young", "age middle",
                    "age old"
                    covid19 = False
        """
        # process query phrases
        if len(user_query.split(':', 1)) > 1:
            prefix = user_query.split(':', 1)[0].strip()
            phrases = user_query.split(':', 1)[1]
            phrases = [f'{prefix} {phrase.strip()}' for phrase in phrases.split(',')]
        else:
            phrases = [phrase.strip() for phrase in user_query.split(',')]
        
        # process searching
        search_return = {} #set()
        for phrase in phrases:
            phrase_result = self.all_sents
            # if covid19:
            #     phrase_result = phrase_result.intersection(self.covid19_sents)

            doc = nlp(phrase.strip())
            for token in doc:
                #print(token)
                if token.is_stop or token.is_punct or token.is_digit: continue
                result = set(self.index_sents.get(token.lemma_.lower(), []))
                phrase_result = phrase_result.intersection(result)
            
            # calculate word count for ranking
            if phrase_result:
                phrase_count = {} #combine returned sentences into each paper
                for snid in phrase_result:
                    pnid, sec, sid = self.nid2sentid[int(snid)].split('|')
                    if pnid in phrase_count:
                        phrase_count[pnid]['sents'].append(snid)
                    else:
                        phrase_count[pnid] = {'count': 0, 'sents':[snid]}
                for paper in phrase_count:
                    for token in doc:
                        if token.is_stop or token.is_punct or token.is_digit: continue
                        phrase_count[paper]['count'] += self.word_counts[str(paper)][token.lemma_.lower()]
                
                for k, v in phrase_count.items():
                    if k in search_return:
                        search_return[k]['count'] += v['count']
                        search_return[k]['sents'].extend(v['sents'])
                    else:
                        search_return[k] = {'count': v['count'], 'sents': v['sents']}
                    
        # rank search return
        if search_return:
            search_return = sorted(search_return.items(), key = lambda x: x[1]['count'], reverse = True)
            search_return = list(set([snid for i in search_return for snid in i[1]['sents']]))
        else:
            search_return = []
            #search_return = search_return.union(phrase_result)
        return search_return


    # sentences display function
    def display_sentences(self, search_return):
        sents = {}
        for snid in search_return:
            pnid, sec, sid = self.nid2sentid[int(snid)].split('|')
            if pnid in sents:
                if sec in sents[pnid]:
                    sents[pnid][sec].append(int(sid))
                else:
                    sents[pnid][sec] = [int(sid)]
            else:
                sents[pnid] = {sec:[int(sid)]}
        
        for pnid, sids in sents.items():
            file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(pnid)]}.json', 'r', encoding = 'utf-8'))
            output_html = "<style type='text/css'>mark { background-color:yellow; color:black; } </style>"
            title = file['title']['text']
            authors = file['authors']
            if len(authors.split(';')) > 5:
                authors = f"{';'.join(authors.split(';')[:5])} et al."
            journal = file['journal']
            publish_time = file['publish_time']
            url = file['url']
            doi = file['doi']
            pmcid = file['pmcid']
            if pmcid != '':
                pmcid = f'<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/", target="_blank">{pmcid}</a>'
            pmid = file['pmid']
            abstract = file['abstract']['text']
            if 't' in sids:
                output_html += f'<h3><mark>{html.escape(title)}</mark></h3>'
            else:
                output_html += f'<h3>{html.escape(title)}</h3>'
            output_html += f'{html.escape(authors)}<br>'
            output_html += f'{html.escape(journal)}, {html.escape(publish_time)}, <a href="{url}", target="_blank">{url}</a><br>'
            output_html += f'doi: {html.escape(doi)}, PMCID: {pmcid}, PMID: {html.escape(pmid)}<br><br>'
            if 'a' in sids:
                doc = nlp(abstract)
                for sid, sent in enumerate(doc.sents):
                    if sid in sids['a']:
                        output_html += f'<mark><b>{html.escape(sent.text)}</b></mark>'
                    else:
                        output_html += f'{html.escape(sent.text)}'
            else:
                output_html += f'{html.escape(abstract)}'
            output_html += '<br><br>'
            display(HTML(output_html))


    # save sentences to csv
    def save_sentences(self, search_return, file_name):
        sents = {}
        for snid in search_return:
            pnid, sec, sid = self.nid2sentid[int(snid)].split('|')
            if pnid in sents:
                sents[pnid].append(self.nid2sentid[int(snid)])
            else:
                sents[pnid] = [self.nid2sentid[int(snid)]]
        with open(f"{self.csv_path}/{file_name}_{datetime.now().timestamp()}.csv", 'w', encoding = 'utf-8') as fcsv:
            csv_writer = csv.writer(fcsv)
            csv_writer.writerow(['sentence', 'Date', 'Study', 'Study Link', 'Journal', 'Study Type',
                                 'Chemicals', 'Sample Size', 'Severity', 'General Outcome',
                                 'Primary Endpoints', 'Clinical Improvement', 'Added On',
                                 'Abstract', 'DOI', 'CORD_UID', 'PMC_Link'])
            sents4write = {}
            for pnid, sids in sents.items():
                file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(pnid)]}.json', 'r', encoding = 'utf-8'))
                date = file['publish_time']
                study = file['title']['text']
                chem_title = set()
                if ('ents' in file['title']) and (file['title']['ents'] != []):
                    for ent in file['title']['ents']:
                        if ent[3] == 'Chemical':
                            chem_title.add(ent[2].lower())
                chem_title = list(chem_title)
                study_link = file['url']
                journal = file['journal']
                abstract = file['abstract']['text']
                # get abstr chems
                chem_abs = set()
                if ('ents' in file['abstract']) and (file['abstract']['ents'] != []):
                    for ent in file['abstract']['ents']:
                        if ent[3] == 'Chemical':
                            chem_abs.add(ent[2].lower())
                chem_abs = list(chem_abs)
                chems = f"{', '.join(chem_title)}|{', '.join(chem_abs)}"
                # get sample size
                re_search = re.search(r'(?<=[\s,.;:!])([1-9]+[\d,]*)\s?.{,40}(patients|cases|records|studies)', abstract, flags = re.I)
                if re_search and re_search[1]:
                    sample_size = int(re_search[1].replace(',', ''))
                else:
                    sample_size = '-'
                # get gen outcome from abstr
                doc = nlp(abstract)
                if len(list(doc.sents)) >= 2:
                    gen_outcome = f"{list(doc.sents)[-2]} {list(doc.sents)[-1]}"
                else:
                    gen_outcome = abstract
                doi = file['doi']
                cord_uid = file['cord_uid']
                pmc_link = file['pmcid']
                if pmc_link != '':
                    pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_link}/"
                pmid = file['pmid']
                study_type = 'Other'
                if pmid != '':
                    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                    soup = BeautifulSoup(get_page(url), 'xml')
                    if soup.find('PublicationTypeList') != None:
                        study_type = ', '.join(list(soup.find('PublicationTypeList').stripped_strings))
                        study_type = 'Review' if 'review' in study_type.lower() else 'Other'
                sentence = '|'.join([self.sentences[sid] for sid in sids])
                sents4write[pnid] = [sentence, date, study, study_link, journal, study_type, chems, sample_size, '',
                                     gen_outcome, '', '', '', abstract, doi, cord_uid, pmc_link]
            rows = sorted(sents4write.values(), key=lambda x:x[5]+'#'+x[1], reverse=True)
            csv_writer.writerows(rows)



    # table display function
    def search_tables(self, user_query):
        # process query phrases
        if len(user_query.split(':', 1)) > 1:
            prefix = user_query.split(':', 1)[0].strip()
            phrases = user_query.split(':', 1)[1]
            phrases = [f'{prefix} {phrase.strip()}' for phrase in phrases.split(',')]
        else:
            phrases = [phrase.strip() for phrase in user_query.split(',')]
        
        # process searching
        search_return = {} #set()
        for phrase in phrases:
            phrase_result = self.all_tables
            # if covid19:
            #     phrase_result = phrase_result.intersection(self.covid19_sents)

            doc = nlp(phrase.strip())
            for token in doc:
                #print(token)
                if token.is_stop or token.is_punct or token.is_digit: continue
                result = set([f'{k}|{tid}' for k,v in self.index_table.get(token.lemma_.lower(), {}).items() for tid in v])
                phrase_result = phrase_result.intersection(result)
    
            # calculate word count for ranking
            if phrase_result:
                phrase_count = {} #combine returned sentences into each paper
                for tstrid in phrase_result:
                    pnid, tid = tstrid.split('|')
                    if pnid in phrase_count:
                        phrase_count[pnid]['sents'].append(tstrid)
                    else:
                        phrase_count[pnid] = {'count': 0, 'sents':[tstrid]}
                for paper in phrase_count:
                    for token in doc:
                        if token.is_stop or token.is_punct or token.is_digit: continue
                        phrase_count[paper]['count'] += self.word_counts[str(paper)][token.lemma_.lower()]
                
                for k, v in phrase_count.items():
                    if k in search_return:
                        search_return[k]['count'] += v['count']
                        search_return[k]['sents'].extend(v['sents'])
                    else:
                        search_return[k] = {'count': v['count'], 'sents': v['sents']}
                    
        # rank search return
        if search_return:
            search_return = sorted(search_return.items(), key = lambda x: x[1]['count'], reverse = True)
            search_return = list(set([snid for i in search_return for snid in i[1]['sents']]))
        else:
            search_return = []
            #search_return = search_return.union(phrase_result)
        return search_return


    # sentences display function
    def display_tables(self, search_return):
        tables = {}
        for table in search_return:
            pnid, tid = table.split('|')
            if pnid in tables:
                if tid not in tables[pnid]:
                    tables[pnid].append(tid)
            else:
                tables[pnid] = [tid]
        
        for pnid, tids in tables.items():
            file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(pnid)]}.json', 'r', encoding = 'utf-8'))
            tids.sort()
            output_html = ""
            title = file['title']['text']
            authors = file['authors']
            if len(authors.split(';')) > 5:
                authors = f"{';'.join(authors.split(';')[:5])} et al."
            journal = file['journal']
            publish_time = file['publish_time']
            url = file['url']
            doi = file['doi']
            pmcid = file['pmcid']
            if pmcid != '':
                pmcid = f'<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/", target="_blank">{pmcid}</a>'
            pmid = file['pmid']
            #abstract = file['abstract']['text']
            tabls = file['tables']
            output_html += f'<h3>{html.escape(title)}</h3>'
            output_html += f'{html.escape(authors)}<br>'
            output_html += f'{html.escape(journal)}, {html.escape(publish_time)}, <a href="{url}", target="_blank">{url}</a><br>'
            output_html += f'doi: {html.escape(doi)}, PMCID: {pmcid}, PMID: {html.escape(pmid)}<br><br>'
            for tid in tids:
                tabl = tabls[int(tid)]
                output_html += f"<b>Table {int(tid)+1}:</b> {html.escape(tabl['text'])}<br><br>"
                if tabl['html'] != '':
                    output_html += f"{tabl['html']}<br><br>"
                else:
                    output_html += '<br><br>'
            display(HTML(output_html))


    # save tables to csv
    def save_tables(self, search_return, file_name):
        tables = {}
        for table in search_return:
            pnid, tid = table.split('|')
            if pnid in tables:
                if tid not in tables[pnid]:
                    tables[pnid].append(tid)
            else:
                tables[pnid] = [tid]
                
        with open(f"{self.csv_path}/{file_name}_{datetime.now().timestamp()}.csv", 'w', encoding = 'utf-8') as fcsv:
            csv_writer = csv.writer(fcsv)
            csv_writer.writerow(['table', 'Date', 'Study', 'Study Link', 'Journal', 'Study Type',
                                 'Chemicals', 'Sample Size', 'Severity', 'General Outcome',
                                 'Primary Endpoints', 'Clinical Improvement', 'Added On',
                                 'Abstract', 'DOI', 'CORD_UID', 'PMC_Link'])
            tabl4write = {}
            for pnid, tids in tables.items():
                file = json.load(open(f'{self.json_path}/{self.nid2corduid[int(pnid)]}.json', 'r', encoding = 'utf-8'))
                tids.sort()
                date = file['publish_time']
                study = file['title']['text']
                chem_title = set()
                if ('ents' in file['title']) and (file['title']['ents'] != []):
                    for ent in file['title']['ents']:
                        if ent[3] == 'Chemical':
                            chem_title.add(ent[2].lower())
                chem_title = list(chem_title)
                study_link = file['url']
                journal = file['journal']
                abstract = file['abstract']['text']
                # get abstr chems
                chem_abs = set()
                if ('ents' in file['abstract']) and (file['abstract']['ents'] != []):
                    for ent in file['abstract']['ents']:
                        if ent[3] == 'Chemical':
                            chem_abs.add(ent[2].lower())
                chem_abs = list(chem_abs)
                chems = f"{', '.join(chem_title)}|{', '.join(chem_abs)}"
                # get sample size
                re_search = re.search(r'(?<=[\s,.;:!])([1-9]+[\d,]*)\s?.{,40}(patients|cases|records|studies)', abstract, flags = re.I)
                if re_search and re_search[1]:
                    sample_size = int(re_search[1].replace(',', ''))
                else:
                    sample_size = '-'
                # get gen outcome from abstr
                doc = nlp(abstract)
                if len(list(doc.sents)) >= 2:
                    gen_outcome = f"{list(doc.sents)[-2]} {list(doc.sents)[-1]}"
                else:
                    gen_outcome = abstract
                doi = file['doi']
                cord_uid = file['cord_uid']
                pmc_link = file['pmcid']
                if pmc_link != '':
                    pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_link}/"
                tabls = file['tables']
                pmid = file['pmid']
                study_type = 'Other'
                if pmid != '':
                    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                    soup = BeautifulSoup(get_page(url), 'xml')
                    if soup.find('PublicationTypeList') != None:
                        study_type = ', '.join(list(soup.find('PublicationTypeList').stripped_strings))
                        study_type = 'Review' if 'review' in study_type.lower() else 'Other'
                tabl = '|'.join([tabls[int(tid)]['text'] for tid in tids])
                tabl4write[pnid] = [tabl, date, study, study_link, journal, study_type, chems, sample_size, '',
                                    gen_outcome, '', '', '', abstract, doi, cord_uid, pmc_link]
            rows = sorted(tabl4write.values(), key=lambda x:x[5]+'#'+x[1], reverse=True)
            csv_writer.writerows(rows)

    # get entity statistics function
    def get_entity_stats(self, search_return):
        # calculate entity counts
        entity_stats = {}
        for pid in search_return:
            for ent_type, ents in self.entity_nodes[str(pid)].items():
                entity_stats[ent_type] = entity_stats.get(ent_type, {})
                for ent, count in ents.items():
                    entity_stats[ent_type][ent] = entity_stats[ent_type].get(ent, 0) + count
        
        return entity_stats


    # entity counts display function
    def display_entities(self, search_return):
        """
        search_return: a list of paper ids returned from search
        """
        # calculate entity counts
        entity_stats = self.get_entity_stats(search_return)
        
        # display entity counts
        for ent_type, ents in entity_stats.items():
            print(f'{ent_type}: ')
            sorted_ents = sorted(ents.items(), key = lambda x:x[1], reverse = True)
            top_10_ents = sorted_ents[:10] if len(sorted_ents) > 10 else sorted_ents
            for ent in top_10_ents:
                print(f'{ent[0]:25}:\t {ent[1]:>5}')
            print('\n')


    # get relations statistics
    def get_relation_stats(self, search_return):
        # calculate relation counts and sentences
        relation_stats = {}
        for pid in search_return:
            for rel_type, rels in self.entity_relations[str(pid)].items():
                relation_stats[rel_type] = relation_stats.get(rel_type, {})
                for rel, rel_info in rels.items():
                    relation_stats[rel_type][rel] = relation_stats[rel_type].get(rel, {'count': 0, 'sents': []})
                    relation_stats[rel_type][rel]['count'] += rel_info['count']
                    relation_stats[rel_type][rel]['sents'].extend(rel_info['sents'])
        
        return relation_stats


    # entity relation counts display function
    def display_relations(self, search_return):
        """
        search_return: a list of paper ids returned from search
        """
        # calculate relation counts and sentences
        relation_stats = self.get_relation_stats(search_return)
        
        # display entity counts
        for rel_type, rels in relation_stats.items():
            print(f'{rel_type}: ')
            sorted_rels = sorted(rels.items(), key = lambda x:x[1]['count'], reverse = True)
            top_10_rels = sorted_rels[:10] if len(sorted_rels) > 10 else sorted_rels
            for rel in top_10_rels:
                sents = list(set(rel[1]['sents']))
                print(f"{rel[0]:25}:\t {rel[1]['count']:>5} in {len(sents):>5} sentences")
                sents = sents[:3] if len(sents) > 3 else sents
                for sent in sents:
                    print(self.sentences[sent])
                print('\n')
            print('\n')
