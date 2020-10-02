# this builds on work from -
# ref https://www.kaggle.com/rohailsyed/consolidating-effects-of-risk-factors-on-covid-19

# V2 handles changes to data for Data updates 20200403, 20200410
#    includes metadata changes and xml parsers for pmc, where both xml pmc and pdf are avail, pmc is used

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import json
from IPython.display import Image
from IPython.core.display import HTML
import re
from re import finditer
import nltk
import spacy
from nltk.stem import PorterStemmer
from collections import defaultdict


'''
set the global variable.

sent_tokens : for each paper_id, keep a cache of the sentence and word tokenized and stemmed paragraphs.
sent_fulls  : for each paper_id, keep a cache of the sentence but NOT word tokenized paragraphs.

reverse_map : target terms into their stemmed versions for compatibility in the matching stage
jdict       : for each paragraph segment found to contain valuable data, store the paper_id and desired segment of text.
htmls       : for each topic html for the topics tags content to display
'''

sent_tokens = defaultdict(lambda: defaultdict(lambda: "")) # for each document, cache the tokenized sentences for easy revisits
sent_fulls = defaultdict(lambda: defaultdict(lambda: ""))

stemmer = PorterStemmer()

reverse_map = {}

# jdict will hold the tokenized sentences for each paragraph in each document
jdict = defaultdict(lambda:[])

# htmls will hold the html for the topics tags content to display
htmls = defaultdict(lambda: "")

# htmlsum will hold the html for the summary of topics journals links to display, also save as csv output
htmlsum = defaultdict(lambda: "")


# from https://www.kaggle.com/ajrwhite/covid-19-thematic-tagging-with-regular-expressions/
def doi_url(d):
    if d.startswith('http'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'

    
'''
validate_segment() is our main function. Each paragraph in each JSON file is passed to this function.
We process and tokenize the text and then look for sentences that mention the desired terms.
Since we are also interested in numeric data, we specifically put a filter to only include segments that have numeric values in them.
'''

def validate_segment(segment, targs, paper_id=None, cnt=None):
    #global targs
    '''
    so the thinking here is that a paragraph that mentions a term related to COVID-19 will mention both in short word-order proximity.
    Particularly, we will work with the hypothesis that the mention of the term and mention of COVID will be no greater than 2
    sentences apart.
    These are the sentences we will keep.
    '''
    
    # quick heuristic to get rid of paragraphs that don't even discuss COVID-19 (or SARS-CoV-2) # may incl coronavirus also TBA
    if not "19" in segment and not "-2" in segment:
        return False, "", set()
        
    # first convert this into tokens
    jtxt = None
    # check if we have already cached the tokenized paragraph.
    # if so, just pick it up and move on.
    if paper_id:
        if paper_id in sent_tokens:
            if cnt in sent_tokens[paper_id]:
                jtxt = sent_tokens[paper_id][cnt]
                jtxt_base = sent_fulls[paper_id][cnt]
            
            
    # if this particular paragraph has not already been cached,
    # perform sentence and word tokenization as well as stemming.
    # then cache it for much faster subsequent processing.
    if jtxt is None:
        jtxt_base = nltk.sent_tokenize(segment)
        jtxt = [[stemmer.stem(y.lower()) for y in nltk.word_tokenize(x)] for x in jtxt_base]
        if not paper_id is None:
            sent_tokens[paper_id][cnt] = jtxt
            sent_fulls[paper_id][cnt] = jtxt_base

    
    
    # for each sentence, determine if the two categories of targets have been matched. If not, try checking the preceding
    # and succeeding sentence.
    
    sent_founds = []
    sent_numerics = []
    for i in range(0, len(jtxt)):
        # for each sentence, check if at least one number is mentioned (stats)
        
        # don't count citations as numeric values (e.g. "according to [12,13] etc.")
        no_bracks = re.sub(r"\[\s*\d+((\s*\,\s*\d+)+)?\]", "", jtxt_base[i])
        matchers = re.search(r"[^A-Z-a-z0-9\-](\d+)[^A-Z-a-z0-9\-]", no_bracks)
        is_numeric = False
        if matchers:
            # as a simple heuristic, we ignore values that might be years.
            # highly unlikely these values will be less than 1900 or greater than 2020.
            if int(matchers.group(1)) < 1900 or int(matchers.group(1)) > 2020:
                is_numeric = True
                    
        # for each sentence, check if any of the words are target words
        tempy = set()
        for k in range(0, len(jtxt[i])):
            word = jtxt[i][k]                    
            # check for match
            for q in range(0, len(targs)):
                if word in targs[q]:
                    tempy.add(q)
        sent_numerics.append(is_numeric)
        sent_founds.append(tempy)
    
    
    # we now have the list of found words. now let's run the heuristic.
    # for each sentence, we check if all terms were located. If not, then we check if the missing terms were in either the preceding
    # of following sentence.
    val_sent = None
    val_tags = None
    tagset = set()
    for i in range(0, len(sent_founds)):
        if len(sent_founds[i])==len(targs):
            if sent_numerics[i]:
                val_sent = jtxt_base[i]
                val_tags = jtxt[i]
                break
        
        # at least one target is missing. check the neighbors
        is_numeric = sent_numerics[i]
        tempset = sent_founds[i].copy()
        if i > 0:
            tempset.update(sent_founds[i-1])
            is_numeric = True if sent_numerics[i] or sent_numerics[i-1] else False
            if len(tempset)==len(targs) and is_numeric:
                val_sent = jtxt_base[i-1] + " " + jtxt_base[i]
                val_tags = jtxt[i] + (jtxt[i-1])
                break
                
        is_numeric = sent_numerics[i]
        tempset = sent_founds[i].copy()
        if i < (len(sent_founds) - 1):
            tempset.update(sent_founds[i+1])
            is_numeric = True if sent_numerics[i] or sent_numerics[i+1] else False
            if len(tempset)==len(targs) and is_numeric:
                val_sent = jtxt_base[i] + " " + jtxt_base[i+1]
                val_tags = jtxt[i] + (jtxt[i+1])
                break          
    
    if not val_sent:
        return False, "", set()
    
    # find the set of tags that were matches
    matchset = set()
    vbase = val_tags
    val_tags = set(val_tags)
    for q in range(0, len(targs)-1):
        matchset = matchset.union(targs[q])
    val_tags = val_tags.intersection(matchset)
    
    return True, val_sent, val_tags


#------------------------------------------------------------------------------
def process_file(jstruct, valid_ids, targs, sources, jrndflt):
    if "paper_id" in jstruct:
        if jstruct["paper_id"] in valid_ids:
            # consolidate the document text and see if there's a match.
            jbod = jstruct["body_text"]
            temp = defaultdict(lambda x: "")                        
            #.... for now, let's keep things simple. We assume that if mentioned, it will be at the paragraph level            
            # loop through each paragraph
            for cnt, x in enumerate(jbod):
                is_valid, val_seg, val_tags = validate_segment(x["text"] , targs, jstruct["paper_id"], cnt)               
                if is_valid:     
                    val_title = jstruct["metadata"]["title"]
                    if jstruct["paper_id"][:3]=="PMC":
                        val_doi = sources.doi[sources.pmcid==jstruct["paper_id"]].values[0] 
                        val_journ = sources.journal[sources.pmcid==jstruct["paper_id"]].fillna(jrndflt).values[0] # 'missing journal'
                        val_pubtime = sources.publish_time[sources.pmcid==jstruct["paper_id"]].values[0] 
                        if val_title == "":
                            val_title = sources.title[sources.pmcid==jstruct["paper_id"]].values[0]
                    else:    
                        val_doi = sources.doi[sources.sha==jstruct["paper_id"]].values[0]
                        val_journ = sources.journal[sources.sha==jstruct["paper_id"]].fillna(jrndflt).values[0] # 'missing journal'
                        val_pubtime = sources.publish_time[sources.sha==jstruct["paper_id"]].values[0] 
                        if val_title == "":
                            val_title = sources.title[sources.sha==jstruct["paper_id"]].values[0]
                    jdict[jstruct["paper_id"]].append({"text":x["text"], "tags":val_tags, "segment":val_seg, "paper_id":jstruct["paper_id"],"doi": val_doi,"journal": val_journ , "publish_time": val_pubtime, "title": val_title}) # jstruct["metadata"]["title"]                     
                    

#-------------------------------------------------------------------------------------------            
 
def json_files_from_directory(osdir, pmc_valid_ids, pdf_valid_ids, targs, sources, testing=False):
    
    pmc_file_list = []
    pdf_file_list = []
    for dirname, _, filenames in os.walk(osdir):  # '/kaggle/input'
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            if filename[-5:]==".json":
                if filename[:3]=="PMC":
                    pmc_file_list.append(os.path.join(dirname, filename))
                else:
                    pdf_file_list.append(os.path.join(dirname, filename))
                

    pmc_file_list.sort()
    pdf_file_list.sort()
    total_files = len(pmc_file_list) + len(pdf_file_list)
    total_pmc_files = len(pmc_file_list)
    total_pdf_files = len(pdf_file_list)
    
    counter = 0
    useds = set()
    for file in pmc_file_list:
        if testing:
            if counter > 1000:  # for testing True
                break           # for testing True
    # jdict will hold the tokenized sentences for each paragraph in each document    
        jrndflt = 'missing journal'
        if 'medrxiv' in file:
            jrndflt = '(medrxiv)' 
        process_file(json.load(open(file, "r")), pmc_valid_ids, targs, sources, jrndflt)
        counter += 1
        perc_complete = round((counter/total_pmc_files)*100)
        if perc_complete%5==0:
            if perc_complete in useds:
                continue
            useds.add(perc_complete)
            print ("{} / {} => {}% complete".format(counter, total_pmc_files, perc_complete))
            
    counter = 0
    useds = set()
    for file in pdf_file_list:
        if testing:
            if counter > 1000:  # for testing True
                break           # for testing True
    # jdict will hold the tokenized sentences for each paragraph in each document 
        jrndflt = 'missing journal'
        if 'medrxiv' in file:
            jrndflt = '(medrxiv)'    
        process_file(json.load(open(file, "r")), pdf_valid_ids, targs, sources, jrndflt)
        counter += 1
        perc_complete = round((counter/total_pdf_files)*100)
        if perc_complete%5==0:
            if perc_complete in useds:
                continue
            useds.add(perc_complete)
            print ("{} / {} => {}% complete".format(counter, total_pdf_files, perc_complete))              
    
#-------------------------------------------------------------------------------------------
def htmlsum_create(dfsum):
    # for topics summary journals
    
    cols = dfsum.columns.values
    sumtopic = dfsum.topic.unique()
    for topicname in sumtopic:
        #print(topicname)
        dftmp = dfsum[dfsum.topic==topicname].copy()
        dftmp = dftmp.reset_index(drop=True)
        idx = dftmp.index
        htmlstr = "<div class='summary_output' >"
        #htmlstr += "<br /><div style='font-weight:bold;'>{}</div><br />".format(topic_name)
        htmlstr += "<div style='display:table; overflow-y: auto; cellpadding:2px; cellspacing:2px; border:1px; border-style:solid;'  >"
   
        htmlstr += "<div style='display:table-row;'>"
        for i in range(len(cols)):  # table column headings
            if cols[i] != 'doi':
                htmlstr += "<div style='display:table-cell; font-weight:bold; padding-left:2px; padding-right:2px; cellspacing:2px; border:1px; border-style:solid; '>" + cols[i] + "</div>"
        htmlstr += "</div>"
        for ix in idx:
            htmlstr += "<div style='display:table-row;'>"
            for i in range(len(cols)):
                if cols[i] != 'doi':
                    if cols[i] == 'title':
                         htmlstr += "<div style='display:table-cell; padding-left:2px; padding-right:2px; cellspacing:2px; border:1px; border-style:solid;'>" +  "<span style='color:#0099cc;'> [" + "<a href=" + dftmp["doi"][ix] + ">"  + dftmp["title"][ix] + "</a>" + "] </span>" + "</div>"
                    else:    
                        htmlstr += "<div style='display:table-cell; padding-left:2px; padding-right:2px; cellspacing:2px; border:1px; border-style:solid;'>" +  dftmp[cols[i]][ix] + "</div>"
            htmlstr += "</div>"                  
   
        htmlstr += "</div>"
        htmlstr += "</div>"
        htmlsum[topicname] = htmlstr 
    
#-------------------------------------------------------------------------------------------        
def htmls_topics_create(targs):
    # for topics summary journals
    tname = []
    tpub = []
    ttitle = []
    tdoi = []
    tjrn = []

    topics = defaultdict(lambda: {"text":[], "title":[], "doi":[], "journal":[], "publish_time":[], "rawtag":""})

    for paper_id, found_objs in jdict.items():
    
        for ele in found_objs:
        
            # for each tag (usually only one) see which topic this falls under
            for tag in ele["tags"]:
                topics[reverse_map[tag]]["text"].append(ele["segment"])
                topics[reverse_map[tag]]["title"].append(ele["title"])
                topics[reverse_map[tag]]["doi"].append(ele["doi"])
                topics[reverse_map[tag]]["journal"].append(ele["journal"])
                topics[reverse_map[tag]]["publish_time"].append(ele["publish_time"])
                topics[reverse_map[tag]]["rawtag"] = tag
            
    for topic_name in topics:
        htmlstr = "<div class='test_output' >"
        htmlstr += "<br /><div style='font-weight:bold;'>{}</div><br />".format(topic_name)
        htmlstr += "<div style='display:table; overflow-y: auto; '>"
       # htmlstr += "<div id='topictab' style='overflow-y: auto;' >"
        for q, entry in enumerate(topics[topic_name]["text"]):
            tname.append(topic_name)
            tpub.append(topics[topic_name]["publish_time"][q])
            ttitle.append(topics[topic_name]["title"][q])
            tdoi.append(topics[topic_name]["doi"][q])
            tjrn.append(topics[topic_name]["journal"][q])
            
            splinter = nltk.word_tokenize(entry)
        
            for i in range(0, len(splinter)):
                if stemmer.stem(splinter[i])==topics[topic_name]["rawtag"]:
                    splinter[i] = "<span style='background-color:#FFDC00;'>" + splinter[i] + "</span>"
                elif stemmer.stem(splinter[i]) in targs[-1]:
                    splinter[i] = "<span style='background-color:#2ECC40 ;'>" + splinter[i] + "</span>"
                
            formatted = " ".join(splinter) + "<span style='color:#0099cc;'> [" + "<a href=" + topics[topic_name]["doi"][q] + ">"  + topics[topic_name]["title"][q] + "</a>"  + "]  </span>"  + "<span style='color:#0074D9;'> [ " + "<i>" +  topics[topic_name]["journal"][q]  + "</i>"  + " ]  "  + topics[topic_name]["publish_time"][q] + "</span>"
            htmlstr += "<div style='display:table-row;'>"
            htmlstr += "<div style='display:table-cell;padding-right:15px;font-size:20px;'>•</div><div style='display:table-cell;'>" + formatted + "</div>"
            htmlstr += "</div>"
        
        htmlstr += "</div>"
        htmlstr += "</div>"
        htmls[topic_name] = htmlstr          
        
    df = pd.DataFrame ( { 'topic' : tname, 'publish_time': tpub, 'title': ttitle, 'doi': tdoi ,'journal': tjrn })
    dfsum = df.drop_duplicates(subset=["topic","doi"])
    dfsum.to_csv('COVID-19_topics_summary.csv', index=False)
    
    htmlsum_create(dfsum)
    
#-------------------------------------------------------------------------------------------                
def target_terms_stemmed(targs):
# convert our target terms into their stemmed versions for compatibility in the matching stage
    for i in range(0, len(targs)):
        newterms = set()
        for ele in targs[i]:
            st = stemmer.stem(ele)
            newterms.add(st)
            reverse_map[st] = ele
        targs[i] = newterms                    
                    

            
    
'''
To use script - Under “File”, click “Add utility script”.
Import into your notebook using import nameof_script
In notebook - use example code below
change paths, filenames, validity, etc. if needed
change targs for topics to create html to display from json files and metatadata
add Display(HTML...) for your targs
'''  

# example code 
# e.g. targs                     
#targs = [set({"sarbecovirus", "zoonosis", "zoonotic",  "host", "spillover"}),
#         set({"covid-19", "covid19", "sars-cov-2", "2019-ncov", "betacoronavirus", "coronavirus"})]

#target_terms_stemmed(targs)

#base_path = "/kaggle/input/CORD-19-research-challenge/"
#sources = pd.read_csv(base_path + "metadata.csv",
#                     dtype={"pubmed_id":str,
#                           "Microsoft Academic Paper ID":str})
#sources.doi = sources.doi.fillna('').apply(doi_url)   

#pmc_valid_ids = set(sources[(sources["has_pmc_xml_parse"]==True) & (sources['pmcid'].notnull() ) ] ["pmcid"].unique().tolist())
#pdf_valid_ids = set(sources[(sources["has_pdf_parse"]==True) & (sources.has_pmc_xml_parse==False) & (sources['sha'].notnull() ) ] ["sha"].unique().tolist())

#osdir = '/kaggle/input'                   
#json_files_from_directory(osdir,pmc_valid_ids, pdf_valid_ids, targs, sources, testing=False ) # testing=True will break on 1000 

#htmls_topics_create(targs)

#display(HTML(htmls["zoonotic"]))   # topic targ to display   

#display(HTML(htmlsum["zoonotic"]))  # topic targ summary to display list of journals

