# %% [code]
# coding: utf-8
# encoding: utf-8

"""
    Copyright (C) 2017 Àlex Bravo and Laura I. Furlong, IBI group.

    BeFree is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    BeFree is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    How to cite BeFree:
    Bravo,À. et al. (2014) A knowledge-driven approach to extract disease-related biomarkers from the literature. Biomed Res. Int., 2014, 253128.
    Bravo,À. et al. (2015) Extraction of relations between genes and diseases from text and large-scale data analysis: implications for translational research. BMC Bioinformatics, 16, 55.

"""
import pickle, tempfile
import traceback, sys, pdb, code, os
from pkg_resources import resource_filename
from tqdm import tqdm
import sys, nltk, codecs, regex, re
from .BeFree_constants import DISEASE_ENTITY, GENE_ENTITY, RESULT_ENTITY_TYPE, RESULT_DOCID, \
    RESULT_SENT_TEXT, RESULT_ENTITY_NORM, \
    RESULT_SENT_NUM, RESULT_ENTITY_OFFSET, RESULT_ENTITY_PARENT, RESULT_ENTITY_TEXT,\
    RESULT_ENTITY_ID, GENE_PATTERN_REGEX, DISEASE_PATTERN_REGEX, RESULT_YEAR,\
    RESULT_JOURNAL, RESULT_ISSN, RESULT_SECTION, RESULT_SECTION_NUM, DICT_DISEASE_NAME,\
    RESULT_COOC_DOCID, RESULT_COOC_SENT_NUM, RESULT_COOC_ENTITY1_OFFSET,\
    RESULT_COOC_ENTITY2_OFFSET, RESULT_COOC_SENT_TEXT, RESULT_COOC_ENTITY1_OFFSET_LIST,\
    RESULT_COOC_ENTITY2_OFFSET_LIST
    
from .BeFree_utils import term_curation, is_overlap, add_elem_dictionary,\
    replace_xml_tags_filtering, get_xref2entrez, get_greek_letters_lw,\
    filt_score, get_mesh_disease_dictionary, get_entity_information,\
    overlap, get_ner_process, get_befree_logo, get_results_screen
from .BeFree_NER import BeFreeNER
from .BeFree_document import DocumentInfo
from time import time
from progressbar import ProgressBar, Percentage, Bar

import json
import scispacy
import spacy

from absl import logging

#logging.get_absl_handler().use_absl_log_file('absl_logging', 'befreeout')
logging.set_verbosity(logging.INFO)

nlp_craft = spacy.load("en_ner_craft_md")
nlp_jnlpba = spacy.load("en_ner_jnlpba_md")

#current_path = sys.path[0]
#mongodb_path = "/".join(current_path.split("/")[:-1]) + "/mongodb"
#sys.path.append(mongodb_path)
#dict_path = "/".join(current_path.split("/")[:-1]) + "/dictionaries"
#sys.path.append(dict_path)
#medline_path = "/".join(current_path.split("/")[:-1]) + "/medline"
#sys.path.append(medline_path)

from ..medline.MedlineUtils import get_pmids_records
from ..medline.medline_constants import MESH_FIELD
from ..mongodb.MongoConnection import MongoConnection
from ..dictionaries.dict_constants import GENE_DB_NAME, ENTREZ_GENE, VER, DICT_ID, DICT_DB_NAME

def entity_extraction(path, file_name, entity_type, BioNER_Object, BioNER_alt, doc_records):
    
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    test = ""
    step = ""
    
    if not "/"== path[-1]:
        path = path + "/"
    
    output_path = path + file_name + test + step + ".log"
    logfile =  codecs.open(output_path,"w",'utf8')
    
    output_path = path + file_name + test + step + ".befree"
    ofile =  codecs.open(output_path,"w",'utf8')
    
    step = step + "_ext"
    output_path = path + file_name + test + step + ".befree"
    ofile_ext = codecs.open(output_path,"w",'utf8')
    
    step = step + "_acrG"
    output_path = path + file_name + test + step + ".befree"
    ofile_acrG = codecs.open(output_path,"w",'utf8')
    
    step = step + "_acr"
    output_path = path + file_name + test + step + ".befree"
    ofile_acr = codecs.open(output_path,"w",'utf8')
    
    step = step + "_ovr1"
    output_path = path + file_name + test + step + ".befree"
    ofile_ovr1 = codecs.open(output_path,"w",'utf8')
    
    step = step + "_ovr2"
    output_path = path + file_name + test + step + ".befree"
    ofile_ovr2 = codecs.open(output_path,"w",'utf8')
    
    step = step + "_filtlongterm"
    output_path = path + file_name + test + step + ".befree"
    ofile_filtlongterm = codecs.open(output_path,"w",'utf8')
    
    if entity_type == GENE_ENTITY:
        step = step + "_filtgene"
        output_path = path + file_name + test + step + ".befree"
        ofile_filtgene = codecs.open(output_path,"w",'utf8')
        gene_dict = get_gene_symbol_dict(use_pickle=True)
        xref2entrez_dict = get_xref2entrez(use_pickle=True)
        char_pat = re.compile('[a-z]')
        not_char_pat = re.compile('[^a-zA-Z]')
        greek_letters = list(get_greek_letters_lw().values())   
    
    step = step + "_total"
    final_output_path = path + file_name + test + step + ".befree"
    ofile_final= codecs.open(final_output_path,"w",'utf8')
    
    # import pdb; pdb.set_trace()
    for sha, record in tqdm(doc_records.items(), total=len(doc_records)):
        #for idx, doc_info in record.items():
            doc_info = record
            pmid = sha #+ ":" + str(idx)
            res = mention_extraction_pmid(entity_type, BioNER_Object, doc_info, pmid, ofile)
            if not len(res):
                continue
            
            res = add_EXTRACT_feature_pmid(res, 0, entity_type, ofile_ext)
            if not len(res):
                continue
            
            if entity_type == DISEASE_ENTITY:
                res = acronym_filtering_diseases_pmid(res, 0, BioNER_alt, ofile_acr)
            elif entity_type == GENE_ENTITY:
                res = acronym_filtering_genes_pmid(res, 0, BioNER_alt, xref2entrez_dict, char_pat, not_char_pat, greek_letters, ofile_acr)
            if not len(res):
                continue
            
            res = overlap_correction_step1_pmid(res, ofile_ovr1)
            while exist_overlap_pmid(res):
                res = overlap_correction_step2_pmid(res, ofile_ovr2)
            if not len(res):
                continue
            
            #filtering
            res = filtering_ambiguity_longterm(res, ofile_filtlongterm)
            if not len(res):
                continue
            
            if entity_type == GENE_ENTITY:
                res = gene_symbol_replacement_entity_level(res, gene_dict, ofile_filtgene)
            if not len(res):
                continue
            
            for lin in res:
                ofile_final.write(lin + "\n")
                ofile_final.flush()
        #except Exception as e:
            #logfile.write(pmid + "\t"+ str(e)+ "\n")
            #logfile.flush()
            #break

    ofile.close()
    ofile_ext.close()
    ofile_acrG.close()
    ofile_acr.close()
    ofile_acr.close()
    ofile_ovr1.close()
    ofile_ovr2.close()
    ofile_final.close()
    logfile.close()
    if entity_type == GENE_ENTITY:
        ofile_filtgene.close()
    
    return final_output_path

def mention_extraction_pmid(entity_type, BioNER_Object, doc_info, pmid, ofile):
    
    BioNER_Object.entity_recognition_cord19(doc_info, pmid)
    res = BioNER_Object.write_result()
    if len(res):
        ofile.write(res + "\n")
        ofile.flush()
        return res.split("\n")
    return res

def add_EXTRACT_feature_pmid(lines, header, entity_type, ofile):
    
    #print "add_EXTRACT_feature"
    disease_filt = []
    disease_filt.append("respiratory depression")
    disease_filt.append("cardiac depression")
    disease_filt.append("myocardial depression")
    disease_filt.append("fewer depression")
    disease_filt.append("vascular depression")
    disease_filt.append("voltage depression")
    disease_filt.append("cardiovascular depression")
    disease_filt.append("periumbilical depression")
    disease_filt.append("spreading depression")
    disease_filt.append("immune depression")
    
    dis_filt_pat = regex.compile(r'(' + "|".join(sorted(disease_filt, reverse = True))+ r')')
        
    genes_filt = []
    genes_filt.append("p3")
    genes_filt.append("p300")
    
    res = []
    
    extracted_dict = {}
    for lin in lines:
        if header:
            header = 0
            continue
        lin_split = lin.strip().split("\t")
        if len(lin_split) < 14:
            continue
        pmid = lin_split[RESULT_DOCID]
        term_norm = lin_split[RESULT_ENTITY_NORM]
        features = lin_split[RESULT_ENTITY_TYPE]
        if "()" in features and "EXTRACTED" in features:
            extracted_dict[pmid+"-"+term_norm] = 1
               
    for lin in lines:
        if header:
            header = 0
            continue
        lin_split = lin.strip().split("\t")
        if len(lin_split) < 14:
            continue
        sent = lin_split[RESULT_SENT_TEXT]
        
        if "</Abstract>" in sent:
            continue
        
        pmid = lin_split[RESULT_DOCID]
        term_norm = lin_split[RESULT_ENTITY_NORM]
        features = lin_split[RESULT_ENTITY_TYPE]
        
        if entity_type == GENE_ENTITY:
            if term_norm in genes_filt and not "GENE" in features:
                continue
            
            if "DISEASE" in features and not "GENE" in features:
                continue
        
        elif entity_type == DISEASE_ENTITY:
            sent = lin_split[RESULT_SENT_TEXT]
            
            if "depression" in term_norm:
                dis_out = False
                matches = dis_filt_pat.finditer(sent.lower())
                
                for m in matches:
                    ini = m.start()
                    end = m.end()
                    if is_overlap([ini,end], lin_split[RESULT_ENTITY_OFFSET].split("#")):
                        dis_out = True
                        break
                
                if dis_out:
                    continue
            
            if not "DISEASE" in features and "GENE" in features:
                continue
                
        features = lin_split[RESULT_ENTITY_TYPE]
        
        if pmid+"-"+term_norm in extracted_dict:
            if not "EXTRACTED" in features:
                lin_split[RESULT_ENTITY_TYPE] = lin_split[RESULT_ENTITY_TYPE] + "|EXTRACTED"
        
        ofile.write("\t".join(lin_split) + "\n")
        res.append("\t".join(lin_split))
        
    return res


def acronym_filtering_general_pmid(lines, header, pmid, medline_conn, entity_type,BioNER_Entity, ofile):
    #print "acronym_filtering_general"
    remove_term_norm = {}
    remove_longterm_norm = {}
    remove_acronym = {}
    remove_acronym_norm = {}
    pmid_terms_dict = {}
    acro_term_dict = {}
    
    doc = medline_conn.find_one({"_id":pmid})
    abbrev_list = doc.get("abbre")
    
    if not len(abbrev_list):
        return lines
    
    for lin in abbrev_list:
        lin_split = lin.split("\t")
        
        pmid = pmid
        num_sent = lin_split[0]
        acr = lin_split[1]
        
        acr_norm  = term_curation(acr)
        lterm = lin_split[2].strip()
        lterm_norm = term_curation(lterm)
        
        if "i.e." in acr:
            continue
        if "</Abstract>" in acr or "</Abstract>" in lterm:
            continue
        
        if not BioNER_Entity.entity_dictionary.has_term(lterm_norm):
            remove_term_norm[pmid+"-"+lterm_norm] = 1
            remove_term_norm[pmid+"-"+acr_norm] = 1
        
            remove_longterm_norm[pmid+"-"+lterm_norm] = 1
            remove_acronym[pmid+"-"+acr] = 1
            remove_acronym_norm[pmid+"-"+acr_norm] = 1
            
            add_elem_dictionary(pmid_terms_dict, pmid+"-"+num_sent, lterm)
            add_elem_dictionary(pmid_terms_dict, pmid+"-"+num_sent, acr)
            
            add_elem_dictionary(acro_term_dict, acr, lterm)
            add_elem_dictionary(acro_term_dict, lterm, acr)
    
    res = []
    for lin in lines:
        if header:
            header = 0
            continue
        
        lin_split = lin.strip().split("\t")
        
        if len(lin_split) == 1:
                continue
        
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        offset = lin_split[RESULT_ENTITY_OFFSET].replace("#", "-")
        mention_id = "-".join([pmid, num_sent, offset])
        term_norm = lin_split[RESULT_ENTITY_NORM]
        
        sent = lin_split[RESULT_SENT_TEXT]#.decode('utf-8')
        offset = lin_split[RESULT_ENTITY_OFFSET]
        overlap = False
        
        if pmid+"-"+num_sent in pmid_terms_dict:
            for term in pmid_terms_dict[pmid+"-"+num_sent]:
                ini = sent.index(term)
                end = ini + len(term)
                
                if is_overlap([ini, end], offset.split("#")):
                    ini2 = int(offset.split("#")[0])
                    if ini2 < ini:
                        new_term = acro_term_dict[term]
                        term = term_curation(term)
                        if pmid+"-"+term in remove_term_norm:
                            remove_term_norm.pop(pmid+"-"+term)
                        for t in new_term:
                            t = term_curation(t)
                            if pmid+"-"+t in remove_term_norm:
                                remove_term_norm.pop(pmid+"-"+t)
                        
                    if ini != ini2:
                        overlap = True
                        break
                    end2 = int(offset.split("#")[1])
                    if end < end2:
                        if entity_type == GENE_ENTITY:
                            if not GENE_PATTERN_REGEX.search(sent[end:end2]):
                                overlap = True
                                break
                        if entity_type == DISEASE_ENTITY:
                            if not DISEASE_PATTERN_REGEX.search(sent[end:end2]):
                                overlap = True
                                break
                    
        if overlap:
            continue
        
        if pmid+"-"+term_norm in remove_term_norm:
            features = lin_split[RESULT_ENTITY_TYPE]
            
            if entity_type == GENE_ENTITY:
                if not "GN" in features and not "GA" in features and not "GP" in features and not regex.search(r'\d', term_norm):
                    continue
            
            if entity_type == DISEASE_ENTITY:
                if not "DN" in features and not "DA" in features and not "DP" in features:
                    continue
            #####################################
            #if not regex.match(r'\d', term_norm):
            #    continue
        ofile.write(lin + "\n")
        res.append(lin)
    return res

    
def acronym_filtering_general_pmid_old(lines, header, acron_path, acron_header, entity_type,BioNER_Entity, ofile):
    #print "acronym_filtering_general"
    remove_term_norm = {}
    remove_longterm_norm = {}
    remove_acronym = {}
    remove_acronym_norm = {}
    pmid_terms_dict = {}
    acro_term_dict = {}
    
    parent_dict = {}
    
    for lin in open(acron_path):
        if acron_header:
            acron_header = 0
            continue
        lin_split = lin.split("\t")
        
        pmid = lin_split[1]
        
        num_sent = lin_split[2]
        acr = lin_split[3]
        
        acr_norm  = term_curation(acr)
        lterm = lin_split[4].strip()
        lterm_norm = term_curation(lterm)
        
        if "i.e." in acr:
            continue
        if "</Abstract>" in acr or "</Abstract>" in lterm:
            continue
        
        if not BioNER_Entity.entity_dictionary.has_term(lterm_norm):
            remove_term_norm[pmid+"-"+lterm_norm] = 1
            remove_term_norm[pmid+"-"+acr_norm] = 1
        
            remove_longterm_norm[pmid+"-"+lterm_norm] = 1
            remove_acronym[pmid+"-"+acr] = 1
            remove_acronym_norm[pmid+"-"+acr_norm] = 1
            
            add_elem_dictionary(pmid_terms_dict, pmid+"-"+num_sent, lterm)
            add_elem_dictionary(pmid_terms_dict, pmid+"-"+num_sent, acr)
            
            add_elem_dictionary(acro_term_dict, acr, lterm)
            add_elem_dictionary(acro_term_dict, lterm, acr)
    
    res = []
    for lin in lines:
        if header:
            header = 0
            continue
        
        lin_split = lin.strip().split("\t")
        
        if len(lin_split) == 1:
                continue
        
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        offset = lin_split[RESULT_ENTITY_OFFSET].replace("#", "-")
        mention_id = "-".join([pmid, num_sent, offset])
        term_norm = lin_split[RESULT_ENTITY_NORM]
        
        sent = lin_split[RESULT_SENT_TEXT]#.decode('utf-8')
        offset = lin_split[RESULT_ENTITY_OFFSET]
        overlap = False
        
        if pmid+"-"+num_sent in pmid_terms_dict:
            for term in pmid_terms_dict[pmid+"-"+num_sent]:
                ini = sent.index(term)
                end = ini + len(term)
                
                if is_overlap([ini, end], offset.split("#")):
                    ini2 = int(offset.split("#")[0])
                    if ini2 < ini:
                        new_term = acro_term_dict[term]
                        term = term_curation(term)
                        if pmid+"-"+term in remove_term_norm:
                            remove_term_norm.pop(pmid+"-"+term)
                        for t in new_term:
                            t = term_curation(t)
                            if pmid+"-"+t in remove_term_norm:
                                remove_term_norm.pop(pmid+"-"+t)
                        
                    if ini != ini2:
                        overlap = True
                        break
                    end2 = int(offset.split("#")[1])
                    if end < end2:
                        if entity_type == GENE_ENTITY:
                            if not GENE_PATTERN_REGEX.search(sent[end:end2]):
                                overlap = True
                                break
                        if entity_type == DISEASE_ENTITY:
                            if not DISEASE_PATTERN_REGEX.search(sent[end:end2]):
                                overlap = True
                                break
                    
        if overlap:
            continue
        
        if pmid+"-"+term_norm in remove_term_norm:
            features = lin_split[RESULT_ENTITY_TYPE]
            
            if entity_type == GENE_ENTITY:
                if not "GN" in features and not "GA" in features and not "GP" in features and not regex.search(r'\d', term_norm):
                    continue
            
            if entity_type == DISEASE_ENTITY:
                if not "DN" in features and not "DA" in features and not "DP" in features:
                    continue
            #####################################
            #if not regex.match(r'\d', term_norm):
            #    continue
        ofile.write(lin + "\n")
        res.append(lin)
    return res


def acronym_filtering_diseases_pmid(lines, header, BioNER_Gene, ofile):
    term_norm_dict = {}
    remove_term = {}
    res = []
    
    for lin in lines:
        try:
            if header:
                header = 0
                continue
            lin_split = lin.strip().split("\t")
            if len(lin_split) == 1:
                continue
            
            pmid = lin_split[RESULT_DOCID]
            num_sent = lin_split[RESULT_SENT_NUM]
            offset = lin_split[RESULT_ENTITY_OFFSET].replace("#", "-")
            mention_id = "-".join([pmid, num_sent, offset])
            
            term_norm = lin_split[RESULT_ENTITY_NORM]
            term_norm_dict[mention_id] = term_norm
            term = lin_split[RESULT_ENTITY_TEXT]
            
            if "<" in term or ">" in term:
                remove_term[pmid+"-"+term_norm] = 1
                continue

            features = lin_split[RESULT_ENTITY_TYPE]
            #if term_norm=='sars':
            #    pass 
            if "()" in features:
                if  "EXTRACTED" in features:
                    if term.islower() and not "DICTIONARY" in features:
                        remove_term[pmid+"-"+term_norm] = 1
                        continue
                    if not lin_split[RESULT_ENTITY_PARENT] in term_norm_dict:
                        remove_term[pmid+"-"+term_norm] = 1
                        continue
                    if pmid+"-"+term_norm in remove_term:
                        remove_term.pop(pmid+"-"+term_norm)
                        continue
                    
                if "SYMBOL" in features and not "EXTRACTED" in features:
                    remove_term[pmid+"-"+term_norm] = 1
                    continue
            else:
                if BioNER_Gene.entity_dictionary.has_term(term_norm) and BioNER_Gene.entity_dictionary.get_symbol(term_norm) and not "EXTRACTED" in features:
                    remove_term[pmid+"-"+term_norm] = 1
                    continue
        except:
            continue
    
    for lin in lines:
        if header:
            header = 0
            continue
        
        lin_split = lin.strip().split("\t")
        
        if len(lin_split) == 1:
                continue
        
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        offset = lin_split[RESULT_ENTITY_OFFSET].replace("#", "-")
        mention_id = "-".join([pmid, num_sent, offset])
        term_norm = lin_split[RESULT_ENTITY_NORM]
        features = lin_split[RESULT_ENTITY_TYPE].split("|")
        sent = lin_split[RESULT_SENT_TEXT]
        if pmid+"-"+term_norm in remove_term:
            if not "DISEASE" in features:
                continue
            
        # Ultimate Filter
        if "SYMBOL" in features and not "EXTRACTED" in features:
            # if term_norm != 'sars' and not "DISEASE" in features:
            if not "DISEASE" in features:
                continue
        
        #filtering_depression_v2
        ini = int(offset.split("-")[0])
        end = int(offset.split("-")[1])
        if "depression" == term_norm:
            if "depression of" in sent[ini:end+5]:
                continue
            if "long term" in sent[ini-15:end].lower():
                continue
            if "long-term" in sent[ini-15:end].lower():
                continue
            if "contractile" in sent[ini-15:end].lower():
                continue
            if "synaptic" in sent[ini-15:end].lower():
                continue
            
        if "ltd" == term_norm:
            if "long-term" in sent or "long term" in sent:
                continue
        
        if "worrying" == term_norm:
            continue     
        
        if filt_score(ini, end, sent):
            continue
        
        ofile.write(lin + "\n")
        res.append(lin)
    return res

    
def acronym_filtering_genes_pmid(lines, header, BioNER_Disease, xref2entrez_dict, char_pat, not_char_pat, greek_letters, ofile):
    #print "acronym_filtering_genes"
    term_norm_dict = {}
    remove_term = {}
    
    for lin in lines:
        try:
            if header:
                header = 0
                continue
            lin_split = lin.strip().split("\t")
            if len(lin_split) == 1:
                continue
            
            pmid = lin_split[RESULT_DOCID]
            num_sent = lin_split[RESULT_SENT_NUM]
            offset = lin_split[RESULT_ENTITY_OFFSET].replace("#", "-")
            mention_id = "-".join([pmid, num_sent, offset])
            
            term_norm = lin_split[RESULT_ENTITY_NORM]
            term_norm_orig = term_norm
            
            term_norm_dict[mention_id] = term_norm
            
            term = lin_split[RESULT_ENTITY_TEXT]
            term_norm = term
            
            if "<" in term or ">" in term:
                remove_term[pmid+"-"+term_norm] = 1
                continue
            
            features = lin_split[RESULT_ENTITY_TYPE]
            #if term_norm_orig=='sars':
            #    remove_term[pmid+"-"+term_norm] = 1 
            if "()" in features:
                if  "EXTRACTED" in features:
                    if term.islower() and not "DICTIONARY" in features:
                        remove_term[pmid+"-"+term_norm] = 1
                        continue
                    if not lin_split[RESULT_ENTITY_PARENT] in term_norm_dict:
                        remove_term[pmid+"-"+term_norm] = 1
                        continue
                    parent_norm = term_norm_dict[lin_split[RESULT_ENTITY_PARENT]]
                    if BioNER_Disease.entity_dictionary.has_term(parent_norm) and not "LONGTERM" in features:
                        remove_term[pmid+"-"+parent_norm] = 1
                        remove_term[pmid+"-"+term_norm] = 1
                        continue
                    if pmid+"-"+term_norm in remove_term:
                        remove_term.pop(pmid+"-"+term_norm)
                else:
                    if "SYMBOL" and not regex.search(r'\d', term_norm):
                        remove_term[pmid+"-"+term_norm] = 1
                    elif BioNER_Disease.entity_dictionary.has_term(term_norm):
                        remove_term[pmid+"-"+term_norm] = 1
            
            elif "LONGTERM" in features:
                if BioNER_Disease.entity_dictionary.has_term(term_norm):
                    remove_term[pmid+"-"+term_norm] = 1
        except:
            #print lin_split
            continue
        
    #Ultimate Filter    
    gene_norm_filt = {}
    gene_norm_filt["insulin"] = 1
    gene_norm_filt["anova"] = 1
    gene_norm_filt["bd i"] = 1
    gene_norm_filt["bd ii"] = 1
    gene_norm_filt["utr"] = 1
    gene_norm_filt["clock"] = 1
    gene_norm_filt["neuronal migration"] = 1
    gene_norm_filt["and"] = 1
    
    res = []
    for lin in lines:
        if header:
            header = 0
            continue
        
        lin_split = lin.strip().split("\t")
        
        if len(lin_split) == 1:
                continue
        
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        offset = lin_split[RESULT_ENTITY_OFFSET].replace("#", "-")
        mention_id = "-".join([pmid, num_sent, offset])
        #term_norm = lin_split[RESULT_ENTITY_NORM]
        term_norm = lin_split[RESULT_ENTITY_TEXT]
        
        sent = lin_split[RESULT_SENT_TEXT]
        
        if pmid+"-"+term_norm in remove_term:
            features = lin_split[RESULT_ENTITY_TYPE]
            if not "GENE" in features:# or not regex.match(r'\d', term_norm):
                continue
        
        #Ultimate Filter
        features = lin_split[RESULT_ENTITY_TYPE].split("|")
        gene_ment_norm = lin_split[RESULT_ENTITY_NORM]
        if gene_ment_norm in gene_norm_filt and not "GENE" in features:
            continue
        
        if not char_pat.search(gene_ment_norm.replace("as", "")):
            continue
        
        term = not_char_pat.sub('',gene_ment_norm)
        if term in greek_letters and not "GENE" in features:
            continue
        
        if len(gene_ment_norm) < 3 and not "EXTRACTED" in features and not "GENE" in features:
            continue
        
        #filtering_depression_v2
        ini = int(offset.split("-")[0])
        end = int(offset.split("-")[1])
        if "-BD" in sent[end:end+3]:
            if not re.match(r'[a-zA-Z]', sent[end+3:]):
                continue 
        
        if filt_score(ini, end, sent):
            continue
        
        # Mapping to Gene ID
        new_gene_list = []
        for xref in lin_split[RESULT_ENTITY_ID].split("|"):
            gene_xref_list = xref2entrez_dict.get(int(xref), [])
            for gene_id in gene_xref_list:
                new_gene_list.append(gene_id)
        
        if len(new_gene_list):
            lin_split[RESULT_ENTITY_ID] = str("|".join(new_gene_list))
            lin = "\t".join(lin_split)
            ofile.write(lin + "\n")
            ofile.flush()
            res.append(lin)
        
    return res


    

def overlap_correction_step1_pmid(lines, ofile):
    #print "overlap_correction_step1"
    iden2offsets = {}
    idenoffset_dict = {}
    
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        
        iden = pmid + "-" + num_sent
        
        add_elem_dictionary(iden2offsets, iden, lin_split[RESULT_ENTITY_OFFSET])
        idenoffset_dict[iden+"-"+lin_split[RESULT_ENTITY_OFFSET]] = lin
    
    remove_offset = {}
    
    for iden in iden2offsets:
        offset_list = iden2offsets[iden]
        i=0
        while i < len(offset_list.keys())-1:
            j = i+1
            while j < len(offset_list.keys()):
                off1 = list(offset_list.keys())[i]
                off2 = list(offset_list.keys())[j]
                if is_overlap(off1.split("#"), off2.split("#")):
                    
                    ini1 = int(off1.split("#")[0])
                    end1 = int(off1.split("#")[1])
                    
                    ini2 = int(off2.split("#")[0])
                    end2 = int(off2.split("#")[1])
                    
                    if ini1 == ini2:
                        if end1 > end2:
                            #off2 OUT!
                            remove_offset[iden+"-"+off2] = 1
                        elif end1 < end2:
                            #off1 OUT!
                            remove_offset[iden+"-"+off1] = 1
                        else:
                            a = 0
                    elif end1 == end2:
                        if ini1 > ini2:
                            #off1 OUT!
                            remove_offset[iden+"-"+off1] = 1
                        elif ini1 < ini2:
                            #off2 OUT!
                            remove_offset[iden+"-"+off2] = 1
                        else:
                            a=0
                    
                    elif end1 > end2 and ini1< ini2:
                        #off2 OUT!
                        remove_offset[iden+"-"+off2] = 1
                    
                    elif end2 > end1 and ini2< ini1:
                        #off1 OUT!
                        remove_offset[iden+"-"+off1] = 1
                j+=1
            i=i+1
    
    res = []    
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        
        iden = pmid + "-" + num_sent
        
        offset = lin_split[RESULT_ENTITY_OFFSET]
        key = iden+"-"+offset
        if key in remove_offset:
            continue
        ofile.write("\t".join(lin_split) + "\n")
        ofile.flush()
        res.append("\t".join(lin_split))
    return res

def exist_overlap_pmid(lines):
    iden2offsets = {}
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        iden = pmid + "-" + num_sent
        add_elem_dictionary(iden2offsets, iden, lin_split[RESULT_ENTITY_OFFSET])
        
    for iden in iden2offsets:
        offset_list = iden2offsets[iden]
        i=0
        while i < len(offset_list.keys())-1:
            j = i+1
            while j < len(offset_list.keys()):
                off1 = list(offset_list.keys())[i]
                off2 = list(offset_list.keys())[j]
                if is_overlap(off1.split("#"), off2.split("#")):
                    return True
                j+=1
            i=i+1
    return False
    
def overlap_correction_step2_pmid(lines, ofile):
    
    iden2offsets = {}
    joder_dict = {}
    
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        
        iden = pmid + "-" + num_sent
        add_elem_dictionary(iden2offsets, iden, lin_split[RESULT_ENTITY_OFFSET])
        joder_dict[iden + "-" + lin_split[RESULT_ENTITY_OFFSET]] = lin_split
        
    dis_del = {}
    
    for iden in iden2offsets:
        offset_list = iden2offsets[iden]
        i=0
        while i < len(offset_list.keys())-1:
            j = i+1
            while j < len(offset_list):
                off1 = list(offset_list.keys())[i]
                off2 = list(offset_list.keys())[j]
                if is_overlap(off1.split("#"), off2.split("#")):
                    spl1 = joder_dict[iden + "-" + off1]
                    spl2 = joder_dict[iden + "-" + off2]
                    mention1 = spl1[RESULT_SENT_TEXT]
                    mention2 = spl2[RESULT_SENT_TEXT]
                    
                    if "," in mention1 and not "," in mention2:
                        dis_del[iden + "-" + off1] = "DEL"
                    
                    elif "," in mention2 and not "," in mention1:
                        dis_del[iden + "-" + off2] = "DEL"
                        
                    elif "&" in mention1 and not "&" in mention2:
                        dis_del[iden + "-" + off1] = "DEL"
                    
                    elif "&" in mention2 and not "&" in mention1:
                        dis_del[iden + "-" + off2] = "DEL"
                    else:
                        
                
                        ini1 = int(off1.split("#")[0])
                        end1 = int(off1.split("#")[1])
                        
                        ini2 = int(off2.split("#")[0])
                        end2 = int(off2.split("#")[1])
                        
                        ini = ini1
                        if ini1 >= ini2:
                            ini = ini2
                        
                        end = end2
                        if end1 >= end2:
                            end = end1
                        new_offset = str(ini) + "#" + str(end)
                        sent = spl1[RESULT_SENT_TEXT]
                        term = sent[int(ini):int(end)]
                        cui2 = spl2[RESULT_ENTITY_ID]
                        
                        features = spl2[RESULT_ENTITY_TYPE]
                        
                        dis_del[iden + "-" + off2] = "DEL"
                        dis_del[iden + "-" + off1] = "\t".join([term, term, new_offset, cui2, features])
                        "pi plc beta 1\tPI-PLC-beta 1\t55#68\t[CUI]"
                j+=1
            i=i+1
    
    res = []
    
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        
        iden = pmid + "-" + num_sent
        
        offset = lin_split[RESULT_ENTITY_OFFSET]
        
        key = iden + "-" + offset
        
        if key in dis_del:
            
            value = dis_del[key]
            if value == "DEL":
                continue
            val_split = value.split("\t")
            
            lin_split[RESULT_ENTITY_NORM] = val_split[0]
            lin_split[RESULT_ENTITY_TEXT] = val_split[1]
            lin_split[RESULT_ENTITY_OFFSET] = val_split[2]
            
            cui_list = lin_split[RESULT_ENTITY_ID].split("|")
            for cui in val_split[3].split("|"):
                if not cui in cui_list:
                    cui_list.append(cui)
            lin_split[RESULT_ENTITY_ID] = "|".join(cui_list)
            
            features_list = lin_split[RESULT_ENTITY_TYPE].split("|")
            for feature in val_split[4].split("|"):
                if not feature in features_list:
                    features_list.append(feature)
            lin_split[RESULT_ENTITY_TYPE] = "|".join(features_list)
            
            new_line = "\t".join(lin_split)
            res.append(new_line)
            ofile.write(new_line + "\n")
            ofile.flush() 
            continue
        
        ofile.write(lin  + "\n")
        res.append(lin)
        ofile.flush() 
    return res


#############################################################
### FILTERING STEPS
#############################################################

def get_ambiguity_list(lines):
    mention_dict = {}
    
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        off = lin_split[RESULT_ENTITY_OFFSET]
        iden = "-".join([pmid, num_sent, off]).replace("#", "-")
        mention_dict[iden] = lin_split
        
    entity_concepts_id = {}
   
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        off = lin_split[RESULT_ENTITY_OFFSET]
        child = "-".join([pmid, num_sent, off]).replace("#", "-")
        parent = lin_split[RESULT_ENTITY_PARENT]
        
        if parent != "NA":
            if not parent in  mention_dict:
                continue
            
            if parent in entity_concepts_id:
                par_joined = entity_concepts_id[parent]
                entity_concepts_id[child] = par_joined
                continue
            if child in entity_concepts_id:
                continue
                
            parent_lin = mention_dict[parent]
            
            parent_iden = parent_lin[RESULT_ENTITY_ID].split("|")
            child_iden = lin_split[RESULT_ENTITY_ID].split("|")
            par = []
            for c1 in child_iden:
                if c1 in parent_iden:
                    par.append(c1)
            if len(par):
                par_joined = "|".join(par)
                #if parent in entity_concepts_id:
                #    if entity_concepts_id[parent] != par_joined:
                #        print parent_iden, child_iden, entity_concepts_id[parent], "-->", par_joined
                #        print lin_split
                entity_concepts_id[parent] = par_joined
                entity_concepts_id[child] = par_joined
    
    return  entity_concepts_id


def filtering_ambiguity_longterm(lines, ofile):
    gene_parents = get_ambiguity_list(lines)
    #ofile = open(output_path, "w")
    
    res = []
    
    for lin in lines:
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        num_sent = lin_split[RESULT_SENT_NUM]
        offset1 = lin_split[RESULT_ENTITY_OFFSET]
        iden1 = "-".join([pmid, num_sent, offset1]).replace("#", "-")
        ide1_dict= {}
        if iden1 in gene_parents:
            for ide1 in lin_split[RESULT_ENTITY_ID].split("|"):
                if ide1 in gene_parents[iden1].split("|"):
                    ide1_dict[ide1]=1
            if not len(ide1_dict):
                continue
        
        if len(ide1_dict):
            lin_split[RESULT_ENTITY_ID] = str("|".join(ide1_dict.keys()))
        
        new_line = "\t".join(lin_split)
        # new_line.decode("utf-8")
        ofile.write(new_line + "\n")
        res.append("\t".join(lin_split))
    
    #ofile.close()
    return res


def get_gene_symbol_dict(use_pickle=False):
    if use_pickle:
        gene_dict = pickle.load(open(resource_filename('befree.in', 'gene_symbol.pkl'), 'rb'))
        return gene_dict 
    gene_conn = MongoConnection(GENE_DB_NAME, ENTREZ_GENE + VER)
    records = gene_conn.find({})
    gene_dict = {}
    for rec in records:
        gene_id = rec["gene_id"]
        gene_symbol = rec["symbol"]
        gene_dict[gene_symbol] = gene_id
    return gene_dict



def gene_symbol_replacement_entity_level(lines, gene_dict, ofile):
    
    #log_file = open(output_path+"_log", "w")
    
    
    #ofile = open(output_path, "w")
    
    res = []
    
    for lin in lines:
        fields = lin.strip().split("\t")
        gene_list = fields[RESULT_ENTITY_ID].split("|")
        #pmid = fields[RESULT_DOCID]
        
        if len(gene_list) > 1:
            gene_mention = fields[RESULT_ENTITY_TEXT]
            gene_id = gene_dict.get(gene_mention, None)
            if not gene_id:
                gene_id = gene_dict.get(gene_mention.upper(), None)
                if not gene_id:
                    #log_file.write(pmid + "\t" + gene_mention + "\t" + str(fields[RESULT_ENTITY_ID]) + "\n")
                    gene_id = str(fields[RESULT_ENTITY_ID])
                    
            fields[RESULT_ENTITY_ID] = str(gene_id)
         
        new_line = "\t".join(fields) 
        ofile.write(new_line + "\n") 
        res.append(new_line)   
    #ofile.close()        
    #log_file.close()   
    return res

#RE
def write_entity_cooccurrence_filtering(entity_path, header, entity_del, output_path, offset_prev = 0):
    
    ofile = open(output_path, "w")
    logfile = open(output_path + ".log", "w")
    
    for lin in open(entity_path):
        if header:
            header = 0
            ofile.write(lin)
            logfile.write(lin)
            continue
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID + offset_prev]
        sent = lin_split[RESULT_SENT_NUM + offset_prev]
        offset = lin_split[RESULT_ENTITY_OFFSET + offset_prev]
        key = "-".join([pmid, sent, offset])
        
        if key in entity_del:
            lin_split.insert(0, entity_del[key])
            logfile.write("\t".join(lin_split)+"\n")
            logfile.flush()
        else:
            ofile.write("\t".join(lin_split)+"\n")
            ofile.flush()
        
    logfile.close()   
    ofile.close()

def gene_disease_cooccurrence_filtering_v2(gene_path, gene_header, dis_path, dis_header, out_gene_path, out_dis_path, offset_prev = 0):
    
    gene_dict = {}
    for lin in open(gene_path):
        if gene_header:
            gene_header = 0
            continue
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        gene_dict = add_elem_dictionary(gene_dict, pmid, lin_split, True)
        
    disease_dict = {}
    for lin in open(dis_path):
        if dis_header:
            dis_header = 0
            continue
        lin_split = lin.strip().split("\t")
        pmid = lin_split[RESULT_DOCID]
        disease_dict = add_elem_dictionary(disease_dict, pmid, lin_split, True)
    
    #mesh_file = INPUT_PATH + "mtrees2014.bin"
    current_path = sys.path[0]
    mesh_file = resource_filename('befree.src.ner', '/mtrees2014.bin')  # "/".join(current_path.split("/")[:-1]) + "/ner/mtrees2014.bin"
    
    mesh_dict = get_mesh_disease_dictionary(mesh_file)
    
    gene_del = {}
    dis_del = {}
    for pmid in disease_dict.keys():
        if pmid in gene_dict:
            
            first_cui_list = []
            
            # GENE
            gene_lines = gene_dict.get(pmid)
            gene_sent, gen_offset_lin = get_entity_information(gene_lines)
            
            # DISEASE
            dis_lines = disease_dict.get(pmid)
            dis_sent, dis_offset_lin = get_entity_information(dis_lines)
            
            for sent in sorted(gene_sent.keys()):
                if sent in dis_sent:
                    
                    for gene_offset in sorted(gene_sent[sent]):
                        for dis_offset in sorted(dis_sent[sent]):
                
                            if is_overlap(gene_offset.split("#"), dis_offset.split("#")):
                                
                                gene_type = gen_offset_lin[sent+ "-" +gene_offset][RESULT_ENTITY_TYPE]
                                dis_type = dis_offset_lin[sent+ "-" +dis_offset][RESULT_ENTITY_TYPE]
                                
                                if gene_offset != dis_offset:
                                    res = overlap(gene_offset.split("#"), dis_offset.split("#"))
                                    if res == 0:
                                        key = "-".join([pmid, sent, dis_offset])
                                        dis_del[key] = "INSIDE"
                                        continue
                                    elif res == 1:
                                        key = "-".join([pmid, sent, gene_offset])
                                        gene_del[key] = "INSIDE"
                                        continue
                                    
                                if "DISEASE" in gene_type and not "GENE" in dis_type:
                                    key = "-".join([pmid, sent, gene_offset])
                                    gene_del[key] = "D"
                                    continue
                                
                                if not "DISEASE" in gene_type and "GENE" in dis_type:
                                    key = "-".join([pmid, sent, dis_offset])
                                    dis_del[key] = "G"
                                    continue
                                
                                if "DISEASE" in gene_type and "GENE" in gene_type:
                                    if "DISEASE" in dis_type and "GENE" in dis_type:
                                        if "GN" in gene_type or "GA" in gene_type:
                                            key = "-".join([pmid, sent, dis_offset])
                                            dis_del[key] = "DG_GNGA"
                                            continue
                                        if "DN" in gene_type or "DA" in gene_type:
                                            key = "-".join([pmid, sent, gene_offset])
                                            gene_del[key] = "DG_DNDA"
                                            continue
                                
                                if "SYMBOL" in gene_type:
                                    
                                    if "GENE" in gene_type or "GENE" in dis_type:
                                        key = "-".join([pmid, sent, dis_offset])
                                        dis_del[key] = "S_G"
                                        continue
                                    
                                    if "DISEASE" in dis_type or "DISEASE" in gene_type:
                                        key = "-".join([pmid, sent, gene_offset])
                                        gene_del[key] = "S_D"
                                        continue
                                    
                                    if "EXTRACTED" in gene_type and not "EXTRACTED" in dis_type:
                                        key = "-".join([pmid, sent, dis_offset])
                                        dis_del[key] = "S_E"
                                        continue
                                    
                                    if "EXTRACTED" in dis_type and not "EXTRACTED" in gene_type:
                                        key = "-".join([pmid, sent, gene_offset])
                                        gene_del[key] = "S_E"
                                        continue
                                    
                                    if "EXTRACTED" in gene_type and "EXTRACTED" in dis_type:
                                        if "CONFIRMED" in gene_type and not "CONFIRMED" in dis_type:
                                            key = "-".join([pmid, sent, dis_offset])
                                            dis_del[key] = "S_E_C"
                                            continue
                                        if not "CONFIRMED" in gene_type and "CONFIRMED" in dis_type:
                                            key = "-".join([pmid, sent, gene_offset])
                                            gene_del[key] = "S_E_C"
                                            continue
                                
                                entity_cui_list = dis_offset_lin[sent+ "-" +dis_offset][RESULT_ENTITY_ID].split("|")
                                            
                                is_dis = 0          
                                for ent_cui in entity_cui_list:
                                    if ent_cui in first_cui_list:
                                        is_dis = 1
                                        break
                                
                                if is_dis:
                                    key = "-".join([pmid, sent, gene_offset])
                                    gene_del[key] = "Mesh"
                                    continue
                                else:
                                    key = "-".join([pmid, sent, dis_offset])
                                    dis_del[key] = "Mesh"
                                    continue
                                
                                #print "FINAL"    
                                #print ""
        
    write_entity_cooccurrence_filtering(gene_path, 0, gene_del, out_gene_path, offset_prev)
    write_entity_cooccurrence_filtering(dis_path, 0, dis_del, out_dis_path, offset_prev)


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def entity_identification(data_path, pub_entity_records, entity_path, add_ents_records=None, loc=None):
    gene_filename = 'genes'
    disease_filename = 'diseases'
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    pub_entity_records = json.load(open(f'{data_path}/{pub_entity_records}', 'r', encoding = 'utf-8'))
    add_ents_records = {} if add_ents_records==None else json.load(open(f'{data_path}/{add_ents_records}', 'r', encoding = 'utf-8'))
    if loc:
       loc = NullContextManager(loc)
    else:
       loc = tempfile.TemporaryDirectory()

    with loc as path:

        path = path + '/'
        
        blacklist_synonyms_path = resource_filename('befree.in', 'synonyms_blacklist.sample') 
        blacklist_synonyms_header = 1
        header_str = 'y'.lower()
        if "0" in header_str or "f" in header_str or "n" in header_str:
            blacklist_synonyms_header = 0
        
        pmid_list = []
        
        print(get_befree_logo())
        print(get_ner_process())
        print("")
        
        gene_synonyms_blacklist = {} #'sars': 1}
        disease_synonyms_blacklist = {}
        if len(blacklist_synonyms_path):
            for lin in open(blacklist_synonyms_path):
                if blacklist_synonyms_header:
                    blacklist_synonyms_header = 0
                    continue
                if "#" in lin:
                    continue
                fields = lin.strip().split("\t")
                
                if fields[1].lower() == "gene":
                    gene_synonyms_blacklist[fields[0].strip()] = 1
                elif fields[1].lower() == "disease":
                    disease_synonyms_blacklist[fields[0].strip()] = 1
        
        print("Loading Gene Dictionary...")
        BioNER_Gene= BeFreeNER(GENE_ENTITY, gene_synonyms_blacklist.keys(), use_pickle=True)
        print("  [OK!]")
        print("Loading Disease Dictionary...")
        BioNER_Disease = BeFreeNER(DISEASE_ENTITY, disease_synonyms_blacklist.keys(), use_pickle=True)
        print("  [OK!]")
        print("")
        start_time_total = time()
        for doc_id, nid in pub_entity_records.items():
            if doc_id in add_ents_records: continue
            file = json.load(open(f'{entity_path}/{doc_id}.json', 'r', encoding = 'utf-8'))
            all_sections = {doc_id: {'title':file['title']['text'], 'abstract':file['abstract']['text']}}
            # extracting entities
            print("Disease Extraction...")
            disease_path = entity_extraction(path, disease_filename, DISEASE_ENTITY, BioNER_Disease, BioNER_Gene, all_sections)
            print("")
            print("Gene Extraction...")
            gene_path = entity_extraction(path, gene_filename, GENE_ENTITY, BioNER_Gene, BioNER_Disease, all_sections)
            print("")
            print("Gene-Disease disambiguation...")
            filtered_gene_path = path + gene_filename + "_FINAL.befree" 
            filtered_disease_path = path + disease_filename + "_FINAL.befree" 
            gene_disease_cooccurrence_filtering_v2(gene_path, 0, disease_path, 0, filtered_gene_path, filtered_disease_path)
            print("")
            print(get_results_screen())
            print("  TOTAL TIME:")
            print("   ", time() - start_time_total, "seconds")
            print("  Gene Results:")
            print("    "+filtered_gene_path)
            print("  Disease Results:")
            print("    "+filtered_disease_path)
            print("")
       
            with open(os.path.join(path, 'genes_FINAL.befree')) as f:
                identified_genes = f.readlines()
            with open(os.path.join(path, 'diseases_FINAL.befree')) as f:
                identified_diseases = f.readlines()
            
            # parsing entities
            sent_start = {}
            for pid, doc in tqdm(all_sections.items(), desc="calculating character offsets"):
                sent_start_ids = []
                for sec_num, section in doc.items():
                    spans = tokenizer.span_tokenize(section)
                    for span in spans:
                        sent_start_ids.append(span[0])
                sent_start[pid] = sent_start_ids
            
            entities = {pid: {sec_id: {'text': doc[sec_id], 'bf_ents': [] } for sec_id in doc} for pid, doc in all_sections.items()}
            if identified_genes:
                for gene in identified_genes:
                    line = gene.strip().split('\t')
                    pid = line[0]
                    sec_id = line[5]
                    sent_id = int(line[6])
                    ident = line[7]
                    name = line[10]
                    loc = line[11]
                    start, end = loc.split('#')
                    start = sent_start[pid][sent_id] + int(start)
                    end = sent_start[pid][sent_id] + int(end)
                    try:
                        assert entities[pid][sec_id]['text'][start:end] == name
                    except AssertionError:
                        logging.info(f'AssertionError with paper {pid} {sec_id}')
                        continue
                    entities[pid][sec_id]['bf_ents'].append([start, end, name, 'Gene', ident])
            
            if identified_diseases:
                for disease in identified_diseases:
                    line = disease.strip().split('\t')
                    pid = line[0]
                    sec_id = line[5]
                    sent_id = int(line[6])
                    ident = line[7]
                    name = line[10]
                    loc = line[11]
                    start, end = loc.split('#')
                    start = sent_start[pid][sent_id] + int(start)
                    end = sent_start[pid][sent_id] + int(end)
                    try:
                        assert entities[pid][sec_id]['text'][start:end] == name
                    except AssertionError:
                        logging.info(f'AssertionError with paper {pid} {sec_id}')
                        continue
                    entities[pid][sec_id]['bf_ents'].append([start, end, name, 'Disease', ident])
            
            file['title']['bf_ents'] = entities[doc_id]['title']['bf_ents']
            file['abstract']['bf_ents'] = entities[doc_id]['abstract']['bf_ents']
            
            # craft model annotation
            title = file['title']['text']
            abstract = file['abstract']['text']
            
            doc = nlp_craft(title)
            ents = []
            for ent in list(doc.ents):
                ents.append([ent.start_char, ent.end_char, ent.text, ent.label_, '-'])
            file['title']['craft_ents'] = ents
            
            doc = nlp_craft(abstract)
            ents = []
            for ent in list(doc.ents):
                ents.append([ent.start_char, ent.end_char, ent.text, ent.label_, '-'])
            file['abstract']['craft_ents'] = ents
            
            doc = nlp_jnlpba(title)
            ents = []
            for ent in list(doc.ents):
                ents.append([ent.start_char, ent.end_char, ent.text, ent.label_, '-'])
            file['title']['jnlpba_ents'] = ents
            
            doc = nlp_jnlpba(abstract)
            ents = []
            for ent in list(doc.ents):
                ents.append([ent.start_char, ent.end_char, ent.text, ent.label_, '-'])
            file['abstract']['jnlpba_ents'] = ents
            
            json.dump(file, open(f'{entity_path}/{doc_id}.json', 'w', encoding = 'utf-8'), indent = 4)
            add_ents_records[doc_id] = nid
    json.dump(add_ents_records, open(f'{data_path}/add_ents_records.json', 'w', encoding = 'utf-8'), indent = 4)
