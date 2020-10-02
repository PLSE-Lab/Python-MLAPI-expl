# Useful functions and constants from
# https://www.kaggle.com/ajrwhite/covid-19-thematic-tagging-with-regular-expressions/notebook

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import re
from IPython.core.display import display, HTML


def load_metadata(metadata_file):
    df = pd.read_csv(metadata_file,
                 low_memory=False)
    df.doi = df.doi.fillna('').apply(doi_url)
    df['authors_short'] = df.authors.apply(shorten_authors)
    df['sorting_date'] = pd.to_datetime(df.publish_time)
    print(f'loaded DataFrame with {len(df)} records')
    return df.sort_values('sorting_date', ascending=False)


def load_full_text(df, data_folder):
    json_list = []
    # We will prefer PMC over PDF
    for filepath in df[df.pmc_json_files.notnull()].pmc_json_files:
        filepath = filepath.split(';')[0]
        with open(os.path.join(data_folder, filepath), 'rb') as f:
            json_list.append(json.load(f))
    # Top up with the PDF files for remaining cases
    for filepath in df[df.pmc_json_files.isnull()
                       & df.pdf_json_files.notnull()].pdf_json_files:
        filepath = filepath.split(';')[0]
        with open(os.path.join(data_folder, filepath), 'rb') as f:
            json_list.append(json.load(f))
    print(f'Found {len(json_list)} full texts for {len(df)} records')
    return json_list


# Fix DOI links
def doi_url(d):
    if d.startswith('http'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'
    
    
# Turn authors list into '<surname>' or '<surname> et al'
def shorten_authors(authors):
    if isinstance(authors, str):
        authors = authors.split(';')
        if len(authors) == 1:
            return authors[0].split(',')[0]
        else:
            return f'{authors[0].split(",")[0]} et al'
    else:
        return authors


def abstract_title_filter(df, search_string):
    return (df.abstract.str.lower().str.replace('-', ' ')
            .str.contains(search_string, na=False) |
            df.title.str.lower().str.replace('-', ' ')
            .str.contains(search_string, na=False))

# Helper function which counts synonyms and adds tag column to DF
def count_and_tag(df: pd.DataFrame,
                  synonym_list: list,
                  tag_suffix: str) -> (pd.DataFrame, pd.Series):
    counts = {}
    df[f'tag_{tag_suffix}'] = False
    for s in synonym_list:
        synonym_filter = abstract_title_filter(df, s)
        counts[s] = sum(synonym_filter)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True
    print(f'Added tag_{tag_suffix} to DataFrame')
    return df, pd.Series(counts)

# Function for printing out key passage of abstract based on key terms
def print_key_phrases(df, key_terms, n=5, chars=300):
    for ind, item in enumerate(df[:n].itertuples()):
        print(f'{ind+1} of {len(df)}')
        print(item.title)
        print('[ ' + item.doi + ' ]')
        try:
            i = len(item.abstract)
            for kt in key_terms:
                kt = kt.replace(r'\b', '')
                term_loc = item.abstract.lower().find(kt)
                if term_loc != -1:
                    i = min(i, term_loc)
            if i < len(item.abstract):
                print('    "' + item.abstract[i-30:i+chars-30] + '"')
            else:
                print('    "' + item.abstract[:chars] + '"')
        except:
            print('NO ABSTRACT')
        print('---')
        
def add_tag_covid19(df):
    # Customised approach to include more complicated logic
    df, covid19_counts = count_and_tag(df, COVID19_SYNONYMS, 'disease_covid19')
    novel_corona_filter = (abstract_title_filter(df, 'novel corona') &
                           df.publish_time.str.startswith('2020', na=False))
    df.loc[novel_corona_filter, 'tag_disease_covid19'] = True
    covid19_counts = covid19_counts.append(pd.Series(index=['novel corona'],
                                                     data=[novel_corona_filter.sum()]))
    return df, covid19_counts


def add_tag_risk(df):
    df, risk_counts = count_and_tag(df, RISK_FACTOR_SYNONYMS, 'risk_factors')
    return df, risk_counts


def display_dataframe(df, title):
    text = f'<h2>{title}</h2><table><tr>'
    text += ''.join([f'<td><b>{col}</b></td>' for col in df.columns.values]) + '</tr>'
    for row in df.itertuples():
        text +=  '<tr>' + ''.join([f'<td valign="top">{v}</td>' for v in row[1:]]) + '</tr>'
    text += '</table>'
    display(HTML(text))
    
    
def term_matcher(full_text_df, meta, case_sens_terms, case_insens_terms):
    case_sens_terms = '|'.join(case_sens_terms)
    case_insens_terms = '|'.join(case_insens_terms)
    output_list = []
    for i, item in enumerate(full_text_df.itertuples()):
        temp_output = {}
        temp_output['doi'] = meta[meta.sha == item.paper_id].doi.values[0]
        try:
            authors = item.metadata['authors'][0]['last']
            if len(item.metadata['authors']) > 1:
                authors += ' et al'
        except:
            authors = 'No author listed'
        design = meta[meta.sha == item.paper_id].design.values[0]
        temp_output['design'] = design
        temp_output['authors'] = authors
        temp_output['title'] = item.metadata['title']
        temp_output['publish_time'] = meta[meta.sha == item.paper_id].publish_time.values[0]
        
        for bt in item.body_text:
            sentence_list = bt['text'].split(r'. ')
            for s in sentence_list:
                if (len(re.findall(case_sens_terms, s)) > 0 or
                    len(re.findall(case_insens_terms, s)) > 0):
                    temp_output['extracted_string'] = s
                    output_list.append(temp_output.copy())
    return pd.DataFrame(output_list)
    

# CONSTANTS

COVID19_SYNONYMS = [
                    'covid',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b'
]

AGE_SYNONYMS = ['median age',
                'mean age',
                'average age',
                'elderly',
                r'\baged\b',
                r'\bold',
                'young',
                'teenager',
                'adult',
                'child'
               ]

SEX_SYNONYMS = ['sex',
                'gender',
                r'\bmale\b',
                r'\bfemale\b',
                r'\bmales\b',
                r'\bfemales\b',
                r'\bmen\b',
                r'\bwomen\b'
               ]

BODYWEIGHT_SYNONYMS = [
    'overweight',
    'over weight',
    'obese',
    'obesity',
    'bodyweight',
    'body weight',
    r'\bbmi\b',
    'body mass',
    'body fat',
    'bodyfat',
    'kilograms',
    r'\bkg\b', # e.g. 70 kg
    r'\dkg\b'  # e.g. 70kg
]

SMOKING_SYNONYMS = ['smoking',
                    'smoke',
                    'cigar', # this picks up cigar, cigarette, e-cigarette, etc.
                    'nicotine',
                    'cannabis',
                    'marijuana']

DIABETES_SYNONYMS = [
    'diabet', # picks up diabetes, diabetic, etc.
    'insulin', # any paper mentioning insulin likely to be relevant
    'blood sugar',
    'blood glucose',
    'ketoacidosis',
    'hyperglycemi', # picks up hyperglycemia and hyperglycemic
]

HYPERTENSION_SYNONYMS = [
    'hypertension',
    'blood pressure',
    r'\bhbp\b', # HBP = high blood pressure
    r'\bhtn\b' # HTN = hypertension
]

IMMUNODEFICIENCY_SYNONYMS = [
    'immune deficiency',
    'immunodeficiency',
    r'\bhiv\b',
    r'\baids\b'
    'granulocyte deficiency',
    'hypogammaglobulinemia',
    'asplenia',
    'dysfunction of the spleen',
    'spleen dysfunction',
    'complement deficiency',
    'neutropenia',
    'neutropaenia', # alternate spelling
    'cell deficiency' # e.g. T cell deficiency, B cell deficiency
]

CANCER_SYNONYMS = [
    'cancer',
    'malignant tumour',
    'malignant tumor',
    'melanoma',
    'leukemia',
    'leukaemia',
    'chemotherapy',
    'radiotherapy',
    'radiation therapy',
    'lymphoma',
    'sarcoma',
    'carcinoma',
    'blastoma',
    'oncolog'
]

CHRONICRESP_SYNONYMS = [
    'chronic respiratory disease',
    'asthma',
    'chronic obstructive pulmonary disease',
    r'\bcopd',
    'chronic bronchitis',
    'emphysema'
]

IMMUNITY_SYNONYMS = [
    'immunity',
    r'\bvaccin',
    'innoculat'
]

CLIMATE_SYNONYMS = [
    'climate',
    'weather',
    'humid',
    'sunlight',
    'air temperature',
    'meteorolog', # picks up meteorology, meteorological, meteorologist
    'climatolog', # as above
    'dry environment',
    'damp environment',
    'moist environment',
    'wet environment',
    'hot environment',
    'cold environment',
    'cool environment'
]

TRANSMISSION_SYNONYMS = [
    'transmiss', # Picks up 'transmission' and 'transmissibility'
    'transmitted',
    'incubation',
    'environmental stability',
    'airborne',
    'via contact',
    'human to human',
    'through droplets',
    'through secretions',
    r'\broute',
    'exportation'
]

REPR_SYNONYMS = [
    r'reproduction \(r\)',
    'reproduction rate',
    'reproductive rate',
    '{r}_0',
    r'\br0\b',
    r'\br_0',
    '{r_0}',
    r'\b{r}',
    r'\br naught',
    r'\br zero'
]

INCUBATION_SYNONYMS = [
    'incubation period',
    'period of incubation',
    'latent period',
    'latency period',
    'period of latency',
    'window period'
]

PERSISTENCE_SYNONYMS = ['persistence',
                        # r'(?<!viral )surface[s]?\b', # THIS DOESN'T WORK
                        'survival surface',
                        'persistence surface',
                        'survival on a surface',
                        'persistence on a surface',
                        'carrier test',
                        'suspension test',
                        'fomite',
                        # 'survival time',
                        'environmental surface',
                        'environmental stability',
                        'environmental reservoirs',
                        'environmental survival',
                        'pathogens in the environment',
                        'environmental pathogen',
                        'contaminated',
                        'contamination',
                        'surface stability',
                        'surface swab',
                        'inanimate surface',
                        'surface disinfection'
                       ]

RISK_FACTOR_SYNONYMS = [
    'risk factor',
    'prognostic factor',
    'prognosis',
    'influencing factor',
    'influence factor',
    'predictive factor',
    'factors predicting',
    'comorbidities',
    'comorbidity',
    r'contributes? to',
    'contributing to',
    r'predictors? of sever',
    r'predictors? of infection',
    r'predictors? of covid',
    r'drivers? of sever',
    r'drivers? of infect',
    r'drivers? of covid',
    r'predicts? sever',
    r'predicts? infection',
    r'predicts? covid'
]

# For mapping from David Mezzetti's Study Design metadata
DESIGNS = [
    'Other',
    'Systematic review',
    'Randomized control trial',
    'Non-randomized trial',
    'Prospective observational',
    'Time-to-event analysis',
    'Retrospective observational',
    'Cross-sectional',
    'Case series',
    'Modeling'
]