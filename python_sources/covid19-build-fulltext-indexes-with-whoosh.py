import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
from whoosh.index import create_in
from whoosh.fields import *

schema = Schema(paper_id = TEXT(stored=True), title=TEXT(stored=True), abstract = TEXT(stored = True), content = TEXT(stored = True))

def create_indexes(schema, papers):
    os.makedirs("./windexes")  
    ix = create_in("./windexes", schema)
    for paper_set in papers:
        writer = ix.writer()
        for index, row in paper_set.iterrows():
            writer.add_document(paper_id = row['paper_id'],
                            title    = row['title'],
                            abstract = row['abstract'],
                            content  = row['text']
                           )
        writer.commit()

def format_name(author):
    middle_name = " ".join(author['middle'])

    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))

    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)


def format_authors(authors, with_affiliation=False):
    name_ls = []

    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)

    return ", ".join(name_ls)


def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}

    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"

    return body


def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []

    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'],
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files


def generate_clean_df(all_files):
    cleaned_files = []

    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text',
                 'bibliography', 'raw_authors', 'raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()

    return clean_df

def clean_biorxiv(biorxiv_dir):
    filenames = os.listdir(biorxiv_dir)
    print("Number of articles retrieved from biorxiv:", len(filenames))
    all_files = []

    for filename in filenames:
        filename = biorxiv_dir + filename
        file = json.load(open(filename, 'rb'))
        all_files.append(file)

    texts = [(di['section'], di['text']) for di in file['body_text']]
    texts_di = {di['section']: "" for di in file['body_text']}
    for section, text in texts:
        texts_di[section] += text

    body = ""
    for section, text in texts_di.items():
        body += "{}\n\n{}\n\n".format(section, text)

    authors = all_files[4]['metadata']['authors']

    cleaned_files = []

    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                       with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]
        cleaned_files.append(features)

    col_names = [
    'paper_id', 'title', 'authors', 'affiliations', 'abstract', 'text', 'bibliography', 'raw_authors', 'raw_bibliography'
    ]
    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    return clean_df

main_path="../input/CORD-19-research-challenge"

biorxiv_dir = main_path+'/biorxiv_medrxiv/biorxiv_medrxiv/'
biorxiv_df = clean_biorxiv(biorxiv_dir)
print(biorxiv_df.head())

pmc_dir = main_path+'/custom_license/custom_license/'
pmc_files = load_files(pmc_dir)
pmc_df = generate_clean_df(pmc_files)
print(pmc_df.head())

comm_dir = main_path+'/comm_use_subset/comm_use_subset/'
comm_files = load_files(comm_dir)
comm_df = generate_clean_df(comm_files)
print(comm_df.head())

noncomm_dir = main_path+'/noncomm_use_subset/noncomm_use_subset/'
noncomm_files = load_files(noncomm_dir)
noncomm_df = generate_clean_df(noncomm_files)
print(noncomm_df.head())

papers = [biorxiv_df, pmc_df, comm_df, noncomm_df]
create_indexes(schema, papers)