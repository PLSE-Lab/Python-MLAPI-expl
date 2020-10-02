"""
Extract only sentences from the body of Hillary Clinton's released emails.
Uses @honnibal's spacy NLP package which uses dependency parse to segment sentences.
Lower-cased and tokenized (including punctuation).
Each line should be a sentence.

First download the 'output' by unzipping the following in the home directory.
  https://www.kaggle.com/c/hillary-clinton-emails/data

Code to:
  https://github.com/benhamner/hillary-clinton-emails

Example:
  python scripts/clean.py
"""
from __future__ import print_function

import re
import pandas as pd
df = pd.read_csv('output/Emails.csv')

from spacy.en import English
print('Loading spacy model..')
nlp = English(load_vectors=False)

STOPSYMBOLS = {'<', '>'}

WORDS_META = {'re:', 'case', 'no.', 'doc no', 'from', 'for:', 'fw:', 'h:', 'to:', 'subject:', 'date:', 'prom:'}
WORDS_DAY = {'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
WORDS_MONTH = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}

pattern_email = re.compile(r"[\w]+@[\.\w]+")
pattern_date = re.compile(r"\d+/\d+/\d+")
pattern_time = re.compile(r"[0-2]?[0-9]:[0-6][0-9]")

is_email = lambda x: sum([i.like_email for i in x]) > 0
is_url = lambda x: sum([i.like_url for i in x]) > 0
is_oov = lambda x: sum([i.cluster == 0 for i in x]) > 1
is_short = lambda x: len(x) <= 3
is_gibberish = lambda x: sum([i.is_digit == i.is_alpha for i in x]) > 1

replace_whitespace = lambda x: re.sub(r'\s+', ' ', x)

dupes = set()
counter = 0

if __name__ == '__main__':
    print('Number of emails:', df.shape[0])

    with open('sentences.txt', 'w') as f:
        for email in df['ExtractedBodyText']:
            # remove nan
            if type(email) != str:
                continue
    
            # emails are sometimes broken up into 
            email_filtered = []
            for line in email.split('\n'):
                tokens = line.lower().split()
                if tokens[0] in WORDS_META:
                    continue
    
                lexemes = [nlp.vocab[unicode(i, errors='ignore')] for i in tokens]
                if lexemes[0].is_digit == lexemes[0].is_alpha:
                    continue
    
                if is_short(tokens):
                    continue
                if is_oov(lexemes):
                    continue
                if is_email(lexemes):
                    continue
                if is_url(lexemes):
                    continue
                # if is_gibberish(lexemes):
                    # continue
    
                # else
                email_filtered.append(line)
    
            email = ' '.join(email_filtered)
            for sent in nlp(unicode(email, errors='ignore')).sents:
                if is_short(sent):
                    continue
    
                tokens = [i.lower_ for i in sent]
                sentence = ' '.join(tokens)
                if sentence in dupes:
                    continue
                else:
                    dupes.add(sentence)
                    counter += 1
                    f.write(sentence)
                    f.write('\n')

    print('Number of extracted sentences:', counter)
    print('Saved extracted sentences to:', 'sentences.txt')

