import json
from pathlib import Path
import string
import shutil

import git
import requests as req
import subprocess
import pandas as pd
# output path
DATA_PATH = Path('/kaggle/working/')

# Request file from wikipedia
bzip_file = req.get('https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2')

# save request content to afile
with open(DATA_PATH/'tawiki-latest-pages-articles.xml.bz2', 'wb') as f:
    f.write(bzip_file.content)

# clone wiki extractor
git.Git("/kaggle/working/").clone("https://github.com/attardi/wikiextractor.git")

EXTRACTED_PATH = DATA_PATH/'extracted'
EXTRACTED_PATH.mkdir()

print('Extracting data from dump archive')
run_stat = subprocess.run(
    ['python',
     # File to run
     str(DATA_PATH/'wikiextractor/WikiExtractor.py'),
     # Processing parameters
     '-s', '--json', '-o',
     # Directory to store Extracted text
     str(DATA_PATH/'extracted'),
     # Archive file to extract from
     str(DATA_PATH/'tawiki-latest-pages-articles.xml.bz2')]
)

# Get list of files extracted from the extraction folder
files_extracted = [str(f) for f in EXTRACTED_PATH.rglob("*/*")]

# Since all files are stored as json we can load them like this 
# LANG_TEXT = [json.loads(line) for _file in files_extracted for line in open(_file)]
lang_text = []
for _file in files_extracted:
    with open(_file, 'r') as f:
        file_lines = f.readlines() 
    for line in file_lines:
        lang_text.append(json.loads(line))

# Function to filter english words
# check each word after removing their punctuations
# filter_english = lambda text: ' '.join([word for word in text.split() if word.translate(str.maketrans('', '', string.punctuation)).isalpha() is False])
def filter_english(text):
    words = []
    for word in text.split():
        word = word.translate(str.maketrans('', '', string.punctuation))
        if not word.isalpha():
            words.append(word)
    return ' '.join(words)


lang_df = pd.DataFrame(lang_text)
lang_df['text'] = lang_df['text'].apply(filter_english)

print(lang_df.info())
print(lang_df.head())

# Store the output in compressed format
lang_df.to_csv(DATA_PATH/'filtered_data.csv.tar.gz', header=True)

# clean up
shutil.rmtree(str(EXTRACTED_PATH))
shutil.rmtree(str(DATA_PATH/'wikiextractor'))
Path(DATA_PATH/'tawiki-latest-pages-articles.xml.bz2').unlink()