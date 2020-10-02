"""
NLPProfiler is a program/library that profiles text data and extracts NLP specific data innate in them, and produce statictical details 
about the text data. Think of pandas.describe() but on text data, you get a dataframe that you then can run the describe() to get 
descriptive statistics from it.
"""
#
# Copyright 2020 Mani Sarkar
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

### Please feel free to fork this work as long as you maintain the above
### license and provide citation.

### Kaggle kernel: https://www.kaggle.com/neomatrix369/nlp-profiler-simple-dataset
### NLP Profiler source: https://github.com/neomatrix369/awesome-ai-ml-dl/blob/master/examples/better-nlp/library/org/neomatrix369/nlp_profiler.py 
### Jupyter Notebook: https://github.com/neomatrix369/awesome-ai-ml-dl/blob/master/examples/better-nlp/notebooks/jupyter/nlp_profiler.ipynb

### Other resources:
### https://github.com/neomatrix369/awesome-ai-ml-dl/blob/master/examples/better-nlp/
### or https://bit.ly/better-nlp

#### Start of the Utility function

### Use shebang to run the pip install commands
#     ! python --version && pip --version
#     ! pip install --upgrade pip && pip --version

#     ! python -m pip install emoji nltk textblob

import os
import struct

### Use os to run the pip install commands
os.system("python --version && pip --version")
print("You are running {} bits Python.".format(8 * struct.calcsize("P")))
os.system("pip install --upgrade pip && pip --version")

print("Installing / updating python packages using pip")
os.system("python -m pip install emoji nltk textblob language-tool-python")
os.system("python -m pip install joblib")

print("Fetching the NLP Profiler source from neomatrix369/awesome-ai-ml-dl")
os.system("wget https://raw.githubusercontent.com/neomatrix369/awesome-ai-ml-dl/master/examples/better-nlp/library/org/neomatrix369/nlp_profiler.py")
os.system("mkdir -p better-nlp/library/org/neomatrix369/")
os.system("mv nlp_profiler.py better-nlp/library/org/neomatrix369/")

import joblib
memory = joblib.Memory("/tmp", compress=9, verbose=0)

import sys
import pandas as pd

sys.path.insert(0, './better-nlp/library') # we need this when running in Google Colab like environments

from org.neomatrix369 import nlp_profiler 
print("Successfully imported, ready to be used.")


class NLPProfiler:
    """
        Profile text data in a dataframe by it's column name
    """
    def apply_text_profiling(self, dataframe, text_column, params={}):
        """
            Perform parsing of the text_column present in the dataframe

            Parameters
            ==========
            dataframe:
               source input dataframe containing the text column specified
            text_column:
               name of the text column to parse
            params: 
                (optional) parameters to calibrate the processing and the resulting dataframe.
                Options: 'high_level' or 'granular'
                Usage: params={'high_level': True, 'granular': False}
                
                high_level: returns the dataframe with high-level concepts like sentiment analysis, spell check, grammar check etc...
                            This operation takes long to finish.
                granular: returns the dataframe with granular/statistical metrics like number of words, number of sentences, number of chars, etc...
                          This operation does not take a long time, even though we have a lot of metrics to produce.

            Return
            ======
            Dataframe with multiple columns, each column represents a metric or categorical representation of the text data parsed
        """
        
        cached_call = memory.cache(nlp_profiler.apply_text_profiling)
        return cached_call(dataframe, text_column, params)