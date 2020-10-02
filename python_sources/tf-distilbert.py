# Kernel to provide a handy utility function to load pretrained DistilBert model and tokenizer

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import transformers as ppb # pytorch transformers


# Any results you write to the current directory are saved as output.

PATH = '/kaggle/input/tfdistilbert-base/'
DATA = {'cased':{ 'model'  : 'distilbert-base-cased-tf_model.h5',
                 'config' : 'distilbert-base-cased-config.json',
                  'vocab'  : 'distilbert-base-cased-vocab.txt'
                },
        'uncased':{ 'model'  : 'distilbert-base-uncased-tf_model.h5',
                    'config' : 'distilbert-base-uncased-config.json',
                    'vocab'  : 'distilbert-base-uncased-vocab.txt'
                  }
        }

    
def get_pretrained_model(cased = True):
    config = ppb.BertConfig.from_json_file(PATH + DATA['cased' if cased else 'uncased']['config'])
    model = ppb.TFDistilBertModel.from_pretrained(PATH + DATA['cased' if cased else 'uncased']['model'], config=config)
    return model

def get_pretrained_tokenizer(cased = True):
    tokenizer = ppb.DistilBertTokenizer.from_pretrained(PATH + DATA['cased' if cased else 'uncased']['vocab'])
    return tokenizer