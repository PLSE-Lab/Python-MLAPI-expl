
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from collections import defaultdict, OrderedDict
import spacy
from numba import cuda, jit,prange
from timeit import default_timer as timer
import json
from config import STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity
from helper_functions import upload_file
import en_core_web_sm

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
nlp = en_core_web_sm.load()

def pooling(embedding, method="mean"):

    if method == "mean":
        embed = np.nanmean(embedding, axis=1)

    elif method == "max":
        embed = np.nanmax(embedding, axis=1)

    else:
        embed = embedding

    return embed


@jit
def embed_text(text, model, tokenizer, max_len):
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    
    encoded_text = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = False,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    #encoded_text_2 = tokenizer.encode(text, add_special_tokens=True)
    input_ids = encoded_text["input_ids"]
    
    # Update the maximum sentence length.
    #max_len = max(max_len, len(encoded_text))
    
    #input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Prediction
    outputs = model(input_ids)

    # The last hidden-state is the first element of the output tuple
    last_hidden_states = outputs[0]

    return last_hidden_states


def get_similarity(em, em2):
    return cosine_similarity(em, em2)


def join_title_abstract(x):

    if np.logical_and(pd.isnull(x["title"]), pd.isnull(x["abstract"])):
        return ""
    elif pd.isnull(x["title"]):
        return x["abstract"]
    elif pd.isnull(x["abstract"]):
        return x["title"]
    else:
        return ". ".join([x["title"], x["abstract"]])


@jit(parallel=True)
def custom_tokenize(x, stopwords, tokenizer, spacytokenizer):
    """
    max_len = 512 is the max tokens that bert can take
    """
    str_to_list = np.array(
        [i for i in str(x).lower().split(" ") if np.logical_not(i in stopwords)]
    )
    
    join_remove_stopwords = " ".join(str_to_list)
    scispacy_tokens = str(spacytokenizer(str(join_remove_stopwords)))
    bert_tokens = tokenizer.tokenize(scispacy_tokens)

    return bert_tokens


@jit(parallel=True)
def make_embedding(
    k, df, nsize, model, tokenizer, stopwords, nbatch, spacytokenizer, l_pooling, max_len
):
    embedded_dict = defaultdict(list)

    for j in range(nbatch):
        if k * nbatch + j < nsize:
            
            df_text = df.iloc[k * nbatch + j]
            
            idx  = df.index[k * nbatch + j]
            
            stokens = custom_tokenize(df_text, stopwords, tokenizer, spacytokenizer)

            embedded_text = embed_text(stokens, model, tokenizer, max_len).detach().tolist()

            if l_pooling:
                embedded_pooled = pooling(embedded_text).tolist()
            
            embedded_dict[idx] = {
                "stokens": stokens,
                "embedding": embedded_text,
                "embedded_pooled":embedded_pooled,
            }

        else:
            break

    return embedded_dict  # stokens_dict,


def embed_loop(df, nsize, model, tokenizer, stopwords, nbatch=128, max_len = 128, l_pooling=True):
    if nsize % nbatch == 0:
        nloops = nsize // nbatch
    else:
        nloops = nsize // nbatch + 1

    if l_pooling:
        df_embedding = pd.DataFrame()

    else:
        df_embedding = None

    spacytokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

    for k in prange(nloops):
        print("%d/%d" % (k + 1, nloops))

        embedded_dict = make_embedding(
            k,
            df,
            nsize,
            model,
            tokenizer, 
            stopwords,
            nbatch,
            spacytokenizer,
            l_pooling,
            max_len,
        )

        list_pooled_embed = [np.squeeze(embedded_dict.get(k)["embedded_pooled"]) for k in embedded_dict.keys()]

        df_embed_temp = pd.DataFrame(list_pooled_embed, index=embedded_dict.keys(), columns=np.arange(768))
        df_embedding = pd.concat([df_embedding, df_embed_temp])

    return df_embedding

def top_n_closest(search_term_embedding, df_embedding, df_meta, n=10):

    proximity_dict = {}
    i = 0

    for k, v in df_embedding.iterrows():
        idx = df_meta.index[i]
        proximity_dict[idx] = {
            "Title": df_meta.loc[idx].title,
            "Abstract": df_meta.loc[idx].abstract,
            "Similarity": get_similarity(
                v.values.reshape(1,-1), search_term_embedding 
            ).flatten()[0],
        }
        i += 1

    order_dict = OrderedDict(
        {
            k: v
            for k, v in sorted(
                proximity_dict.items(),
                key=lambda item: item[1]["Similarity"],
                reverse=True,
            )
        }
    )

    proper_list = list(order_dict.keys())[:n]

    return proper_list, order_dict