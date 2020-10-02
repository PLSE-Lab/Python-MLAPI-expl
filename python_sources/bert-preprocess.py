import numpy as np
import sys

import os
import json
import collections
import itertools
import tensorflow as tf

#sys.path.append('../input/bert-joint-baseline/')
#import tokenization

def file_iter(input_file, tqdm=None, num_lines=None):
    """
    Yields jsonl file lines.
    """
    
    input_paths = tf.io.gfile.glob(input_file)

    def _open(path):
        if path.endswith(".gz"):
            return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
        else:
            return tf.io.gfile.GFile(path, "r")

    for path in input_paths:
        print("Reading: %s"% path)
        with _open(path) as input_file:
            if tqdm is not None:
                input_file = tqdm(input_file)
            for line in input_file:
                yield line
                if num_lines is not None:
                    if num_lines<=1:
                        raise StopIteration
                    num_lines -= 1
    

            

class FeatureWriter(object):
    """
    Writes a feature dictionary to TF example file.
    """

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1
        tf_train =  tf.train

        tf_feature = collections.OrderedDict()
        for k,v in feature.items():
            if not hasattr(v,'__iter__'):
                v = [v]
            tf_feature[k] = tf_train.Feature(int64_list=tf_train.Int64List(value=list(v)))

        tf_example = tf_train.Example(features=tf_train.Features(feature=tf_feature))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()

class Counter:
    def __init__(self,max_value=None):
        self.cnt = 0
        self.max_value=max_value
    @property
    def value(self):
        if self.max_value is None  or self.cnt<self.max_value:
            self.cnt +=1
        return self.cnt
    def reset(self):
        self.cnt=0

class ContextType:
    """
    Creates the context type token
    """
    def __init__(self,max_value=None):
        self.counters = {
            'Table':     Counter(max_value),
            'Paragraph': Counter(max_value),
            'List':      Counter(max_value),
            'Other':     Counter(max_value)
        }
        self.types = {
            '<table>': 'Table',
            '<p>':  'Paragraph',
            '<ol>': 'List',
            '<ul>': 'List',
            '<dl>': 'List'
        }
        
    def reset(self):
        for cnt in self.counters.values():
            cnt.reset()
        return self

    def __call__(self,x):
        cntx_type = self.types.get(x,'Other')
        return '[%s=%d]'%(cntx_type,self.counters[cntx_type].value)

def new_feature(uid, seq_len, q_toks):
    """
    Creates a new feature dictionary.
    The query part is filled
    """
    input_ids   = np.zeros(seq_len,dtype=np.int32)
    input_mask  = np.ones(seq_len,dtype=np.int8)
    segment_ids = np.ones(seq_len,dtype=np.int8)
    
    q_len = len(q_toks)
    input_ids[:q_len] = q_toks
    segment_ids[:q_len] = 0 
    
    token_map = np.empty(seq_len,dtype=np.int32)
    token_map.fill(-1)
    
    return dict(unique_id=uid,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                token_map=token_map)
    
class JSON2Features:
    """
    Converts a line from a jsonl file to a list of feature dictionaries. 
    Each feature is a dictionary with
        unique_id, input_ids, input_mask, segment_ids, token_map
    keys. Other elements of the InputFeature object are not computed as 
    they are not needed for inference in BERT model.
    
    tokenizer: a FullTokenizer from tokenization.
    other parameters are the elements of FLAGS needed here.
    These are defaults for BERT joint baseline
    FLAGS = (   skip_nested_contexts=True,
                max_position=50,
                max_contexts=48,
                max_query_length=64,
                max_seq_length=512,
                doc_stride=128,
                include_unknowns=-1.0,
            )

    """
    def __init__(self,
                 tokenizer,
                 skip_nested_contexts = True,
                 max_query_length = 64,
                 max_seq_length = 512,
                 max_position = 50,
                 max_contexts = 48,
                 doc_stride = 128,
                 **kwargs):
        self.tokenizer = tokenizer
        self.skip_nested_contexts = skip_nested_contexts
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.max_contexts = max_contexts
        self.doc_stride = doc_stride
        self.cntx_type = ContextType(max_value=max_position)
        self.unk = tokenizer.vocab.get(tokenizer.wordpiece_tokenizer.unk_token)
        self.cls = tokenizer.vocab.get('[CLS]')
        self.q   = tokenizer.vocab.get('[Q]')
        self.sep = tokenizer.vocab.get('[SEP]')
        self.token_start = np.array([tokenizer.vocab.get('[ContextId=-1]'),
                                     tokenizer.vocab.get('[NoLongAnswer]')])
        
    def tokenize_special_token(self,x):
        tokenizer =  self.tokenizer
        y = tokenizer.vocab.get(x,self.unk)
        return y
    
    def __call__(self,line):

        record = json.loads(line,object_pairs_hook=collections.OrderedDict)
        self.process_long_answer_candidates(record)
        self.join_contexts(record)
        self.process_question(record)
        return self.create_features(record)
    
    def join_contexts(self,record):
        lans = record.pop('long_answer_candidates')
        record['token_id']  = np.concatenate([self.token_start]+[x['token_id'] for x in lans])
        record['token_pos'] = np.concatenate([np.array([-1,-1])]+[x['token_pos'] for x in lans])
        return record
    
    def create_features(self,record):
        seq_len = self.max_seq_length
        q_toks = record['query_tokens']
        len_q = len(q_toks)
        feature_free_space = seq_len-len_q-1 ## final [SEP]
        doc_stride = self.doc_stride
        
        uid  = int(record['example_id'])
        token_id = record['token_id']
        token_map = record['token_pos']
        
        features = []
        
        ## doc_spans (0,ffs),dots, k*doc_stride,k*doc_stride+ffs
        ## where k*doc_stride < len(token_id)<= k*doc_stride+ffs
        ## i.e. len(token_id)-ffs <= k*doc_stride < len(token_id)-ffs + doc_stride
        for i,start in enumerate(range(0,
                                       max(0,len(token_id)-feature_free_space)+doc_stride,
                                       doc_stride)):
            feature = new_feature(uid=uid+i,seq_len=seq_len,q_toks=q_toks)
            size = min(feature_free_space, len(token_id)-start)
            feature['input_ids'][len_q:len_q+size] = token_id[start:start+size]
            feature['token_map'][len_q:len_q+size] = token_map[start:start+size]
            feature['input_ids'][len_q+size] = self.sep
            features.append(feature)
        
        if size<feature_free_space:
            feature['input_mask'][len_q+size+1:] = 0
            feature['segment_ids'][len_q+size+1:] = 0
        
        return features
    
    def process_question(self,record):
        tokenizer = self.tokenizer
        text = record.pop('question_text')
        token_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        # print(token_list)
        # token_list = list(itertools.chain(*token_list))
        if len(token_list) > self.max_query_length:
            token_list = token_list[-self.max_query_length:]
        token_list = [self.cls,self.q] + token_list + [self.sep] 
        token_ids = np.array(token_list,dtype=np.int32)
        record['query_tokens'] = token_ids 
        
    def process_answers(self,record):
        pass

    def process_long_answer_candidates(self,record):
        token_list = record.pop('document_text').lower().split()
        token_pos  = np.arange(len(token_list))

        tokens,token_map = np.unique(token_list,return_inverse=True)

        ## remove html tags
        ## html_idx = ((tokens>='<')*(tokens<'='))
        ## This is not perfect, but that is the way it is done in the original script
        ## It matters as some links has the form < http:...>,<https:.../ >
        html_idx = np.array(['<' in tok for tok in tokens])
        token_pos[html_idx[token_map]] = -1
        html_idx = html_idx.nonzero()[0]

        ## processing long answer candidates
        lans = record['long_answer_candidates']
        cntx_type = self.cntx_type.reset()
        for i,x in enumerate(lans):
            x['id'] = i

        if self.skip_nested_contexts:
            lans = [x for x in lans if x['top_level']]

        for x in lans:
            idx = token_pos[x['start_token']:x['end_token']]
            x['token_pos']   = idx[idx>-1]

        lans = [x for x in lans if len(x['token_pos'])>0]
        for x in lans:
            x['context_type'] = cntx_type(token_list[x['start_token']])
            
        ## there is an initial context '[ContextId=-1] [NoLongAnswer]'
        if len(lans)>=self.max_contexts:
            lans = lans[:self.max_contexts-1]
        
        for x in lans:
            x['context_head'] = [
                self.tokenize_special_token('[ContextId=%d]' % x['id']),
                self.tokenize_special_token(x.pop('context_type'))
                                ] 
            
        ## record for each token if it should be further tokenized by the tokenizer
        keep_token = np.empty(len(tokens))
        keep_token.fill(False)
        for x in lans:
            keep_token[token_map[x['token_pos']]] = True

        ## inv_token_map tells for each token position, the index of that token in tokens_ids
        token_idx = keep_token.nonzero()[0]
        inv_token_map = np.zeros(len(tokens),dtype=np.int32)
        inv_token_map[token_idx] = np.arange(len(token_idx))
        inv_token_map = inv_token_map[token_map]

        ## token_ids is the list of further tokenized tokens
        tokenizer = self.tokenizer
        token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in tokens[token_idx]]
        token_ids_len = [len(x) for x in token_ids]

        for x in lans:
            x_idx = inv_token_map[x['token_pos']]
            c_head = x.pop('context_head')
            x['token_pos'] = np.fromiter(itertools.chain([-1]*len(c_head),
                                                         *[[idx]*token_ids_len[k] 
                                                              for idx,k in zip(x['token_pos'],x_idx)]),
                                            dtype=np.int32)
            x['token_id']    = np.fromiter(itertools.chain(c_head,
                                                           *[token_ids[k] for k in x_idx]),
                                           dtype=np.int32)
        
        record['long_answer_candidates'] = lans
        return record

#if __name__ == '__main__':
#   some test    