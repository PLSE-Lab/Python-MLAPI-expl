#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, "../input/sentencepiece-pb2/")


# In[ ]:


import os
import sentencepiece as spm
import sentencepiece_pb2


# In[ ]:


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path, "spiece.model"))
    
    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return tokens, offsets


# In[ ]:


spt = SentencePieceTokenizer("../input/albert-large-v1/")


# In[ ]:


spt.encode("hi, how are you?")

