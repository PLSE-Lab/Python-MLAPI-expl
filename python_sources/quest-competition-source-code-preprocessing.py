#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re, html
import numpy as np
import pandas as pd


# Source code had a large coverage in the SO data. In the write-ups I've seen so far, most people did not first further train (masked language model) their Bert models on the StackExchange data prior to finetuning their classifiers, which would mean that the models wouldn't really have a clear understanding of what's happening in the source. I didn't have the time or compute to do a SX LM either. Sometimes there would be blocks of source code that 10, 20, even 40 lines long. I felt that this would cause overfitting. Even more so in cases like RoBERTa that include whitespace in the tokenizers. Due to this, early on I invested some time in creating some parsers that would identify source code.

# Once source is identified, you have some options. For example, if you have a link, do you just want to replace it with URL? Would you rather replace it with a special token, e.g. [URL]? Or would you rather do further regex parsing, such as convert it to the domain name.

# Some code is easy to identify such as php blocks, or javascript embedded into HTML. Other code is very difficult to identify because there are no opening and closing tags and people just start typing code right into their sentences. Do handle this, I split the text into lines and then run a bunch of regex statements on each line to see if the line should be 'classified' as code or not. If there are multiple concurrent source code lines then we do something. In the example below, any additional lines after a threshold number of lines is truncated, as an example to curve overfitting. This could be extended by, e.g. replacing the entire code block with a special token.

# In[ ]:


# allow first line not to be counted as source(?)
multiline = re.compile('\n{2,}')

code_html = re.compile(
    r'(\wclass\s*=)|(\wid\s*=)|(\whref\s*=)|(^\s*<[a-z]+\w.*[a-z]>\s*$)|(</a>)|(</ul>)|(</li>)|(<hmtl)|(</div>)|(</span>)|(<[a-z]+.*[a-z]+\s*=.*>)', re.IGNORECASE
)
code_python = re.compile(
    r'("\s*,\s*")|(\'\s*,\s*\')|((\'|")\s*:)|([\'"][\])])|((\[|\()(\'|"))|(^\s*[{}\(\)\[\]]\s*$)|(=\s*")|(=\s*\')|(=\s*\()|(=\s*{)|(=\s*\[)|(^\s*return\b)|(\[\s*[0-9xyzijk\-]+\s*\])|([a-z_0-9]+\.[a-z_0-9]+\()|(^\s*else\s*:)|(=\s*[{\[(])|(^\s*[a-z_0-9]+\s*\.[a-z_0-9]+\s*\()|(^\s*[a-z_0-9]+\s*\.[a-z_0-9]+\s*[=+/*])|(^\s*#+)|(^\s*def\s*[a-z_0-9]+)|(\welif\w)|(\s*if.*:\s*$)|(^\s*class [a-z_0-9\(\)]:\s*$)|(for\s+.*\s+in\s+.*:\s*$)|(^\s*[a-z_0-9]+\(.*\)\s*$)|(^\s*def\s*[a-z_0-9]*\s*\()|([a-z_0-9]+\.[a-z_0-9]+\s*[=+/\-])', re.IGNORECASE
)
code_php = re.compile(
    # We get 99% of c/c++/php for free by catching ending on ";"
    r'(\(\s*[\'"])|([\'"]\s*\))|({\s*$)|(\)\s*{)|(;\s*$)|(=>)|(else\s*[\({])|(\$[a-z]+)|([a-z0-9_]{3,}\()|(^\s*(//|/\*))|(\*/$)', re.IGNORECASE
)
code_java = re.compile(
    r'(^\s*(var|const)\b)|(optional<|list<|set<|map<|queue<)|(\(\s*{|}\s*\))|(^\s*\@[a-z])', re.IGNORECASE
)

def detsrc(line):
    strip_len = len(line.strip())

    if strip_len==0: return True    
    if code_html.search(line): return True
    
    if strip_len > 150: return False
    if code_python.search(line): return True
    if code_php.search(line): return True
    if code_java.search(line): return True
    return False

# @numba.jit
def strip_code(text):
    lowertext = text.lower()
 
    # Strip HTML first, because it can encapsulate JS and PHP
    while True:
        pos1 = lowertext.find('<html')
        if pos1 == -1: break
            
        pos2 = lowertext.find('/html>', pos1+5)
        if pos2 == -1: break
        
        # Add the token
        codelen = pos2-pos1-5
        tag = f' sample code  '
        text = text[:pos1] + tag + text[pos2+6:]
        lowertext = text.lower()
        
    # Strip PHP next cause it's easy
    while True:
        pos1 = lowertext.find('<?php')
        if pos1 == -1: break
            
        pos2 = lowertext.find('?>', pos1+5)
        if pos2 == -1: break
        
        # Add the token
        codelen = pos2-pos1-5
        tag = f' sample code '
        text = text[:pos1] + tag + text[pos2+2:]
        lowertext = text.lower()
        
    # Strip Script next cause it's easy
    while True:
        pos1 = lowertext.find('<script')
        if pos1 == -1: break
            
        pos2 = lowertext.find('</script>', pos1+7)
        if pos2 == -1: break
        
        # Add the token
        codelen = pos2-pos1-7
        tag = f' sample code '
        text = text[:pos1] + tag + text[pos2+9:]
        lowertext = text.lower()
        
    # Strip links
    while True:
        pos1 = lowertext.find('<a ')
        if pos1 == -1: break
            
        pos2 = lowertext.find('</a>', pos1+3)
        if pos2 == -1: break
        
        # TODO: Add the domain
        tag = f' url '
        text = text[:pos1] + tag + text[pos2+4:]
        lowertext = text.lower()
        
    # Strip images
    while True:
        pos1 = lowertext.find('<img')
        if pos1 == -1: break
            
        pos2 = lowertext.find('>', pos1+4)
        if pos2 == -1: break
        
        # TODO: Add the alt if present, otherwise the domain
        tag = f' image '
        text = text[:pos1] + tag + text[pos2+1:]
        lowertext = text.lower()

    # Detect and strip specific languages
    # Replace multi-whitespace with single whitespace using multiline
    # We do this hear rather than above for various reasons:
    codelines = []
    lines = multiline.sub('\n', text).split('\n')
    
    stretch = 0
    for line in lines:
        if detsrc(line):
            stretch += 1
        else:
            stretch = 0
        codelines.append(stretch)

    # TODO: IMPORTANT
    # IF WE FILTER ALL LINES, THEN WE HAVE A PROBLEM.
    # WE SHOULD HAVE AT LEAST 1 NON-CODE LINE, OTHERWISE, DROP FILTERING
    
    return '\n'.join([
        line
        for idx_line, line in enumerate(lines)
        if codelines[idx_line]<5
    ])


# In[ ]:


train = pd.read_csv("../input/google-quest-challenge/train.csv")
test  = pd.read_csv("../input/google-quest-challenge/test.csv")


# In[ ]:


train.question_body = train.question_body.apply(html.unescape).apply(strip_code)
train.answer        = train.answer.apply(html.unescape).apply(strip_code)
test.question_body  = test.question_body.apply(html.unescape).apply(strip_code)
test.answer         = test.answer.apply(html.unescape).apply(strip_code)


# The above code isn't optimized at all but runs fast enough on the competition data.

# In[ ]:


train[['question_body','answer']].to_csv('train_clean.csv')
test[['question_body','answer']].to_csv('test_clean.csv')

