# %% [code]
#!/usr/bin/env python
# coding: utf-8
"""
this is a collection of dependencies for CORD-19 challenge
"""

import os

from concurrent.futures import ThreadPoolExecutor, as_completed
n_cpus = os.cpu_count()
print(f'Number of CPUs: {n_cpus}')
executor = ThreadPoolExecutor(max_workers=n_cpus)

import numpy as np
from pathlib import Path
import json
import sys
import joblib
import ahocorasick

datapath = Path('/kaggle/input/')

print("Loading Aho Corasic Automata")
A=joblib.load("/kaggle/input/cord19acautomatalargemodel/automata_ent_only.pkl")
print("Automata properties")
print(A.get_stats())

def find_matches(sent_text, A=A):
    """
    find longest match using Aho Corasic automata
    """
    matched_ents = []
    for char_end, (eid, ent_text) in A.iter(sent_text):
        char_start = char_end - len(ent_text)
        matched_ents.append((eid, ent_text, char_start, char_end))
    # remove shorter subsumed matches
    longest_matched_ents = []
    for matched_ent in sorted(matched_ents, key=lambda x: len(x[1]), reverse=True):
        # print("matched_ent:", matched_ent)
        longest_match_exists = False
        char_start, char_end = matched_ent[2], matched_ent[3]
        for _, _, ref_start, ref_end in longest_matched_ents:
            # print("ref_start:", ref_start, "ref_end:", ref_end)
            if ref_start <= char_start and ref_end >= char_end:
                longest_match_exists = True
                break
        if not longest_match_exists:
            # print("adding match to longest")
            longest_matched_ents.append(matched_ent)
    return [t for t in longest_matched_ents if len(t[1])>3] 


