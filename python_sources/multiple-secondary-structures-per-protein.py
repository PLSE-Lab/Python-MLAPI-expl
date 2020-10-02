# Just run the whole script to see how many proteins have ambiguous secondary structures

import numpy as np
import pandas as pd

df = pd.read_csv('../input/2018-06-06-ss.cleaned.csv')

# Check for duplicate amino acid sequences that have different secondary structures

aa_sequence_sst_map = {}
aa_sequences_with_ambiguous_ssts = {}

for itertuple in df.itertuples():
    aa_sequence = getattr(itertuple, 'seq')
    sst = getattr(itertuple, 'sst3')
    try:
        if sst not in aa_sequence_sst_map[aa_sequence]:
            aa_sequence_sst_map[aa_sequence].append(sst)
            aa_sequences_with_ambiguous_ssts[aa_sequence] = 0
    except:
        aa_sequence_sst_map[aa_sequence] = [sst]

print('Found', len(aa_sequences_with_ambiguous_ssts.keys()), 'of a total of', len(df['seq'].unique()), 'proteins to have multiple secondary structures.')