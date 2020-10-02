import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt

# get each chromosome's name & sequence from genome.fa
# only a few really matter (2L,2R,3L,3R,4,X,Y, )
chroms = ["chr2L","chr2R","chr3L","chr3R","chr4","chrX","chrY"]
chrom_seqs = []
for seq_record in SeqIO.parse("../input/genome.fa","fasta"):
    chrom_seqs.append(seq_record.lower())
    #print("Id: " + seq_record.id + " \t " + "Length:" + \
    #    str("{:,d}".format(len(seq_record))))
    #print(repr(seq_record.lower().seq) + "\n")

# get meta-simple-repeat.csv
repeats = pd.read_csv("../input/meta-simple-repeat.csv")

# look at some stats
stats_by_chrom = pd.DataFrame(columns=chroms)
stats_by_chrom.loc["num_repeats"] = 0
stats_by_chrom.loc["num_repeated_letters"] = 0
stats_by_chrom.loc["length"] = 0
stats_by_chrom.loc["prop_repeat"] = 0

for chrom in chroms:
    # count repeats for each chrom
    repeats_chrom = repeats[repeats["chrom"]==chrom]
    stats_by_chrom.set_value("num_repeats",chrom,len(repeats_chrom))
    
    # count the number of letters that repeats account for
    num_repeated_letters = 0
    for index,row in repeats_chrom.iterrows():
        num_repeated_letters += row["chromEnd"]-row["chromStart"]+1
    stats_by_chrom.set_value("num_repeated_letters",chrom,num_repeated_letters)
    
    # calculate percentage repeated
    chrom_seq = [x for x in chrom_seqs if x.name == chrom][0]
    stats_by_chrom.set_value("length",chrom,len(chrom_seq))
    stats_by_chrom.set_value("prop_repeat",chrom,\
        stats_by_chrom.at["num_repeated_letters",chrom]/len(chrom_seq))
#print(stats_by_chrom)

# some plots
ind = np.arange(len(chroms))
width = 0.45 # bar width
fig,ax = plt.subplots()
rects = ax.bar(ind,stats_by_chrom.loc["num_repeats"],width,color='b',align='center')
ax.set_ylabel("tandem repeats")
ax.set_xlabel("chromosome")
ax.set_title("number of tandem repeats by chromosome")
ax.set_xticklabels([""]+chroms) # [""]+ is to correct the shift caused by center align
ax.set_yticks([0,2000,4000,6000,8000,10000,12000])
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,1.03*height,\
            '%d' % int(height), ha='center',va='bottom')
autolabel(rects)
plt.savefig("num_repeats.png")
#
ind = np.arange(len(chroms))
width = 0.45 # bar width
fig,ax = plt.subplots()
rects = ax.bar(ind,stats_by_chrom.loc["num_repeated_letters"],width,color='b',align='center')
ax.set_ylabel("tandem repeat letters")
ax.set_xlabel("chromosome")
ax.set_title("number of letters that make up tandem repeats")
ax.set_xticklabels([""]+chroms)
#plt.autoscale() # so the y-axis label doesn't get cut off
ax.set_yticks([0,200000,400000,600000,800000,1000000,1200000])
autolabel(rects)
plt.tight_layout()
plt.savefig("num_repeated_letters.png")
#
ind = np.arange(len(chroms))
width = 0.45 # bar width
fig,ax = plt.subplots()
rects = ax.bar(ind,stats_by_chrom.loc["length"],width,color='b',align='center')
ax.set_ylabel("length")
ax.set_xlabel("chromosome")
ax.set_title("length of chromosome sequence")
ax.set_xticklabels([""]+chroms)
autolabel(rects)
plt.savefig("length.png")
#
ind = np.arange(len(chroms))
width = 0.45 # bar width
fig,ax = plt.subplots()
rects = ax.bar(ind,stats_by_chrom.loc["prop_repeat"],width,color='b',align='center')
ax.set_ylabel("proportion repeated")
ax.set_xlabel("chromosome")
ax.set_title("proportion of chromosomes consisting of tandem repeats")
ax.set_xticklabels([""]+chroms)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,1.03*height,\
            round(height,3), ha='center',va='bottom')
autolabel(rects)
plt.savefig("prop_repeats.png")
#