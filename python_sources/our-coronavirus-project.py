#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os.path as path
import os

cwd = os.getcwd() # get the current working directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt # graphs plotting
get_ipython().run_line_magic('matplotlib', 'inline')
from Bio import SeqIO # some BioPython that might come in handy


# taken from [Zhang Lab](https://zhanglab.ccmb.med.umich.edu/FASTA/)
# 
# FASTA format is a text-based format for representing either nucleotide sequences or peptide sequences, in which base pairs or amino acids are represented using single-letter codes. A sequence in FASTA format begins with a single-line description, followed by lines of sequence data. The description line is distinguished from the sequence data by a greater-than (">") symbol in the first column. It is recommended that all lines of text be shorter than 80 characters in length.
# An example sequence in FASTA format is:
# 
# >S|spike_protein|MG996765|AWV66922.1|A0A3G1RFS9|P138/72|NA|1972|Horse|Switzerland|NA|Equine_torovirus
# MFLCFCAATVLCFWINSGGADVVPNGTLIFSEPVPYPFSLDVLRSFSQHVVLRNKRAVTTISWSYSYQIT
# TSSLSVNSWYVTFTAPLGWNYYTGQSFGTVLNQNAMMRASQSTFTYDVISYVGQRPNLDCQVNSLVNGGL
# DGWYSTVRVDNCFNAPCHVGGRPGCSIGIPYMSNGVCTRVLSTTQSPGLQYEIYSGQQFAVYQITPYTQY
# 
# 
# Sequences are expected to be represented in the standard IUB/IUPAC amino acid and nucleic acid codes, with these exceptions:
# lower-case letters are accepted and are mapped into upper-case;
# a single hyphen or dash can be used to represent a gap of indeterminate length;
# in amino acid sequences, U and * are acceptable letters (see below).
# any numerical digits in the query sequence should either be removed or replaced by appropriate letter codes (e.g., N for unknown nucleic acid residue or X for unknown amino acid residue).
# The nucleic acid codes are:
# 
#         A --> adenosine           M --> A C (amino)
#         C --> cytidine            S --> G C (strong)
#         G --> guanine             W --> A T (weak)
#         T --> thymidine           B --> G T C
#         U --> uridine             D --> G A T
#         R --> G A (purine)        H --> A C T
#         Y --> T C (pyrimidine)    V --> G C A
#         K --> G T (keto)          N --> A G C T (any)
#                                   -  gap of indeterminate length
# The accepted amino acid codes are:
# 
#     A ALA alanine                         P PRO proline
#     B ASX aspartate or asparagine         Q GLN glutamine
#     C CYS cystine                         R ARG arginine
#     D ASP aspartate                       S SER serine
#     E GLU glutamate                       T THR threonine
#     F PHE phenylalanine                   U     selenocysteine
#     G GLY glycine                         V VAL valine
#     H HIS histidine                       W TRP tryptophan
#     I ILE isoleucine                      Y TYR tyrosine
#     K LYS lysine                          Z GLX glutamate or glutamine
#     L LEU leucine                         X     any
#     M MET methionine                      *     translation stop
#     N ASN asparagine                      -     gap of indeterminate length

# # **Reading the data**

# In[ ]:


# Read the fasta-file and create a dictionary of its protein sequences

# path for the input file on Kiril's laptop
input_file_name = 'E:\!2020 SPRING\8630 Advanced Bioinformatics\Project\!DATA\Aligned Spike Proteins (StrainName-AccessionNumber-HostSpecies-VirusSpecies).fasta'

# However, if we are on kaggle, we use kaggle directory
if cwd == '/kaggle/working': 
    input_file_name = '../input/spike-proteins-in-conona-viruses/Aligned Spike Proteins (StrainName-AccessionNumber-HostSpecies-VirusSpecies).fasta'
#     input_file_name = '../input/spike-proteins-in-conona-viruses/upd1300.fasta'

    
# The fasta defline (name of a sequence) has the following format:
# Strain Name | Accession Number | Host Species | Virus Species) 
sequences_dictionary = {sequence.id : sequence.seq for sequence in SeqIO.parse(input_file_name,'fasta')}


# In[ ]:


# From the newly formed sequences_dictionary, we create 3 lists:
# a list of deflines,
# a list of sequences,
# and a list of target values

# We want to mark all sequences that belong to the viruses that can infect humans as 1 (i.e., target = 1), 
# all other sequences as 0 (i.e., target = 0)

deflines = [entry for entry in sequences_dictionary.keys()]             # create a list of deflines
protein_sequences = [entry for entry in sequences_dictionary.values()]  # create a list of protein sequences 
targets = [1 if 'Human' in entry else 0 for entry in deflines]          # create a list of target values


# In[ ]:


# Then we create a class fasta_sequence so that we would be able to use the sequence data easily 

class fasta_sequence:
    def __init__(self, defline, sequence, target, type_of_encoding = "onehot"):
        
        # we read the input data
        
        self.defline = defline
        self.sequence = sequence
        self.target = target
        
        # and create more descriptions of the input data
        
        # report the strain name (the 1st fiel of the defline)
        self.strain_name = defline.split("|")[0]
        # report the accession number (the 2nd fiel of the defline)
        self.accession_number = defline.split("|")[1]        
        # report the host species (the 3rd fiel of the defline)
        self.host_species = defline.split("|")[2]    
        # report the virus species (the 4th fiel of the defline)
        self.virus_species = defline.split("|")[3]
        
        
# We convert a string with the alphabet = 'ABCDEFGHIKLMNPQRSTUVWXYZ-' 
# !!!(B,X,Z are ``extra'' letters; we have to address this in the future) 
# into either a list mapping chars to integers (called integer encoding),
# or a sparce list. In the latter, each amino acid is represented as an one-hot vector of length 20, 
# where each position, except one, is set to 0.  E.g., alanine is encoded as 10000000000000000000, cystine is encoded as 01000000000000000000
# See the full table above.
# Symbol '-' is encoded as a zero-vector.

        def encoding(sequence, type_of_encoding):

            # define universe of possible input values
            alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-'
            # define a mapping of chars to integers
            char_to_int = dict((c, i) for i, c in enumerate(alphabet))


            # integer encoding
            integer_encoded = [char_to_int[char] for char in sequence]

            # one-hot encoding
            onehot_encoded = list()
            for value in integer_encoded:
                letter = [0 for _ in range(len(alphabet)-1)]
                if value != len(alphabet)-1:
                    letter[value] = 1
                onehot_encoded.append(letter)
            flat_list = [item for sublist in onehot_encoded for item in sublist]

            if type_of_encoding == "onehot":
                return flat_list
            else:
                return integer_encoded
            
        #  we use the encoding function to create a new attribute for the sequence -- its encoding        
        self.encoded = encoding(sequence, type_of_encoding)


# In[ ]:


# we create a list of sequences as objects of the class fasta_sequence
# all sequences are encoded with one-hot encoding (it is the default option of the constructor of the class)
sequences = []
for i in range(0, len(deflines)):
    current_sequence = fasta_sequence(deflines[i],protein_sequences[i],targets[i])
    sequences.append(current_sequence)


# # **Getting familiar with the data**

# In[ ]:


# Let's get the first look at out data

print("There are", len(sequences), "sequences in the fasta file")
print("Each of the sequences has", len(sequences[0].sequence), "cites")

import random
random_id = random.randrange(0,len(sequences),1) # generates a random number within the range of sequence numbers
print("\n\nBelow you can see an example of a random sequence number",random_id,"\n")
print("The Defline:\n",sequences[random_id].defline)
print("\nThe details of the Defline:\nThe strain name: ", sequences[random_id].strain_name), 
print("The accession number: ",sequences[random_id].accession_number)
print("The host species: ",sequences[random_id].host_species)
print("The virus species: ",sequences[random_id].virus_species)
print("\nThe Sequence:\n",sequences[random_id].sequence)
print("\nThe Target Value:\n",sequences[random_id].target)
# print("\nEncoding:\n",sequences[random_id].encoded)


# In[ ]:


# Here we print the names of the sequences infective to humans among the first 100 sequence 
for i in range(0,100):
    if sequences[i].target == 1:
        print(sequences[i].defline)


# In[ ]:


# create a list of strain names (the 4th fiel of the defline)
virus_species = [entry.virus_species for entry in sequences]    
# convert the list of strain names into a set 
virus_species_set = set(virus_species)

print("There are", len(virus_species_set)-1, "unique virus species in our dataset") # we don't count the NA entry
print("The list of all unique virus species in our dataset:\n", virus_species_set)


# In[ ]:


# create a list of strain names of the viruses that can affect humans
human_virus_species = [entry.virus_species for entry in sequences if 'Human' in entry.defline]
# turn this list into a set
human_virus_species_set = set(human_virus_species)

print("There are",len(human_virus_species_set)-1, "unique virus human-infective species in our dataset") # we don't count the NA entry
print("The list of all unique human-infective virus species in our dataset:\n", human_virus_species_set)


# There are only 7 known to be able to infect humans. 
# 
# The virus classification is gives as Genus -> Subgenus -> Species -> Subspecies -> Strain.
# The names that are in our database are hilighted **in bold**.
# 
# 
# * 229E (a.k.a. HCoV-229E, Human coronavirus 229E) belongs to  Alphacoronavirus -> Duvinacovirus -> **Human coronavirus 229E** -> NA -> NA
# * NL63 (a.k.a. HCoV-NL63, Human coronavirus NL63) belongs to  Alphacoronavirus -> Setracovirus -> **Human coronavirus NL63** -> NA -> NA
# * OC43 (a.k.a. HCoV-OC43, Human coronavirus OC43) belongs to  Betacoronavirus -> NA -> **Betacoronavirus 1** -> Human coronavirus OC43 -> NA
# * HKU1 (a.k.a. HCoV-HKU1, **Human coronavirus HKU1**) belongs to  Betacoronavirus -> Embecovirus -> **Human coronavirus HKU1** -> NA -> NA
# * MERS-CoV (a.k.a. EMC/2012, HCoV-EMC/2012, Middle East respiratory syndrome-related coronavirus)
#    belongs to  Betacoronavirus -> Merbecovirus -> **Middle East respiratory syndrome-related coronavirus** -> NA -> NA
# * SARS-CoV (a.k.a. SARS-CoV, SARS-CoV-1, Severe acute respiratory syndrome coronavirus)
#    belongs to  Betacoronavirus -> NA-> **Severe acute respiratory syndrome-related coronavirus** -> NA -> Severe acute respiratory syndrome coronavirus
# * **SARS-CoV-2** (a.k.a. 2019-nCoV, Severe acute respiratory syndrome coronavirus 2)
#    belongs to  Betacoronavirus -> Sarbecovirus -> Severe acute respiratory syndrome-related coronavirus -> NA -> Severe acute respiratory syndrome coronavirus 2
#                   
#                   
# 
# 

# In[ ]:


idx = pd.Index(virus_species) # creates an index which allows counting the entries easily
print('Here are all of the viral species in the dataset: \n', len(idx),"entries in total")
idx.value_counts()


# In[ ]:


# here we prepare the data to be plootted as a bar graph
y_labels = idx.value_counts().index.values # virus species names
counts = idx.value_counts().values    # numbers of occuriencies
counts_as_series = pd.Series(counts)


# In[ ]:


# plot the bar graph

plt.figure(figsize=(12, 9))
ax = counts_as_series.plot(kind ='barh')
ax.set_title('Distribution of viral species')
ax.set_xlabel('Number of entries')
#ax.set_ylabel('Species of virus')
ax.set_yticklabels(y_labels)
ax.set_xlim(-10, 295) # we change the x limits in order to make labels more readable

rectangles = ax.patches

# we place a label for each bar
for rectangle in rectangles:
    
    # we obtain x and y positions for the current label
    x_value = rectangle.get_width()
    y_value = rectangle.get_y() + rectangle.get_height()/2
    
    # we annotate a current bar in the bar graph
    plt.annotate(
        x_value,                    # we use x_value as a label
        (x_value, y_value),         # we place labels at end of the bars
        xytext=(5, 0),              # we shift the label horizontally by 5
        textcoords="offset points", # we interpret xytext as an offset in points
        va='center',                # we center the labels vertically 
        ha='left')                  # we specify the alignment for the labels                                   


# In[ ]:


#determining unique viral hosts
host_species = [entry.host_species for entry in sequences]    
host_species_set = set(host_species)

print("There are",len(host_species_set)-1, "unique viral hosts in our dataset") # we don't count the NA entry
print("The list of unique viral hosts in our dataset is: \n", host_species_set)


# In[ ]:


print('Here are all of the host species in the dataset:')
#display host species 
idx2 = pd.Index(host_species)
print(idx2.value_counts())

# here we prepare the data to be plootted as a bar graph
y_labels = idx2.value_counts().index.values # viral host species names
counts = idx2.value_counts().values    # numbers of occuriencies
counts_as_series = pd.Series(counts)


plt.figure(figsize=(12, 9))
ax = counts_as_series.plot(kind ='barh')
ax.set_title('The data distribution of viral hosts')
ax.set_xlabel('Number of entries')
#ax.set_ylabel('Species of viral host')
ax.set_yticklabels(y_labels)
ax.set_xlim(-10, 295) # we change the x limits in order to make labels more readable

rectangles = ax.patches

# we place a label for each bar
for rectangle in rectangles:
    
    # we obtain x and y positions for the current label
    x_value = rectangle.get_width()
    y_value = rectangle.get_y() + rectangle.get_height()/2
    
    # we annotate a current bar in the bar graph
    plt.annotate(
        x_value,                    # we use x_value as a label
        (x_value, y_value),         # we place labels at end of the bars
        xytext=(5, 0),              # we shift the label horizontally by 5
        textcoords="offset points", # we interpret xytext as an offset in points
        va='center',                # we center the labels vertically 
        ha='left')                  # we specify the alignment for the labels       


# In[ ]:


# How many viral sequences are marked as infective to humans?
# To answer it, we can simply count 1's in the target list:

print("We have got", targets.count(1), "entries that can infect humans") 


# In[ ]:


human_related_sequences = [entry for entry in sequences if 'Human' in entry.defline]  # create a list of human related sequences


# In[ ]:


# NA's in human sequences
countNAs = 0
human_NAs = []
for entry in human_related_sequences:
    if entry.virus_species == "NA":
        human_NAs.append(entry.accession_number)
        countNAs += 1 

print("In total there are",countNAs,"NA's in human related sequences.")
print("Assension numbers of NA's in human related sequences:")
print(human_NAs)


# In[ ]:


# create a list of strain names (the 4th fiel of the defline)
virus_species_human = [entry.virus_species for entry in  human_related_sequences]  
idx_human = pd.Index(virus_species_human) # creates an index which allows counting the entries easily
idx_human.value_counts()


# In[ ]:


multi_targets = []

for i in range(len(deflines)):
    vtl = deflines[i].lower()
    if 'human' in vtl:
        multi_targets.append('Human')
    elif 'bat' in vtl:
        multi_targets.append('Bat')
    elif 'avian' in vtl:
        multi_targets.append('Bird')
    elif 'camel' in vtl:
        multi_targets.append('Camel')
    elif 'porcine' in vtl or 'swine' in vtl or 'sus_scrofa_domesticus' in vtl:
        multi_targets.append("Pig")
    else:
        multi_targets.append('Other')


# In[ ]:


coro_types = []
for i in range(len(deflines)):
    vtl = deflines[i].lower()
    if 'alpha' in vtl:
        coro_types.append("Alpha")
    elif 'hku10' in vtl:
        coro_types.append("Alpha")
    # do these before hku1 and hku2 so it counts it right
    elif 'hku12' in vtl:
        coro_types.append("Delta")
    elif 'hku13' in vtl:
        coro_types.append("Delta")
    elif 'hku14' in vtl:
        coro_types.append("Beta")
    elif 'hku15' in vtl:
        coro_types.append("Delta")
    elif 'hku16' in vtl:
        coro_types.append("Delta")
    elif 'hku17' in vtl:
        coro_types.append("Delta")
    elif 'hku18' in vtl:
        coro_types.append("Delta")
    elif 'hku19' in vtl:
        coro_types.append("Delta")
    elif 'hku20' in vtl:
        coro_types.append("Delta")
    elif 'hku21' in vtl:
        coro_types.append("Delta")

    elif 'fcov' in vtl:
        coro_types.append("Alpha")
    elif 'nl63' in vtl:
        coro_types.append("Alpha")
    elif 'hku2' in vtl:
        coro_types.append("Alpha")
    elif 'hku11' in vtl:
        coro_types.append("Delta")
    elif 'beta' in vtl:
        coro_types.append("Beta")
    elif 'cattle' in vtl:
        coro_types.append("Beta")
    elif 'mouse' in vtl:
        coro_types.append("Beta")
    elif 'hku1' in vtl:
        coro_types.append("Beta")
    elif 'hku4' in vtl:
        coro_types.append("Beta")
    elif 'hku5' in vtl:
        coro_types.append("Beta")
    elif 'hku9' in vtl:
        coro_types.append("Beta")
    elif 'severe_acute_respiratory_syndrome' in vtl:
        coro_types.append("Beta")
    elif 'SARS' in vtl:
        coro_types.append("Beta")
    elif 'middle_east_respiratory_syndrome' in vtl:
        coro_types.append("Beta")
    elif 'camel' in vtl:
        coro_types.append("Beta")
    elif 'murine' in vtl:
        coro_types.append("Beta")
    elif 'chicken' in vtl:
        coro_types.append("Gamma")
    elif 'pheasant' in vtl:
        coro_types.append("Gamma")
    elif 'pigeon' in vtl:
        coro_types.append("Gamma")
    elif 'beluga' in vtl:
        coro_types.append("Gamma")
    elif 'tcov' in vtl:
        coro_types.append("Gamma")
    elif 'duck' in vtl:
        coro_types.append("Gamma")
    elif 'porcine' in vtl:
        coro_types.append("Delta")
    elif 'swine' in vtl:
        coro_types.append("Delta")
    elif 'sus_scrofa_domesticus' in vtl:
        coro_types.append("Delta")
    elif 'sparrow' in vtl:
        coro_types.append('Delta')
    elif 'human' in vtl:
        coro_types.append("Human Uncategorized")
    else:
        coro_types.append("N/A")


# In[ ]:


labels = ['Human', 'Bat','Camel', 'Other', 'Bird', 'Pig']
overall_size = len(multi_targets)
sizes = [multi_targets.count('Human') / overall_size,          multi_targets.count('Bat') / overall_size,          multi_targets.count('Camel') / overall_size,          multi_targets.count('Other') / overall_size,          multi_targets.count('Bird') / overall_size,          multi_targets.count('Pig') / overall_size]

explode = (0.1, 0, 0, 0, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[ ]:


ct_labels = ['Alpha', 'Beta', 'Gamma', 'Delta', 'N/A', 'Human Uncategorized']
ct_overall_size = len(coro_types)
ct_sizes = [coro_types.count('Alpha') / ct_overall_size,             coro_types.count('Beta') / ct_overall_size,             coro_types.count('Gamma') /ct_overall_size,             coro_types.count('Delta') / ct_overall_size,             coro_types.count('N/A') / ct_overall_size,             coro_types.count('Human Uncategorized') / ct_overall_size]

ct_explode = (0, 0, 0, 0, 0, 0)

ct_fig1, ct_ax1 = plt.subplots()
ct_ax1.pie(ct_sizes, explode = ct_explode, labels = ct_labels, autopct = '%1.1f%%',
        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ct_ax1.axis('equal')  
plt.tight_layout()
plt.show()


# # Encoding

# In[ ]:


# for a list of sequences, returns a list of encoded sequences and a list of targets
def EncodeAndTarget(list_of_sequences):
    # encoding the sequences
    list_of_encoded_sequences = [entry.encoded for entry in list_of_sequences]
    # creating lists of targets
    list_of_targets = [entry.target for entry in list_of_sequences]
    return list_of_encoded_sequences, list_of_targets   


# In[ ]:


# encoding ALL the sequences
encoded_sequences = [entry.encoded for entry in sequences]


# ### Random 20% of all sequences

# In[ ]:


number_of_seq = int(len(sequences)*0.20)
random20 = random.sample(sequences, number_of_seq)

# obtaining encoded sequences and lists of targets
encoded_random20, targets_random20 = EncodeAndTarget(random20)


# ### Human related

# In[ ]:


# obtaining encoded sequences and lists of targets
encoded_human_sequences, human_targets = EncodeAndTarget(human_related_sequences)


# ### Muting Human_coronavirus_HKU1

# In[ ]:


# creating two lists of sequenses: with and without Human_coronavirus_HKU1
sequences_without_HKU1 = [entry for entry in sequences if 'Human_coronavirus_HKU1' not in entry.defline] 
sequences_HKU1 = [entry for entry in sequences if 'Human_coronavirus_HKU1' in entry.defline] 

# obtaining encoded sequences and lists of targets
encoded_without_HKU1, targets_without_HKU1 = EncodeAndTarget(sequences_without_HKU1)
encoded_HKU1, targets_HKU1 = EncodeAndTarget(sequences_HKU1)


#  ### Muting SARS1

# In[ ]:


# creating two lists of sequenses: with and without Severe_acute_respiratory_syndrome_related_coronavirus
sequences_without_SARS1 = [entry for entry in sequences if 'Severe_acute_respiratory_syndrome_related_coronavirus' not in entry.defline] 
sequences_SARS1 = [entry for entry in sequences if 'Severe_acute_respiratory_syndrome_related_coronavirus' in entry.defline] 

# obtaining encoded sequences and lists of targets
encoded_without_SARS1, targets_without_SARS1 = EncodeAndTarget(sequences_without_SARS1)
encoded_SARS1, targets_SARS1 = EncodeAndTarget(sequences_SARS1)


#  ### Balanced Avian vs Human

# In[ ]:


# we create a list were as many avian related sequences as many human relared are in human_related_sequences
Avian_dataset = [entry for entry in sequences if 'Avian' in entry.defline]
Human_Avian_dataset = human_related_sequences + random.sample(Avian_dataset, k = targets.count(1))

# obtaining encoded sequences and lists of targets
balanced_encoded_sequences, balancedtargets = EncodeAndTarget(Human_Avian_dataset)


#  ### Mammals

# In[ ]:


# Manual observation of viral hosts species show 25 hosts with following names which are mammals: 
# Chimpanzee, Anteater, Rat, Mink, Hedgehog, 
# Alpaca, Human, Dolphin, Pig, Buffalo, 
# Sus_scrofa_domesticus
# Mus_Musculus__Severe_Combined_Immunedeficiency__Scid___Female__6_8_Weeks_Old__Liver__Sample_Id:_E4m31, 
# Dog, Mouse, bat_BF_506I, 
# Rabbit, Camel, Goat, Cattle, Horse, 
# Cat, Bat, bat_BF_258I, Swine, Ferret        

list_of_mammals_but_human = ['Alpaca', 'Anteater', 'Bat', 'Buffalo', 'Camel',
                   'Cat', 'Cattle', 'Chimpanzee', 'Dog', 'Dolphin',
                   'Ferret', 'Goat', 'Hedgehog', 'Horse', 'Mink', 'Mouse',
                   'Mus_Musculus__Severe_Combined_Immunedeficiency__Scid___Female__6_8_Weeks_Old__Liver__Sample_Id:_E4m31,'
                   'Pig', 'Rabbit', 'Rat', 'Rhinolophus_blasii', 'Sus_scrofa_domesticus',
                   'Swine', 'bat_BF_258I', 'bat_BF_506I']

mammal_dataset_but_human = []
mammal_human = []
for sequence in sequences:
    if sequence.host_species == "Human":
        mammal_human.append(sequence)
        continue
    for entry in list_of_mammals_but_human:
        if sequence.host_species == entry:
            mammal_dataset_but_human.append(sequence)
            continue
            
mammal_dataset = mammal_dataset_but_human + mammal_human

# obtaining encoded sequences and lists of targets
mammal_encoded_sequences, mammaltargets = EncodeAndTarget(mammal_dataset)


# # alpha + betacoronavirus data sets

# In[ ]:


#adding alpha and betacoronaviruses to separate lists based on criteria outlined in "vizualization" section


list_of_alpha_1 = [entry for entry in sequences if 'alpha' in entry.defline.lower()]
list_of_alpha_2 = [entry for entry in sequences if 'hku10' in entry.defline.lower()]
list_of_alpha_3 = [entry for entry in sequences if 'fcov' in entry.defline.lower()]
list_of_alpha_4 = [entry for entry in sequences if 'nl63' in entry.defline.lower()]
list_of_alpha_5 = [entry for entry in sequences if 'hku2' in entry.defline.lower()]


list_of_beta_1 = [entry for entry in sequences if 'beta' in entry.defline.lower()]
list_of_beta_2 = [entry for entry in sequences if 'hku14' in entry.defline.lower()]
list_of_beta_3 = [entry for entry in sequences if 'cattle' in entry.defline.lower()]
list_of_beta_4 = [entry for entry in sequences if 'mouse' in entry.defline.lower()]
list_of_beta_5 = [entry for entry in sequences if 'hku1' in entry.defline.lower()]
list_of_beta_6 = [entry for entry in sequences if 'hku4' in entry.defline.lower()]
list_of_beta_7 = [entry for entry in sequences if 'hku5' in entry.defline.lower()]
list_of_beta_8 = [entry for entry in sequences if 'hku9' in entry.defline.lower()]
list_of_beta_9 = [entry for entry in sequences if 'severe_acute_respiratory_syndrome' in entry.defline.lower()]
list_of_beta_10 = [entry for entry in sequences if 'SARS' in entry.defline.lower()]
list_of_beta_11 = [entry for entry in sequences if 'middle_east_respiratory_syndrome' in entry.defline.lower()]
list_of_beta_12 = [entry for entry in sequences if 'camel' in entry.defline.lower()]
list_of_beta_13 = [entry for entry in sequences if 'murine' in entry.defline.lower()]

#combining datasets and removing duplicate sequences

alpha_coronavirus = list_of_alpha_1 + list_of_alpha_2 + list_of_alpha_3 + list_of_alpha_4 + list_of_alpha_5  
beta_coronavirus = list_of_beta_1 + list_of_beta_2 + list_of_beta_3 + list_of_beta_4 + list_of_beta_5 + list_of_beta_6 + list_of_beta_7 + list_of_beta_8 + list_of_beta_9 + list_of_beta_10 + list_of_beta_11 + list_of_beta_12 + list_of_beta_13

alpha_and_beta_coronavirus = alpha_coronavirus + beta_coronavirus
alpha_and_beta_coronavirus = list(dict.fromkeys(alpha_and_beta_coronavirus))

print('Our data set has', len(alpha_and_beta_coronavirus), 'alphacoronavirus + betacoronavirus sequences')


# Creating lists of all non-alphacoronavirus and non-betacoronavirus sequences based on "vizualization" section 

non_alpha_beta = []
    
for entry in sequences:
    if entry  not in alpha_and_beta_coronavirus:
        non_alpha_beta.append(entry)


print(len(non_alpha_beta), 'are not alphacoronavirus or betacoronavirus')

#combining into one list, this is the entire data set. Previous lists allow for individual access to alphacoroanvirus/betacoronavirus 
#or their specific titles indicated in the defline (which is represented within our visualization)

alpha_beta_targeted_list = alpha_and_beta_coronavirus + non_alpha_beta

#creating a list of custom targets: 1's refer to alpha and betacoronavirus sequences, 0's refer to non-alpha and betacoronavirus sequences

alpha_beta_targets = []

for entry in alpha_and_beta_coronavirus:
    alpha_beta_targets.append(1)
    
for entry in non_alpha_beta:
    alpha_beta_targets.append(0)
    


# encoding
# we will only make use of the encoded (encoded_alpha_beta) dataset
# targets will be represented in alpha_beta_targets

encoded_alpha_beta, alpha_beta_targets2 = EncodeAndTarget(alpha_beta_targeted_list)



# > # **Visualization**

# In[ ]:


from sklearn.manifold import TSNE
import seaborn as sns

# We embed all our sequences into 2D vectors with help of TSNE
X_embedded = TSNE(n_components=2).fit_transform(encoded_sequences)


# In[ ]:


# We visualize the embeddings of ALL sequences
# Type: 0 is not infective
#       1 is infective

data_frame = pd.DataFrame({'1st coordinate of the embedded vector': X_embedded[:,0], 
                           '2nd coordinate of the embedded vector': X_embedded[:,1], 
                           'Type': targets})
sns.relplot(x = '1st coordinate of the embedded vector', 
            y = '2nd coordinate of the embedded vector', 
            hue = 'Type', 
            data = data_frame, 
            legend = "full",
            style = 'Type')
plt.show()


# In[ ]:


# We visualize the embeddings of ALL sequences
# Type is a type of host

data_frame = pd.DataFrame({'1st coordinate of the embedded vector': X_embedded[:,0], 
                           '2nd coordinate of the embedded vector': X_embedded[:,1], 
                           'Types of host': multi_targets})
sns.relplot(x = '1st coordinate of the embedded vector', 
            y = '2nd coordinate of the embedded vector', 
            hue = 'Types of host', 
            data = data_frame, 
            legend = "full",
            style = 'Types of host')
plt.show()


# In[ ]:


# We visualize the embeddings of ALL sequences
# Type is a type of virus species

data_frame_coro_types = pd.DataFrame({'1st coordinate of the embedded vector': X_embedded[:,0], 
                           '2nd coordinate of the embedded vector': X_embedded[:,1], 
                           'Types of virus': coro_types})
sns.relplot(x = '1st coordinate of the embedded vector', 
            y = '2nd coordinate of the embedded vector', 
            hue = 'Types of virus', 
            data = data_frame_coro_types, 
            legend = "full",
            style = 'Types of virus')
plt.show()


# In[ ]:


# We embed human related sequences into 2D vectors with help of TSNE

X_embedded_human = TSNE(n_components=2).fit_transform(encoded_human_sequences)


# In[ ]:


# We visualize the embeddings of human related sequences
# Type is a type of virus species

human_related_viral_species = [entry.virus_species for entry in human_related_sequences]

data_frame = pd.DataFrame({'1st coordinate of the embedded vector': X_embedded_human[:,0], 
                           '2nd coordinate of the embedded vector': X_embedded_human[:,1], 
                           'Types of virus': human_related_viral_species})
sns.relplot(x = '1st coordinate of the embedded vector', 
            y = '2nd coordinate of the embedded vector', 
            hue = 'Types of virus', 
            data = data_frame, 
            legend = "full",
            style = 'Types of virus')
plt.show()


# Looks like all NAs are the Middle_east beast :-)

# # CLUSTERING

# In[ ]:


from sklearn import cluster

number_of_clusters = 7

our_dictionary_of_clustering = {"kMeans": cluster.KMeans(n_clusters = number_of_clusters, random_state = 0, 
                                                         init = "k-means++", max_iter = 300, n_init = 10), 
                                 "Birch": cluster.Birch(n_clusters = number_of_clusters)
                                }


# In[ ]:


# Search for elbow for k-means clustering

def ElbowSearch_kMeans(X, min_number_of_clusters = 2, 
                        max_number_of_clusters = 7, plotting = True):

    wcss = [] # wcss = Within Cluster Sum of Squares 
    for i in range(min_number_of_clusters, max_number_of_clusters + 1):
        our_clustering_method = cluster.KMeans(n_clusters = i, random_state = 0, 
                                               init = "k-means++", max_iter = 300, n_init = 10)
        our_clustering_method.fit(X)
       
        # inertia_ calculates sum of squared distances of samples to their closest cluster center
        wcss.append(our_clustering_method.inertia_) 
   
    if plotting == True:
        plt.plot(range(min_number_of_clusters, max_number_of_clusters + 1), wcss)
        plt.title("Elbow method for kMeans")
        plt.xlabel("Number of clusters")
        plt.ylabel("Within Cluster Sum of Squares (WCSS)")
        plt.show()
        
    return wcss


# In[ ]:


from sklearn.decomposition import TruncatedSVD

def Clustering(X, n_clusters = number_of_clusters,
               list_of_chosen = ["kMeans"], plotting = True,
               dictionary_of_clustering = our_dictionary_of_clustering):
    dict_labeling = {}
    for name, clustering_method in dictionary_of_clustering.items():
        if name in list_of_chosen:
            labels = clustering_method.fit_predict(X)    
            dict_labeling[name] = labels
            
            if plotting == True:
                svd = TruncatedSVD(n_components = 2, n_iter = 7, random_state = 42)
                X_embedded_SVD = svd.fit_transform(X)
                data_frame_clustering = pd.DataFrame({'1st SVD coordinate': X_embedded_SVD[:,0],
                                                      '2nd SVD coordinate': X_embedded_SVD[:,1], 
                                                      'Cluster labels':  labels})
                sns.relplot(x = '1st SVD coordinate', 
                            y = '2nd SVD coordinate', 
                            hue = 'Cluster labels', 
                            data = data_frame_clustering, 
                            legend = "full",
                            style = 'Cluster labels') 
                plt.title("The results of " + str(name) + " clustering:")
                plt.show()
    return dict_labeling


# In[ ]:


# we apply elbow method to human related sequences

ElbowSearch_kMeans(encoded_human_sequences, 
                   min_number_of_clusters = 2, 
                   max_number_of_clusters = 7)


# In[ ]:


# we demonstrate the results of clustering for the kMeans and Birch methods

dict_labeling = Clustering(encoded_human_sequences, number_of_clusters, ['kMeans','Birch'])


# # How we assess the performance

# In[ ]:


# Statictical parameters for the performance of a classifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def scores(y_test, y_predicted):  
    precision = precision_score(y_test, y_predicted, pos_label = None,
                                    average = 'weighted')             
    recall = recall_score(y_test, y_predicted, pos_label = None, average = 'weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    accuracy = accuracy_score(y_test, y_predicted)
    print("accuracy = %.4f, f1 = %.4f, precision = %.4f, recall = %.4f" % (accuracy, f1, precision, recall))
    return accuracy, f1, precision, recall


# In[ ]:


# Confusion matrix

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion Matrix',
                          cmap = plt.cm.spring):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 26)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 20)
    plt.yticks(tick_marks, classes, fontsize = 20)
    
    
    fmt = '.2f' if normalize else 'd'
    
    
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", 
                 color = "white" if cm[i, j] < thresh else "black", fontsize = 40)
    

    plt.ylabel('True labels', fontsize = 24)
    plt.xlabel('Predicted labels', fontsize = 24)
    
    
    # this is just a fix for the mpl bug that cuts off top/bottom of the heatmap matrix
    # this fix is not needed for kaggle
    if cwd != '/kaggle/working':
        bottom, top = plt.ylim() # current values for bottom and top
        bottom += 0.5 # change the bottom
        top -= 0.5 # change the top
        plt.ylim(bottom, top) # update the values

    return plt


# #### Dictionary of classifiers

# In[ ]:


# this is our dicrionary of classifiers

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

our_dictionary_of_classifiers = {"SVM": SVC(probability = True, gamma = 'scale'), 
                                 "Logistic Regression": LogisticRegression(C=30.0, class_weight = 'balanced', 
                                                                           solver = 'newton-cg', multi_class = 'multinomial', 
                                                                           n_jobs = -1, random_state = 42),
                                 "Decision Tree": DecisionTreeClassifier(random_state = 42),
                                 "Random Forest": RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 15, 
                                                                         n_jobs= -1, random_state = 42)
                            }


# In[ ]:


# Receiver operating characteristic (ROC) curve

from sklearn.metrics import roc_curve

def ROC_curve(xTrain, xTest, yTrain, yTest,
              list_of_chosen = ["SVM"],
              dictionary_of_classifiers = our_dictionary_of_classifiers):
    
    # preparing the figure for plotting the ROC Curves
    
    sns.set_style('whitegrid')
    plt.figure(figsize = (10, 8))
    plt.plot([0, 1], [0, 1], color = 'blue', linestyle = '--', label = 'Flip a coin')
   
    # calculating the ROC Curves

    for name, classifier in dictionary_of_classifiers.items():
        if name in list_of_chosen:
            classifier.fit(xTrain, yTrain)
            # predict_proba gives the probabilities for the target 
            # (0 and 1 in our case) as a list (array). 
            # The number of probabilities for each row is equal 
            # to the number of categories in target variable (2 in our case).
            probabilities = classifier.predict_proba(xTest)
            probability_of_ones = probabilities[:,1] 
            # roc_curve returns:
            # - false positive rates (FP_rates), i.e., 
            # the false positive rate of predictions with score >= thresholds[i]
            
            # - true positive rates (TP_rates), i.e., 
            # the true positive rate of predictions with score >= thresholds[i]
            
            # - thresholds 
            FP_rates, TP_rates, thresholds = roc_curve(yTest, probability_of_ones)
            # plotting the ROC Curve to visualize all the methods
            plt.plot(FP_rates, TP_rates, label = name)
            
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.title('ROC Curve', fontsize = 14)
    plt.legend(loc = "lower right")
    plt.show()


# In[ ]:


# Box plot after CV-fold crossvalidation
from sklearn.model_selection import cross_validate

# this function returns a dataframe with  'accuracy', 'f1', 'sensitivity', 'specificity' 
# calculated within k-fold cross validation of a list of classifiers

# models = list of classifiers
# X = list of feature vectors
# y = list of labels
# models_names = custom names of the classifiers for the dataframe
# CV = number of folds in validation
# number_of_processors = number of processors used for peerforming the cross validation

# if number_of_processors = -1, all processors are used 
# (DISCLAIMER: it may cause a warning ``timeout or by a memory leak'') 

def Crossvalidating(X, y, CV = 5, list_of_chosen = ["SVM"], plotting = True,
                          dictionary_of_classifiers = our_dictionary_of_classifiers,
                          number_of_processors = 1):
    
                  
    table = pd.DataFrame(index = range(CV*len(list_of_chosen)))
    table_entries = []
    y_series = pd.Series(y)
    
    for name, classifier in dictionary_of_classifiers.items():
        if name in list_of_chosen:
            scoring = {'accuracy','f1','recall','balanced_accuracy'}
            scores = cross_validate(classifier, X, y_series, cv = CV, n_jobs = number_of_processors,
                                    scoring = scoring, return_train_score = False)
            accuracy = scores['test_accuracy']
            f1 = scores['test_f1']
            sensitivity = scores['test_recall']
            balanced_accuracy = scores['test_balanced_accuracy'] 
            # specificity = 2 * balanced_accuracy - sensitivity
            specificity = []
            for j in range(0,len(balanced_accuracy)):
                specificity.append(2*balanced_accuracy[j]-sensitivity[j])
            for j in range(0,len(balanced_accuracy)):
                table_entries.append((name, accuracy[j], f1[j], sensitivity[j], specificity[j]))
                
    table = pd.DataFrame(table_entries, columns = ['classifier', 'accuracy', 'F1', 'sensitivity', 'specificity'])
    
    if plotting == True:
        df_melted = pd.melt(table,id_vars = ['classifier'], 
                            value_vars = ['accuracy', 'F1', 'sensitivity', 'specificity'], 
                            var_name = 'scores')
        sns.boxplot(x = 'scores', y = 'value', data = df_melted, hue = 'classifier', palette = "Set3")
        # Put the legend out of the figure
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    return table


# # CLASSIFIERS

# In[ ]:


def Classifying(xTrain, xTest, yTrain, yTest,
                list_of_chosen = ["SVM"],
                dictionary_of_classifiers = our_dictionary_of_classifiers, 
                conf_matr = True):
    for name, classifier in dictionary_of_classifiers.items():
        if name in list_of_chosen:
            classifier.fit(xTrain, yTrain)
            y_predicted = classifier.predict(xTest)  

            print("Results for " + str(name))
            # report accuracy, f1, precision, recall for the classifier
            accuracy, f1, precision, recall = scores(yTest, y_predicted)
            
            if conf_matr == True:
                # The confusion matrix for the test data for SVM
                cm = confusion_matrix(yTest, y_predicted)
                fig = plt.figure(figsize=(7, 7))
                plot = plot_confusion_matrix(cm, classes=['Not Infective','Infective'], normalize=False, title='Confusion matrix\n for ' + str(name))
                plt.show()


# In[ ]:


# On random 20% of ALL sequences

# Split the dataset into a training set (80%) and a test set (20%)
xTrain20, xTest20, yTrain20, yTest20 = train_test_split(encoded_random20, targets_random20, test_size = 0.2, random_state = 42)

classifiers_we_use = ["SVM", "Decision Tree"]
Classifying(xTrain20, xTest20, yTrain20, yTest20, list_of_chosen = classifiers_we_use)


# In[ ]:


# On ALL sequences

# Split the dataset into a training set (80%) and a test set (20%)
xTrain, xTest, yTrain, yTest = train_test_split(encoded_sequences, targets, test_size = 0.2, random_state = 42)

Classifying(xTrain, xTest, yTrain, yTest,["SVM", "Logistic Regression", "Decision Tree", "Random Forest"])


# In[ ]:


# SVM-classifier for muted HKU_1

xTrain_without_HKU1, xTest_without_HKU1, yTrain_without_HKU1, yTest_without_HKU1 = train_test_split(encoded_without_HKU1, 
                                                                                                    targets_without_HKU1, 
                                                                                                    test_size = 0.2, 
                                                                                                    random_state = 42)
x_Test_HKU1 = xTest_without_HKU1 + encoded_HKU1
y_Test_HKU1 = yTest_without_HKU1 + targets_HKU1

Classifying(xTrain_without_HKU1, x_Test_HKU1, yTrain_without_HKU1, y_Test_HKU1)


# In[ ]:


# SVM-classifier for TOTALLY muted HKU_1
xTrain_without_HKU1_total = encoded_without_HKU1
yTrain_without_HKU1_total = targets_without_HKU1
x_Test_HKU1_total = encoded_HKU1
y_Test_HKU1_total = targets_HKU1

Classifying(xTrain_without_HKU1_total, x_Test_HKU1_total, yTrain_without_HKU1_total, y_Test_HKU1_total)


# **Oh, dear!**

# In[ ]:


# SVM-classifier for muted SARS1

xTrain_without_SARS1, xTest_without_SARS1, yTrain_without_SARS1, yTest_without_SARS1 = train_test_split(encoded_without_SARS1, 
                                                                                                        targets_without_SARS1, 
                                                                                                        test_size = 0.2,
                                                                                                        random_state = 42)
x_Test_SARS1 = xTest_without_SARS1 + encoded_SARS1
y_Test_SARS1 = yTest_without_SARS1 + targets_SARS1  

Classifying(xTrain_without_SARS1, x_Test_SARS1, yTrain_without_SARS1, y_Test_SARS1)


# In[ ]:


# SVM-classifier for TOTALLY muted SARS1
xTrain_without_SARS1_total = encoded_without_SARS1
yTrain_without_SARS1_total = targets_without_SARS1
x_Test_SARS1_total = encoded_SARS1
y_Test_SARS1_total = targets_SARS1

Classifying(xTrain_without_SARS1_total, x_Test_SARS1_total, yTrain_without_SARS1_total, y_Test_SARS1_total)


# In[ ]:


# SVM-classifier for Balanced Data set

xTrain_Balanced, xTest_Balanced, yTrain_Balanced, yTest_Balanced = train_test_split(balanced_encoded_sequences, 
                                                                                                    balancedtargets, 
                                                                                                    test_size = 0.2, 
                                                                                                    random_state = 42)

Classifying(xTrain_Balanced, xTest_Balanced, yTrain_Balanced, yTest_Balanced)


# Everything (accuracy, f1, precision, recall) is close to 1 here, meaning: we are experiencing overfitting

# In[ ]:


# SVM-classifier for the mammalian dataset
xTrain_mammal, xTest_mammal, yTrain_mammal, yTest_mammal = train_test_split(mammal_encoded_sequences, 
                                                                            mammaltargets, 
                                                                            test_size = 0.2, 
                                                                            random_state = 42)

Classifying(xTrain_mammal, xTest_mammal, yTrain_mammal, yTest_mammal)


# In[ ]:


#Classifier for alpha + betacoroanviruses vs others

xTrainAB, xTestAB, yTrainAB, yTestAB = train_test_split(encoded_alpha_beta, alpha_beta_targets, test_size = 0.2, random_state = 42)

classifiers_we_use = ["SVM", "Logistic Regression"]
Classifying(xTrainAB, xTestAB, yTrainAB, yTestAB, list_of_chosen = classifiers_we_use)


# # Receiver operating characteristic (ROC) curve

# In[ ]:


ROC_curve(xTrain, xTest, yTrain, yTest, list_of_chosen = ["SVM", "Logistic Regression", "Decision Tree", "Random Forest"])


# # Box plots

# In[ ]:


table = Crossvalidating(xTrain, yTrain, CV = 5, list_of_chosen = ["SVM", "Logistic Regression", "Decision Tree", "Random Forest"])


# In[ ]:


box_plots = sns.boxplot(x = 'classifier', y ='accuracy', data = table, width = 0.8)
plt.setp(box_plots.get_xticklabels(), rotation = 90)
box_plots.set(xlabel = None)


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,6))  
f1 = sns.boxplot(x = 'classifier', y = 'accuracy', data = table, ax = ax1, width=0.6, linewidth = 1)
f1.set(xticklabels=[])
f1.set(xlabel=None)
f2 = sns.boxplot(x = 'classifier', y = 'F1', data = table, ax = ax2, width = 0.6, linewidth = 1)
f2.set(xticklabels=[])
f2.set(xlabel=None)
f3 = sns.boxplot(x = 'classifier', y = 'sensitivity', data = table, ax = ax3, width = 0.6, linewidth = 1)
plt.setp(ax3.get_xticklabels(), rotation = 90)
f4 = sns.boxplot(x = 'classifier', y = 'specificity', data = table, ax = ax4, width = 0.6, linewidth = 1)
plt.setp(ax4.get_xticklabels(), rotation = 90)

