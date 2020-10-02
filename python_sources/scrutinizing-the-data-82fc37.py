#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

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

input_file_name = '../input/spike-proteins-in-conona-viruses/Aligned Spike Proteins (StrainName-AccessionNumber-HostSpecies-VirusSpecies).fasta'
# input_file_name = '../input/spike-proteins-in-conona-viruses/upd1300.fasta'

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
    def __init__(self,defline,sequence,target):
        
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
    
    


# In[ ]:


#create a list of sequences as objects of the class fasta_sequence
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

print("There are",len(virus_species_set)-1, "unique virus species in our dataset") # we don't count the NA entry
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
# In the table belong the visur classification in gives as Genus -> Subgenus -> Species -> Subspecies -> Strain
# The name that are in our database are hilighted **in bold**
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
ax.set_title('The data distribution')
ax.set_xlabel('Number of entries')
ax.set_ylabel('Species of virus')
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


# > # **Encoding**

# In[ ]:


# We convert a string with the alphabet = 'ABCDEFGHIKLMNPQRSTUVWXYZ-' 
# !!!(B,X,Z are ``extra'' letters; we have to address this in the future) 
# into either a list mapping chars to integers (called integer encoding),
# or a sparce list. In the latter, each amino acid is represented as an one-hot vector of length 20, 
# where each position, except one, is set to 0.  E.g., alanine is encoded as 10000000000000000000, cystine is encoded as 01000000000000000000
# See the full table above.
# Symbol '-' is encoded as a zero-vector.

def encoding(sequence, type_of_encoding = "onehot"):

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
    


# In[ ]:


# Encode sequences with a one-hot encoding
encoded_sequences = []
for entry in sequences:
    encoded_sequences.append(encoding(entry.sequence))


# In[ ]:


encoded_human_sequences = []
human_targets = []
for entry in human_related_sequences:
    encoded_human_sequences.append(encoding(entry.sequence))
    human_targets.append(entry.virus_species)


# Muting Human_coronavirus_HKU1

# In[ ]:


# We prepare to mute Human_coronavirus_HKU1
sequences_without_HKU1 = [entry for entry in sequences if 'Human_coronavirus_HKU1' not in entry.defline] 
sequences_HKU1 = [entry for entry in sequences if 'Human_coronavirus_HKU1' in entry.defline] 

print(len(sequences_without_HKU1))
print(len(sequences_HKU1))


# In[ ]:


# Encode sequences without HKU1 with a one-hot encoding
encoded_without_HKU1 = []
encoded_HKU1 = []

for entry in sequences_without_HKU1:
    encoded_without_HKU1.append(encoding(entry.sequence))

for entry in sequences_HKU1:
    encoded_HKU1.append(encoding(entry.sequence))


# In[ ]:


# take care of targets while muting HKU1
targets_without_HKU1 = []
for entry in sequences_without_HKU1:
    targets_without_HKU1.append(entry.target)
    
targets_HKU1 = []
for entry in sequences_HKU1:
    targets_HKU1.append(entry.target)
    
print(targets_HKU1.count(1))


# Muting Severe_acute_respiratory_syndrome_related_coronavirus

# In[ ]:


# We prepare to mute Severe_acute_respiratory_syndrome_related_coronavirus
sequences_without_SARS1 = [entry for entry in sequences if 'Severe_acute_respiratory_syndrome_related_coronavirus' not in entry.defline] 
sequences_SARS1 = [entry for entry in sequences if 'Severe_acute_respiratory_syndrome_related_coronavirus' in entry.defline] 

print(len(sequences_without_SARS1))
print(len(sequences_SARS1))


# In[ ]:


for item in sequences_SARS1:
    print(item.defline)


# In[ ]:


# Encode sequences without SARS1 with a one-hot encoding
encoded_without_SARS1 = []
encoded_SARS1 = []

for entry in sequences_without_SARS1:
    encoded_without_SARS1.append(encoding(entry.sequence))

for entry in sequences_SARS1:
    encoded_SARS1.append(encoding(entry.sequence))


# In[ ]:


# take care of targets while muting SARS1
targets_without_SARS1 = []
for entry in sequences_without_SARS1:
    targets_without_SARS1.append(entry.target)
    
targets_SARS1 = []
for entry in sequences_SARS1:
    targets_SARS1.append(entry.target)
    
print(targets_SARS1.count(1))


# > # **Visualization**

# In[ ]:


# Visualize all the sequences (encoded) as feature vectors with help of TSNE
from sklearn.manifold import TSNE
import seaborn as sns

X_embedded = TSNE(n_components=2).fit_transform(encoded_sequences)


# In[ ]:


palette = sns.color_palette("bright", 2)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=targets, legend='full', palette = palette)


# In[ ]:


# we use TSNE to embed sequences that are infective for humans
from sklearn.manifold import TSNE
import seaborn as sns
X_embedded_human = TSNE(n_components=2).fit_transform(encoded_human_sequences)


# In[ ]:


data_frame = pd.DataFrame({'1st coordinate of the embedded vector': X_embedded_human[:,0], 
                           '2nd coordinate of the embedded vector': X_embedded_human[:,1], 
                           'Types of virus': human_targets})
sns.relplot(x = '1st coordinate of the embedded vector', 
            y = '2nd coordinate of the embedded vector', 
            hue = 'Types of virus', 
            data = data_frame, 
            legend = "full",
            style = 'Types of virus')
plt.show()


# Looks like all NAs are the Middle_east beast :-)

# # Some insights from Kiki

# In[ ]:


multi_targets = []

for i in range(len(deflines)):
    if 'human' in deflines[i].lower():
        multi_targets.append('Human')
    elif 'bat' in deflines[i].lower():
        multi_targets.append('Bat')
    elif 'avian' in deflines[i].lower():
        multi_targets.append('Bird')
    elif 'camel' in deflines[i].lower():
        multi_targets.append('Camel')
    else:
        multi_targets.append('Other')
        
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
    elif 'porcine' in vtl:
        coro_types.append("Delta")
    else:
        coro_types.append("N/A")


# In[ ]:


data_frame = pd.DataFrame({'1st coordinate of the embedded vector': X_embedded[:,0], 
                           '2nd coordinate of the embedded vector': X_embedded[:,1], 
                           'Types of virus': multi_targets})
sns.relplot(x = '1st coordinate of the embedded vector', 
            y = '2nd coordinate of the embedded vector', 
            hue = 'Types of virus', 
            data = data_frame, 
            legend = "full",
            style = 'Types of virus')
plt.show()


# In[ ]:


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


import matplotlib.pyplot as plt

labels = ['Human', 'Bat','Camel', 'Other', 'Bird']
overall_size = len(multi_targets)
sizes = [multi_targets.count('Human') / overall_size,          multi_targets.count('Bat') / overall_size,          multi_targets.count('Camel') / overall_size,          multi_targets.count('Other') / overall_size,          multi_targets.count('Bird') / overall_size]

explode = (0.1, 0, 0, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[ ]:


ct_labels = ['Alpha', 'Beta', 'Gamma', 'Delta', 'N/A']
ct_overall_size = len(coro_types)
ct_sizes = [coro_types.count('Alpha') / ct_overall_size,             coro_types.count('Beta') / ct_overall_size,             coro_types.count('Gamma') /ct_overall_size,             coro_types.count('Delta') / ct_overall_size,             coro_types.count('N/A') / ct_overall_size]

ct_explode = (0, 0, 0, 0, 0)

ct_fig1, ct_ax1 = plt.subplots()
ct_ax1.pie(ct_sizes, explode=ct_explode, labels=ct_labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle
ct_ax1.axis('equal')  
plt.tight_layout()
plt.show()


# > # **How we assess the performance**

# In[ ]:


# Calculate statictical parameters for the performance of a classifier
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


# Plot a confusion matrix
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.spring):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=26)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True labels', fontsize=24)
    plt.xlabel('Predicted labels', fontsize=24)

    return plt


# # SVM

# In[ ]:


# Use the default SVM-classifier

# Split the dataset into a training set (80%) and a test set (20%)
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(encoded_sequences, targets, test_size = 0.2, random_state = 42)

from sklearn.svm import SVC
classifier_SVM = SVC(probability = True)
classifier_SVM.fit(xTrain, yTrain)
y_predicted_SVM_default = classifier_SVM.predict(xTest)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(yTest, y_predicted_SVM_default)


# In[ ]:


# The confusion matrix for the test data for SVM
cm = confusion_matrix(yTest, y_predicted_SVM_default)
fig = plt.figure(figsize=(8, 8))
plot = plot_confusion_matrix(cm, classes=['Not Infective','Infective'], normalize=False, title='Confusion matrix\n for SVM')
plt.show()


# In[ ]:


# SVM-classifier for muted HKU_1

xTrain_without_HKU1, xTest_without_HKU1, yTrain_without_HKU1, yTest_without_HKU1 = train_test_split(encoded_without_HKU1, 
                                                                                                    targets_without_HKU1, 
                                                                                                    test_size = 0.2, 
                                                                                                    random_state = 42)
x_Test_HKU1 = xTest_without_HKU1 + encoded_HKU1
y_Test_HKU1 = yTest_without_HKU1 + targets_HKU1

from sklearn.svm import SVC
classifier_SVM = SVC(probability = True)
classifier_SVM.fit(xTrain_without_HKU1,yTrain_without_HKU1)
y_predicted_SVM_HKU1 = classifier_SVM.predict(x_Test_HKU1)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(y_Test_HKU1, y_predicted_SVM_HKU1)


# In[ ]:


# SVM-classifier for TOTALLY muted HKU_1
xTrain_without_HKU1_total = encoded_without_HKU1
yTrain_without_HKU1_total = targets_without_HKU1
x_Test_HKU1_total = encoded_HKU1
y_Test_HKU1_total = targets_HKU1

from sklearn.svm import SVC
classifier_SVM = SVC(probability = True)
classifier_SVM.fit(xTrain_without_HKU1_total,yTrain_without_HKU1_total)
y_predicted_SVM_HKU1_total = classifier_SVM.predict(x_Test_HKU1_total)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(y_Test_HKU1_total, y_predicted_SVM_HKU1_total)


# **Oh, dear!**

# In[ ]:


# SVM-classifier for muted SARS_1


xTrain_without_SARS1, xTest_without_SARS1, yTrain_without_SARS1, yTest_without_SARS1 = train_test_split(encoded_without_SARS1, 
                                                                                                        targets_without_SARS1, 
                                                                                                        test_size = 0.2,
                                                                                                        random_state = 42)

x_Test_SARS1 = xTest_without_SARS1 + encoded_SARS1
y_Test_SARS1 = yTest_without_SARS1 + targets_SARS1  
from sklearn.svm import SVC
classifier_SVM = SVC(probability = True)
classifier_SVM.fit(xTrain_without_SARS1,yTrain_without_SARS1)
y_predicted_SVM_SARS1 = classifier_SVM.predict(x_Test_SARS1)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(y_Test_SARS1, y_predicted_SVM_SARS1)


# In[ ]:


# SVM-classifier for TOTALLY muted SARS1
xTrain_without_SARS1_total = encoded_without_SARS1
yTrain_without_SARS1_total = targets_without_SARS1
x_Test_SARS1_total = encoded_SARS1
y_Test_SARS1_total = targets_SARS1

from sklearn.svm import SVC
classifier_SVM = SVC(probability = True)
classifier_SVM.fit(xTrain_without_SARS1_total,yTrain_without_SARS1_total)
y_predicted_SVM_SARS1_total = classifier_SVM.predict(x_Test_SARS1_total)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(y_Test_SARS1_total, y_predicted_SVM_SARS1_total)


# # New insights from Arthur (balancing the dataset)
# 
# edited and slightly modified by Kiril

# In[ ]:


Avian_dataset = [entry for entry in sequences if 'Avian' in entry.defline]

Human_Avian_dataset = human_related_sequences + random.choices(Avian_dataset, k = targets.count(1))


aviantargets = [0 for entry in range(0,targets.count(1))]
humandtargets =[1 for entry in range(0,targets.count(1))]
balancedtargets = humandtargets + aviantargets


#encoding balanced sequences with one-hot encoding

balanced_encoded_sequences =[]

for entry in Human_Avian_dataset:
    balanced_encoded_sequences.append(encoding(entry.sequence))


# In[ ]:


# SVM-classifier for Balanced Data set
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
xTrain_Balanced, xTest_Balanced, yTrain_Balanced, yTest_Balanced = train_test_split(balanced_encoded_sequences, 
                                                                                                    balancedtargets, 
                                                                                                    test_size = 0.2, 
                                                                                                    random_state = 42)

classifier_SVM_Balanced = SVC(probability = True)
classifier_SVM_Balanced.fit(xTrain_Balanced, yTrain_Balanced)
y_predicted_SVM_Balanced = classifier_SVM_Balanced.predict(xTest_Balanced)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(yTest_Balanced, y_predicted_SVM_Balanced)


# Everything (accuracy, f1, precision, recall) is close to 1 here, meaning: we are experiencing overfitting

# # Creating a mammal host species dataset   

# In[ ]:


#determining unique viral hosts
host_species = [entry.host_species for entry in sequences]    
host_species_set = set(host_species)

print("There are",len(host_species_set)-1, "unique viral hosts in our dataset") # we don't count the NA entry
print("The list of unique viral hosts in our dataset is: \n", host_species_set)


# In[ ]:


#Manual observation of unique viral hosts show 25 hosts with following names which are mammals: Chimpanzee, Anteater, Rat, Mink, Hedgehog, Alpaca, Human, Dolphin, Pig, Buffalo, Sus_scrofa_domesticus
#Mus_Musculus__Severe_Combined_Immunedeficiency__Scid___Female__6_8_Weeks_Old__Liver__Sample_Id:_E4m31, Dog, Mouse, bat_BF_506I, Rabbit, Camel, Goat, Cattle, Horse, Cat, Bat, bat_BF_258I, Swine, Ferret        


#manually adding each host's viral sequence to a single dataset
mammal_dataset1 = [entry for entry in sequences if 'Chimpanzee' in entry.defline]
mammal_dataset2 = [entry for entry in sequences if 'Anteater' in entry.defline]
mammal_dataset3 = [entry for entry in sequences if 'Rat' in entry.defline]
mammal_dataset4 = [entry for entry in sequences if 'Mink' in entry.defline]
mammal_dataset5 = [entry for entry in sequences if 'Hedgehog' in entry.defline]
mammal_dataset6 = [entry for entry in sequences if 'Alpaca' in entry.defline]
mammal_dataset7 = [entry for entry in sequences if 'Ferret' in entry.defline]
mammal_dataset8 = [entry for entry in sequences if 'Dolphin' in entry.defline]
mammal_dataset9 = [entry for entry in sequences if 'Pig' in entry.defline]
mammal_dataset10 = [entry for entry in sequences if 'Buffalo' in entry.defline]
mammal_dataset11 = [entry for entry in sequences if 'Sus_scrofa_domesticus' in entry.defline]
mammal_dataset12 = [entry for entry in sequences if 'Mus_Musculus__Severe_Combined_Immunedeficiency__Scid___Female__6_8_Weeks_Old__Liver__Sample_Id:_E4m31' in entry.defline]
mammal_dataset13 = [entry for entry in sequences if 'Dog' in entry.defline]
mammal_dataset14 = [entry for entry in sequences if 'Mouse' in entry.defline]
mammal_dataset15 = [entry for entry in sequences if 'bat_BF_506I' in entry.defline]
mammal_dataset16 = [entry for entry in sequences if 'Rabbit' in entry.defline]
mammal_dataset17 = [entry for entry in sequences if 'Camel' in entry.defline]
mammal_dataset18 = [entry for entry in sequences if 'Goat' in entry.defline]
mammal_dataset19 = [entry for entry in sequences if 'Cattle' in entry.defline]
mammal_dataset20 = [entry for entry in sequences if 'Horse' in entry.defline]
mammal_dataset21 = [entry for entry in sequences if 'Cat' in entry.defline]
mammal_dataset22 = [entry for entry in sequences if 'Bat' in entry.defline]
mammal_dataset23 = [entry for entry in sequences if 'bat_BF_258I' in entry.defline]
mammal_dataset24 = [entry for entry in sequences if 'Swine' in entry.defline]
mammal_Human_dataset25 = [entry for entry in sequences if 'Human' in entry.defline]

mammal_dataset = mammal_dataset1 + mammal_dataset2 + mammal_dataset3 + mammal_dataset4 + mammal_dataset5 + mammal_dataset6 + mammal_dataset7 + mammal_dataset8 + mammal_dataset9 + mammal_dataset10 + mammal_dataset11 + mammal_dataset12 + mammal_dataset13 + mammal_dataset14 + mammal_dataset15 + mammal_dataset16 + mammal_dataset17 + mammal_dataset18 + mammal_dataset19 + mammal_dataset20 + mammal_dataset21 + mammal_dataset22 + mammal_dataset23 + mammal_dataset24 + mammal_Human_dataset25


#encodeing mammal dataset

mammal_encoded_sequences = []

for entry in mammal_dataset:
    mammal_encoded_sequences.append(encoding(entry.sequence))


# In[ ]:


#targets
nonhumanmammaltargets = [0 for entry in mammal_dataset1 + mammal_dataset2 + mammal_dataset3 + mammal_dataset4 + mammal_dataset5 + mammal_dataset6 + mammal_dataset7 + mammal_dataset8 + mammal_dataset9 + mammal_dataset10 + mammal_dataset11 + mammal_dataset12 + mammal_dataset13 + mammal_dataset14 + mammal_dataset15 + mammal_dataset16 + mammal_dataset17 + mammal_dataset18 + mammal_dataset19 + mammal_dataset20 + mammal_dataset21 + mammal_dataset22 + mammal_dataset23 + mammal_dataset24]
humanmammaltargets = [1 for entry in mammal_Human_dataset25]

mammaltargets = nonhumanmammaltargets + humanmammaltargets

print(mammaltargets)
print(len(mammaltargets))


# In[ ]:


print('There are', len(mammal_dataset), 'mammals in the total dataset \n')
print('Here are all of the host specieis in the dataset')

#display host species 
idx2 = pd.Index(host_species)
print(idx2.value_counts())


# In[ ]:


# here we prepare the data to be plootted as a bar graph
y_labels = idx2.value_counts().index.values # viral host species names
counts = idx2.value_counts().values    # numbers of occuriencies
counts_as_series = pd.Series(counts)


# In[ ]:


plt.figure(figsize=(12, 9))
ax = counts_as_series.plot(kind ='barh')
ax.set_title('The data distribution of viral hosts')
ax.set_xlabel('Number of entries')
ax.set_ylabel('Species of viral host')
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


# SVM-classifier for mammal Data set
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
xTrain_mammal, xTest_mammal, yTrain_mammal, yTest_mammal = train_test_split(mammal_encoded_sequences, 
                                                                                                    mammaltargets, 
                                                                                                    test_size = 0.2, 
                                                                                                    random_state = 42)

classifier_SVM_mammal = SVC(probability = True)
classifier_SVM_mammal.fit(xTrain_mammal, yTrain_mammal)
y_predicted_SVM_mammal = classifier_SVM_mammal.predict(xTest_mammal)

#report accuracy, f1, precision, recall for SVM
accuracy, f1, precision, recall = scores(yTest_mammal, y_predicted_SVM_mammal)


# # Boxt plots from Lake Li

# In[ ]:


# import numpy as nu
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.metrics import roc_curve,auc
# from sklearn.model_selection import StratifiedKFold
# from scipy import interp
# import matplotlib.pylab as plt

# from sklearn.ensemble import AdaBoostClassifier as ad
# from sklearn.svm import SVC #svm
# from sklearn.ensemble import RandomForestClassifier as RF #random forest
# from sklearn.ensemble import ExtraTreesClassifier as ET #extratree
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.multiclass import OneVsRestClassifier as OR


# In[ ]:


# rf = RF(max_depth=2, random_state=0)
# et=ET(n_estimators=100, random_state=0)
# svc=SVC(probability=True)
# pool=[rf,et,svc]

# x=feature ####please change this part to fit your code#####
# y=Label

# cv = StratifiedKFold(n_splits=10,shuffle=False)


# In[ ]:


def ROC(method):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    
    acc=[]
    sen=[]
    spec=[]
    f1=[]  
    
    def confusion_metrics (conf_matrix):
        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)
    
        #accuracy
        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))

        # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))
        
        # calculate the specificity
        conf_specificity = (TN / float(TN + FP))
        
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    
        acc.append(conf_accuracy)
        sen.append(conf_sensitivity)
        spec.append(conf_specificity)
        f1.append(conf_f1)
    
    
    for train,test in cv.split(x,y):
        prediction = method.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
        y_pred=method.fit(x.iloc[train],y.iloc[train]).predict(x.iloc[test])
        #create confusion matrix
        cm = metrics.confusion_matrix(y.iloc[test], y_pred)
        cm_df = pd.DataFrame(cm)
        confusion_metrics (cm_df)
      
        fpr, tpr, t = roc_curve(y.iloc[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1
        
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    method1=str(method)
    title="ROC of "+method1[:method1.find("(")]
    plt.title(title)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.show()
    
    #boxplot
    plt.plot()
    labelY=["accurancy","sensitivity","specificity","f1 score"]
    data=[acc,sen,spec,f1]
    title3="performance of "+method1[:method1.find("(")]
    plt.title(title3)
    plt.ylim=(0,1,0.1)
    plt.boxplot(data,labels=labelY)
    plt.show()

