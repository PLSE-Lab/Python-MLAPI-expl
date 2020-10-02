#!/usr/bin/env python
# coding: utf-8

# # What do we know about drugs and therapeautics for Covid19?
# 
# ## Creating a novel knowledge graph by using state-of-the-art NLP tools to measure the efficacy valence of sentences in academic articles.
# 
# Our tool is useful to create a landscape of current therapeutics, with the ability to drill-down into individual drugs and treatments for more details.
# 
# # Results
# 
# Run the cell below to see the interactive graph!

# In[ ]:


import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import Image, display, HTML

files = ['CORE_RESULTS', 'chlordiazepoxide', 'covid19', 'chloroquine', 'azithromycin', 'hydroxychloroquine', 'tocilizumab', 'testosterone', 'remdesivir', 'oseltamivir', 'ribavirin', 'interferon', 'lopinavir', 'olaparib', 'emtricitabine', 'ritonavir', 'darunavir', 'adenosine', 'ciclesonide', 'nelfinavir', 'sofosbuvir', 'tenofovir', 'dipeptidyl', 'teriflunomide', 'diltiazem', 'arginine', 'nafamostat', 'malaria', 'china', 'trial', 'pneumonia', 'sars', 'rna', 'ear', 'mers', 'polymerase', 'serum', 'kidney', 'fibrosis', 'influenza', 'anemia', 'cytokine', 'macrophages', 'inflammation', 'rig-i', 'danoprevir', 'liver', 'hiv', 'umifenovir', 'lung', 'nucleotides', 'plasmid', 'triphosphate', 'hepatitis', 'glucose', 'penicillin', 'streptomycin', 'blood']
directory = widgets.Dropdown(options=files, description='node:')
images = widgets.Dropdown(options=files)

def update_images(*args):
    images.options = files

directory.observe(update_images, 'value')

def show_images(file):
    display(Image('/kaggle/input/imagesdata/' + file + '.png'))
    
_ = interact(show_images, file=directory)


# # Summary
# 
# ## These knowledge graphs display the interconnections amongs drugs, coronaviruses, and other relevant biomedical entities. Our knowledge graph approach emphasizes scientific results and findings over basic medical knowledge due to our use of an "efficacy valence" approach, which we describe below. 
# 
# ### How to Use this Widget: The "CORE_RESULTS" option in the drop-down menu displays our main findings. Use the drop-down menu to explore the connections related to other relevant biomedical entities.
# 
# ### Data: The above knowledge graphs use only articles in bioXriv and medRxiv, two of the preprint archives provided for this challenge, due to the long runtimes associated with larger datasets.
# 
# ### Methods: Our approach uses a novel algorithm intended to create connections that result from scientific findings ("this study indicates that drug X leads to medical outcome Y") rather than background knowledge ("drug X and drug Y have been used to treat condition  Z"). We accomplish this novel weighting scheme by using "efficacy valence" scores, a numerical score of the extent to which a statement emphasizes the creation of novel knowledge. We further explain our approach and methodology below.
# 
# #### Run the code above to see the cached results that were created by this notebook earlier. If you want to generate new results, run the rest of this notebook!

# ## A Method for Evaluating Our Knowledge Graph: Evaluate against a baseline graph
# 
# "Evaluating" a knowledge graph is difficult. Unlike supervised machine learning tasks such as object detection, there is no objective, universal ground truth for a knowledge graph. There are some theoretical possibilites though. One method for evaluating a knowledge graph might be to interview actual persons, experts or non-experts, and ask whether the knowledge graph accurately represents that person's understanding of a phenomenon or even reveals new knowledge. 
# 
# We took a more modest approach. We had one team member who did not work on the graphing algorithm select a reputable survey of COVID-19-related drugs and manually create a knowledge graph. We could then compare our algorithmically-derived results to this manually-created knowledge graph. Our manually-created knowledge graph is below:

# In[ ]:


print("Our target graph from our external literature survey, to validate our results:")

from IPython.display import Image
Image("../input/goalgraph/goal.png")


# Our own assessment (and you should never have a student grade themself) is that our computer-generated knowledge graphs did suprisingly well at discovering the links present in this hand-crafted knowledge graph. Our knowledge graph found most, if not all, all of the therapeutics that appeared in this custom knowledge graph (see below). We should mention, however, that the two drugs our knowledge graphs did not find never appeared the raw article dataset. We view this as a limitation of the source data and not the algorithm. 
# 
# Although we did not formally present 'interleukin-6' in our result graph, we did mine the word 'cytokine,' which is a superclass of this and other interleukins. Similarly, assuming that the importance in the goal graph of plasma as a therapeutic stems from its ability to deliver antibodies, our graph returns serum, which, as we understand it, is the same as plasma, but without fibrogens (clotting material).
# 
# Similarly, the Kaggle challenge task itself mentions the drugs naproxen, clarithromycin, and minocycline, however, the first two didn't occur in any paper in the biorxiv dataset provided (we only used these papers), while minocycline was mentioned just twice (in the same sentence with many other drugs), making it almost impossible for our algorithm to tease out that this particular drug (and not others in the sentence, which we did report on), happened to be relevant. As for the remaining drugs, all but one (favipiravir) of them appeared in our CORE_RESULTS main graph, with favipiravir being the only drug that appeared only in one or more of the flyout graphs instead.
# 
# In addition to finding all the drugs in our goal, we were able to find 18 additional drugs that may be relevant treatment options for Covid19!
# 
# We used this literature review, specifically the sub-section on "investigational approaches."
# 
# Kenneth McIntosh, Martin S Hirsch, Allyson Bloom, "Coronaviruses," Uptodate.com, https://www.uptodate.com/contents/coronaviruses, accessed April 5, 2020.
# 
# # Pros and cons of this approach
# 
# Our methodology for intelligently choosing edges based on efficacy valence is going to be far superior to simpler text mining approaches that look for word frequency and other statistical-based approaches. The resultant graph we achieved provides a relatively simple, but highly complete picture of the current landscape of available theraputics and drugs. We also specifically focus of biomedical Named Entity Recognition by using MeSH keywords and drug databases to mine topics.
# 
# There are two limitations we face: the first is time; it takes several hours, even if you have a GPU and parallelize multiple CPUs, to generate this graph (which is why we uploaded cached versions of our results). We envision a world where a researcher would probably run this code daily. 
# 
# We also, with more time, could have more intelligently performed word stemming, tokenization, and other basic NLP hygiene, but found that our results exceeded our expectations. We also could have applied more sophisticated Named Entity Recognition approaches to reduce a dependency on the need to update MeSH and drug keywords whenever they come out by using tools like BioBERT.
# 
# Our second limitation is that we did not have time to attempt any text summarization into English sentences for each of the drugs. We anticipate that other research groups that have focused on this sort of thing could be tasked with developing such a module, which could be attached to our graph and provide more detailed information for each treatment we identify.
# 
# # Next steps
# 
# At the bottom of this notebook, we repeat the code for generating the knowledge graph, and include a display that will return all the papers and relevant sentences, along with their efficacy valence score, that were used to generate the knowledge graph. A researcher can then feed these papers and/or the relevant sentences into another tool that performs elegant text summarization, which was not the focus of our project.

# # Analytical Method
# 
# The code we used to conduct this analysis and generate the knowledge graphs above can be found below.
# 
# ## Note on running the notebook yourself
# 
# ### Cached versions versus running from scratch
# 
# Due to the size of the dataset, we've provided three options in the cell below for you to choose how you want to run this notebook:
# 1. "cached": Run a cached version that we have uploaded, which generates edges from the entire biorxiv_medrxiv dataset as of 4/11/2020. This will recreate the graph you saw above. This should take about fifteen minutes (most of the time is for automatically installing libraries, dependencies, and training a BERT model for sentence efficacy valence).
# 2. "mini" : Run a mini version, which shows you a proof-of-concept on two articles from the dataset. This takes about as long to run as above.
# 3. "full" : Run a full version, on a dataset of your choice. This can take a few hours or longer, depending on your hardware.
# 
# ### Requirements and runtime
# 
# In order to process thousands of articles, it is best if you have access to multiple CPU cores, and at least one GPU. If you are just running a cached version of this notebook, a single CPU will suffice. 
# 
# ### Running options
# 
# 1. **Using paper abstracts**: Besides choosing which dataset to run on (cached, mini, or full), you can also specify if sentences should be pulled from the paper abstracts; right now, we have turned this option off, because it adds to the runtime.
# 2. **Choosing a different edge weighting** : Currently, edge weighting is done using the efficacy valence model we built. You can also choose to weight edges by paper citation count, by weighting edges based on how many "interesting words" are in the sentence (we define this later in the notebook), or a custom combination of all three. See the cells at the end of this notebook for options.

# In[ ]:


# Do you want to run this notebook using the uploaded saved checkpoints, or from scratch?
# Note: it takes several hours (more than Kaggle limits) to run from scratch on all the articles from the biorxiv_medrxiv
# dataset, unless you are able to parallelize across multiple CPUs. GPUs are also required for part of the notebook.
RUNTYPE = "cached" # options are {"mini, "full", "cached"}

# Do you want to include paper abstracts in the analysis, or just the main text of the articles?
ABSTRACT = False

print("Running the notebook with the " + RUNTYPE + " dataset...")
Image("../input/diagrams/flow.png")


# # Analytical Steps
# 
# There are four steps necessary to generate the knowledge graphs we presented above:
# 1. [Edge collection](#section1): Process all sentences from the specified articles into raw edges.
# 2. [Edge conversion](#section2): Convert the sentence on each edge into a format that can be labelled with an efficacy valence score.
# 3. [Edge labelling](#section3): Train an efficacy valence model and then predict the efficacy valence score for each edge and display the interactive graph.
# 4. [Drawing the graph](#section4): Applying a weighting algorithm to the edges, using the efficacy valence score, and displaying the main graph and options to drill-down on each of its nodes.
# 

# <a id="section1"></a>
# # 1. Edge collection
# 
# The code below will go through all the articles specified, and looks for any edges that involve a drug. Then, it will repeat the process for all those edges.
# 
# 

# ## 1.1 Edge collection: pre-processing the articles and other datasets
# 
# The code below requires a "seed" of drugs to begin the edge discovery process. A seed refers to a list of relevant search terms. We used PolySearch, a free, online tool, to generate a list of drugs related to COVID-19. PolySearch found "chloroquine" and "chlordiazepoxide," which we deemed sufficient for our seed.
# 
# Our analysis used several external datasets, which we have made public.
# 1. "covid19.json" : The PolySearch query results for 'covid' mentioned above. We only used the 'drugs' component.
# 2. "MeSH.txt" : MeSH keywords. From NIH, "The Medical Subject Headings (MeSH) thesaurus is a controlled and hierarchically-organized vocabulary produced by the National Library of Medicine. It is used for indexing, cataloging, and searching biomedical and health-related information." We use these MeSH keywords to perform Named Entity Recognition (NER) in this biomedical dataset.
# 3. "bioNERs.txt" : bioNERS we mined from the PolySearch results mentioned above, however, in retrospect these are probably just a subset of MeSH keywords. 
# 4. "verbs.txt" : Any "interesting words" we found when we were labelling sentences from the PolySearch results for our efficacy valence model descrived later. You can use these interesting words as another way to measure efficacy valence for the edges in the settings at the top of this file. (Note: they include other parts of speech besides verbs.)
# 
# We also used a set of ad-hoc stopwords to avoid adding certain nodes to our knowledge graphs.
# 
# Finally, we extend our Named Entity Recognition (NER) to include both the domestic (USAN) and international (INN) drug names as of 2019 that we collected from the WHO and other online sources.

# In[ ]:


import numpy as np 
import pandas as pd 
get_ipython().system('pip install beautifulsoup4')
import json
from bs4 import BeautifulSoup
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import os
import random
import pandas
import traceback
from copy import deepcopy

from joblib import Parallel, delayed
import multiprocessing

# #########################################################################################################################################################
# DATASET PRE-PROCESSING
# #########################################################################################################################################################

root_dir = '../input/'

# which of the PolySearch results do you want to use as the seed for this algorithm?
# from PolySearch2.0: http://polysearch.ca/index
# Liu Y., Liang Y., Wishart D.S. (2015) PolySearch 2.0: A significantly improved text-mining system for discovering associations between human diseases, genes, drugs, metabolites, toxins, and more. Nucleic Acids Res. 2015 Jul 1;43(Web Server Issue):W535-42.
with open(root_dir + 'polysearch-covid19-results-json/covid.json') as f:
  polysearch = json.load(f)

# populate the metadata and contents of the articles in the dataset (has some, but not all, overlap with PolySearch)
metadata = pandas.read_csv(root_dir + "CORD-19-research-challenge/metadata.csv")
files1 = os.listdir(root_dir + 'CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/')
#files2 = os.listdir('./noncomm_use_subset/noncomm_use_subset')
#files3 = os.listdir('./pmc_custom_license/pmc_custom_license')

articles = {}
for file in files1:
	f = open(root_dir + 'CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/' + file)
	article = json.load(f)
	f.close()
	if file not in articles.keys():
		articles[file] = article
'''for file in files2:
	f = open('./noncomm_use_subset/noncomm_use_subset/' + file)
	article = json.load(f)
	f.close()
	articles[file] = article
for file in files3:
	f = open('./pmc_custom_license/pmc_custom_license/' + file)
	article = json.load(f)
	f.close()
	articles[file] = article'''

# if you want to just see the proof-of-concept on a mini dataset of just two articles
if RUNTYPE == "mini":
    mini = {}
    for a in list(articles.keys())[:2]:
        mini[a] = articles[a]
    articles = mini

# calculate the citation counts of papers in this dataset; code copied from
# https://www.kaggle.com/beatrandom/all-papers-sorted-by-their-citation-count
def createCitationCounts(articles):
	paperAndCount = {}
	for data in articles.values():
		if len(data["metadata"]["title"]) != 0:
			if data["metadata"]["title"] not in paperAndCount :
				paperAndCount[data["metadata"]["title"]] = 1
			else :
				oldCount = paperAndCount[data["metadata"]["title"]]
				paperAndCount[data["metadata"]["title"]] = oldCount+1
		for reference in data["bib_entries"].items():
			if (not reference[1]["title"].startswith("Submit your next manuscript to BioMed Central")) and ("This article is an" not in reference[1]["title"]) and (len(reference[1]["title"])) != 0:
				if reference[1]["title"] not in paperAndCount :
					paperAndCount[reference[1]["title"]] = 1
				else :
					oldCount = paperAndCount[reference[1]["title"]]
					paperAndCount[reference[1]["title"]] = oldCount+1
	return paperAndCount
citationCounts = createCitationCounts(articles)

# #########################################################################################################################################################
# EXTERNAL DATASET PRE-PROCESSING
# #########################################################################################################################################################

# collect the MeSH keywords downloaded from NIH
NERs = []
file = open(root_dir + "keywords-mesh-bioner-verbs-txt/MeSH.txt")
data = file.readlines()
file.close()
for d in data:
	if d[:-1] not in NERs and '(' not in d:
		NERs.append(d[:-1].rstrip().lower())

# add in additional bioNERs found in the PolySearch dataset review that might be specific to this disease, collected during efficacy_valence labelling
bioNERs = []
file = open(root_dir + "keywords-mesh-bioner-verbs-txt/bioNERs.txt")
data = file.readlines()
file.close()
for d in data:
	bioNERs.append(d[:-1].lower())

for b in bioNERs:
	if b not in NERs:
		#print(b)
		NERs.append(b.lower())

# create the set of interesting words (affects, results, more, lower, show, etc) mined during efficacy_valence labelling that serve as a proxy for
# efficacy
interestingWords = []
file = open(root_dir + "keywords-mesh-bioner-verbs-txt/verbs.txt")
data = file.readlines()
file.close()
for d in data:
	interestingWords.append(d[:-1].lower())

# some manually-chosen stopwords from the MeSH corpora...this could probably be refined with more time
stopwords = ['disease', 'dates', 'control', 'measures', 'infection', 'epidemic', 'health', 'literature', 'review', 'control', 'preventive medicine',
	'association', 'diagnosis', 'treatment', 'methods', 'patients', 'education', 'epidemiology', 'role', 'balance', 'medicine', 'supplies', 'public health',
	'muscle', 'cells', 'human', 'gene', 'animals', 'tissues', 'phenotype', 'elements', 'pathology', 'incidence', 'risk', 'probability', 'staistics', 'binding',
	'protein', 'forms', 'result', 'incidence', 'emergencies', 'children', 'duration', 'host', 'noise', 'reports', 'models', 'safety', 'feedback', 'back', 'biology',
	'region', 'demography', 'community', 'virus', 'light', 'safety', 'feedback', 'immunity', 'back', 'breeding', 'exercise', 'uncertainty', 'axis', 'biology',
	'region', 'tail', 'statistics', 'bias', 'success', 'drugs', 'pain', 'prevalence', 'sex', 'mice', 'diet', 'suppression', 'accounting', 'convergence', 'lion',
	'environment', 'ecology', 'hospitals', 'metabolism', 'reproduction', 'hand', 'cold', 'abnormalities', 'cities', 'evolution', 'transformation', "3'",
	'trigger', 'adults', 'signaling', 'concentrations', 'recruitment', 'diffusion', 'translocation', 'atmosphere', 'solutions', 'finger', 'powder', 'radius',
	'transduction', 'suspensions', 'kb', 'technology', 'face', 'meta-analysis', 'trees', 'chromosomes', 'polymorphism', 'genetics', 'vegetables', 'hygiene',
	'biochemistry', 'biophysics', 'frameshifting', 'genomics', 'serology', 'arm', 'plastic', 'heating', 'virology', 'physiology', 'induction', 'chromosome',
	'ca', 'consultant', 'skin', 'touch', 'nose', 'disinfection', 'anticipation', 'mouth', 'dominance', 'surgery', 'absorption', 'document', 'poly', 'house',
	'ecosystem', 'microbiology', 'antibiotics', 'disasters', 'pharmacogenetics', 'viruses', 'genome', 'coronavirus', 'water', 'bacteria', 'syndrome',
	'mutation', 'lead', 'infections', 'diseases', 'vaccines', 'ether', 'mab']

diseaseSynonyms = ["Covid19", "Covid-19", "coronavirus", "SARS-coronavirus", "2019-nCoV", "SARS-CoV-2", 'COVID19', "Coronavirus", 'covid19', 'sars-cov-2', '2019-ncov',
	"sars-coronavirus", '2019ncov', 'ncov2019', 'ncov-2019', 'covid-19', 'COVID-19']

# Look up the lists of all international and USA drug names (collected from the two sources below)
#http://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/tools-to-assist-with-the-classification-in-the-hs/hs_classification-decisions/inn-table.aspx
#https://www.genome.jp/kegg/brite.html
def drugLookup(word):  
	drugs = ['celecoxib', 'diclofenac', 'diflunisal', 'etodolac', 'fenoprofen', 'flurbiprofen', 'ibuprofen', 'indomethacin', 'ketoprofen', 'ketorolac', 'meclofenamate', 'mefenamic', 'meloxicam', 'nabumetone', 'naproxen', 'oxaprozin', 'piroxicam', 'sulindac', 'tolmetin', 'buprenorphine', 'fentanyl', 'hydrocodone', 'hydromorphone', 'levorphanol', 'methadone', 'morphine', 'oxycodone', 'oxymorphone', 'tapentadol', 'tramadol', 'butorphanol', 'codeine', 'meperidine', 'nalbuphine', 'pentazocine', 'acetaminophen', 'benzhydrocodone', 'butalbital', 'lidocaine', 'acamprosate', 'disulfiram', 'naltrexone', 'lofexidine', 'naloxone', 'bupropion', 'nicotine', 'varenicline', 'amikacin', 'gentamicin', 'neomycin', 'paromomycin', 'plazomicin', 'streptomycin', 'tobramycin', 'cefadroxil', 'cefazolin', 'cephalexin', 'cefaclor', 'cefotetan', 'cefoxitin', 'cefprozil', 'cefuroxime', 'cefdinir', 'cefixime', 'cefotaxime', 'cefpodoxime', 'ceftazidime', 'ceftriaxone', 'cefepime', 'ceftaroline', 'ceftolozane', 'amoxicillin', 'ampicillin', 'piperacillin', 'penicillin', 'dicloxacillin', 'nafcillin', 'oxacillin', 'ertapenem', 'imipenem', 'meropenem', 'azithromycin', 'clarithromycin', 'erythromycin', 'fidaxomicin', 'besifloxacin', 'ciprofloxacin', 'delafloxacin', 'gatifloxacin', 'gemifloxacin', 'levofloxacin', 'moxifloxacin', 'ofloxacin', 'sulfacetamide', 'sulfadiazine', 'demeclocycline', 'doxycycline', 'eravacycline', 'minocycline', 'omadacycline', 'sarecycline', 'tetracycline', 'sulfamethoxazole', 'trimethoprim', 'dalbavancin', 'oritavancin', 'telavancin', 'vancomycin', 'clindamycin', 'lincomycin', 'nitrofurantoin', 'linezolid', 'tedizolid', 'lefamulin', 'aztreonam', 'daptomycin', 'metronidazole', 'secnidazole', 'tinidazole', 'tigecycline', 'acetic', 'bacitracin', 'chloramphenicol', 'fosfomycin', 'methenamine', 'polymyxin', 'rifaximin', 'ethosuximide', 'methsuximide', 'benzodiazepines', 'barbituates', 'gaba', 'carbamazepine', 'eslicarbazepine', 'ethotoin', 'fosphenytoin', 'lacosamide', 'oxcarbazepine', 'phenytoin', 'rufinamide', 'zonisamide', 'cannabidiol', 'potassium', 'multiple', 'donepezil', 'galantamine', 'rivastigmine', 'memantine', 'ergoloid', 'isocarboxazid', 'phenelzine', 'selegiline', 'tranylcypromine', 'citalopram', 'desvenlafaxine', 'duloxetine', 'escitalopram', 'fluoxetine', 'fluvoxamine', 'levomilnacipran', 'nefazodone', 'paroxetine', 'sertraline', 'trazodone', 'venlafaxine', 'vilazodone', 'vortioxetine', 'amitriptyline', 'amoxapine', 'clomipramine', 'desipramine', 'doxepin', 'imipramine', 'nortriptyline', 'protriptyline', 'trimipramine', 'maprotiline', 'mirtazapine', 'aripiprazole', 'quetiapine', 'esketamine', 'chlordiazepoxide', 'olanzapine', 'perphenazine', 'rolapitant', 'granisetron', 'ondansetron', 'palonosetron', 'aprepitant', 'fosnetupitant', 'netupitant', 'dronabinol', 'nabilone', 'chlorpromazine', 'diphenhydramine', 'doxylamine', 'hydroxyzine', 'meclizine', 'metoclopramide', 'prochlorperazine', 'promethazine', 'scopolamine', 'trimethobenzamide', 'dihydroergotamine', 'ergotamine', 'almotriptan', 'eletriptan', 'frovatriptan', 'lasmiditan', 'naratriptan', 'rizatriptan', 'sumatriptan', 'zolmitriptan', 'erenumab', 'fremanezumab', 'galcanezumab', 'timolol', 'divalproex', 'topiramate', 'valproic', 'guanidine', 'pyridostigmine', 'aminosalicylic', 'bedaquiline', 'capreomycin', 'cycloserine', 'ethambutol', 'ethionamide', 'isoniazid', 'pyrazinamide', 'rifampin', 'rifapentine', 'dapsone', 'rifabutin', 'bendamustine', 'chlorambucil', 'mechlorethamine', 'melphalan', 'lomustine', 'busulfan', 'cyclophosphamide', 'procarbazine', 'temozolomide', 'trabectedin', 'apalutamide', 'bicalutamide', 'darolutamide', 'enzalutamide', 'flutamide', 'nilutamide', 'abiraterone', 'lenalidomide', 'pomalidomide', 'thalidomide', 'estramustine', 'fulvestrant', 'tamoxifen', 'toremifene', 'azacitidine', 'capecitabine', 'mercaptopurine', 'thioguanine', 'hydroxyurea', 'anastrozole', 'exemestane', 'letrozole', 'etoposide', 'topotecan', 'pexidartinib', 'gefitinib', 'binimetinib', 'cobimetinib', 'alectinib', 'brigatinib', 'ceritinib', 'crizotinib', 'lorlatinib', 'dabrafenib', 'encorafenib', 'trametinib', 'vemurafenib', 'acalabrutinib', 'ibrutinib', 'abemaciclib', 'palbociclib', 'ribociclib', 'ramucirumab', 'afatinib', 'dacomitinib', 'erlotinib', 'neratinib', 'osimertinib', 'erdafitinib', 'glasdegib', 'sonidegib', 'vismodegib', 'ivosidenib', 'ruxolitinib', 'cabazitaxel', 'everolimus', 'bosutinib', 'dasatinib', 'imatinib', 'nilotinib', 'ponatinib', 'lapatinib', 'axitinib', 'cabozantinib', 'lenvatinib', 'pazopanib', 'regorafenib', 'sorafenib', 'sunitinib', 'vandetanib', 'fedratinib', 'gilteritinib', 'midostaurin', 'entrectinib', 'larotrectinib', 'alpelisib', 'copanlisib', 'duvelisib', 'idelalisib', 'niraparib', 'olaparib', 'rucaparib', 'talazoparib', 'venetoclax', 'belinostat', 'panobinostat', 'moxetumomab', 'bevacizumab', 'mogamulizumab', 'blinatumomab', 'daratumumab', 'dinutuximab', 'elotuzumab', 'ipilimumab', 'pertuzumab', 'trastuzumab', 'cetuximab', 'necitumumab', 'panitumumab', 'olaratumab', 'atezolizumab', 'avelumab', 'cemiplimab', 'durvalumab', 'nivolumab', 'pembrolizumab', 'obinutuzumab', 'ofatumumab', 'rituximab', 'ado-trastuzumab', 'brentuximab', 'gemtuzumab', 'inotuzumab', 'polatuzumab', 'alitretinoin', 'bexarotene', 'tretinoin', 'allopurinol', 'leucovorin', 'levoleucovorin', 'mesna', 'rasburicase', 'zoledronic', 'methotrexate', 'pemetrexed', 'aldesleukin', 'asparaginase-erwinia', 'calaspargase', 'denileukin', 'enasidenib', 'fludarabine', 'ixazomib', 'mitoxantrone', 'pegaspargase', 'selinexor', 'tagraxofusp', 'trifluridine', 'uridine', 'vorinostat', 'albendazole', 'ivermectin', 'mebendazole', 'moxidectin', 'praziquantel', 'triclabendazole', 'miltefosine', 'artemether', 'atovaquone', 'chloroquine', 'hydroxychloroquine', 'mefloquine', 'primaquine', 'quinine', 'tafenoquine', 'benznidazole', 'nitazoxanide', 'pentamidine', 'pyrimethamine', 'benztropine', 'trihexyphenidyl', 'apomorphine', 'bromocriptine', 'pramipexole', 'ropinirole', 'rotigotine', 'carbidopa', 'levodopa', 'rasagiline', 'safinamide', 'istradefylline', 'entacapone', 'tolcapone', 'amantadine', 'fluphenazine', 'haloperidol', 'loxapine', 'pimozide', 'thioridazine', 'thiothixene', 'trifluoperazine', 'asenapine', 'brexpiprazole', 'cariprazine', 'iloperidone', 'lurasidone', 'pimavanserin', 'paliperidone', 'risperidone', 'ziprasidone', 'clozapine', 'cidofovir', 'foscarnet', 'ganciclovir', 'letermovir', 'valganciclovir', 'adefovir', 'entecavir', 'lamivudine', 'ribavirin', 'tenofovir', 'sofosbuvir', 'dasabuvir', 'elbasvir', 'glecaprevir', 'ledipasvir', 'ombitasvir', 'velpatasvir', 'acyclovir', 'famciclovir', 'valacyclovir', 'delavirdine', 'doravirine', 'efavirenz', 'etravirine', 'nevirapine', 'rilpivirine', 'abacavir', 'didanosine', 'emtricitabine', 'stavudine', 'zidovudine', 'atazanavir', 'darunavir', 'fosamprenavir', 'indinavir', 'nelfinavir', 'ritonavir', 'saquinavir', 'tipranavir', 'dolutegravir', 'raltegravir', 'ibalizumab', 'cobicistat', 'enfuvirtide', 'maraviroc', 'bictegravir', 'elvitegravir', 'lopinavir', 'baloxavir', 'oseltamivir', 'peramivir', 'rimantadine', 'zanamivir', 'alprazolam', 'clonazepam', 'clorazepate', 'diazepam', 'lorazepam', 'oxazepam', 'buspirone', 'meprobamate', 'lamotrigine', 'lithium', 'biguanides', 'dipeptidyl', 'glp-1', 'sodium-glucose', 'sulfonylureas', 'thiazolidinedione', 'glucose', 'alogliptin', 'canagliflozin', 'dapagliflozin', 'empagliflozin', 'ertugliflozin', 'glipizide', 'glyburide', 'insulin', 'linagliptin', 'pioglitazone', 'repaglinide', 'saxagliptin', 'sitagliptin', 'diazoxide', 'glucagon', 'desmopressin', 'emicizumab-kxwh', 'adenosine', 'clonidine', 'droxidopa', 'guanfacine', 'methyldopa', 'methyldopate', 'midodrine', 'doxazosin', 'phenoxybenzamine', 'prazosin', 'terazosin', 'benazepril', 'captopril', 'enalapril', 'fosinopril', 'lisinopril', 'moexipril', 'perindopril', 'quinapril', 'ramipril', 'trandolapril', 'azilsartan', 'candesartan', 'eprosartan', 'irbesartan', 'losartan', 'olmesartan', 'telmisartan', 'valsartan', 'amlodipine', 'felodipine', 'isradipine', 'nicardipine', 'nifedipine', 'nimodipine', 'nisoldipine', 'diltiazem', 'verapamil', 'bumetanide', 'ethacrynate', 'ethacrynic', 'furosemide', 'torsemide', 'amiloride', 'eplerenone', 'spironolactone', 'triamterene', 'chlorothiazide', 'chlorthalidone', 'hydrochlorothiazide', 'indapamide', 'methyclothiazide', 'metolazone', 'choline', 'fenofibrate', 'fenofibric', 'gemfibrozil', 'atorvastatin', 'fluvastatin', 'lovastatin', 'pitavastatin', 'pravastatin', 'rosuvastatin', 'simvastatin', 'cholestyramine', 'colesevelam', 'colestipol', 'ezetimibe', 'icosapent', 'lomitapide', 'niacin', 'omega-3-acid', 'omega-3', 'alirocumab', 'evolocumab', 'direct-acting', 'acetazolamide', 'aliskiren', 'digoxin', 'ivabradine', 'mecamylamine', 'pentoxifylline', 'ranolazine', 'atenolol', 'bisoprolol', 'hydralazine', 'metoprolol', 'nadolol', 'nebivolol', 'propranolol', 'sacubitril', 'amphetamine', 'dextroamphetamine', 'lisdexamfetamine', 'methamphetamine', 'atomoxetine', 'dexmethylphenidate', 'methylphenidate', 'milnacipran', 'pregabalin', 'interferon', 'peginterferon', 'dalfampridine', 'teriflunomide', 'fingolimod', 'siponimod', 'dimethyl', 'diroximel', 'alemtuzumab', 'cladribine', 'glatiramer', 'natalizumab', 'ocrelizumab', 'daclizumab', 'amifampridine', 'deutetrabenazine', 'dextromethorphan', 'edaravone', 'gabapentin', 'tetrabenazine', 'riluzole', 'valbenazine', 'desogestrel', 'estradiol', 'drospirenone', 'ethynodiol', 'levonorgestrel', 'norethindrone', 'norgestimate', 'copper', 'etonogestrel', 'medroxyprogesterone', 'ulipristal', 'norelgestromin', 'segesterone', 'benzyl', 'crotamiton', 'lindane', 'malathion', 'permethrin', 'spinosad', 'deferasirox', 'deferiprone', 'penicillamine', 'succimer', 'tolvaptan', 'trientine', 'prussian', 'zinc', 'ammonium', 'carglumic', 'chromic', 'citric', 'fluoride', 'magnesium', 'manganese', 'sodium', 'calcium', 'ferric', 'lanthanum', 'sevelamer', 'sucroferric', 'patiromer', 'folic', 'alosetron', 'crofelemer', 'eluxadoline', 'loperamide', 'rifamycin', 'telotristat', 'difenoxin', 'diphenoxylate', 'dicyclomine', 'glycopyrrolate', 'methscopolamine', 'lactulose', 'polyethylene', 'lubiprostone', 'methylnaltrexone', 'naloxegol', 'naldemedine', 'prucalopride', 'tegaserod', 'linaclotide', 'plecanatide', 'tenapanor', 'cimetidine', 'famotidine', 'nizatidine', 'ranitidine', 'misoprostol', 'sucralfate', 'dexlansoprazole', 'esomeprazole', 'lansoprazole', 'omeprazole', 'pantoprazole', 'rabeprazole', 'alvimopan', 'bezlotoxumab', 'chenodiol', 'metreleptin', 'ursodiol', 'bismuth', 'mirabegron', 'darifenacin', 'fesoterodine', 'flavoxate', 'oxybutynin', 'solifenacin', 'tolterodine', 'trospium', 'alfuzosin', 'silodosin', 'tamsulosin', 'dutasteride', 'finasteride', 'tadalafil', 'bethanechol', 'tiopronin', 'pentosan', 'oxandrolone', 'oxymetholone', 'danazol', 'fluoxymesterone', 'methyltestosterone', 'testosterone', 'estropipate', 'ethinyl', 'hydroxyprogesterone', 'megestrol', 'progesterone', 'clomiphene', 'raloxifene', 'ospemifene', 'methimazole', 'propylthiouracil', 'bradykinin', 'kallikrein', 'calcineurin', 'budesonide', 'cortisone', 'dexamethasone', 'hydrocortisone', 'methylprednisolone', 'prednisolone', 'prednisone', 'triamcinolone', 'balsalazide', 'mesalamine', 'olsalazine', 'sulfasalazine', 'alendronate', 'etidronate', 'ibandronate', 'pamidronate', 'risedronate', 'abaloparatide', 'calcitonin', 'parathyroid', 'teriparatide', 'cinacalcet', 'etelcalcetide', 'denosumab', 'calcifediol', 'calcitriol', 'doxercalciferol', 'paricalcitol', 'romosozumab', 'cenegermin', 'cyclopentolate', 'hydroxypropyl', 'lifitegrast', 'tropicamide', 'hydroxyamphetamine', 'loteprednol', 'phenylephrine', 'brinzolamide', 'brimonidine', 'dorzolamide', 'fluocinolone', 'beclomethasone', 'ciclesonide', 'flunisolide', 'fluticasone', 'mometasone', 'montelukast', 'zafirlukast', 'zileuton', 'aclidinium', 'ipratropium', 'revefenacin', 'tiotropium', 'umeclidinium', 'albuterol', 'arformoterol', 'epinephrine', 'formoterol', 'indacaterol', 'levalbuterol', 'metaproterenol', 'salmeterol', 'terbutaline', 'olodaterol', 'benzonatate', 'guaifenesin', 'tetrahydrozoline', 'triprolidine', 'brompheniramine', 'cromolyn', 'aminophylline', 'theophylline', 'roflumilast', 'endothelin', 'phosphodiesterase-5', 'prostacyclin', 'guanylate', 'acetylcysteine', 'immunomodulators', 'nintedanib', 'pirfenidone', 'avanafil', 'sildenafil', 'vardenafil', 'alprostadil', 'bremelanotide', 'flibanserin', 'prasterone', 'estazolam', 'flurazepam', 'quazepam', 'temazepam', 'triazolam', 'eszopiclone', 'zaleplon', 'zolpidem', 'suvorexant', 'tasimelteon', 'ramelteon', 'armodafinil', 'modafinil', 'pitolisant', 'solriamfetol', '\ufeffabacavir', 'abafungin', 'abagovomab', 'abametapir', 'abaperidone', 'abarelix', 'abatacept', 'abciximab', 'abecomotide', 'abediterol', 'abetimus', 'abexinostat', 'abicipar pegol', 'abitesartan', 'abituzumab', 'abrilumab', 'abrineurin', 'acalisib', 'acarbose', 'acebilustat', 'acemannan', 'acetorphine', 'acetylmethadol', 'aciclovir', 'acipimox', 'acitazanolast', 'acitretin', 'aclantate', 'aclerastide', 'aclidinium bromide', 'acolbifene', 'acorafloxacin', 'acotiamide', 'acoziborole', 'acreozast', 'acrisorcin', 'acrivastine', 'acrizanib', 'acrocinonide', 'acronine', 'actoxumab', 'acumapimod', 'adafosbuvir', 'adalimumab', 'adaprolol', 'adargileukin alfa', 'adarigiline', 'adarotene', 'adatanserin', 'adavivint', 'adavosertib', 'adecatumumab', 'adegramotide', 'adekalant', 'adelmidrol', 'aderbasib', 'adibendan', 'adimlecleucel', 'adipiplon', 'adomeglivant', 'adomiparin sodium', 'adoprazine', 'adrogolide', 'aducanumab', 'afabicin', 'afacifenacin', 'afamelanotide', 'afasevikumab', 'afegostat', 'afeletecan', 'afelimomab', 'afimoxifene', 'aflibercept', 'afovirsen', 'afoxolaner', 'aftobetin', 'afuresertib', 'afutuzumab', 'agalsidase alfa', 'agalsidase beta', 'aganepag', 'aganirsen', 'agatolimod', 'agerafenib', 'aglatimagene besadenovec', 'aglepristone', 'agomelatine', 'alacizumab pegol', 'aladorian', 'alagebrium chloride', 'alalevonadifloxacin', 'alamifovir', 'alatrofloxacin', 'albaconazole', 'albenatide', 'albiglutide', 'albinterferon alfa-2b', 'albitiazolium bromide', 'albusomatropin', 'albutrepenonacog alfa', 'alcaftadine', 'alclofenac', 'aldoxorubicin', 'alefacept', 'aleglitazar', 'alemcinal', 'aleplasinin', 'alfacalcidol', 'alfatradiol', 'alfentanil', 'alferminogene tadenovec', 'alfimeprase', 'alglucosidase alfa', 'alicaforsen', 'alicapistat ', 'alicdamotide', 'alidornase alfa ', 'alilusem', 'alinastine', 'alipogene tiparvovec', 'alirinetide', 'alisertib', 'alisporivir', 'allobarbital', 'allylprodine', 'almagate', 'almorexant', 'almurtide', 'alnespirone', 'alniditan', 'alobresib', 'alofanib', 'aloxistatin', 'alpertine', 'alphacetylmethadol', 'alphameprodine', 'alphamethadol', 'alphaprodine', 'alprafenone', 'alprenolol', 'alsactide', 'altapizone', 'altinicline', 'altiratinib', 'altizide', 'altumomab', 'alvameline', 'alvelestat', 'alvespimycin', 'alvocidib', 'amadinone', 'amatuximab', 'ambasilide', 'ambrisentan', 'ambruticin', 'amcasertib', 'amcinafal', 'amcinafide', 'amcinonide', 'amdoxovir', 'amebucort', 'amediplase', 'amelometasone', 'amelubant', 'amenamevir', 'amfepramone', 'amfetamine', 'amfomycin', 'amibegron', 'amiglumide', 'amilintide', 'amilomotide', 'amiloxate', 'aminoethyl nitrate', 'aminorex', 'amiselimod', 'amitifadine', 'amitriptylinoxide', 'amlintide', 'amobarbital', 'amolimogene bepiplasmid', 'amopyroquine', 'amotosalen', 'amoxydramine camsilate', 'amphenidone', 'amprenavir', 'amsilarotene', 'amuvatinib', 'amylmetacresol', 'anacetrapib', 'anagestone', 'anagliptin', 'anakinra', 'anamorelin', 'anaritide', 'anatibant', 'anatumomab mafenatox', 'anaxirone', 'anazocine', 'ancestim', 'ancriviroc', 'ancrod', 'andecaliximab ', 'andexanet alfa', 'anecortave', 'anetumab ravtansine', 'angiotensinamide', 'anidulafungin', 'anifrolumab', 'anileridine', 'anisperimus', 'anistreplase', 'anivamersen', 'anrukinzumab', 'anseculin', 'antazonite', 'antithrombin alfa', 'antithrombin gamma', 'apabetalone', 'apadenoson', 'apadoline', 'apaflurane', 'apararenone ', 'apatorsen', 'apaxifylline', 'apaziquone', 'apilimod', 'apimostinel ', 'apitolisib', 'apixaban', 'aplaviroc', 'aplindore ', 'apolizumab', 'apratastat', 'apremilast', 'apricitabine', 'apricoxib', 'aprinocarsen', 'aprocitentan', 'aprutumab ', 'aprutumab ixadotin ', 'aptazapine', 'aptiganel', 'arasertaconazole', 'arbaclofen', 'arbaclofen placarbil', 'arbekacin', 'arcitumomab', 'ardenermin', 'ardeparin sodium', 'arfalasin', 'arfolitixorin', 'arginine', 'argipressin', 'argiprestocin', 'arhalofenate', 'arimoclomol', 'arnolol', 'arofylline', 'arotinolol', 'arprinocid', 'artefenomel', 'arteflene', 'artemisone', 'artemotil', 'artenimol', 'arterolane', 'arundic acid', 'arzoxifene', 'asapiprant', 'asciminib ', 'ascorbyl gamolenate', 'ascrinvacumab', 'aselizumab', 'aseripide', 'asfotase alfa', 'asimadoline', 'asivatrep', 'asobamast', 'asoprisnil', 'asoprisnil ecamate', 'astemizole', 'astodrimer', 'asudemotide', 'asunaprevir', 'asunercept', 'asvasiran', 'atabecestat', 'atacicept', 'ataciguat', 'atagabalin', 'ataluren', 'atamestane', 'ataquimast', 'atecegatran', 'atecegatran fexenetil', 'atelocantel', 'atesidorsen', 'atexakin alfa', 'atibeprone', 'atidortoxumab', 'atigliflozin', 'atilmotin', 'atinumab', 'atiprimod', 'atiprosin', 'atiratecan', 'atizoram', 'atliprofen', 'atocalcitol', 'atogepant', 'atopaxar', 'atorolimumab', 'atrasentan', 'atreleuton', 'atrimustine', 'atromepine', 'atuveciclib ', 'audencel', 'auriclosene', 'avacincaptad pegol', 'avacopan', 'avadomide', 'avagacestat', 'avalglucosidase alfa', 'avapritinib', 'avasimibe', 'avatrombopag', 'avibactam', 'aviptadil', 'aviscumine', 'avitriptan', 'avoralstat', 'avorelin', 'avosentan', 'avotermin', 'axalimogene filolisbac', 'axelopran', 'axicabtagene ciloleucel', 'axitirome', 'axomadol', 'azabon', 'azalanstat', 'azatadine', 'azathioprine', 'azeliragon', 'azeloprazole', 'azetirelin', 'azilsartan medoxomil', 'azimilide', 'azintuxizumab', 'azintuxizumab vedotin', 'azoximer bromide', 'bacmecillinam', 'bafetinib', 'balaglitazone', 'balamapimod', 'balapiravir', 'balazipone', 'balicatib', 'baliforsen', 'balipodect', 'balixafortide', 'balofloxacin', 'balovaptan', 'baloxavir marboxil', 'baltaleucel', 'balugrastim', 'bamaquimast', 'baminercept', 'bamirastine', 'bamosiran', 'banoxantrone', 'bapineuzumab', 'barasertib', 'barbital', 'bardoxolone', 'baricitinib', 'barixibat', 'barusiban', 'basifungin', 'basiliximab', 'basimglurant', 'basmisanil', 'batabulin', 'batefenterol', 'batimastat', 'bavisant', 'bavituximab', 'bazedoxifene', 'bazlitoran', 'becampanel', 'becaplermin', 'becatecarin', 'beclanorsen', 'beclavubir', 'becocalcidiol', 'bectumomab', 'bederocin', 'bedoradrine', 'befetupitant', 'befiradol', 'begacestat', 'begelomab', 'beinaglutide', 'belaperidone', 'belatacept', 'belizatinib', 'belnacasan', 'beloranib', 'belotecan', 'beloxepin', 'bemarituzumab', 'bemcentinib', 'beminafil', 'bemiparin sodium', 'bemotrizinol', 'bempedoic acid', 'benfurodil hemisuccinate', 'benolizime', 'benperidol', 'benralizumab', 'bentamapimod', 'bentemazole', 'benzalkonium chloride', 'benzethidine', 'benzfetamine', 'benzindopyrine', 'benzodrocortisone', 'benzopyrronium bromide', 'benzpiperylone', 'beperminogene perplasmid', 'bepotastine', 'berdazimer sodium', 'berlafenone', 'berlimatoxumab', 'beroctocog alfa', 'bertilimumab', 'berubicin', 'berupipam', 'bervastatin', 'berzosertib', 'besifovir', 'besilesomab', 'besipirdine', 'betacetylmethadol', 'betameprodine', 'betamethadol', 'betamicin', 'betaprodine', 'betasizofiran', 'betaxolol', 'betiatide', 'betibeglogene darolentivec', 'betoxycaine', 'betrixaban', 'bevacizumab beta', 'bevasiranib', 'bevenopran', 'bevirimat', 'bexaglifozin', 'bexlosteride', 'bezafibrate', 'bezitramide', 'biapenem', 'bibapcitide', 'bifarcept', 'bifeprunox', 'bilastine', 'bimagrumab', 'bimatoprost', 'bimekizumab', 'bimiralisib', 'bimoclomol', 'bimosiamose', 'binetrakin', 'binfloxacin', 'binizolast', 'binodenoson', 'birabresib ', 'biricodar', 'birinapant', 'biriperone', 'bisaramil', 'bisegliptin', 'bisnafide', 'bisoctrizole', 'bivalirudin', 'bivatuzumab', 'bixalomer', 'bleselumab', 'blisibimod', 'blonanserin', 'blontuvetmab', 'blosozumab', 'boceprevir', 'bococizumab', 'bornaprolol', 'bortezomib', 'bosentan', 'bovhyaluronidase azoximer', 'bradanicline', 'branaplam ', 'brasofensine', 'brazergoline', 'brazikumab ', 'brecanavir', 'bremazocine', 'brentuximab vedotin', 'brexanolone', 'briakinumab', 'briciclib', 'brifentanil', 'brilacidin', 'brilanestrant ', 'brimapitide', 'brincidofovir', 'briobacept', 'brivanib alaninate', 'brivaracetam', 'brivoligide', 'brobactam', 'brodalumab', 'brolamfetamine', 'brolucizumab', 'bromazepam', 'bromerguride', 'brometenamine', 'bromhexine', 'bromindione', 'brontictuzumab', 'broperamole', 'brostallicin', 'brotizolam', 'bucelipase alfa', 'buciclovir', 'budiodarone', 'budotitane', 'bulaquine', 'bumepidil', 'bunaprolast', 'bunazosin', 'bunolol', 'buparlisib', 'bupranolol', 'buquiterine', 'burapitant', 'burixafor', 'burlulipase', 'burosumab ', 'buserelin', 'butamirate', 'buthalital sodium', 'butixocort', 'butofilolol', 'butropium bromide', 'butylphthalide', 'cabiotraxetan', 'cabiralizumab', 'cabotegravir', 'cadazolid', 'cadrofloxacin', 'calaspargase pegol', 'calcium carbimide', 'calcobutrol', 'caldaret', 'caloxetic acid', 'camazepam', 'camicinal', 'camidanlumab', 'camidanlumab tesirine', 'camobucol', 'camrelizumab ', 'canakinumab', 'candocuronium iodide', 'canerpaturev', 'canertinib', 'canfosfamide', 'cangrelor', 'cannabidiol ', 'canoctakin', 'canosimibe', 'cantuzumab ravtansine', 'capadenoson', 'capeserod', 'capivasertib', 'caplacizumab', 'capmatinib', 'capravirine', 'capromab', 'capromorelin', 'capsaicin', 'carabersat', 'carafiban', 'carbaldrate', 'carbasalate calcium', 'carbazocine', 'carbetimer', 'carboplatin', 'carfenazine', 'carfilzomib', 'carglumic acid', 'caricotamide', 'cariporide', 'carisbamate', 'carlumab', 'carmegliptin', 'carmofur', 'carmoterol', 'carotegrast', 'carotuximab', 'cartasteine', 'carumonam', 'casimersen ', 'casopitant', 'caspofungin', 'cathine', 'cathinone', 'catramilast', 'catridecacog', 'catumaxomab', 'cavosonstat', 'cebranopadol', 'ceclazepide', 'cedelizumab', 'cediranib', 'cefiderocol', 'cefilavancin', 'cefluprenam', 'cefmatilen', 'cefoselis', 'cefovecin', 'cefquinome', 'ceftaroline fosamil', 'cefteram', 'ceftizoxime alapivoxil', 'ceftobiprole', 'ceftobiprole medocaril', 'cefuzonam', 'celgosivir', 'celivarone', 'cemadotin', 'cemdisiran', 'cenderitide', 'cenegermin ', 'cenerimod', 'cenersen', 'cenicriviroc', 'cenisertib', 'cenobamate', 'cenplacel', 'censavudine', 'centanafadine', 'cepeginterferon alfa-2b', 'ceralifimod', 'cerdulatinib', 'cergutuzumab amunaleukin', 'cerivastatin', 'cerlapirdine', 'cerliponase alfa', 'certolizumab pegol', 'certoparin sodium', 'ceruletide', 'cetermin', 'cethromycin', 'cetilistat', 'cetrimide', 'cevimeline', 'cevipabulin', 'cevoglitazar', 'chloralodol', 'chlorotrianisene', 'choline fenofibrate', 'choriogonadotropin alfa', 'ciaftalan zinc', 'cibinetide', 'cicloprolol', 'ciclosporin', 'cilengitide', 'cilmostim', 'cilobradine', 'cilofungin', 'cilomilast', 'ciluprevir', 'cimaglermin alfa', 'cinaciguat', 'cinalukast', 'cinamolol', 'cindunistat', 'cinhyaluronate sodium', 'cintredekin besudotox', 'cipamfylline', 'cipargamin', 'cipemastat', 'cipralisant', 'ciraparantag', 'cisatracurium besilate', 'cismadinone', 'citarinostat', 'citatuzumab bogatox', 'cixutumumab', 'cizolirtine', 'clamikalant', 'clazakizumab', 'clazosentan', 'clenoliximab', 'clevidipine', 'clevudine', 'cliropamine', 'clivatuzumab tetraxetan', 'clobazam', 'clofarabine', 'clofenoxyde', 'clomifenoxide', 'clonitazene', 'cloprednol', 'cloranolol', 'clotiazepam', 'cloxazolam', 'cloxestradiol', 'cobiprostone', 'cobitolimod', 'cobomarsen', 'codactide', 'codoxime', 'codrituzumab', 'cofetuzumab', 'cofetuzumab pelidotin', 'coleneuramide', 'colestilan', 'colimecycline', 'coltuximab ravtansine', 'coluracetam', 'conatumumab', 'conbercept', 'concizumab', 'condoliase', 'conestat alfa', 'conivaptan', 'contusugene ladenovec', 'corifollitropin alfa', 'cortodoxone', 'cosdosiran', 'cosfroviximab', 'cositecan', 'crenezumab', 'crenigacestat', 'crenolanib', 'cridanimod', 'crilvastatin', 'crisaborole', 'crisantaspase', 'crisnatol', 'crizanlizumab ', 'crobenetine', 'crolibulin', 'cromakalim', 'cromoglicate lisetil', 'crotedumab', 'crotoniazide', 'cupabimod ', 'custirsen', 'cutamesine', 'cyclobarbital', 'cystine', 'dabelotine', 'dabigatran', 'dabigatran etexilate', 'dacetuzumab', 'daclatasvir', 'daclizumab beta', 'dactolisib', 'daglutril', 'dagrocorat', 'dalantercept', 'dalatazide', 'dalbraminol', 'dalcetrapib', 'dalcotidine', 'dalotuzumab', 'dalteparin sodium', 'damoctocog alfa pegol', 'danaparoid sodium', 'danegaptide', 'daniplestim', 'daniquidone', 'danirixin', 'danoprevir', 'danusertib', 'danvatirsen', 'dapabutan', 'dapaconazole', 'dapansutrile', 'dapiclermin', 'dapirolizumab pegol', 'dapitant', 'dapivirine', 'daporinad', 'daprodustat', 'darapladib', 'darbepoetin alfa', 'darbufelone', 'darexaban', 'darglitazone', 'darinaparsin', 'darolutamide ', 'darotropium bromide', 'darsidormine', 'darusentan', 'dasantafil', 'dasiglucagon', 'dasolampanel', 'dasotraline', 'datelliptium chloride', 'davalintide', 'davamotecan pegadexamer', 'davasaicin', 'davunetide', 'daxalipram', 'decernotinib', 'decitropine', 'declenperone', 'declopramide', 'decoglurant', 'decominol', 'dectrekumab', 'defactinib', 'deferitazole', 'deferitrin', 'deflazacort', 'defoslimod', 'degarelix', 'delamanid', 'delanzomib', 'delcasertib', 'deldeprevir', 'deleobuvir', 'delequamine', 'delgocitinib', 'deligoparin sodium', 'delimotecan', 'delmitide', 'delorazepam', 'deloxolone', 'delparantag', 'delpazolid', 'deltibant', 'delucemine', 'dematirsen', 'demcizumab', 'demegestone', 'demiditraz', 'demplatin pegraglumer', 'denagliptin', 'denaverine', 'denbufylline', 'denenicokin', 'denibulin', 'denileukin diffitox', 'denintuzumab mafodotine', 'denotivir', 'denufosol', 'deoxycholic acid', 'depatuxizumab ', 'depatuxizumab mafodotin ', 'depelestat', 'depreotide', 'deptropine', 'deracoxib', 'derazantinib', 'derenofylline', 'derpanicate', 'derquantel', 'dersalazine', 'desaspidin', 'desciclovir', 'descinolone', 'desfesoterodine', 'desidustat', 'desirudin', 'deslanoside', 'desloratadine', 'desmeninol', 'desmetramadol', 'desmoteplase', 'desomorphine', 'desoximetasone', 'detanosal', 'detiviciclovir', 'detumomab', 'deudextromethorphan', 'deutolperisone', 'dexamethasone cipecilate', 'dexamfetamine', 'dexbudesonide', 'dexecadotril', 'dexefaroxan', 'dexelvucitabine', 'dexisometheptene ', 'dexketoprofen', 'dexmecamylamine', 'dexnebivolol', 'dexpemedolac', 'dexpramipexole', 'dexpropranolol', 'dexsotalol', 'dextiopronin', 'dextofisopam', 'dextromoramide', 'dextropropoxyphene', 'dextrorphan', 'dezaguanine', 'dezamizumab ', 'dezapelisib', 'diampromide', 'dianexin', 'dianhydrogalactitol', 'dianicline', 'diaplasinin', 'dibotermin alfa', 'dibunate', 'dichlorisone', 'diciferron', 'dicirenone', 'diclofenac etalhyaluronate', 'dicobalt edetate', 'dienestrol', 'dienogest', 'diethylstilbestrol', 'diethylthiambutene', 'dietifen', 'difelikefalin', 'difemerine', 'difenoximide', 'difeterol', 'diflomotecan', 'difluprednate', 'dihydrocodeine', 'dilmapimod', 'dilopetine', 'dimabefylline', 'dimadectin', 'dimelazine', 'dimenoxadol', 'dimepheptanol', 'dimesone', 'dimethylthiambutene', 'dinaciclib', 'dinalbuphine sebacate', 'dinutuximab beta', 'dioxaphetyl butyrate', 'dipipanone', 'diprafenone', 'dipraglurant', 'diprogulic acid', 'diridavumab', 'dirithromycin', 'dirlotapide', 'diroximel fumarate', 'dirucotide', 'disitertide', 'disogluside', 'disomotide', 'disufenton sodium', 'disulergine', 'disuprazole', 'ditekiren', 'ditercalinium chloride', 'dithranol', 'divaplon', 'dizocilpine', 'docetaxel', 'dociparstat sodium', 'dofequidar', 'dolcanatide', 'domagrozumab', 'domitroban', 'domoprednate', 'domperidone', 'donaperminogene seltoplasmid', 'donitriptan', 'doqualast', 'doramapimod', 'doramectin', 'doranidazole', 'dorastine', 'doripenem', 'dornase alfa', 'dorzagliatin', 'dosergoside', 'dotinurad', 'dovitinib', 'draquinolol', 'drinabant', 'drisapersen', 'droloxifene', 'dronedarone', 'drotebanol', 'drotrecogin alfa (activated)', 'droxinavir', 'drozitumab', 'dulaglutide', 'dulanermin', 'duligotumab', 'dupilumab', 'dusigitumab', 'dusquetide', 'dutacatib', 'duteplase', 'dutogliptin', 'duvoglustat', 'duvortuxizumab', 'ebalzotan', 'ecalcidene', 'ecallantide', 'ecamsule', 'ecenofloxacin', 'ecipramidil', 'ecogramostim', 'ecomustine', 'ecopipam', 'ecopladib', 'ecraprost', 'ecromeximab', 'eculizumab', 'edaglitazone', 'edasalonexent', 'edelfosine', 'edifolone', 'edivoxetine', 'edodekin alfa', 'edonentan', 'edonerpic', 'edotecarin', 'edotreotide', 'edoxaban', 'edratide', 'edrecolomab', 'edronocaine', 'efalizumab', 'efaproxiral', 'efegatran', 'efepoetin alfa', 'efepristin', 'efgartigimod alfa', 'efinaconazole', 'efipladib', 'efizonerimod alfa', 'eflapegastrim ', 'eflenograstim alfa', 'efletirizine', 'eflucimibe', 'efmoroctocog alfa', 'efpeglenatide', 'efpegsomatropin', 'eftilagimod alfa', 'eftrenonacog alfa', 'efungumab', 'eganoprost', 'egaptivon pegol', 'eglumetad', 'elacestrant ', 'elacridar', 'elacytarabine', 'elafibranor', 'elagolix', 'elamipretide', 'elapegademase ', 'elarofiban', 'elbimilast', 'eldacimibe', 'eldecalcitol', 'eldelumab', 'eleclazine', 'eledoisin', 'elenbecestat', 'elesclomol', 'elezanumab ', 'elgemtumab', 'eliglustat', 'elinafide', 'elinogrel', 'elisartan', 'elisidepsin', 'elivaldogene tavalentivec ', 'elliptinium acetate', 'elobixibat', 'elocalcitol', 'elomotecan', 'elopiprazole', 'elosulfase alfa', 'elpamotide', 'elpetrigine', 'elsamitrucin', 'elsibucol', 'elsiglutide', 'elsulfavirine', 'eltanexor', 'eltrapuldencel', 'eltrombopag', 'elubrixin', 'elzasonan', 'emactuzumab', 'emapalumab ', 'emapticap pegol', 'emapunil', 'embeconazole', 'embusartan', 'emeramide', 'emfilermin', 'emibetuzumab', 'emicerfont', 'emicizumab', 'emideltide', 'emivirine', 'emixustat', 'emoctakin', 'emodepside', 'empegfilgrastim', 'empesertib', 'emricasan', 'enadenoticirev', 'enarodustat', 'enavatuzumab', 'encaleret', 'encenicline', 'endixaprine', 'enecadin', 'enerisant', 'enestebol', 'enfortumab vedotin', 'eniluracil', 'eniporide', 'enisamium iodide', 'enlimomab', 'enlimomab pegol', 'enoblituzumab', 'enobosarm', 'enocitabine', 'enokizumab', 'enoticumab', 'enoxacin', 'enoxaparin sodium', 'enoximone', 'enprofylline', 'enrasentan', 'ensartinib ', 'ensereptide', 'ensituximab', 'ensulizole', 'entasobulin', 'entinostat', 'entolimod', 'entospletinib', 'enzacamene', 'enzaplatovir ', 'enzastaurin', 'epacadostat', 'epafipase', 'epelsiban', 'eperezolid', 'epertinib ', 'epervudine', 'epetirimod', 'epetraborole', 'epicriptine', 'epiestriol', 'epimestrol', 'epipropidine', 'epirubicin', 'epitizide', 'epitumomab', 'eplivanserin', 'epoetin alfa', 'epoetin beta', 'epoetin delta', 'epoetin epsilon', 'epoetin gamma', 'epoetin kappa', 'epoetin omega', 'epoetin theta', 'epoetin zeta', 'epratuzumab', 'eprinomectin', 'eprociclovir', 'eprodisate', 'eprotirome', 'epsiprantel', 'eptacog alfa (activated)', 'eptacog alfa pegol (activated)', 'eptacog beta (activated)', 'eptapirone', 'eptaplatin', 'eptifibatide', 'eptinezumab ', 'eptotermin alfa', 'erenumab ', 'eretidigene velentivec ', 'ergometrine', 'eribaxaban', 'eribulin', 'ericolol', 'eritoran', 'eritrityl tetranitrate', 'erlizumab', 'ersentilide', 'erteberel', 'ertiprotafib', 'ertumaxomab', 'esatenolol', 'esaxerenone', 'esmirtazapine', 'esmolol', 'esonarimod', 'esorubicin', 'esoxybutynin', 'esreboxetine', 'estetrol', 'esuberaprost', 'etafedrine', 'etafenone', 'etalocib', 'etamicastat', 'etanercept', 'etaracizumab', 'eteplirsen', 'ethchlorvynol', 'ethinamate', 'ethyl loflazepate', 'ethylcellulose', 'ethylestrenol', 'ethylmethylthiambutene', 'ethynerone', 'ethypicone', 'eticyclidine', 'etiguanfacine', 'etilamfetamine', 'etilevodopa', 'etiprednol dicloacetate', 'etirinotecan pegol', 'etisomicin', 'etisulergine', 'etofenamate', 'etofylline', 'etofylline clofibrate', 'etonitazene', 'etoricoxib', 'etorphine', 'etoxeridine', 'etoxybamide', 'etrabamine', 'etrasimod', 'etriciguat', 'etripamil', 'etrolizumab', 'eufauserase', 'evacetrapib', 'evandamine', 'evatanepag', 'evenamide', 'evernimicin', 'evinacumab', 'evobrutinib ', 'evocalcet', 'evodenoson', 'evofosfamide', 'evogliptin', 'exametazime', 'examorelin', 'exaprolol', 'exatecan', 'exatecan alideximer', 'exbivirumab', 'exebacase', 'exenatide', 'exeporfinium chloride', 'exisulind', 'ezatiostat', 'ezlopitant', 'ezutromid', 'fabesetron', 'facinicline', 'fadolmidine', 'faldaprevir', 'falnidamol', 'famiraprinium chloride', 'fampridine', 'fampronil', 'fanapanel', 'fandofloxacin', 'fandosentan', 'faralimomab', 'farampator', 'farglitazar', 'farletuzumab', 'fasidotril', 'fasiglifam', 'fasinumab', 'fasitibant chloride', 'fasobegron', 'fasoracetam', 'favipiravir', 'faxeladol', 'fazarabine', 'febuxostat', 'fedotozine', 'fedovapagon', 'felbinac', 'feloprentan', 'felvizumab', 'fenalcomine', 'fencamfamin', 'fenebrutinib', 'fenetylline', 'fenfluthrin', 'fenleuton', 'fenproporex', 'fentonium bromide', 'fermagate', 'ferric (59Fe) citrate injection', 'ferric carboxymaltose', 'ferric derisomaltose', 'ferric maltol', 'ferroquine', 'ferrotrenine', 'fevipiprant', 'fexapotide', 'fexofenadine', 'fezakinumab', 'fezolinetant ', 'fiacitabine', 'fiboflapon', 'ficlatuzumab', 'fidarestat', 'fidexaban', 'fiduxosin', 'figitumumab', 'figopitant', 'filaminast', 'filanesib', 'filgotinib', 'filibuvir', 'filociclovir', 'filorexant', 'fimaporfin', 'fimasartan', 'finafloxacin', 'finerenone', 'finrozole', 'fipamezole', 'firategrast', 'firibastat', 'firivumab', 'firtecan peglumer', 'firtecan pegol', 'firuglipel', 'fispemifene', 'fitusiran', 'flanvotumab', 'flestolol', 'fletikumab', 'flindokalner', 'flomoxef', 'flopristin', 'florbenazine (18F)', 'florbetaben (18F)', 'florbetapir (18F)', 'florfenicol', 'florilglutamic acid (18F)', 'flortanidazole (18F)', 'flortaucipir (18F)', 'flotegatide (18F)', 'flovagatran', 'fluciclatide (18F)', 'fluciclovine (18F)', 'fludalanine', 'fludeoxyglucose (18F)', 'fludiazepam', 'flugestone', 'fluindarol', 'fluindione', 'flumazenil', 'flumoxonide', 'flunitrazepam', 'fluocortin', 'fluorfenidine (18F)', 'fluralaner', 'flurdihydroergotamine ', 'flurpiridaz (18F)', 'flusoxolol', 'fluspirilene', 'flutafuranol (18F)', 'flutemetamol (18F)', 'fluticasone furoate', 'flutriciclamide (18F)', 'fodipir', 'foliglurax', 'folitixorin', 'follitropin alfa', 'follitropin beta', 'follitropin delta', 'follitropin epsilon ', 'follitropin gamma', 'fomidacillin', 'fomivirsen', 'fonadelpar', 'fondaparinux sodium', 'fontolizumab', 'fonturacetam', 'foralumab', 'forasartan', 'foravirumab', 'foretinib', 'forigerimod', 'formebolone', 'forodesine', 'foropafant', 'fosalvudine tidoxil', 'fosaprepitant', 'fosbretabulin', 'fosdagrocorat', 'fosdevirine', 'fosfluconazole', 'fosfluridine tidoxil', 'fosfructose', 'fosmetpantotenate', 'fosopamine', 'fospropofol', 'fosquidone', 'fosravuconazole', 'fostamatinib', 'fostemsavir', 'fostriecin', 'fosveset', 'fozivudine tidoxil', 'fradafiban', 'frakefamide', 'fremanezumab ', 'fresolimumab', 'frunevetmab', 'fruquintinib', 'fudosteine', 'fulacimstat', 'fuladectin', 'fulranumab', 'funapide', 'furaprevir', 'furethidine', 'furomine', 'futuximab', 'gabapentin enacarbil', 'gacyclidine', 'gadocoletic acid', 'gadodenterate', 'gadodiamide', 'gadofosveset', 'gadomelitol', 'gadopenamide', 'gadopentetic acid', 'gadoteric acid', 'gadoversetamide', 'gadoxetic acid', 'galarubicin', 'galdansetron', 'galeterone', 'galidesivir', 'galsulfase', 'galunisertib', 'gamithromycin', 'ganaxolone', 'gandotinib', 'ganetespib', 'ganglefene', 'ganitumab', 'ganstigmine', 'gantacurium chloride', 'gantenerumab', 'gantofiban', 'garenoxacin', 'garnocestim', 'garvagliptin', 'gataparsen', 'gatipotuzumab', 'gavestinel', 'gavilimomab', 'gaxilose', 'geclosporin', 'gedatolisib', 'gedivumab', 'gemcabene', 'gemcitabine', 'gemcitabine elaidate', 'gemigliptin', 'gemilukast', 'gemopatrilat', 'gemtuzumab ozogamicin ', 'gepotidacin', 'gevokizumab', 'gilvetmab', 'gimatecan', 'gimeracil', 'giminabant', 'gimsilumab', 'giractide', 'girentuximab', 'giripladib', 'gisadenafil', 'givinostat', 'givosiran', 'glaspimod', 'glaziovine', 'glembatumumab', 'glembatumumab vedotin', 'glenvastatin', 'glepaglutide', 'gleptoferron', 'glesatinib', 'gloximonam', 'glucalox', 'glucarpidase', 'glufosfamide', 'glusoferron', 'glutethimide', 'glycerol phenylbutyrate', 'golimumab', 'golnerminogene pradenovec', 'golodirsen', 'golotimod', 'golvatinib', 'goralatide', 'goserelin', 'gosogliptin', 'goxalapladib', 'granotapide', 'grapiprant', 'graunimotide', 'grazoprevir', 'guadecitabine', 'guafecainol', 'guaiactamine', 'guaraprolose', 'guselkumab', 'halazepam', 'halopredone', 'haloxazolam', 'hemoglobin betafumaril (bovine) ', 'hemoglobin crosfumaril', 'hemoglobin crosfumaril (bovine)', 'hemoglobin glutamer', 'hemoglobin raffimer', 'hexobendine', 'histrelin', 'homidium bromide', 'hydroflumethiazide', 'hydromorphinol', 'hydroxycarbamide', 'hydroxypethidine', 'hyetellose', 'hymetellose', 'hyprolose', 'hypromellose', 'ianalumab', 'ibacitabine', 'ibafloxacin', 'ibandronic acid', 'ibazocine', 'iberdomide', 'ibipinabant', 'iboctadekin', 'ibodutant', 'ibritumomab tiuxetan', 'ibrolipim', 'ibudilast', 'ibutamoren', 'icaridin', 'iclaprim', 'icometasone enbutate', 'icomucret', 'icopezil', 'icosabutate', 'icospiramide', 'icrucumab', 'idalopirdine', 'idarubicin', 'idarucizumab', 'idasanutlin', 'idrabiotaparinux sodium', 'idramantone', 'idraparinux sodium', 'idremcinal', 'idronoxil', 'idropranolol', 'idursulfase', 'idursulfase beta', 'ifabotuzumab', 'iferanserin', 'ifetroban', 'iganidipine', 'igovomab', 'iguratimod', 'iladatuzumab', 'iladatuzumab vedotin', 'ilaprazole', 'ilepatril', 'ilepcimide', 'ilixadencel', 'ilmetropium iodide ', 'ilodecakin', 'ilomastat', 'ilonidap', 'iloprost', 'ilorasertib', 'imagabalin', 'imalumab', 'imarikiren', 'imazodan', 'imeglimin', 'imepitoin', 'imetelstat', 'imgatuzumab', 'imidafenacin', 'imidaprilat', 'imiglitazar', 'imiglucerase', 'imirestat', 'imisopasem manganese', 'imitrodast', 'imlatoclax ', 'imlifidase', 'implitapide', 'inakalant', 'inarigivir soproxil', 'incadronic acid', 'inclacumab', 'inclisiran', 'incyclinide', 'indalpine', 'indanidine', 'indantadol', 'indatuximab ravtansine', 'indeglitazar', 'indenolol', 'indibulin', 'indimilast', 'indiplon', 'indisetron', 'indisulam', 'indolidan', 'indoprofen', 'indoramin', 'indoximod', 'indusatumab', 'indusatumab vedotin', 'inebilizumab', 'inecalcitol', 'infigratinib', 'infliximab', 'ingenol disoxate', 'ingenol mebutate', 'ingliforib', 'iniparib', 'inocoterone', 'inogatran', 'inolimomab', 'inolitazone', 'inositol', 'inotersen ', 'inotuzumab ozogamicin', 'insulin argine', 'insulin aspart', 'insulin degludec', 'insulin detemir', 'insulin glargine', 'insulin glulisine', 'insulin lispro', 'insulin peglispro', 'insulin tregopil', 'intedanib', 'intepirdine', 'interferon alfacon-1', 'intetumumab', 'intiquinatine', 'iocanlidic acid', 'iodine (124I) girentuximab', 'iodine (131I) derlotuximab biotin', 'iodofiltic acid (123I)', 'iodophthalein sodium', 'ioflubenzamide (131I)', 'ioflupane (123I)', 'iofolastat (123I)', 'ioforminol', 'iolopride (123I)', 'iometopane (123I)', 'iosimenol', 'ipafricept', 'ipamorelin', 'ipatasertib', 'ipazilide', 'ipenoxazone', 'ipidacrine', 'ipragliflozin', 'ipragratine', 'ipratropium bromide', 'ipravacaine', 'iproxamine', 'ipsapirone', 'iralukast', 'irampanel', 'iratumumab', 'irbinitinib', 'irdabisant', 'irinotecan', 'irofulven', 'iroplact', 'irosustat', 'iroxanadine', 'irtemazole', 'isalmadol', 'isamoltan', 'isatoribine', 'isatuximab', 'isavuconazole', 'isavuconazonium chloride', 'isbufylline', 'iseganan', 'isepamicin', 'ismomultin alfa', 'isoetarine', 'isoflupredone', 'isometamidium chloride', 'isomethadone', 'isophane insulin', 'isoprednidene', 'isoprenaline', 'isosorbide', 'isosulpride', 'ispinesib', 'ispronicline', 'israpafant', 'istaroxime', 'istiratumab', 'isunakinra', 'itacitinib ', 'itameline', 'itanapraced', 'itolizumab', 'itriglumide', 'itrocainide', 'iturelix', 'ivacaftor', 'ivoqualine', 'ixabepilone', 'ixekizumab', 'izonsteride', 'josamycin', 'kalafungin', 'kallidinogenase', 'kanamycin', 'keliximab', 'keracyanin', 'ketanserin', 'ketazocine', 'ketazolam', 'ketobemidone', 'ketocaine', 'ketocainol', 'ketorfanol', 'ketotrexate', 'khelloside', 'kitasamycin', 'labetuzumab', 'labetuzumab govitecan', 'labradimil', 'lacnotuzumab', 'lactalfate', 'ladarixin', 'ladiratuzumab', 'ladiratuzumab vedotin', 'ladirubicin', 'laflunimus', 'lafutidine', 'lagatide', 'lagociclovir', 'laidlomycin', 'lamifiban', 'lampalizumab', 'lamtidine', 'lanabecestat', 'lanacogene vosiparvovec', 'lanadelumab', 'lancovutide', 'landiolol', 'landipirdine', 'landogrozumab', 'lanepitant', 'lanicemine', 'lanifibranor', 'lanimostim', 'laninamivir', 'laniquidar', 'lanopepden', 'lanoteplase', 'lanperisone', 'lanproston', 'lapaquistat', 'lapisteride', 'laprafylline', 'laprituximab', 'laprituximab emtansine', 'laquinimod', 'larazotide', 'larcaviximab', 'laromustine', 'laronidase', 'laropiprant', 'larotaxel', 'larotrectinib ', 'lasalocid', 'lascufloxacin', 'lasinavir', 'lasofoxifene', 'latamoxef', 'latanoprostene bunod', 'latidectin', 'latrepirdine', 'latromotide', 'laurcetium bromide', 'lauroguadine', 'lauromacrogol 400', 'lavamilast', 'lavoltidine', 'lazertinib', 'lebrikizumab', 'lecimibide', 'leconotide', 'lecozotan', 'ledismase', 'ledoxantrone', 'lefetamine', 'lefitolimod', 'leflutrozole', 'lefradafiban', 'lemalesomab', 'lemborexant', 'lemidosul', 'lemoxinol', 'lemuteporfin', 'lenadogene nolparvovec', 'lenampicillin', 'lenapenem', 'lenercept', 'leniolisib', 'lenomorelin', 'lensiprazine', 'lenzilumab', 'lepirudin', 'lerdelimumab', 'leridistim', 'lerimazoline', 'lersivirine', 'lesinidase alfa', 'lesinurad', 'lesofavumab', 'lesogaberan', 'lestaurtinib', 'letaxaban', 'leteprinim', 'letolizumab', 'leuprorelin', 'levacetylmethadol', 'levallorphan', 'levamfetamine', 'levamlodipine', 'levisoprenaline', 'levmetamfetamine', 'levobetaxolol', 'levobunolol', 'levobupivacaine', 'levocetirizine', 'levoglucose', 'levoketoconazole', 'levolansoprazole', 'levomefolic acid', 'levomequitazine', 'levomethorphan', 'levomoprolol', 'levomoramide', 'levonadifloxacin', 'levonebivolol', 'levophenacylmorphan', 'levopropoxyphene', 'levopropylhexedrine', 'levormeloxifene', 'levosalbutamol', 'levosemotiadil', 'levosulpiride', 'levotofisopam', 'lexacalcitol', 'lexanopadol', 'lexaptepid pegol', 'lexatumumab', 'lexibulin', 'lexipafant', 'liafensine', 'liatermin', 'libenzapril', 'libivirumab', 'licarbazepine', 'licofelone', 'licostinel', 'lidadronic acid', 'lidamidine', 'lidorestat', 'lifastuzumab vedotin', 'lifibrol', 'lificiguat', 'lifirafenib', 'ligelizumab', 'lilopristone', 'lilotomab', 'limiglidole', 'linaprazan', 'linetastine', 'linifanib', 'linopristin', 'linsitinib', 'lintitript', 'lintuzumab', 'liothyronine', 'lipegfilgrastim', 'liraglutide', 'lirequinil', 'lirexapride', 'lirilumab', 'lirimilast', 'lisadimate', 'lisavanbulin ', 'lisofylline', 'lisuride', 'litenimod', 'litomeglovir', 'litronesib', 'livaraparin calcium', 'lividomycin', 'lixazinone', 'lixisenatide', 'lixivaptan', 'lobeglitazone', 'lobeline', 'lobucavir', 'lodelcizumab', 'lodenafil carbonate', 'lodenosine', 'lodoxamide', 'lokivetmab', 'lometraline', 'lometrexol', 'lomevactone', 'lomibuvir', 'lomifylline', 'lonafarnib', 'lonapalene', 'lonapegsomatropin', 'lonaprisan', 'loncastuximab', 'loncastuximab tesirine', 'lonoctocog alfa', 'lopobutan', 'loprazolam', 'loprodiol', 'loracarbef', 'lorajmine', 'lorcaserin', 'loreclezole', 'lorediplon', 'lormetazepam', 'lorpiprazole', 'lorvotuzumab mertansine', 'losatuxizumab', 'losatuxizumab vedotin', 'losigamone', 'losmapimod', 'losmiprofen', 'lotilaner', 'lotilibcin', 'lotrafiban', 'loviride', 'loxicodegol', 'loxoprofen', 'lozilurea', 'lubabegron', 'lubazodone', 'lubeluzole', 'lucatumumab', 'lucerastat', 'lucimycin', 'lucitanib', 'luliconazole', 'lulizumab pegol', 'lumacaftor', 'lumasiran', 'lumateperone', 'lumefantrine', 'lumicitabine ', 'lumiliximab', 'luminespib', 'lumiracoxib', 'lumretuzumab', 'lunacalcipol', 'lupartumab ', 'lupartumab amadotin ', 'lupitidine', 'lurbinectedin', 'lurtotecan', 'lusaperidone', 'luseogliflozin', 'luspatercept', 'lusupultide', 'lusutrombopag', 'lutetium (177Lu) lilotomab satetraxetan', 'lutetium (177Lu) oxodotreotide', 'lutikizumab ', 'lutrelin', 'lutropin alfa', 'lymecycline', 'lynestrenol', 'lypressin', 'lysergide', 'macimorelin', 'macitentan', 'macrogol ester', 'macrosalb (131 I)', 'macrosalb (99m Tc)', 'maduramicin', 'magaldrate', 'magnesium clofibrate', 'maletamer', 'maleylsulfathiazole', 'malotilate', 'managlinat dialanetil', 'mangafodipir', 'manifaxine', 'manitimus', 'mannitol hexanitrate', 'mannosulfan', 'manozodil', 'mantabegron', 'mapatumumab', 'mapinastine', 'mapracorat', 'maraciclatide', 'maralixibat chloride', 'mardepodect', 'margetuximab', 'maribavir', 'maridomycin', 'marimastat', 'mariptiline', 'marizomib', 'maropitant', 'marzeptacog alfa (activated)', 'masilukast', 'masitinib', 'matuzumab', 'mavacamten', 'mavacoxib', 'mavatrep', 'mavoglurant', 'mavrilimumab', 'maxacalcitol', 'mazapertine', 'mazindol', 'mazipredone', 'mazokalim', 'mebeverine', 'mebolazine', 'mebutamate', 'mebutizide', 'mecapegfilgrastim', 'mecasermin rinfabate', 'mecillinam', 'meclinertant', 'meclocycline', 'mecloqualone', 'meclorisone', 'mecobalamin', 'medazepam', 'medifoxamine', 'medorinone', 'medorubicin', 'medrogestone', 'medrylamine', 'mefenorex', 'mefeserpine', 'mefruside', 'megalomicin', 'meglucycline', 'melagatran', 'meldonium', 'melengestrol', 'melevodopa', 'melogliptin', 'melphalan flufenamide', 'melquinast', 'meluadrine', 'menfegol', 'menogaril', 'meobentine', 'mephenytoin', 'mepitiostane', 'mepolizumab', 'mepramidil', 'meproscillarin', 'meprylcaine', 'meradimate', 'meralein sodium', 'meralluride', 'merbromin', 'mercaptomerin', 'mercumatilin sodium', 'mercurophylline', 'merestinib', 'mergocriptine', 'meribendan', 'mericitabine', 'merimepodib', 'merisoprol (197 Hg)', 'merotocin', 'mesabolone', 'mesmulogene ancovacivec', 'mesocarb', 'mespiperone (11C)', 'mestanolone', 'mesterolone', 'mestranol', 'mesulergine', 'mesulfamide', 'mesulfen', 'mesuprine', 'mesuximide', 'metaglycodol', 'metahexamide', 'metamfetamine', 'metaraminol', 'metaterol', 'metazocine', 'metelimumab', 'metenkefalin', 'metenolone', 'metergoline', 'metergotamine', 'metescufylline', 'metesculetol', 'metesind', 'metformin glycinate', 'methandriol', 'methaqualone', 'metharbital', 'methazolamide', 'methdilazine', 'methiodal sodium', 'methoprene', 'methyldesorphine', 'methyldihydromorphine', 'methylergometrine', 'methylnaltrexone bromide', 'methylphenobarbital', 'methylrosanilinium chloride', 'methylsamidorphan chloride', 'methylthiouracil', 'methyprylon', 'methysergide', 'metipirox', 'metipranolol', 'metkefamide', 'metogest', 'metopon', 'metoquizine', 'metralindole', 'metribolone', 'metynodiol', 'mexafylline', 'mexrenoate potassium', 'mezacopride', 'mezilamine', 'mibampator', 'mibefradil', 'mibenratide', 'mibolerone', 'micafungin', 'midafotel', 'midaglizole', 'midaxifylline', 'midazolam', 'midomafetamine', 'mifamurtide', 'mifepristone', 'migalastat', 'miglustat', 'milacainide', 'milademetan', 'milameline', 'milataxel', 'milatuzumab', 'milciclib', 'milfasartan', 'milipertine', 'milodistim', 'milveterol', 'mimopezil', 'minalrestat', 'mindodilol', 'mindoperone', 'minepentate', 'minesapride', 'minodronic acid', 'minolteparin sodium', 'minopafant', 'minretumomab', 'mipeginterferon alfa-2b', 'mipitroban', 'mipomersen', 'miproxifene', 'mipsagargin', 'miralimogene ensolisbac', 'miransertib', 'miravirsen', 'miridesap ', 'mirikizumab', 'miriplatin', 'mirisetron', 'mirococept', 'mirodenafil', 'mirogabalin', 'mirostipen', 'mirvetuximab', 'mirvetuximab soravtansine', 'mitapivat', 'mitemcinal', 'mitiglinide', 'mitomalcin', 'mitomycin', 'mitoquidone', 'mitratapide', 'mitumomab', 'mivebresib ', 'mivobulin', 'mivotilate', 'mizagliflozin', 'mobenakin', 'mocetinostat', 'mocimycin', 'mocravimod', 'modaline', 'modimelanotide', 'modithromycin', 'modotuximab', 'mofarotene', 'mofebutazone', 'molibresib', 'molidustat', 'momelotinib', 'monalizumab', 'monepantel', 'mongersen', 'monoethanolamine oleate', 'monophosphothiamine', 'monteplase', 'moprolol', 'moracizine', 'moroctocog alfa', 'morolimumab', 'morpheridine', 'morphine glucuronide', 'mosunetuzumab', 'motavizumab', 'motesanib', 'motexafin', 'motolimod', 'moxestrol', 'moxetumomab pasudotox', 'moxilubant', 'moxisylyte', 'mozavaptan', 'mozenavir', 'mubritinib', 'muplestim', 'murabutide', 'muraglitazar', 'mureletecan', 'murepavadin', 'murodermin', 'mycophenolic acid', 'myrophine', 'nacartocin', 'nacolomab tafenatox', 'nacubactam ', 'nadofaragene firadenovec', 'nadorameran', 'nadroparin calcium', 'nafamostat', 'nafarelin', 'nafetolol', 'nafithromycin', 'naflocort', 'naftalofos', 'naftazone', 'nagrestipen', 'nalfurafine', 'nalmefene', 'nalmexone', 'nalorphine', 'naltalimide', 'naluzotan', 'namilumab', 'naminidil', 'namitecan', 'namodenoson', 'namoxyrate', 'nangibotide', 'nanterinone', 'napabucasin', 'napitane', 'naproxcinod', 'napsagatran', 'naptumomab estafenatox', 'naquotinib ', 'naratuximab', 'naratuximab emtansine', 'narlaprevir', 'narnatumab', 'naronapride', 'nasaruplase beta', 'nastorazepide', 'nateglinide', 'nateplase', 'navamepent', 'navarixin', 'naveglitazar', 'navicixizumab', 'navitoclax', 'navivumab', 'navoximod ', 'navuridine', 'naxifylline', 'nazartinib', 'nebentan', 'nebicapone', 'neboglamine', 'nebotermin', 'necuparanib', 'neflamapimod', 'neladenoson bialanate', 'nelarabine', 'nelatimotide ', 'nelezaprine', 'nelivaptan', 'nelociguat', 'nelonicline', 'nelotanserin', 'nemadectin', 'nemifitide', 'nemiralisib', 'nemolizumab', 'nemonoxacin', 'nemorubicin', 'nepadutant', 'nepafenac', 'nepaprazole', 'nepicastat', 'nepidermin', 'neramexane', 'nerelimomab', 'nerispirdine', 'nesbuvir', 'nesiritide', 'nesvacumab', 'netazepide', 'netilmicin', 'netivudine', 'netoglitazone', 'neutramycin', 'nexeridine', 'nicanartine', 'nicocodine', 'nicocortonide', 'nicodicodine', 'nicodicosapent', 'nicomorphine', 'nicoracetam', 'nicotredole', 'nifekalant', 'nifungin', 'nimetazepam', 'nimorazole', 'nimotuzumab', 'niperotidine', 'niprofazone', 'niraxostat', 'nirogacestat ', 'nitisinone', 'nitraquazone', 'nitrazepam', 'nitricholine perchlorate', 'nivimedone', 'nivocasan', 'nobiprostolan', 'nofecainide', 'nogalamycin', 'nolasiban', 'nolatrexed', 'nolomirole', 'nolpitantium besilate', 'nonacog alfa', 'nonacog beta pegol', 'nonacog gamma', 'nonathymulin', 'nonoxinol', 'noracymethadol', 'norclostebol', 'norcodeine', 'nordazepam', 'nordinone', 'norgesterone', 'norgestomet', 'norgestrienone', 'norletimol', 'norleusactide', 'norlevorphanol', 'normethadone', 'normorphine', 'norpipanone', 'nortopixantrone', 'norvinisterone', 'nosantine', 'nupafant', 'nusinersen', 'nuvenzepine', 'obatoclax', 'obenoxazine', 'oberadilol', 'obeticholic acid', 'obicetrapib ', 'obiltoxaximab', 'obinepitide', 'oblimersen', 'ocaratuzumab', 'ocinaplon', 'oclacitinib', 'ocriplasmin', 'octinoxate', 'octisalate', 'octocog alfa', 'octopamine', 'octotiamine', 'octoxinol', 'odalasvir', 'odanacatib', 'odulimomab', 'ofranergene obadenovec ', 'oftasceine', 'oglemilast', 'oglufanide', 'olamkicept', 'olamufloxacin', 'olanexidine', 'olaptesed pegol', 'olcegepant', 'olcorolimus', 'oleclumab', 'olendalizumab', 'olesoxime', 'oletimol', 'oliceridine', 'olinciguat', 'olipudase alfa', 'olmutinib', 'olodanrigan', 'olokizumab', 'olopatadine', 'olorofim', 'olpadronic acid', 'olprinone', 'olumacostat glasaretil', 'omacetaxine mepesuccinate', 'omaciclovir', 'omalizumab', 'omapatrilat', 'omarigliptin', 'omaveloxolone', 'omberacetam', 'ombrabulin', 'omecamtiv mecarbil', 'omidenepag', 'omigapil', 'omiloxetine', 'omipalisib', 'omocianine', 'omoconazole', 'ompinamer', 'omtriptolide', 'onalespib', 'onapristone', 'onartuzumab', 'onasemnogene abeparvovec', 'ondelopran', 'onercept', 'ontazolast', 'ontuxizumab', 'opaganib', 'opanixil', 'opaviraline', 'opebacan', 'opicapone', 'opicinumab', 'opiranserin', 'opolimogene capmilisbac', 'oportuzumab monatox', 'opratonium iodide', 'oprelvekin', 'oprozomib', 'orantinib', 'orazipone', 'orbofiban', 'orciprenaline', 'ordopidine', 'oregovomab', 'oreptacog alfa (activated)', 'orgotein', 'orientiparcin', 'orilotimod', 'orlistat', 'ornipressin', 'ortataxel', 'orteronel', 'orticumab', 'orvepitant', 'osanetant', 'osemozotan', 'osilodrostat', 'osutidine', 'otamixaban', 'otelixizumab', 'otenabant', 'otenzepad', 'oteracil', 'oteseconazole', 'otlertuzumab', 'ovemotide', 'oxabolone cipionate', 'oxabrexine', 'oxaliplatin', 'oxatomide', 'oxazolam', 'oxeclosporin', 'oxeglitazar', 'oxeladin', 'oxelumab', 'oxetacillin', 'oxifentorex', 'oxilofrine', 'oximonam', 'oxiperomide', 'oxipurinol', 'oxitefonium bromide', 'oxitropium bromide', 'oxogestone', 'oxprenolol', 'ozanezumab', 'ozanimod', 'ozarelix', 'ozenoxacin', 'ozogamicin', 'ozoralizumab', 'paclitaxel', 'paclitaxel ceribate', 'paclitaxel poliglumex', 'paclitaxel trevatide', 'pacrinolol', 'pacritinib', 'padeliporfin', 'padoporfin', 'padsevonil ', 'pafenolol', 'pafuramidine', 'pagibaximab', 'pagoclone', 'palifermin', 'paliflutine', 'palifosfamide', 'palinavir', 'paliroden', 'palivizumab', 'palosuran', 'palovarotene', 'palucorcel', 'pamapimod', 'pamaqueside', 'pamicogrel', 'pamiparib', 'pamiteplase', 'pamrevlumab', 'panamesine', 'pancopride', 'panobacumab', 'panomifene', 'panthenol', 'panulisib', 'paquinimod', 'pararosaniline embonate', 'parathyroid hormone', 'pardoprunox', 'parecoxib', 'parethoxycaine', 'pargeverine', 'pargolol', 'paritaprevir', 'parnaparin sodium', 'parogrelil', 'parsaclisib', 'parsatuzumab', 'pascolizumab', 'pasireotide', 'pasotuxizumab', 'pateclizumab', 'patiromer calcium', 'patisiran', 'patritumab', 'patupilone', 'paulomycin', 'paxamate', 'pazinaclone', 'pazufloxacin', 'pefcalcitol', 'peficitinib', 'peforelin', 'pegacaristim', 'pegamotecan', 'pegapamodutide', 'pegaptanib', 'pegargiminase', 'pegbovigrastim', 'pegcantratinib', 'pegdarbepoetin beta', 'pegdinetanib', 'pegfilgrastim', 'pegilodecakin', 'peginesatide', 'peginterferon alfa-2a', 'peginterferon alfa-2b', 'peginterferon alfacon-2', 'peginterferon beta-1a', 'peginterferon lambda-1a', 'pegloticase', 'pegmusirudin', 'pegnartograstim', 'pegnivacogin', 'pegorgotein', 'pegpleranib', 'pegrisantaspase', 'pegsiticase', 'pegsunercept', 'pegteograstim', 'pegunigalsidase alfa ', 'pegvaliase', 'pegvisomant', 'pegvorhyaluronidase alfa ', 'pegzilarginase', 'peldesine', 'peliglitazar', 'pelitinib', 'pelitrexol', 'pelubiprofen', 'pemafibrate', 'pemaglitazar', 'pemlimogene merolisbac', 'pemoline', 'penbutolol', 'penciclovir', 'pentamorphone', 'pentobarbital', 'pentoxyverine', 'peplomycin', 'perakizumab', 'perampanel', 'peretinoin', 'perflenapent', 'perflexane', 'perflisobutane', 'perflisopent', 'perflubrodec', 'perflubutane', 'perflutren', 'perifosine', 'perospirone', 'perzinfotel', 'petesicatib', 'pethidine', 'petrichloral', 'pevonedistat', 'pexacerfont', 'pexastimogene devacirepvec', 'pexelizumab', 'pexiganan', 'pexmetinib', 'phenadoxone', 'phenampromide', 'phenazocine', 'phencyclidine', 'phendimetrazine', 'phenmetrazine', 'phenobarbital', 'phenobarbital sodium', 'phenomorphan', 'phenoperidine', 'phentermine', 'phenythilone', 'pholcodine', 'piboserod', 'pibrentasvir', 'pibrozelesin', 'pibutidine', 'piclamilast', 'piclidenoson', 'piclozotan', 'picoplatin', 'pictilisib', 'pidilizumab', 'pifonakin', 'pilaralisib', 'pimasertib', 'pimilprost', 'piminodine', 'pimodivir ', 'pinatuzumab vedotin', 'pinazepam', 'pinokalant', 'pinometostat', 'pipendoxifene', 'piragliatin', 'pirarubicin', 'pirazmonam', 'piridoxilate', 'piritramide', 'piromelatine', 'pitolisant ', 'pitrakinra', 'pivmecillinam', 'pixatimod', 'placulumab', 'pleconaril', 'plerixafor', 'pleuromulin', 'plevitrexed', 'plinabulin', 'plitidepsin', 'plocabulin', 'plozalizumab', 'plusonermin', 'pobilukast', 'pocapavir', 'podilfen', 'polaprezinc', 'polatuzumab vedotin', 'policapram', 'policresulen', 'polidexide sulfate', 'poligeenan', 'poliglecaprone', 'polixetonium chloride', 'polmacoxib', 'poloxalene', 'polysorbate 20', 'polysorbate 21', 'polysorbate 40', 'polysorbate 60', 'polysorbate 61', 'polysorbate 65', 'polysorbate 80', 'polysorbate 81', 'polysorbate 85', 'pomaglumetad methionil', 'pomisartan', 'ponazuril', 'ponesimod', 'ponezumab', 'porgaviximab', 'posaconazole', 'posaraprost', 'posatirelin', 'poseltinib ', 'posizolid', 'poskine', 'pozanicline', 'poziotinib', 'pracinostat', 'pradefovir', 'pradigastat', 'pradimotide', 'pradofloxacin', 'pralatrexate', 'praliciguat', 'pralmorelin', 'pralnacasan', 'pramiconazole', 'pramiracetam', 'pramlintide', 'pranazepide', 'prasinezumab', 'prasugrel', 'pratosartan', 'prazarelix', 'prazepam', 'prednazoline', 'prednimustine', 'preladenant', 'premafloxacin', 'prenalterol', 'prenoxdiazine', 'presatovir', 'pretiadil', 'pretomanid', 'prexasertib', 'prexigebersen', 'prezalumab', 'pridopidine', 'priliximab', 'prinaberel', 'prinomastat', 'pritelivir', 'pritoxaximab', 'procinolol', 'proheptazine', 'promegestone', 'propafenone', 'propentofylline', 'properidine', 'propetandrol', 'propikacin', 'propiram', 'propisergide', 'propoxycaine', 'propyl docetrizoate', 'prorenoate potassium', 'prosultiamine', 'proterguride', 'prulifloxacin', 'pruvanserin', 'pseudoephedrine', 'psilocybine', 'pumafentrine', 'pumaprazole', 'pumosetrag', 'pyridoxine', 'pyrithyldione', 'pyronaridine', 'pyrovalerone', 'quadazocine', 'quarfloxin', 'quifenadine', 'quiflapon', 'quilizumab', 'quilostigmine', 'quilseconazole', 'quinagolide', 'quinazosin', 'quinbolone', 'quinelorane', 'quinezamide', 'quisinostat', 'quisultazine', 'quizartinib', 'rabacfosadine', 'rabeximod', 'rabusertib', 'racecadotril', 'racemethorphan', 'racemoramide', 'racemorphan', 'racotumomab', 'ractopamine', 'radafaxine', 'radalbuvir', 'radavirsen', 'radequinil', 'radezolid', 'radiprodil', 'radotermin', 'radotinib', 'radretumab', 'rafabegron', 'rafigrelide', 'rafivirumab', 'ragaglitazar', 'ralaniten', 'ralimetinib', 'ralinepag', 'ralpancizumab', 'raltitrexed', 'ramatercept', 'ramatroban', 'ramciclane', 'ramixotidine', 'ramosetron', 'ranelic acid', 'ranevetmab ', 'ranibizumab', 'ranirestat', 'ranpirnase', 'rapacuronium bromide', 'rapastinel', 'raseglurant', 'rathyronine', 'ravidasvir', 'ravoxertinib ', 'ravuconazole', 'ravulizumab', 'raxibacumab', 'razaxaban', 'razupenem', 'razuprotafib', 'rebamipide', 'rebastinib', 'recainam', 'recanaclotide ', 'recilisib', 'recoflavone', 'redaporfin', 'redasemtide', 'refametinib', 'refanezumab', 'regadenoson', 'regavirumab', 'reglitazar', 'regrelor', 'relacatib', 'relacorilant', 'relamorelin', 'relcovaptan', 'relebactam', 'relenopride', 'relmapirazin', 'reloxaliase', 'reltecimod ', 'relugolix', 'remacemide', 'remdesivir', 'remeglurant', 'remetinostat ', 'remimazolam', 'remlarsen', 'remogliflozin etabonate', 'remtolumab ', 'renadirsen', 'renytoline', 'renzapride', 'reparixin', 'repifermin', 'repinotan', 'resatorvid', 'rescimetol', 'resiquimod', 'reslizumab', 'resminostat', 'resocortol', 'retapamulin', 'retaspimycin', 'retelliptine', 'reteplase', 'retigabine', 'retosiban', 'revamilast', 'revaprazan', 'revatropate', 'reveglucosidase alfa', 'revexepride', 'reviparin sodium', 'revusiran', 'rezafungin acetate', 'rezatomidine', 'riamilovir', 'ribaminol', 'ribaxamase', 'ribuvaptan', 'ricasetron', 'ricolinostat', 'ridaforolimus', 'ridinilazole', 'rifalazil', 'riferminogene pecaplasmid', 'rigosertib', 'rilapladib', 'rilimogene galvacirepvec', 'rilimogene glafolivec', 'rilonacept', 'rilotumumab', 'rimacalib', 'rimegepant', 'rimeporide', 'rimexolone', 'rimiducid', 'rimigorsen', 'rimonabant', 'rintatolimod', 'rinucumab', 'riociguat', 'ripasudil', 'ripisartan', 'risankizumab', 'risarestat', 'rislenemdaz', 'rismorelin', 'ritobegron', 'ritrosulfan', 'rivabazumab', 'rivabazumab pegol', 'rivanicline', 'rivaroxaban', 'rivenprost', 'riviciclib', 'rivipansel', 'rivoceranib', 'rivogenlecleucel', 'rivoglitazone', 'robalzotan', 'robatumumab', 'robenacoxib', 'rocepafant', 'rociletinib', 'rociverine', 'rodorubicin', 'rofecoxib', 'rofleponide', 'rogaratinib ', 'roledumab', 'rolicyclidine', 'rolicyprine', 'rolipoltide', 'rolofylline', 'rolziracetam', 'romidepsin', 'romiplostim', 'ronacaleret', 'roneparstat', 'roniciclib', 'ronopterin', 'rontalizumab', 'ropeginterferon alfa-2b', 'ropidoxuridine', 'rosabulin', 'rosiglitazone', 'rosiptor ', 'rosmantuzumab ', 'rosomidnar ', 'rosonabant', 'rostafuroxin', 'rostaporfin', 'rotamicillin', 'rotigaptide', 'rovalpituzumab', 'rovalpituzumab tesirine', 'rovatirelin', 'rovazolac', 'rovelizumab', 'roxadustat', 'roxibolone', 'roxifiban', 'roxindole', 'rozanolixizumab ', 'rubitecan', 'ruboxistaurin', 'ruclosporin', 'rufloxacin', 'rupatadine', 'rupintrivir', 'ruplizumab', 'rurioctocog alfa pegol', 'rusalatide', 'rutamycin', 'ruzadolane', 'ruzasvir', 'sabarubicin', 'sabcomeline', 'sabiporide', 'sacituzumab ', 'sacituzumab govitecan', 'sacrosidase', 'sacubitrilat', 'safotibant', 'sagopilone', 'salcaprozic acid', 'salclobuzic acid', 'salinomycin', 'salirasib', 'salnacedin', 'samalizumab', 'samarium (153Sm) lexidronam', 'samatasvir', 'samidorphan', 'samixogrel', 'sampatrilat', 'sampeginterferon beta-1a', 'sanfetrinem', 'sapacitabine', 'sapanisertib', 'sapitinib', 'saprisartan', 'saracatinib', 'sarakalim', 'sardomozide', 'saredutant', 'saridegib', 'sarilumab', 'sarizotan', 'saroglitazar', 'sarolaner', 'sarpogrelate', 'sarsagenin', 'saruplase', 'satavaptan', 'satoreotide ', 'satoreotide trizoxetan', 'satralizumab', 'satraplatin', 'satumomab', 'savolitinib', 'scopinast', 'sebelipase alfa', 'secbutabarbital', 'secobarbital', 'secretin human', 'secukinumab', 'securinine', 'sedecamycin', 'seglitide', 'seladelpar ', 'selamectin', 'selepressin', 'seletalisib', 'seletracetam', 'selexipag', 'seliciclib', 'selicrelumab', 'seliforant', 'selisistat', 'selodenoson', 'selonsertib', 'selprazine', 'seltorexant ', 'selumetinib', 'selurampanel', 'semagacestat', 'semaglutide', 'semapimod', 'semaxanib', 'sembragiline', 'semparatide', 'semuloparin sodium', 'senazodan', 'senicapoc', 'senrebotase', 'seocalcitol', 'sepantronium bromide', 'sepetaprost', 'sepranolone', 'seprilose', 'serabelisib ', 'seractide', 'seratrodast', 'serdemetan', 'serelaxin', 'sergliflozin etabonate', 'seribantumab', 'seridopidine', 'serlopitant', 'sermorelin', 'serum gonadotrophin', 'setileuton', 'setipafant', 'setipiprant', 'setmelanotide', 'setoxaximab', 'setrobuvir', 'setrusumab', 'seviteronel', 'sevitropium mesilate', 'sevuparin sodium', 'siagoside', 'sibenadet', 'sibrafiban', 'sibrotuzumab', 'sifalimumab', 'silmitasertib', 'silperisone', 'siltuximab', 'simenepag', 'simeprevir', 'simeticone', 'simfibrate', 'simoctocog alfa', 'simotaxel', 'simtuzumab', 'sinapultide', 'sincalide', 'sinitrodil', 'sipatrigine', 'siplizumab', 'sipoglitazar', 'siramesine', 'sirtratumab', 'sirtratumab vedotin', 'sirukumab', 'sisapronil', 'sitafloxacin', 'sitalidone', 'sitamaquine', 'sitaxentan', 'sitimagene ceradenovec', 'sitofibrate', 'sitravatinib', 'sivelestat', 'sivifene', 'smilagenin', 'sobetirome', 'soblidotin', 'sodelglitazar', 'sodium borocaptate (10B)', 'sodium chromate (51 Cr)', 'sodium feredetate', 'sodium iopodate', 'sodium morrhuate', 'sodium stibocaptate', 'sodium tetradecyl sulfate', 'sodium timerfonate', 'sofigatran', 'sofinicline', 'sofituzumab vedotin', 'sofpironium bromide ', 'solabegron', 'solanezumab', 'solcitinib', 'solimastat', 'solithromycin', 'solitomab', 'solnatide', 'solpecainol', 'somagrebove', 'somalapor', 'somapacitan', 'somatorelin', 'somatrogon ', 'somatropin pegol', 'somavaratan', 'somenopor', 'soneclosan', 'sonedenoson', 'sonepiprazole', 'sonolisib', 'sontuzumab', 'soraprazan', 'sorbinicate', 'soretolide', 'sotagliflozin', 'sotatercept', 'sothrombomodulin alfa', 'sotirimod', 'sotrastaurin', 'sovaprevir', 'spanlecortemlocel', 'sparsentan', 'spartalizumab', 'spebrutinib', 'spirendolol', 'spirofylline', 'spiroglumide', 'spiroplatin', 'sprifermin', 'sprodiamide', 'squalamine', 'stacofylline', 'stamulumab', 'stannsoporfin', 'stenbolone', 'stibamine glucoside', 'succinobucol', 'sucralox', 'sudismase', 'sudoxicam', 'sufentanil', 'sufugolix', 'sugammadex', 'sulamserod', 'sulbactam', 'suleparoid sodium', 'sulesomab', 'sulfiram', 'sulfogaiacol', 'sultimotide alfa', 'sultroponium', 'sumanirole', 'sunepitron', 'supidimide', 'suptavumab ', 'surinabant', 'surotomycin', 'susalimod', 'susoctocog alfa', 'sutezolid', 'suvizumab', 'suvratoxumab', 'tabalumab', 'tabelecleucel', 'taberminogene vadenovec', 'tabimorelin', 'tacapenem', 'tacedinaline', 'tadekinig alfa', 'tadocizumab', 'tafamidis', 'tafluposide', 'tafluprost', 'tafoxiparin sodium', 'tagorizine', 'talabostat', 'talacotuzumab', 'talactoferrin alfa', 'taladegib', 'talaglumetad', 'talampanel', 'talaporfin', 'talarozole', 'talibegron', 'taliglucerase alfa', 'talimogene laherparepvec', 'talinexomer', 'talmapimod', 'talnetant', 'talotrexin', 'talsaclidine', 'taltirelin', 'taltobulin', 'talviraline', 'tamibarotene', 'tamtuvetmab', 'tanaproget', 'tandutinib', 'taneptacogin alfa', 'tanespimycin', 'tanezumab', 'tanogitran', 'tanomastat', 'tanurmotide', 'tanzisertib', 'tapinarof', 'taplitumomab paptox', 'taprenepag', 'taprizosin', 'tarafenacin', 'taranabant', 'tarenflurbil', 'tarextumab', 'taribavirin', 'tariquidar', 'tarloxotinib bromide', 'tasadenoturev', 'taselisib', 'tasidotin', 'tasipimidine', 'tasisulam', 'tasonermin', 'tasosartan', 'taspoglutide', 'tasquinimod', 'tavaborole', 'tavilermide', 'tavolimab', 'tazarotene', 'tazemetostat', 'tazofelone', 'tazomeline', 'tebanicline', 'tebipenem pivoxil', 'tecadenoson', 'tecalcet', 'tecarfarin', 'tecastemizole', 'teceleukin', 'tecemotide', 'technetium (99m Tc) fanolesomab', 'technetium (99m Tc) furifosmin', 'technetium (99mTc) apcitide', 'technetium (99mTc) etarfolatide', 'technetium (99mTc) nofetumomab merpentan', 'technetium (99mTc) pintumomab', 'technetium (99mTc) trofolastat chloride', 'tecovirimat', 'tedalinab', 'tedatioxetine', 'tedisamil', 'teduglutide', 'tefibazumab', 'tefinostat', 'teglarinad chloride', 'teglicar', 'tegobuvir', 'tegoprazan', 'telacebec', 'telaprevir', 'telapristone', 'telatinib', 'telbermin', 'telbivudine', 'telcagepant', 'telinavir', 'telisotuzumab ', 'telisotuzumab vedotin ', 'telithromycin', 'telmapitant', 'teludipine', 'temanogrel', 'temavirsen', 'temefos', 'temiverine', 'temocaprilat', 'temoporfin', 'temsavir', 'temsirolimus', 'tenalisib', 'tenamfetamine', 'tenatoprazole', 'tenatumomab', 'tenecteplase', 'teneligliptin', 'teneliximab', 'tenifatecan', 'teniposide', 'tenivastatin', 'tenocyclidine', 'tenofovir alafenamide', 'tenofovir exalidex ', 'teplizumab', 'tepotinib', 'teprasiran', 'teprotumumab', 'terameprocol', 'terbogrel', 'terestigmine', 'terofenamate', 'tertomotide', 'terutroban', 'tesaglitazar', 'tesamorelin', 'tesetaxel', 'tesevatinib', 'tesidolumab', 'teslexivir', 'tesmilifene', 'tesofensine', 'testolactone', 'tetomilast', 'tetracosactide', 'tetrazepam', 'tetrodotoxin', 'teverelix', 'tezacaftor', 'tezacitabine', 'tezampanel', 'tezepelumab', 'tezosentan', 'thebacon', 'thiocolchicoside', 'thiomersal', 'thiopental sodium', 'threonine', 'thrombin alfa', 'thrombomodulin alfa', 'thymalfasin', 'thymocartin', 'thymotrinan', 'thyrotropin alfa', 'tiazotic acid', 'tibulizumab', 'ticagrelor', 'ticalopride', 'ticolubant', 'tideglusib', 'tidembersat', 'tifacogin', 'tifenazoxide', 'tifuvirtide', 'tigapotide', 'tigatuzumab', 'tigemonam', 'tigestol', 'tigilanol tiglate', 'tigloidine', 'tigolaner', 'tilapertin', 'tilarginine', 'tildipirosin', 'tildrakizumab', 'tilidine', 'tilivapram', 'tilmacoxib', 'tilnoprofen arbamel', 'tilorone', 'tilsotolimod', 'tiludronic acid', 'timapiprant', 'timcodar', 'timigutuzumab', 'timolumab', 'timrepigene emparvovec', 'tinostamustine', 'tinzaparin sodium', 'tiomolibdic acid', 'tipapkinogene sovacivec', 'tipelukast', 'tipifarnib', 'tipiracil', 'tiplasinin', 'tiplimotide', 'tipredane', 'tiprelestat', 'tiprotimod', 'tirabrutinib ', 'tiragolumab', 'tirasemtiv', 'tirofiban', 'tirvalimogene teraplasmid', 'tisagenlecleucel', 'tislelizumab', 'tisocalcitate', 'tisotumab', 'tisotumab vedotin', 'tivanisiran', 'tivantinib', 'tiviciclovir', 'tivirapine', 'tivozanib', 'tixocortol', 'tobicillin', 'toborinone', 'tocamphyl', 'toceranib', 'tocilizumab', 'tocladesine', 'tocofenoxate', 'tofacitinib', 'tofimilast', 'tofogliflozin', 'tolafentrine', 'tolevamer', 'tolfamide', 'tolindate', 'toliprolol', 'tolonium chloride', 'toloxychlorinol', 'tomeglovir', 'tomicorat', 'tomoglumide', 'tomopenem', 'tomuzotuximab', 'tonabacase ', 'tonabersat', 'tonapofylline', 'tonogenconcel', 'topilutamide', 'topiroxostat', 'topixantrone', 'toprilidine', 'topsalysin', 'topterone', 'toralizumab', 'torapsel', 'torbafylline', 'torcetrapib', 'torcitabine', 'toreforant', 'toripristone', 'tosactide', 'tosagestin', 'tosatoxumab', 'tosedostat', 'tositumomab', 'tosufloxacin', 'totrombopag', 'tovetumab', 'tozadenant', 'tozasertib', 'tozuleristide ', 'trabedersen', 'trabodenoson', 'tradecamide', 'tradipitant', 'trafermin', 'tralesinidase alfa', 'tralokinumab', 'tramiprosate', 'transcrocetin', 'transferrin aldifitox', 'trastuzumab deruxtecan', 'trastuzumab duocarmazine ', 'trastuzumab emtansine', 'travoprost', 'traxoprodil', 'trebananib', 'trecetilide', 'trecovirsen', 'tregalizumab', 'trelagliptin', 'trelanserin', 'tremacamra', 'tremelimumab', 'trempamotide', 'trengestone', 'trenonacog alfa', 'treosulfan', 'treprostinil', 'tresperimus', 'trestolone', 'tretazicar', 'trevogrumab', 'triampyzine', 'trichlormethiazide', 'triclonide', 'tricosactide', 'tridecactide', 'tridolgosir', 'trifarotene', 'trifosmin', 'trilaciclib', 'trilostane', 'trimebutine', 'trimedoxime bromide', 'trimeperidine', 'triparanol', 'triplatin tetranitrate', 'triptorelin', 'trodusquemine', 'trofinetide', 'tromantadine', 'tropabazate', 'tropantiol', 'tropapride', 'tropifexor', 'tropigline', 'tropirine', 'tropisetron', 'troplasminogen alfa', 'tropodifene', 'trospium chloride', 'trovafloxacin', 'trovirdine', 'troxacitabine', 'tryptophan', 'tucidinostat ', 'tucotuzumab celmoleukin', 'tulathromycin', 'tulinercept', 'tulrampator', 'turoctocog alfa', 'turoctocog alfa pegol', 'turofexorate isopropyl', 'tuvatidine', 'tyloxapol', 'tylvalosin', 'tyromedan', 'ubenimex', 'ublituximab', 'ubrogepant', 'udenafil', 'ularitide', 'ulifloxacin', 'ulimorelin', 'ulixertinib', 'ulobetasol', 'ulocuplumab', 'ulodesine', 'umeclidinium bromide', 'umifenovir', 'umirolimus', 'upadacitinib ', 'upamostat', 'upenazime', 'upidosin', 'uprifosbuvir ', 'uprosertib', 'urefibrate', 'urelumab', 'uridine triacetate', 'urokinase alfa', 'ursodeoxycholic acid', 'urtoxazumab', 'usistapide', 'ustekinumab', 'utomilumab ', 'vabicaserin', 'vaborbactam', 'vactosertib', 'vadacabtagene leraleucel', 'vadadustat', 'vadastuximab', 'vadastuximab talirine', 'vadimezan', 'valategrast', 'valdecoxib', 'valnemulin', 'valnivudine ', 'valoctocogene roxaparvovec', 'valomaciclovir', 'valopicitabine', 'valrocemide', 'valrubicin', 'valspodar', 'valtorcitabine', 'valziflocept', 'vamorolone ', 'vandefitemcel', 'vandortuzumad vedotin', 'vangatalcite', 'vaniprevir', 'vantictumab', 'vanucizumab', 'vanutide cridificar', 'vapaliximab', 'vapendavir', 'vapitadine', 'vapreotide', 'varespladib', 'varfollitropin alfa', 'varisacumab', 'varlilumab', 'varlitinib', 'varodarsen', 'vasopressin injection', 'vatalanib', 'vatanidipine', 'vatelizumab', 'vatinoxan', 'vatiquinone', 'vatreptacog alfa (activated)', 'vecabrutinib', 'vedaclidine', 'vedaprofen', 'vedolizumab', 'vedroprevir', 'velafermin', 'velagliflozin ', 'velaglucerase alfa', 'velcalcetide', 'veldoreotide', 'veliflapon', 'velimogene aliplasmid', 'veliparib', 'velmanase alfa', 'velneperit', 'veltuzumab', 'velusetrag', 'venglustat', 'vepalimomab', 'veradoline', 'vercirnon', 'verdinexor', 'verdiperstat', 'vericiguat', 'verinurad', 'vernakalant', 'verosudil', 'verpasep caltespen', 'versetamide', 'verteporfin', 'verubecestat', 'verubulin', 'verucerfont', 'vesatolimod', 'vesencumab', 'vesnarinone', 'vestipitant', 'vestronidase alfa ', 'vibegron', 'vicriviroc', 'vidofludimus', 'vidupiprant', 'vilanterol', 'vilaprisan', 'vildagliptin', 'vinflunine', 'vinmegallate', 'vintafolide', 'vinylbital', 'vipadenant', 'viqualine', 'viquidacin', 'visilizumab', 'vistusertib', 'vixotrigine', 'vobarilizumab', 'vocimagene amiretrorepvec', 'voclosporin', 'vofopitant', 'volanesorsen', 'volasertib', 'volinaserin', 'volixibat', 'volociximab', 'volpristin', 'vonapanitase', 'vonicog alfa', 'vonlerolizumab', 'vonoprazan', 'vorapaxar', 'vorasidenib', 'voretigene neparvovec ', 'vorhyaluronidase alfa', 'voriconazole', 'vorolanib ', 'vorsetuzumab', 'vorsetuzumab mafodotin', 'voruciclib', 'vosaroxin', 'vosoritide', 'votrisiran', 'votucalis', 'votumumab', 'voxelotor', 'voxergolide', 'voxilaprevir', 'voxtalisib', 'vunakizumab ', 'xaliproden', 'xanomeline', 'xantofyl palmitate', 'xemilofiban', 'xenon (133 XE)', 'xentuzumab', 'xenysalate', 'xibenolol', 'xidecaflur', 'ximelagatran', 'xipranolol', 'xyloxemine', 'yttrium (90Y) clivatuzumab tetraxetan', 'yttrium (90Y) tacatuzumab tetraxetan', 'zabofloxacin', 'zalutumumab', 'zamicastat', 'zanapezil', 'zankiren', 'zanolimumab', 'zanubrutinib', 'zastumotide', 'zaurategrast', 'zelandopam', 'zeniplatin', 'zenocutuzumab', 'zibotentan', 'zibrofusidic acid', 'ziconotide', 'zicronapine', 'zidebactam', 'zifrosilone', 'zilantel', 'zilascorb (2H)', 'zilpaterol', 'zindotrine', 'zinostatin stimalamer', 'zipeprol', 'ziralimumab', 'zocainone', 'zolasartan', 'zolbetuximab', 'zoledronic acid', 'zoleprodolol', 'zoliflodacin', 'zonampanel', 'zoniporide', 'zoptarelin doxorubicin', 'zosuquidar', 'zotarolimus', 'zoticasone', 'zucapsaicin', 'zuretinol acetate', 'etoposide toniribate', 'evagenretcel', 'darvadstrocel', 'emiplacel', 'fexuprazan', 'abrezekimab', 'adalimumab beta', 'apadamtase alfa', 'apraglutide', 'arazasetron', 'belantamab', 'belantamab mafodotin', 'belvarafenib', 'bersanlimab', 'bifikafusp alfa', 'bizalimogene ralaplasmid', 'borofalan (10B)', 'bulevirtide', 'cedazuridine', 'cetrelimab', 'cevidoplenib', 'cibisatamab', 'ciforadenant', 'cilofexor', 'cligosiban', 'conteltinib', 'contezolid', 'cusatuzumab', 'dalcinonacog alfa', 'delolimogene mupadenorepvec', 'deutivacaftor', 'difamilast', 'diroleuton', 'domatinostat', 'edicotinib', 'efavaleukin alfa', 'efineptakin alfa', 'efinopegdutide', 'eftansomatropin alfa', 'elismetrep', 'enapotamab', 'enapotamab vedotin', 'enexasogaol', 'epaminurad', 'epeleuton', 'etidaligide', 'etigilimab', 'faricimab', 'fidanacogene elaparvovec', 'fimepinostat', 'firsocostat', 'flotetuzumab', 'gadopiclenol', 'ganaplacide', 'gefapixant', 'ibrexafungerp', 'imaprelimab', 'iscalimab', 'lanraplenib', 'lenabasum', 'lenvervimab', 'leronlimab', 'licogliflozin', 'lifileucel', 'linerixibat', 'linzagolix', 'livoletide', 'lotamilast', 'macozinone', 'mavelertinib', 'mavilimogene ralaplasmid', 'mavorixafor', 'mosedipimod', 'nalotimagene carmaleucel', 'peposertib', 'daridorexant', 'netakimab', 'nidufexor', 'onfekafusp alfa', 'onvatilimab', 'opigolix', 'opinercept', 'otaplimastat', 'parimifasor', 'pavinetant', 'pegcetacoplan', 'pemigatinib', 'praconase', 'ravagalimab', 'rebisufligene etisparvovec', 'revosimeline', 'risdiplam', 'roblitinib', 'romilkimab', 'samrotamab', 'samrotamab vedotin', 'satoreotide tetraxetan', 'seclidemstat', 'setafrastat', 'surufatinib', 'sutimlimab', 'tavokinogene telseplasmid', 'tebentafusp', 'tegavivint', 'telratolimod', 'tengonermin', 'tepilamide fumarate', 'tepoditamab', 'timbetasin', 'tomivosertib', 'trastuzumab beta', 'tricaprilin', 'umbralisib', 'upacicalcet', 'uproleselan', 'valanafusp alfa', 'valemetostat', 'mitazalimab', 'viltolarsen', 'vopratelimab', 'zilucoplan', 'abelacimab', 'abivertinib', 'adriforant', 'alteminostat', 'amelparib', 'amlivirsen', 'ampreloxetine', 'asalhydromorphone', 'aticaprant', 'avasopasem manganese', 'avoplacel', 'azelaprag', 'bamadutide', 'bempegaldesleukin', 'bevifimod', 'bintrafusp alfa', 'birtamimab', 'brilaroxazine', 'budigalimab', 'camsirubicin', 'cenupatide', 'ceralasertib', 'cimlanod', 'cintirorgon', 'coblopasvir', 'cotadutide', 'crovalimab', 'danicopan', 'dersimelagon', 'dilanubicel', 'dilpacimab', 'dostarlimab', 'durlobactam', 'eftozanermin alfa', 'eladocagene exuparvovec', 'elopultide', 'eluforsen', 'encequidar', 'ensifentrine', 'exicorilant', 'fosgemcitabine palabenamide', 'fosifloxuridine nafalbenamide', 'foslinanib', 'fosmanogepix', 'frovocimab', 'futibatinib', 'galicaftor', 'gancotamab', 'golexanolone', 'gosuranemab', 'hydromethylthionine', 'iadademstat', 'idecabtagene vicleucel', 'ilginatinib', 'iodine (131I) apamistamab', 'lenzumestrocel', 'leriglitazone', 'linrodostat', 'lisocabtagene maraleucel', 'marstacimab', 'masupirdine', 'miricorilant', 'mivavotinib', 'murlentamab', 'neluxicapone', 'nerinetide', 'nevanimibe', 'nirsevimab', 'nomacopan', 'obexelimab', 'odevixibat', 'olacaftor', 'olenasufligene relduparvovec', 'olinvacimab', 'olorinab', 'omburtamab', 'ontamalimab', 'orilanolimab', 'osocimab', 'otilimab', 'prademagene zamikeracel', 'prolgolimab', 'redipultide', 'relatlimab', 'reldesemtiv', 'reproxalap', 'resmetirom', 'ripretinib', 'rocacetrapib', 'rodatristat', 'rolinsatamab', 'rolinsatamab talirine', 'roluperidone', 'rovafovir etalafenamide', 'ruxotemitide', 'selatogrel', 'setogepram', 'sintilimab', 'siremadlin', 'soticlestat', 'spesolimab', 'tabituximab', 'tabituximab barzuxetan', 'tafasitamab', 'talditercept alfa', 'taniborbactam', 'tavapadon', 'telaglenastat', 'temelimab', 'teserpaturev', 'tildacerfont', 'tirbanibulin', 'tirzepatide', 'tofersen', 'toripalimab', 'umibecestat', 'vafidemstat', 'valecobulin', 'zampilima']
	if word.lower() in drugs:
		return True
	return False

print("Using " + str(len(articles)) + " articles for this knowledge graph...")


# ## 1.2 Edge collection: generating an edge from a sentence
# 
# Our edge collection algorithm finds connections between different biomedical entities by looking for keywords in the same sentences or an adjacent sentences.
# 
# Each edge (i.e. connection between two keywords) is assigned values for the following properties:
# 1. node1 and node2 (the two keywords)
# 2. A sentiment score based on Python's Natural Language Toolkit (NLTK) library. We don't actually use this score in this notebook. We used this score for a separate analysis.
# 3. The number of citations for the article from which the the keywords were selected.
# 4. The year of publication for the paper from which the keywords were selected, if applicable
# 5. A one-hot encoding of all the interesting words found in the sentence pair
# 6. A binary flag (i.e. a 0/1 variable) indicating whether at least one of the keywords was a drug
# 
# Scoring edges/sentences for efficacy valence using our novel algorithm comes later.

# In[ ]:


# #########################################################################################################################################################
# FUNCTIONS TO CALCULATE THE FEATURES OF AN EDGE
# #########################################################################################################################################################

# checks the sentence against the list of verbs and nouns manually identified as relevant to a sentence with "academic merit." A proxy for efficacy_valence
def getInterestingEmbedding(sentence, interestingWords):
	soup = BeautifulSoup(sentence, features="html.parser")
	text = soup.get_text()
	words = text.split(" ")
	found = { i : 0 for i in interestingWords }
	for w in words:
		if w.lower() in interestingWords:
			found[w.lower()] = 1
	embedding = []
	for k in sorted(found.keys()):
		embedding.append(found[k])
	return embedding

# pulls a cheesy NLTK sentiment analyzer score; might be interesting to
def non_bioSentiment(sentence):
	sid = SentimentIntensityAnalyzer()
	ss = sid.polarity_scores(sentence)
	return ss['compound']

# calculates an edge, and all its features, from a sentence-pair and a sourceNode (which must occur in the sentences to trigger edge generation)
def generateEdgesFromSentence(sourceNode, title, sentence, diseaseSynonyms, interestingWords, citationCounts, NERs, stopwords):
	sentence = sentence.lower()
	if sourceNode not in sentence:
		return []

	# split the sentence pair into the two basic sentences
	pieces = sentence.split("###")
	sentence1 = pieces[0]
	sentence2 = pieces[1]

	# pull all the keywords out of the sentences
	keywords = []
	for word in sentence.split(" "):
		word = word.rstrip().lower().replace('(','').replace(')','').replace(';','').replace(',','').replace('.','')
		if (word in NERs or word in diseaseSynonyms or drugLookup(word)) and word != '':
			keywords.append(word)

	# feed the sentence into the sentiment analysis to get the sentiment score
	sentiment = (non_bioSentiment(sentence1) + non_bioSentiment(sentence2)) * 1.0 / 2

	# look up the paper in the 30K dataset to get the year
	try:
		paper = (metadata[metadata['title'] == title]).iloc[0]
		year = paper['publish_time']
		if not str(year).isnumeric():
			year = 0

		# look up the number of citations this paper has had (will need to create this as a dict from the dataset)
		citations = citationCounts[paper['title']]
	except Exception as e:
		year = 0
		citations = -1

	# scan the sentence pair for any of the applicable verbs/context
	embedding = getInterestingEmbedding(sentence, interestingWords)

	# creates a list of edges from the sourceNode and keywords
	edges = []
	for key in keywords:
		isDrug = drugLookup(key) or drugLookup(sourceNode)

		if sourceNode.rstrip() == key.rstrip():
			continue

		if title != 'PolySearch':
			edge = {'node1':sourceNode.lower(), 'node2':key.lower(), 'paperUID':title, 'paperYear':year, 'context':embedding, 'paperCitationCount':citations, 'sentiment':sentiment, 'sentence':sentence, 'isDrug':isDrug}
			if len(sourceNode.rstrip()) > 2 and len(key.rstrip()) > 2:
				if key.rstrip().lower() not in stopwords and key.rstrip().lower()+'s' not in stopwords and key.rstrip().lower()[:-1] not in stopwords:
					edges.append(edge)
		else:
			edge = {'node1':sourceNode.lower(), 'node2':key.lower(), 'paperUID':title, 'paperYear':year, 'context':[1]*len(embedding), 'paperCitationCount':100, 'sentiment':1, 'sentence':sentence, 'isDrug':True}
			edges.append(edge)
			edge = {'node2':sourceNode.lower(), 'node1':key.lower(), 'paperUID':title, 'paperYear':year, 'context':[1]*len(embedding), 'paperCitationCount':100, 'sentiment':1, 'sentence':sentence, 'isDrug':True}
			edges.append(edge)

	return edges


# ## 1.3 Edge collection: parallelizing the code to collect edges
# 
# Because mining all sentences from this dataset and identifying keyword relationships is resource-intensive, our code enables parallelization across multiple CPUs. This reduces the runtime for edge generation using the full biorxiv_medrxiv dataset to under an hour.
# 
# The code below checks if the user wants to include papers abstracts for the edge mining.

# In[ ]:


# #########################################################################################################################################################
# HELPER FUNCTIONS TO PARALLELIZE EDGE GENERATION
# #########################################################################################################################################################

# flattens an arbitrarily-nested list of lists
def flatten(items, seqtypes=(list)):
	for i, x in enumerate(items):
		while i < len(items) and isinstance(items[i], seqtypes):
			items[i:i+1] = items[i]
	return items

# used to parallelize generate an edge from a sentence-pait
def process_helper(s, node2, article, diseaseSynonyms=diseaseSynonyms, interestingWords=interestingWords, citationCounts=citationCounts, NERs=NERs, stopwords=stopwords):
	return generateEdgesFromSentence(node2.lower(), article['metadata']['title'], s, diseaseSynonyms, interestingWords, citationCounts, NERs, stopwords)

# used to parallelize processing each article; for now, it looks at the body of the article, but could be made to also include the abstract (commented out)
def process_article(article, node2, diseaseSynonyms=deepcopy(diseaseSynonyms), interestingWords=deepcopy(interestingWords), citationCounts=deepcopy(citationCounts),
 NERs=deepcopy(NERs), stopwords=deepcopy(stopwords)):
	temp_level2 = []
	if ABSTRACT == True:
		try:
			sentences = article['abstract'][0]['text'].split('.')
			for s in sentences:
				#print(article)
				temp_level2.extend(process_helper(s, edge, article))
		except:
			print("\t error processing abstract")
	sections = article['body_text']
	for s in sections:
		sentences = s['text'].split('.')
		sentCount = 0
		while sentCount + 1 < len(sentences):
		#for s in sentences:
			s1 = sentences[sentCount]
			s2 = sentences[sentCount + 1]
			s = s1 + " ### " + s2
			temp_level2.extend(process_helper(s, node2, article))
			sentCount += 1
	return temp_level2


# ## 1.4 Edge collection: Running the code to collect all edges
# 
# Our edge collection has three steps:
# 1. Collect all edges from drugs and/or COVID-19 synonyms found in the PolySearch dataset articles
# 2. Collect all edges where a keyword in any of our dataset matches any keyword in (1) above, to generate "level2" edges
# 3. Repeat step (2) on all "level2" edges, to generate "level3" edges
# 
# These edges are then saved to a file, edges.csv, as a checkpoint. The code below will not run (to save time) if you chose to run a cached version of this notebook.

# In[ ]:


# #########################################################################################################################################################
# MAIN EDGE GENERATION CODE
# #########################################################################################################################################################

if RUNTYPE != "cached":
    edges = []

    # uses the PolySearch results as a seed; this can be replaced with another other edge generation tool
    drugs = []
    for drug in polysearch['hits']['drug']:
        print("processing " + str(len(drug)) + " drug articles for drug " + drug['ename'] + ":")
        drugs.append(drug['ename'].lower())
    NERs.extend(drugs)

    for drug in polysearch['hits']['drug']:
        print("processing " + str(len(drug)) + " drug articles for drug " + drug['ename'] + ":")
        for disease in diseaseSynonyms:
            edges.extend(generateEdgesFromSentence(drug['ename'].lower(), 'PolySearch', drug['ename'].lower() + " ### " + disease.lower(), diseaseSynonyms, interestingWords, citationCounts, NERs, stopwords))
    for e in edges:
        print(e)
        print("")
    print("finished "+ str(len(edges)) + " PolySearch edges, now doing level2...")

    def edge_helper(node2, articles=deepcopy(articles)):
        level_t = []
        print("in parallel " + str(node2))
        for article in list(articles.values()):
            level_t.extend(process_article(article, node2))
        return level_t

    # parallelize the code on multiple CPUs that will collect all the edges for the next level(s)
    num_cores = 14
    #num_cores = multiprocessing.cpu_count()
    node2s = []

    # will collect everything matching things that matched coronavirus and or the PolySearch drugs, like "lung" or "some other drug"
    level2 = []
    new_2 = []
    c = 0
    print("everything matching covid/PolySearch-drug: on " + str(len(edges)) + "edges: ")
    for edge in edges:
        print("at edge " + str(c) + " :" + edge['node2'] + ":")
        if len(edge['node2'].lower()) > 0 and edge['node2'].lower() not in node2s: # if we already saw this term, don't process it again across all the documents
            node2s.append(edge['node2'].lower())
            new_2.append(edge['node2'])
            print(edge['node2'])
            #print("\tprocessing " +str(len(list(articles.values()))) +  " articles...")
        c += 1

    print("parallel runs on " + str(len(new_2)))
    level2.extend(Parallel(n_jobs=14)(delayed(edge_helper)(node2) for node2 in new_2))
    level2 = flatten(level2)
    print(len(level2))


    # will collect the next level of edges from any term found in the search above
    print("everything matching items found on level2: on " + str(len(level2)) + " edges: ")
    level3 = []
    node3s = []
    new_3 = []
    for edge in level2:
        if len(edge['node2'].lower()) > 0 and edge['node2'].lower() not in node3s: # if we already saw this term, don't process it again across all the documents
            print("at level2 " + str(c) + " out of " + str(len(level2)) + ":" + edge['node2'] + ":")
            node3s.append(edge['node2'].lower())
            new_3.append(edge['node2'])
            print(edge['node2'])

    print("parallel runs on " + str(len(new_3)))
    level3.extend(Parallel(n_jobs=14)(delayed(edge_helper)(node2) for node2 in new_3))
    level3 = flatten(level3)
    print(len(level3))

    # save whatever data was fininshed (or recovered) to a csv that is separated by the # sign; we need to feed these sentences into a clearner, and then to the
    # efficacy_valence labeler
    file = open("edges.csv", "w")
    file.write("node1#node2#paperUID#paperYear#context#paperCitationCount#sentiment#sentence#utility#isDrug#level\n")
    for e in edges:
        line = e['node1'].replace('#', '')  + "#" + e['node2'].replace('#', '')  + "#" + e['paperUID'].replace('#', '')  + "#" + str(e['paperYear']) + "#" +             str(e['context']) + "#" + str(e['paperCitationCount']) + "#" + str(e['sentiment']) + "#" + e['sentence'].replace('#', '--') + "#0#" + str(e['isDrug']) + "#1\n"
        file.write(line)
    for e in level2:
        print(e['node2'])
        line = e['node1'].replace('#', '')  + "#" + e['node2'].replace('#', '')  + "#" + e['paperUID'].replace('#', '')  + "#" + str(e['paperYear']) + "#" +             str(e['context']) + "#" + str(e['paperCitationCount']) + "#" + str(e['sentiment']) + "#" + e['sentence'].replace('#', '--') + "#0#" + str(e['isDrug']) + "#2\n"
        file.write(line)
    for e in level3:
        print(e['node2'])
        line = e['node1'].replace('#', '')  + "#" + e['node2'].replace('#', '')  + "#" + e['paperUID'].replace('#', '')  + "#" + str(e['paperYear']) + "#" +             str(e['context']) + "#" + str(e['paperCitationCount']) + "#" + str(e['sentiment']) + "#" + e['sentence'].replace('#', '--') + "#0#" + str(e['isDrug']) + "#3\n"
        file.write(line)
    file.close()

    print("finished creating " + str(len(edges) + len(level2) + len(level3)) + " edges.")
else:
    print("using cached edges: warning, the dataset has probably been updated since this snapshot, so cache is likely out of date and on fewer articles")


# <a id="section2"></a>
# # 2. Edge conversion
# 
# Above, we collected raw edges between keywords in our dataset, and annotated these edges with various features such as the citation count of the paper the edge came from, and how many interesting words we saw in that sentence.
# 
# Now, we are ready to label each edge using our novel efficacy valence algorithm, which will be how we weight the edges in our final graph. To do so, we have to prepare the sentences in the edges to feed into a state-of-the-art NLP model we describe in more detail later.
# 
# In the cell below, we convert a sentence such that every keyword/NER is replaced with the string "VOID". We do this because of how we train our efficacy valence model: we want it to learn what sentences have more "academic merit" and "novelty" than others, but we don't want it to accidentially learn relationships about specific diseases, drugs, or other topics. Therefore, we attempt to cross out all these keywords and replace them with VOID instead, so our model can focus on the grammar and other words in the sentence. In theory, our efficacy model aims to be disease and treatment agnostic.
# 
# The code below is not parallelized, and can take a few hours to run on the full dataset. You can try skipping this step (on a small subset of data) and seeing how much not having VOID impacts the efficacy valence tool we use later.

# In[ ]:


# a simple function just to open a dataframe with # as the separator; we had trouble getting pandas to do this cleanly.
def getDataframe(file, columns):
    file = open(file)
    data = file.readlines()
    file.close()

    lists = []
    for d in data[1:]:
        list_row = []
        d = d[:-1].split("#")

        if len(d) == len(columns):
            lists.append(d)
        else:
            print("error processing line " + str(d))

    edges = pd.DataFrame(lists, columns=columns)
    edges['context'].apply(lambda x: np.array(eval(str(x))))
    edges['paperYear'].apply(lambda x: x if str(x).isnumeric() else 0)
    return edges

# replaces the keywords with VOID in the sentence-pair.
def void(sentence, mesh, ctr):
	ctr[0] += 1
	if ctr[0] % 1000 == 0:
		print(ctr[0])
	pieces = sentence.split(' ')
	result = []
	for p in pieces:
		clean = p.replace(',', '').replace('.', '').replace(';', '')
		if clean.lower() in mesh or drugLookup(clean.lower()):
			result.append("VOID")
		else:
			result.append(p)
	result = ' '.join(result)
	#print(result)
	return result

columns = ['node1', 'node2', 'paperUID', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 
           'sentence', 'utility', 'isDrug', 'level']

if RUNTYPE == "cached":
    df = pandas.read_csv(root_dir + "edges-cached/voided_edges_cached.csv")
else:
    df = getDataframe("edges.csv", columns)
    ctr = [0]
    print("Processing " + str(len(df)) + " edges...")
    df['sentence'] = df['sentence'].apply(lambda x: void(x, NERs, ctr))
    df.to_csv("voided_edges.csv")
    
print("finished preparing data for passing to BERT efficacy valence.")


# <a id="section3"></a>
# # 3. Edge labelling: Using BERT (state-of-the-art NLP tool) to predict the "efficacy valence" of sentences in academic articles
# 
# ## TLDR: The code below labels each sentence in our dataset with how likely it is to contain experimental results, as opposed to background or commonly-known facts.
# 
# ## Background
# 
# We would like to weight the nodes in our graph by how much "academic merit" and "novelty" exists in the sentence they came from. For example, in an academic paper, giving a background fact such as "The flu kills thousands of people a year." is less interesting for our purposes than an experimental results such as "Our study demonstrates that drug ABC was more effective at reducing symptoms than drug XYZ, but with fewer side effects." You can think of this as 'sentiment analysis' (like judging if a movie review is positive or negative), but for deciding how useful a sentence from an academic paper is for knowledge engineering.
# 
# We want to be able to use the underlying grammatical structure of sentences, the context of words, as well as the words themselves to build a predictive model that can label a sentence with a likelihood of containing a useful fact (so assign a 0 or 1). We call the concept of a useful academic fact or outcome as "efficacy valence."
# 
# ## Modeling Efficacy Valence by training a BERT model
# 
# BERT, which stands for Bidirectional Encoder Representations from Transformers, is a state-of-the-art NLP tool that can be used to generate rich embeddings of sentences, to be used in downstream NLP tasks: https://en.wikipedia.org/wiki/BERT_(language_model). You can use BERT to generate features to feed into a classifier, among many other tasks. In our case, we'll train BERT to generate features/word-embeddings that we can then feed into a binary classifier that labels the source sentence as useful (1) or not useful (0).
# 
# ## BERT code (below)
# 
# The code below uses an open source sentiment analysis BERT model that we fine-tuned for our dataset. The original code tried to predict the sentiment of movie reviews. We made minor changes to the code base, to fine-tune the pre-trained BERT model on our sentences and labels, rather than movie reviews.
# https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

# ## 3.1 Edge labelling: training the BERT model on our sentences
# 
# To train BERT, we needed sentences that were labelled by their human-judged utility; we then simply feed these sentences into a pre-trained BERT model (available for download), and fine-tune BERT on our particular task. We obtained ~1000 training sentences by manually going through example sentences from the PolySearch database for diseases we thought were similar to covid19, such as sars, mers, influenza, hiv, malaria, and others, and manually labelling them on a 1-4 scale ('negative', 'neutral', 'weak positive', and 'strong positive'). Some examples are below:
# * Strong positive: The results showed that VOID significantly induced VOID expression in a time- and dose-dependent manner in the VOID. 
# * Weak positive: The results showed that VOID could up-regulate VOID expression time- and dose- dependently in VOID.
# * Neutral: VOID has been persistent in the VOID VOID since 2012. 
# * Negative: The efficacy of VOID fortification against VOID is uncertain in VOID-endemic settings. 
# 
# For binary classification, we converted these scores into positive or not positive.
# 
# We used 900 out of the 1000 samples to train the model, reserving the rest for evaluation. On our evaluation set, our trained model had an F1-score of 94% and an AUC of 93% -- not bad at all given how little time we had! Not terribly surprsing either, given how powerful BERT is compared to other word-embeddings, and that we feel that the task of deciding efficacy valence isn't a particulary difficult one from an NLP perspective. Once we trained BERT on our labelled dataset, we could then label all the sentences we mined thusfar on this Covid19 dataset.
# 
# ### Training runtime
# 
# Training the BERT model below on a GPU takes less than 3 minutes. It also takes some time (around ten minutes) for python to install the libraries and resolve dependencies below.

# In[ ]:


# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the ssmallpecific language governing permissions and
# limitations under the License.

# Import the correct versions of tensorflow and cuda drivers to match with this code, and install bert.

if RUNTYPE != 'cached':
    get_ipython().system('apt install -y cuda-toolkit-10-0')
    get_ipython().system('pip install tensorflow==1.15')
    get_ipython().system('pip install tensorflow-gpu==1.15')
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import tensorflow as tf
    import tensorflow_hub as hub
    from datetime import datetime
    import numpy as np
    from tqdm import tqdm
    tqdm.pandas()

    get_ipython().system('pip install bert-tensorflow')
    import bert
    from bert import run_classifier
    from bert import optimization
    from bert import tokenization

    from tensorflow import keras
    import os
    import re

    # check to make sure that this notebook recognizes that we have a GPU
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    device_name = tf.test.gpu_device_name()
    if "GPU" not in device_name:
        print("GPU device not found")
    print('Found GPU at: {}'.format(device_name))
else:
    print("Running on cached version without GPU support; using cached version of BERT labels.")


# The code below trains BERT; you can just think of BERT as a module here for doing sentence classification. There are many online resources available for understanding BERT which are better than what I could try to fit in here.

# In[ ]:


#####################################################################################################################
#
# TRAIN the BERT model to predict efficacy valence of sentences
#
#####################################################################################################################

# Set the output directory for saving model file
# Optionally, set a GCP bucket location

if RUNTYPE != 'cached':

    OUTPUT_DIR = '.' #@param {type:"string"}

    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'
    label_list = [0,1] # we are doing binary classification

    '''# Code to train and evaluate the model on the training dataset (90/10 split)
    train = temp_df
    test = pd.read_csv(root_dir + "bio-sentiments/all_sentiments_bert.csv", sep=",")
    splitter = np.random.rand(len(test)) < 0.1
    test = test[splitter]
    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                       text_a = x[DATA_COLUMN], 
                                                                       text_b = None, 
                                                                       label = x[LABEL_COLUMN]), axis = 1)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    '''

    # Our kaggle-competition specific training dataset used to fine-tune our model (made public):
    train = pd.read_csv(root_dir + "bio-sentiments/all_sentiments_bert.csv")
    train = train.drop(columns=['Unnamed: 0'])
    train['polarity'] = train['polarity'].apply(lambda x: int(x))

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                       text_a = x[DATA_COLUMN], 
                                                                       text_b = None, 
                                                                       label = x[LABEL_COLUMN]), axis = 1)

    # This is a path to an uncased (all lowercase) version of BERT
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    # BERT has its own way to break up a sentence into word-embeddings, using its tokenizer
    def create_tokenizer_from_hub_module():
      with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
          vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])
      return bert.tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)

    tokenizer = create_tokenizer_from_hub_module()

    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 128

    # Convert our training data to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

    # Set up the BERT model
    def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
      bert_module = hub.Module(
          BERT_MODEL_HUB,
          trainable=True)
      bert_inputs = dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids)
      bert_outputs = bert_module(
          inputs=bert_inputs,
          signature="tokens",
          as_dict=True)

      # Use "pooled_output" for classification tasks on an entire sentence.
      # Use "sequence_outputs" for token-level output.
      output_layer = bert_outputs["pooled_output"]
      hidden_size = output_layer.shape[-1].value

      # Create our own layer to tune for politeness data.
      output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

      with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:
          (loss, predicted_labels, log_probs) = create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
          train_op = bert.optimization.create_optimizer(
              loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
          # Calculate evaluation metrics. 
          def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            f1_score = tf.contrib.metrics.f1_score(
                label_ids,
                predicted_labels)
            auc = tf.metrics.auc(
                label_ids,
                predicted_labels)
            recall = tf.metrics.recall(
                label_ids,
                predicted_labels)
            precision = tf.metrics.precision(
                label_ids,
                predicted_labels) 
            true_pos = tf.metrics.true_positives(
                label_ids,
                predicted_labels)
            true_neg = tf.metrics.true_negatives(
                label_ids,
                predicted_labels)   
            false_pos = tf.metrics.false_positives(
                label_ids,
                predicted_labels)  
            false_neg = tf.metrics.false_negatives(
                label_ids,
                predicted_labels)
            return {
                "eval_accuracy": accuracy,
                "f1_score": f1_score,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "true_negatives": true_neg,
                "false_positives": false_pos,
                "false_negatives": false_neg
            }
          eval_metrics = metric_fn(label_ids, predicted_labels)

          if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=loss,
              train_op=train_op)
          else:
              return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
          (predicted_labels, log_probs) = create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
          predictions = {
              'probabilities': log_probs,
              'labels': predicted_labels
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # Return the actual model function in the closure
      return model_fn

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    WARMUP_PROPORTION = 0.1
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    model_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)
else:
    print("skipped BERT training because cached version running")


# ## 3.2 Edge labelling: Running the trained BERT model on all the sentences in our edges to get efficacy valence labels
# 
# Once we have trained BERT, we can then use it to label our sentences. First, we break up the sentence pairs into individual sentences, as this is what BERT was originally trained on. Then, we generate labels for each of the sentences, and store these in our dataframe of edges.
# 
# ### Runtime
# BERT should be run with a GPU, and can take about an hour on the full dataset, and runs in a few minutes on the mini dataset. If you are running this notebook with the cached setting on, it will skip running BERT on the GPU, and will just load the pre-labelled sentences that we created with the same code offline.

# In[ ]:


#####################################################################################################################
#
# Run the trained BERT model to predict efficacy valence of sentences from our edges
#
#####################################################################################################################

if RUNTYPE != 'cached':

    # Our test data contains two senteces for each edge; we need to split these up using the separator below, and then record
    # a score for each of the two sentences in the dataframe
    separator = '------'
    def splitter(sentence, separator, index):
       if separator in sentence:
           return sentence.split(separator)[index]
       else:
           return sentence

    def getPrediction(in_sentences):
      labels = ["Negative", "Positive"]
      input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
      input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
      predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
      predictions = estimator.predict(predict_input_fn)
      return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

    def util_calc(x):
        if x == 'Negative': 
            return 0
        return 1

    # to avoid running BERT forever, find the unique senteces from all sentence pairs, and only label those
    test = pd.read_csv("voided_edges.csv") 
    test['polarity'] = test['utility']
    sentenceHash = {}
    sentences = test['sentence']

    sep="------"
    single_sentences = []
    for a in sentences:
        splitt = a.split(sep)
        if len(splitt) == 2: #there are a few outliers that had three, just ignore them
            a, b = splitt
            single_sentences.append(a)
            single_sentences.append(b)
    single_sentences = list(set(single_sentences))

    # prepare data for BERT of just single, unique sentences
    labels = [0 for s in single_sentences]
    single_test = pd.DataFrame(list(zip(single_sentences, labels)), columns=['sentence','polarity'])
    test_InputExamples = single_test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                           text_a = x[DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[LABEL_COLUMN]), axis = 1)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print("finished tokenizing unique sentences...")

    # pass the tokenized sentences to BERT, label them
    test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=True)
    preds = getPrediction(single_test['sentence'])
    preds = [util_calc(p[2]) for p in preds]

    def hash_helper(x):
        if x in sentenceHash.keys(): return sentenceHash[x]
        else:
            print(x)
            return 'ERROR'

    test.to_csv('worksaver.csv')

    # find the single sentences in the original sentence pairs, and assign them their two labels
    for i, s in enumerate(single_sentences):
        sentenceHash[s] = preds[i]
    test = test[test['sentence'].map(lambda x: len(x.split(sep)) != 3)]
    test['sentence0'] = test['sentence'].apply(lambda x: x.split(sep)[0])
    test['sentence1'] = test['sentence'].apply(lambda x: x.split(sep)[1])
    test['utility'] = test['sentence0'].apply(lambda x: hash_helper(x))
    test['utility2'] = test['sentence1'].apply(lambda x: hash_helper(x))
    test = test[test['utility'] != 'ERROR']
    test = test[test['utility2'] != 'ERROR']

    print(test.head())
    print(len(test))
    print((test.utility.value_counts()))
    print((test.utility2.value_counts()))
    test.to_csv("labled_one_million.csv")
    print("finished labelling all sentences with BERT.")
else:
    test = pandas.read_csv(root_dir + "labelled-cached/labelled_edges_cached.csv")
    test = test[test['utility'] != 'ERROR']
    test = test[test['utility2'] != 'ERROR']
    print("skipped running BERT on sentences because we're running the cached version.")


# <a id="section4"></a>
# # 4. Drawing the graph
# 
# Once we have all our edges labelled with the efficacy valence of the sentences that generated them, we can then build our knowledge graph using these scores as weights.
# 
# ## 4.1 Drawing the graph: Cleaning the raw edges
# 
# We first do some housekeeping around the edges dataframe, converting strings to numerics as needed, coalescing the disease synonyms into a single node (we only did this for covid19, but had we more time we should have done it for all drugs/keywords), and manually provide labels for the edges that came directly from PolySearch results, as we didn't create them using any source sentences (we could have in theory, but just took this shortcut given our time constraints).
# 
# If you have many edges (over a million), this can take around ten minutes.

# In[ ]:


import numpy as np
import pandas as pd
import networkx as nx
import ast
import matplotlib.pyplot as plt
import datetime
from random import randint
import copy
import os
#from tqdm import tqdm
#tqdm.pandas()
import sys
sys.executable

edges = test

diseaseSynonyms = ["Covid19", "Covid-19", "coronavirus", "SARS-coronavirus", "2019-nCoV", "SARS-CoV-2", 'COVID19', "Coronavirus", 'covid19', 'sars-cov-2', '2019-ncov',
	"sars-coronavirus", '2019ncov', 'ncov2019', 'ncov-2019', 'covid-19']
edges['node1'] = edges['node1'].apply(lambda x: "covid19" if x in diseaseSynonyms else x)
edges['node2'] = edges['node2'].apply(lambda x: "covid19" if x in diseaseSynonyms else x)

def isDrug(n1, n2, paperUID):
    return drugLookup(n1) or drugLookup(n2) or paperUID == 'PolySearch'

def cleanPoly(utility, paper):
    if paper == 'PolySearch':
        return 1
    return int(utility)

# convert the columns into numeric types as needed
edges['isDrug'] = edges[['node1', 'node2', 'paperUID']].apply(lambda x: isDrug(*x), axis=1)


edges['utility'] = edges[['utility', 'paperUID']].apply(lambda x: cleanPoly(*x), axis=1)
edges['utility2'] = edges[['utility2', 'paperUID']].apply(lambda x: cleanPoly(*x), axis=1)
edges['paperCitationCount'] = edges['paperCitationCount'].apply(lambda x: 0 if x == '-1' else int(x))
edges['context_sum'] = edges.context.apply(lambda x: sum(eval(x)))

# Below, we make some (admittedly minimal and ad-hoc) attempts to coalesce synonyms into single nodes.
edges['node1'] = edges['node1'].apply(lambda x: "receptor" if 'receptor' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "receptor" if 'receptor' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "lopinavir" if 'lopinavir' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "lopinavir" if 'lopinavir' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "hiv" if 'hiv' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "hiv" if 'hiv' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "cytokine" if 'cytokine' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "cytokine" if 'cytokine' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "china" if 'chinese' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "china" if 'chinese' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "trial" if 'trial' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "trial" if 'trial' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "interferon" if 'interferon' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "interferon" if 'interferon' in x else x)

edges.head()


# ## 4.2 Drawing the graph: creating fly-out edges for final nodes
# 
# We want the user to be able to drill down into each node in our graph, beyond just the most meaningful drugs and topics we find. Therefore, we will get additional edges where node1 matches the final node in our graph, and node2 is a drug and/or topic that occurs above some threshold in our entire dataset. 

# In[ ]:


# calculate the frequency for all node2 keywords
index = list(edges['node2'].value_counts().index)
counts = list(edges['node2'].value_counts())

ctr = 0
while ctr < len(index):
    print(str(counts[ctr]) + "\t" + index[ctr] + " "  )
    ctr += 1    
value_counts = {index[i]: counts[i] for i in range(len(index))}

# mine additional edges from a source node1, where the node2 values are a drug, and occur above some threshold
threshold_additional = 10
def getAdditionalEdges(node):
    mini1 = edges[edges['node2'] == node]
    mini1 = mini1[mini1['utility'] == 1]
    mini2 = edges[edges['node1'] == node]
    mini2 = mini2[mini2['utility'] == 1]
    result = set(list(mini1['node1']) + list(mini2['node2']))
    cleaned = []
    for r in result:
        if value_counts[r] > threshold_additional and drugLookup(r):
            cleaned.append(r)
    #print((cleaned))
    return cleaned

print(edges.isDrug.value_counts())
print(len(edges))


# ## 4.3 Drawing the graph: Generate the goal graph as a baseline
# 
# Recall, we used a reputable literature survey paper on Covid19 to generate a goal graph that we can compare our graph to. Below, we encode the drugs and concepts from that survey paper into a graph, and provide a function that can measure how many nodes two graphs have in common.
# 
# We commented out any nodes (drugs) below that never showed up in the dataset, as we shouldn't use them to measure the quality of our graphs (since it would be impossible for us to ever have seen them).

# In[ ]:


G_goal = nx.Graph()
G_goal.add_edge('remdesivir', 'covid19')
G_goal.add_edge('remdesivir', 'trials')
G_goal.add_edge('remdesivir', 'sars')
G_goal.add_edge('remdesivir', 'mers')
G_goal.add_edge('remdesivir', 'nausea')
G_goal.add_edge('remdesivir', 'vomiting')
G_goal.add_edge('remdesivir', 'transaminase')
G_goal.add_edge('remdesivir', 'renal')
G_goal.add_edge('chloroquine', 'hydroxychloroquine')
G_goal.add_edge('chloroquine', 'sars')
G_goal.add_edge('hydroxychloroquine', 'sars')
G_goal.add_edge('hydroxychloroquine', 'emergency')
G_goal.add_edge('chloroquine', 'emergency')
G_goal.add_edge('chloroquine', 'toxicity')
G_goal.add_edge('hydroxychloroquine', 'toxicity')
G_goal.add_edge('chloroquine', 'interactions')
G_goal.add_edge('hydroxychloroquine', 'interactions')
G_goal.add_edge('chloroquine', 'dosing')
G_goal.add_edge('hydroxychloroquine', 'dosing')
G_goal.add_edge('hydroxychloroquine', 'trials')
G_goal.add_edge('hydroxychloroquine', 'fever')
G_goal.add_edge('hydroxychloroquine', 'cough')
G_goal.add_edge('hydroxychloroquine', 'chest')
G_goal.add_edge('chloroquine', 'symptoms')
G_goal.add_edge('chloroquine', 'china')
G_goal.add_edge('hydroxychloroquine', 'azithromycin')
G_goal.add_edge('azithromycin', 'rna')
G_goal.add_edge('azithromycin', 'trials')
G_goal.add_edge('interleukin-6', 'covid19')
G_goal.add_edge('interleukin-6', 'anecdotal')
G_goal.add_edge('interleukin-6', 'covid19')
G_goal.add_edge('interleukin-6', 'tocilizumab')
#G_goal.add_edge('interleukin-6', 'sarilumab')
#G_goal.add_edge('interleukin-6', 'siltuximab')
G_goal.add_edge('tocilizumab', 'trials')
#G_goal.add_edge('sarilumab', 'trials')
#G_goal.add_edge('siltuximab', 'trials')
G_goal.add_edge('plasma', 'emergency')
G_goal.add_edge('plasma', 'covid19')
G_goal.add_edge('plasma', 'emergency')
G_goal.add_edge('plasma', 'oxygenation')
G_goal.add_edge('favipiravir', 'influenza')
G_goal.add_edge('favipiravir', 'trials')
G_goal.add_edge('favipiravir', 'covid19')
G_goal.add_edge('favipiravir', 'clearance')
G_goal.add_edge('favipiravir', 'trials')
G_goal.add_edge('lopinavir-ritonavir', 'sars')
G_goal.add_edge('lopinavir-ritonavir', 'hiv')
G_goal.add_edge('lopinavir-ritonavir', 'mers')
G_goal.add_edge('lopinavir-ritonavir', 'trials')

# measures how similar two graphs are based on shared nodes.
def compareGraphs(G_baseline, G_test):
    totalNodes = 0
    foundNodes = 0
    for n in G_baseline.nodes:
        if n in G_test.nodes:
            foundNodes += 1
        totalNodes += 1
    return foundNodes * 100.0 / totalNodes

f = plt.figure(figsize=(15,15))
nx.draw(G_goal,arrows=None, with_labels=True)


# ## 4.4 Drawing the graph: Pruning edges by weight
# 
# Our algorithm goes through all the edges, and calculates an aggregate weight for each unique node1-node2 pair it sees (order doesn't matter, we sort the node names alphabetically). Recall, the same edge may have been generated more than once if it appears multiple times in the articles.
# 
# We will build the graph using a weight of our choice:
# * "polarity": will take the average of all the efficacy valence scores across the same edge, using the max score of the sentence-pair
# * "polarity_sum" : sums, rather than average, all efficacy valence scores across the same edge, using the max score of the sentence-pair
# * "interestingWords" : sums up all the interesting words ever seen for this edge, including duplicates
# * "citations" : calculates the average citation count for this edge's sentences/papers, NOT weighted by publication year (you could change this)
# * "drug" : sums all the times at least one keyword in the edge was a drug
# * otherwise, it just sums the all the above features (in an unintelligent way; you could optimize this, we ran out of time and were satisfied with our results with "polarity_sum")
# 
# The code below implements these functions.

# In[ ]:


# calculates the weight the first time an edge is seen, using the choice specified by the user
def calcInitialWeight(interestingWords, paperCitationCount, paperYear, polarity, polarity2, drug, choice):
    if choice == 'interestingWords':
        return eval(interestingWords)
    elif choice == 'citations':
        return [int(paperCitationCount)]
    elif choice == 'polarity':
        return [max(int(polarity), int(polarity2))]
    elif choice == 'polarity_sum':
        return max(int(polarity), int(polarity2))
    elif choice == 'drug':
        if drug == True:
            return 1
        else:
            return 0
    else:
        drug = 1 if drug == True else 0
        return sum(eval(interestingWords)) + int(polarity) + int(paperCitationCount) + drug

# updates the respective edge weight
def updateWeight(weight, interestingWords, paperCitationCount, paperYear, polarity, polarity2, drug, choice):
    if choice == 'interestingWords':
        ctr = 0
        interestingWords = eval(interestingWords)
        while ctr < len(interestingWords):
            if interestingWords[ctr] == 1:
                weight[ctr] = 1
            ctr += 1
        return weight
    elif choice == 'citations':
        weight.append(int(paperCitationCount))
        return weight
    elif choice == 'polarity':
        weight.append(max(int(polarity), int(polarity2)))
        return weight
    elif choice == 'polarity_sum':
        weight += (max(int(polarity), int(polarity2)))
        return weight    
    elif choice == 'drug':
        if drug == True:
            return 1 + weight
        else:
            return weight
    else:
        drug = 1 if drug == True else 0
        return weight + sum(eval(interestingWords)) + int(polarity) + int(paperCitationCount) + drug

def sorted(x, y):
    if x < y:
        return x, y
    return y, x
    
#function to add edges, summing one-hot encoding of context using numpy arrays 
def edge_add(node1, node2, paperYear, context, paperCitationCount, sentiment, polarity, drug, polarity2, context_sum, G, choice):
    a, b = sorted(node1,node2)
    if type(paperYear) != type('string'):
        paperYear = "2020"

    if G.has_edge(a, b):
        old_weight = G[a][b]['weight']
        G[a][b]['weight'] = updateWeight(old_weight, context, paperCitationCount, paperYear, polarity, polarity2, drug, choice)
    else:
        G.add_edge(a, b, weight=calcInitialWeight(context, paperCitationCount, paperYear, polarity, polarity2, drug, choice))


# ## 4.5 Drawing the graph: generate the graph(s)
# 
# We can now generate one or more graphs using the pruned edges. The code below allows you to conduct a GridSearch on the potential weighting schemes and thresholds, finding the graph that has the highest overlap in nodes with the goal graph.
# 
# Currently, the code below is set to use the "polarity_sum" weighting scheme (efficacy valence) with a threshold of 20, but you could try the other weighting schemes and thresholds and find the best graph automatically.
# 
# At this point, we restrict this graph to only be built from edges that have at least one drug as a node, to limit its size, but you could play around with this!

# In[ ]:


# restrict this graph to just covid19 treatments
mini_edges = edges[edges['isDrug'] == True]
print('length of drug edges: ', len(mini_edges))

# removes all edges that do not meet the minimum weight requirements
def cleanGraphUsingThreshold(G, threshold, choice):
    for e in G1.edges:
        weights = G1.edges[e]['weight']
        # if we need to average the scores in a list (for some weight calculations)
        if choice == 'polarity' or choice == 'interestingWords' or choice == 'citations':
            mean = sum(weights) * 1.0 / len(weights)
            G1.edges[e]['weight'] = mean
            
        if G1.edges[e]['weight'] < threshold:
            G.remove_edge(e[0], e[1])

    # remove all nodes with no edges after cleaning
    nodes = list(G.nodes).copy()
    for n in nodes:
        if len(list(G.adj[n])) < 1:
            G.remove_node(n)

# set the weighting scheme and thresholds to build a graph, using the GridSearch below
choices = ['polarity_sum'] # you would normally provide a list here if you want to do GridSearch
if RUNTYPE == 'mini': # if we're just building a graph from two articles, there won't be much there, so keep weights low
    thresholds = [0]
else:
    thresholds = [20] # you would normally provide a list here if you want to do GridSearch
    
# use GridSearch to find the optimum thresholds and weighting scheme, compared to the goal graph you created earlier
for choice in choices:
    for thresh in thresholds:
        subset = mini_edges.copy(deep=True)
        G1 = nx.Graph()
        subset[['node1', 'node2', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 'utility', 'isDrug', 'utility2', 'context_sum']].apply(lambda x: edge_add(*x, G1, choice), axis = 1)
        cleanGraphUsingThreshold(G1, thresh, choice)
        print('score: ' + str(compareGraphs(G_goal, G1))[:3] + " choice: " + choice + " thresh: " + str(thresh))


# Let's take a look at our draft graph, knowing that we will still need to do some pruning:

# In[ ]:


# draws the first graph we made
f = plt.figure(figsize=(15,15))
nx.draw(G1,arrows=None, with_labels=True)


# ## 4.6 Drawing the graph: Cleaning the best graph
# 
# Given the little time we had for this project, we could have done a more formal and/or better job of identifying stopwords that should not be nodes, based on word frequency, parts of speech, etc. Here, we have a function that you can customize to remove any superfluous nodes and edges from the graph that you are confident aren't really meaningful.
# 
# We also remove any nodes that don't have any edges coming out of them after the pruning process.

# In[ ]:


clutter = ['membrane', 'rights', 'multiple', 'therapeutics', 'condition', 'population', 'screening', 'limit', 'growth',
          'intracellular', 'data', 'strains', 'therapeutic', 'time', 'nuclei', 'evaluation', 'work', 'transport', 'conductance',
          'immune', 'inhibition', 'degradation', 'antibodies', 'receptor', 'populations', 'mast', 'family', 'alleles', 'production',
          'synthesis', 'future', 'cost', 'low', 'kinetics', 'tablets', 'alpha', 'regulation', 'unknown', 'fraction', 'nature',
          'pathogenesis', 'selective', 'strain', 'short', 'expansion', 'play', 'observation', 'sensitivity', 'males', 'females',
          'injury', 'factor', 'transcription', 'aged', 'history', 'secretion', 'survival', 'plays', 'antibody', 'culture',
          'behavior', 'formation', 'recombination', 'ubiquitination', 'vertebrates', 'beta', 'injection', 'white', 'up-regulation',
          'heat', 'enzymes', 'memory', 'electron', 'transfer', 'eating', 'ions', 'transfection', 'conjugated', 'vertebrate', 'abl',
          'acid', 'sodium', 'calcium', 'cations', 'ant', 'ice', 'rain', 'air', 'potassium', 'hydrogen', 'acids', 'magnesium',
          'charge', 'zinc']

# remove any nodes manually that should have ideally be in our stoplist in the first place.
G2a = nx.Graph()
for edge in G1.edges:
    if edge[0] not in clutter and edge[1] not in clutter:
        G2a.add_edge(edge[0], edge[1], weight=1)
        
# also remove the nodes we see above that aren't connected to the main graph, for legibility
# depends on what you set your threshold to above
outliers = ['digoxin', 'vancomycin', 'stress', 'vagina', 'ethyl', 'prednisolone', 'kinase']
G2b = nx.Graph()
for edge in G2a.edges:
    if edge[0] not in outliers and edge[1] not in outliers:
        G2b.add_edge(edge[0], edge[1], weight=1)

# remove any nodes that no longer have any edges after pruning
G2 = copy.deepcopy(G2b)
for n in nx.isolates(G2b):
    G2.remove_node(n)
    
f = plt.figure(figsize=(15,15))
nx.draw(G2,arrows=None, with_labels=True)
print(G2.nodes)


# ## 4.7 Drawing the graph: Displaying the graph
# 
# To display our graph, our code will generate an image for each node that contains its fly-out graph; we create a directory to store these images that is separate from the cached directory.
# 
# Recall, the fly-out graphs for each drug show current and additional connections not in the graph above; we chose to use fly-outs on individual nodes to reduce the clutter in the previous graph, and to allow for focused attention on edges for a fly-out node that may be below the global threshold setting chosen earlier.
# 

# In[ ]:


# create/clean the directory (was too lazy to look up how to check if dir is there)
try:
    os.system("mkdir images_live")
except:
    os.system("rm -r images_live")
    

# call the code to generate fly-out nodes from every node in the main graph, all in a single graph (will be very dense!)
G3 = copy.deepcopy(G2)
for node in G2.nodes:
    for n in getAdditionalEdges(node):
        if str(node) not in clutter:
            G3.add_edge(node, n, weight=0)

# generate individual fly-out graphs that mimic what will take place when someone "clicks" an individual node
# most of the effort here is to customize the node and edge coloroing depending on what the selected node is
for seed_node in G2.nodes:
    G_specific = copy.deepcopy(G3)
    node_colors = {}
    labels = {}
    sizes = {}
    for node in G_specific.nodes:
        node_colors[node] = "aliceblue"
        labels[node] = ""
        sizes[node] = value_counts[node] / 10.0 + 100
    edge_colors = {}
    for edge in G_specific.edges:
        if edge[0] == seed_node or edge[1] == seed_node:
            node_colors[edge[0]] = "deepskyblue"
            node_colors[edge[1]] = "deepskyblue"
            labels[edge[0]] = str(edge[0])
            labels[edge[1]] = str(edge[1])
            edge_colors[edge] = "black"
            sizes[edge[0]] = value_counts[edge[0]] / 10.0 + 100
            sizes[edge[1]] = value_counts[edge[1]] / 10.0 + 100
            
    sizes['covid19'] = 400
    node_colors['covid19'] = "red"

    nodelist = []
    node_color = []
    node_size = []
    for k in node_colors.keys():
        nodelist.append(k)
        node_color.append(node_colors[k])
        node_size.append(sizes[k])
    edgelist = []
    edge_color = []
    for k in edge_colors.keys():
        edgelist.append(k)
        edge_color.append(edge_colors[k])
    f = plt.figure(figsize=(15,15))
    nx.draw_spring(G_specific,arrows=None, with_labels=True, ax=f.add_subplot(111), nodelist=nodelist,
                   edgelist=edgelist,node_size=node_size,node_color=node_color,node_shape='o', alpha=1.0,
                   cmap=None, vmin=None,vmax=None, linewidths=None, width=1.0, edge_color=edge_color,
                   edge_cmap=None, edge_vmin=None,edge_vmax=None, style='solid', labels=labels, font_size=12, 
                   font_color='black', font_weight='normal', font_family='sans-serif', label='COVID19 treatments/drugs')
    f.savefig("./images_live/" + str(seed_node) + ".png")


# colorize the core graph we made earlier and save it to the same directory as the other graphs
node_colors = {}
labels = {}
sizes = {}
edge_colors = {}
for edge in G2.edges:
    node_colors[edge[0]] = "deepskyblue"
    node_colors[edge[1]] = "deepskyblue"
    labels[edge[0]] = str(edge[0])
    labels[edge[1]] = str(edge[1])
    edge_colors[edge] = "black"
    sizes[edge[0]] = value_counts[edge[0]] / 10.0 + 100
    sizes[edge[1]] = value_counts[edge[1]] / 10.0 + 100
sizes['covid19'] = 400
node_colors['covid19'] = "red"
nodelist = []
node_color = []
node_size = []
for k in node_colors.keys():
    nodelist.append(k)
    node_color.append(node_colors[k])
    node_size.append(sizes[k])
edgelist = []
edge_color = []
for k in edge_colors.keys():
    edgelist.append(k)
    edge_color.append(edge_colors[k])
f = plt.figure(figsize=(15,15))
nx.draw_spring(G2,arrows=None, with_labels=True, ax=f.add_subplot(111), nodelist=nodelist,
                   edgelist=edgelist,node_size=node_size,node_color=node_color,node_shape='o', alpha=1.0,
                   cmap=None, vmin=None,vmax=None, linewidths=None, width=1.0, edge_color=edge_color,
                   edge_cmap=None, edge_vmin=None,edge_vmax=None, style='solid', labels=labels, font_size=12, 
                   font_color='black', font_weight='normal', font_family='sans-serif', label='COVID19 treatments/drugs')
f.savefig("./images_live/CORE_RESULTS.png")


# ## 4.8 Drawing the graph: Preparing a dataframe to select papers/sentences based on a selected node
# 
# This is meant as a placeholder for future work that can use automated text summarization and highlighting to coalesce the "search results" we show here
# 

# In[ ]:


columns = ['node1', 'node2', 'paperUID', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 
               'sentence', 'utility', 'isDrug', 'level']
raw_edges = edges.copy(deep=True)

raw_edges['utility1'] = edges['utility']
raw_edges['utility2'] = edges['utility2']

def max1(a, b):
    a = int(a)
    b = int(b)
    if a > b:
        return a
    return b

raw_edges['utility'] = raw_edges[['utility1', 'utility2']].apply(lambda x: max1(*x), axis=1)
    
def mineSentences(keyword, edges, utility):
    subset1 = edges[edges['node1'] == keyword]
    subset2 = edges[edges['node2'] == keyword]
    joined = pd.concat([subset1, subset2])
    joined = joined.reset_index()
    joined = joined.drop(columns=['sentence', 'sentence0', 'sentence1', 'utility1', 'utility2', 'index', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 'isDrug'])
    joined = joined[joined['utility'] >= utility]
    joined = joined.drop_duplicates()
    display(joined)
    #display(set(joined['paperUID']))


# ## 4.9 Drawing the graph: Drawing the interactive graph
# 
# We will use some widgets to allow for a drop-down menu to select either the CORE_RESULTS (default), or select one of the nodes in that graph and show its fly-out edges.

# In[ ]:


import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import Image, display, HTML

# choose which directory to load images from <--- I think this should be rewritten to be live, based on code above
if RUNTYPE != 'cached':
    files = os.listdir("images_live/")
    cleaned = []
    for f in files:
        cleaned.append(f.split(".png")[0])
    files = cleaned
    files.pop(files.index('CORE_RESULTS'))
    files = ['CORE_RESULTS'] + files
else:
    files = ['CORE_RESULTS'] + list(G2.nodes)
directory = widgets.Dropdown(options=files, description='node:')
images = widgets.Dropdown(options=files)

def update_images(*args):
    images.options = files

directory.observe(update_images, 'value')

def show_images(file):
    display(Image('./images_live/' + file + '.png'))

    drug = file
    @interact
    def show_articles_more_than(node=drug, utility=1):
        mineSentences(node, raw_edges, utility)
    
_ = interact(show_images, file=directory)

