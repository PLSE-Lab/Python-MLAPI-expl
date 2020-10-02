#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Semi-Automated Rapid Review Workflow
# ## Task: What is the best method to combat the hypercoagulable state seen in COVID-19?
# 
# This notebook implements an **iterative, semi-automated workflow** for key components of a **rapid systematic review**.
# 
# **[Systematic reviews](https://en.wikipedia.org/wiki/Systematic_review)** are a comprehensive and transparent method for synthesizing evidence from the published literature. What primarily distinguishes systematic reviews from other types of literature reviews is the level of detail and thoroughness; researchers involved in systematic reviews take painstaking care into developing a precise review question, inclusion and exclusion criteria, search strategy, assessments of study quality, and synthesis of results (often in the form of a formal meta-analysis).   
# 
# **[Rapid reviews](https://guides.temple.edu/c.php?g=78618&p=4156608)** are a timely alternative when a full systematic review is too resource or time intensive to conduct (systematic reviews often take [12-24 months](https://www.heardproject.org/news/rapid-review-vs-systematic-review-what-are-the-differences/) to complete). Rapid reviews attempt to balance the need for systematic and transparent methods with expediency by simplifying the review components.  This may include reducing the number of databases for search, assigning a single reviewer in each step while another reviewer verifies the results, excluding grey literature, or narrowing the scope of the review.
# 
# This submission deals with two fundamental components of a rapid review: (1) [searching the literature for relevant studies](#Filter-papers), and (2) [extracting key information](#Extract-information) from studies relevant to the research question. While the specific task here is to extract various information related to hypercoagulability and COVID-19, this approach could be applied to any rapid systematic review process.
# 
# ![Process Diagram](https://i.imgur.com/vc2AVsQ.png)
# 
# ### Table of Contents
# * [Filter Papers](#Filter-papers)
# * [Extract Information](#Extract-information)
#    * [Sample Size](#Sample-Size)
#    * [Study Type](#Study-Type)
#    * [Severity](#Severity)
#    * [Therapeutic Methods](#Therapeutic-Methods)
#    * [Outcome/Conclusion Excerpt](#Outcome/Conclusion-Excerpt)
#    * [Primary Endpoint](#Primary-Endpoint)
#    * [Clinical Improvement](#Clinical-Improvement)
# * [Generate Final Output](#Generate-Final-Output)

# First, fix versions of all our dependencies for reproducibility:

# In[ ]:


get_ipython().system('pip install -q pandas==1.0.3 spacy==2.2.1 scispacy==0.2.4 https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz numpy==1.18.1')


# In[ ]:


import json
import warnings
import shutil
import en_core_sci_md
import pandas as pd
from enum import Enum
from spacy.tokens import Doc, Token, Span
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Set
from pprint import pprint

CORD19_INPUT_DIR = Path("/kaggle/input/CORD-19-research-challenge")
ANNOTATIONS_INPUT_FILE = Path("/kaggle/input/generate-umls-annotations/umls_annotations.zip")
ANNOTATION_DIR = Path("/tmp/umls_annotations")
ANNOTATION_DIR.mkdir(exist_ok=True, parents=True)
shutil.unpack_archive(ANNOTATIONS_INPUT_FILE, ANNOTATION_DIR.parent)

OUTPUT_DIR = Path("/kaggle/working")


# For filtering papers and most of the information extraction tasks, we'll need to identify specific biological concepts in the text and potentially relate them to each other or other words.  Doing this via manual text matching (regexes, string searching) is difficult and error-prone because of the wide variety of synonyms and related terms for any given biological concept.  Entirely data-driven approaches that might address the synonym problem (similarity calculations/searches using word/sentence embeddings) are also not ideal due to the inability to tweak/tune the results using valuable subject matter expertise.
# 
# Therefore, our chosen approach for concept annotation is to use scispacy's [EntityLinker](https://github.com/allenai/scispacy#entitylinker) to annotate [UMLS](https://www.nlm.nih.gov/research/umls/index.html) concepts in the text via string matching.  The UMLS vocabulary encodes a wealth of biological knowledge about concepts and how they relate, saving us a lot of time coming up with synonym lists and identifying related terms.  The annotations are imperfect, since there are many cases where string matching can fail to disambiguate between concepts that are named the same but mean different things (e.g., someone named Parkinson vs Parkinson's disease).  However, the results are fairly good overall and provide a decent baseline.  Improved results could potentially be obtained by training and applying a model which uses machine learning to disambiguate entity mentions, such as [MedCAT](https://github.com/CogStack/MedCAT).
# 
# In the [generate-umls-annotations kernel](https://www.kaggle.com/jasonnance/generate-umls-annotations), we applied the EntityLinker to all the articles in the CORD-19 dataset and saved the annotations.  The result is a directory full of JSON objects containing lists of annotated term data.  We'll load some data in below and display an example annotation:

# In[ ]:


for annotation_file in ANNOTATION_DIR.iterdir():
    with open(annotation_file, "r") as f:
        paper_annotations = json.load(f)
        
    if len(paper_annotations) > 0:
        pprint((annotation_file.name, paper_annotations[:2]))
        break


# Here, we define a function that provides an iterator over all the JSON metadata documents in the competition input dataset and some other functions that return the title/abstract/other data from the JSON format.  We'll use these to cleanly access the data in each paper.

# In[ ]:


def all_json_iter() -> Tuple[str, Dict[str, Any]]:
    """
    Iterate over all data files across all text subsets
    """
    all_files = CORD19_INPUT_DIR.glob(
        "document_parses/*/*.json"
    )

    for json_file in all_files:
        if json_file.name.startswith("."):
            # There are some garbage files in here for some reason
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            try:
                article_json = json.load(f)
            except Exception:
                raise RuntimeError(f"Failed to parse json from {json_file}")

        # PMC XML, PDF, etc
        text_type = json_file.parent.name

        yield text_type, article_json
        
        
def get_annotations(sha: str) -> List[Dict[str, Any]]:
    with open(ANNOTATION_DIR / f"{sha}.json", "r") as f:
        return json.load(f)
        
        
def get_article_json(sha: str) -> Dict[str, Any]:
    article_files = list(CORD19_INPUT_DIR.glob(f"document_parses/*/{sha}.json"))
    if len(article_files) == 0:
        raise RuntimeError(f"No JSON file found for SHA {sha}")
    else:
        # If there are multiple parses for this document, we'll take the last one
        # This is intended to match up with how the annotations are generated --
        # If there are multiple papers with the same ID (SHA), the last one will end
        # up in the annotations
        article_file = article_files[-1]
        
    with open(article_file, "r") as f:
        return json.load(f)
        
def get_paper_id(article_json: Dict[str, Any]) -> str:
    return article_json["paper_id"]


def get_title(article_json: Dict[str, Any]) -> str:
    return article_json["metadata"]["title"]


def get_abstract(article_json: Dict[str, Any]) -> str:
    if "abstract" not in article_json:
        return ""
    return "\n\n".join(a["text"] for a in article_json["abstract"])


def get_full_text(article_json: Dict[str, Any]) -> str:
    if "body_text" not in article_json:
        return ""
    return "\n\n".join(a["text"] for a in article_json["body_text"])


def get_all_text(article_json: Dict[str, Any]) -> str:
    return f"{get_abstract(article_json)} {get_full_text(article_json)}"


# ## Filter papers
# 
# The first step of our workflow is to use the annotated terms to filter the papers.  We're interested in all papers that discuss both COVID-19 and hypercoagulability.
# 
# We identify hypercoagulability using a set of related concepts from the UMLS vocabulary identified by the Concept Unique Identifiers (CUIs) in the code below.  Unfortunately, the UMLS release available in scispacy's EntityLinker is from 2017, so it doesn't have any of the newer concepts that were added for COVID-19 and related terms.  Instead, we do a text search based on a list of synonyms.
# 
# Note this is the first place in the workflow where iteration is necessary.  New relevant terms may be discovered, and terms which were previously thought to be relevant may turn out to be irrelevant.  The filter criteria here should be easy to edit going forward, so iteration can be rapid.  The final state of this notebook submission represents multiple rounds of iterating on filter terms.

# In[ ]:


filtered_annotations = {}

COVID19_TERMS = tuple(t.lower() for t in (
    "covid",
    "covid-",
    "SARS-CoV-2",
    "HCoV-19",
    "coronavirus 2",
))

HYPERCOAG_CUIS = set((
    "C0398623",  # Thrombophilia
    "C2984172",  # F5 Leiden Allele
    "C0311370",  # Lupus anticoagulant disorder
    "C1704321",  # Nephrotic Syndrome, Minimal Change
    "C3202971",  # Non-Infective Endocarditis
    "C0040053",  # Thrombosis
    "C2826333",  # D-Dimer Measurement
    "C0060323",  # Fibrin fragment D
    "C3536711",  # Anti-coagulant [EPC]
    "C2346807",  # Anti-Coagulation Factor Unit
    "C0012582",  # dipyridamole
))

num_papers_missing_text = 0
num_annotations_found = 0
num_annotations_missing = 0

metadata = pd.read_csv(CORD19_INPUT_DIR / "metadata.csv")
shas = set()
publish_times = {}
dois = {}
# The sha field may have multiple SHAs delimited by semicolons
for multi_sha, publish_time, doi in zip(metadata["sha"], metadata["publish_time"], metadata["doi"]):
    if not pd.isnull(multi_sha):
        for sha in multi_sha.split("; "):
            shas.add(sha)
            publish_times[sha] = publish_time
            dois[sha] = doi

for sha in shas:
    try:
        paper_json = get_article_json(sha)
    except RuntimeError:
        # This paper doesn't have a JSON file for its full text
        num_papers_missing_text += 1
        continue
        
    paper_annotations = get_annotations(sha)
    if paper_annotations is None:
        # Missing annotations here indicates a desync between our generated annotations
        # and the input data -- keep track but continue
        num_annotations_missing += 1
        continue
    else:
        num_annotations_found += 1
    
    found_covid19 = False
    found_hypercoag = False
    
    search_text = f"{get_title(paper_json)}\n{get_abstract(paper_json)}".lower()
    
    for term in COVID19_TERMS:
        if term in search_text:
            found_covid19 = True
            break
    if not found_covid19:
        continue
    
    paper_concepts = set(concept["cui"] for concept in paper_annotations)
    for hypercoag_cui in HYPERCOAG_CUIS:
        if hypercoag_cui in paper_concepts:
            found_hypercoag = True
            break
            
    if found_covid19 and found_hypercoag:
        filtered_annotations[sha] = paper_annotations[:]


print(f"Checked {num_annotations_found} paper parses with annotations.\n"
      f"Ignored {num_annotations_missing} paper parses without annotations and {num_papers_missing_text} papers without text available.\n"
      f"Identified {len(filtered_annotations)} papers related to COVID-19 and hypercoagulability.")


# Now we have the final set of papers we're interested in.

# In[ ]:


with open(OUTPUT_DIR / "filtered_annotations.json", "w") as f:
    json.dump(filtered_annotations, f)


# ## Extract information
# 
# Once we have the set of papers filtered to only those containing our topics of interest, we need to extract the relevant information.  This is a tricky process which is going to require entirely different methods for different pieces of information.  Without a large training set to use data-driven methods (like training a machine learning model for classification), we're limited to heuristics developed using subject matter expertise.  We fully acknowledge they're imperfect, but we can tweak them as we identify gaps, and being able to automatically extract at least some of the required information can save human reviewers a lot of time.
# 
# This is the other part of the workflow which requires a lot of iteration.  The performance of each heursitic can only be improved through back-and-forth with a subject matter expert after reviewing the performance and identifying potential areas of improvement.  The goal of the heuristics should be to err on the side of capturing too much information -- then it's easy for human reviewers to quickly look through and identify the correct info, whereas if info is missing, they have to go back and read the abstract and potentially full text.
# 
# We use the title and abstract for information extraction due to memory limitations generating annotations for the full text of all papers.

# In[ ]:


nlp = en_core_sci_md.load()


# In[ ]:


# https://stackoverflow.com/a/493788
def text2int(textnum, numwords={}):
    """
    Convert anything that matches the spaCy "like_num" rule to an integer.
    
    https://spacy.io/api/token#attributes
    """
    try:
        # Commas trip up the int parser, so remove them if there are any
        return int(textnum.replace(",", ""))
    except ValueError:
        pass
    
    if not numwords:
        units = [
          "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            raise ValueError("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def get_span(annotation: Dict[str, Any], doc: Doc) -> Span:
    """
    Return the span corresponding to the given annotation.
        
    Assumes the start/end indices in the annotation line up correctly with the document
    (i.e., the document was constructed exactly the same way in the original annotation process
    as it was in the given parsed document).
    """
    return doc.char_span(annotation["start"], annotation["end"])

def get_context(annotation: Dict[str, Any], doc: Doc) -> str:
    """
    Return the context (sentence) in the document containing the given annotation.
    """
    return get_span(annotation, doc).sent.text

def get_root_token(annotation: Dict[str, Any], doc: Doc) -> Token:
    """
    Return the root token for the given annotation.
    """
    return get_span(annotation, doc).root


# ### Sample Size
# 
# To identify the sample size heuristically, we'll use spaCy's dependency parsing to find nouns representing a study participant ("subject", "patient", etc) and record any numbers that were associated with them.  For each extracted sample size, we return the containing sentence, so a human reviewer can quickly determine whether they agree without reading the full text.
# 
# This produces a lot of false positives which are difficult to disambiguate from the language features alone -- ex.  a study may mention various subsets of its cohort, or a review paper may be describing the sample size of the papers it's reviewing.  We'll do a best-effort disambiguation by taking the maximum number if there are multiple numbers, to attempt to avoid picking up subsets of the sample.
# 
# We note this approach fails in some cases on this dataset because line numbers are embedded in many PDFs which confuse spaCy's dependency parser.  For example:
# 
# > This 95 model is first trained on patients from a single hospital and then externally validated on 96 patients from four other hospitals. We achieve strong performance, notably predicting 97 mortality
# 
# Our somewhat naive approach doesn't recognize that "96" in the above excerpt is a line number and reports it incorrectly.  Implementing a method to clean the line numbers (or parsing the PDFs in a way which doesn't include them in the first place) should improve the performance.
# 
# We noticed other failure cases which aren't easily addressable using this approach:
# 
#  - Case studies with a single patient
#  - Sentences where the spaCy model's part-of-speech tagger fails (ex. in "83 confirmed patients", "confirmed" could be parsed as a verb rather than an adjective)
#  - Papers which don't explicitly list the full sample size (ex. "100 COVID-19 patients and 200 non-COVID-19 patients")
#  
# However, we find that in general, the heuristic performs fairly well, and returning the context of the sentence along with the extracted sample size allows a human reviewer to very quickly confirm whether the extracted value is correct.

# In[ ]:


SAMPLE_SIZE_NOUNS = set(("patient", "subject", "case", "birth"))

def find_sample_size(doc: Doc) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    For a parsed spaCy document representing a paper, try to identify the sample size heuristically.
    If possible, return a tuple containing the sample size, noun describing
    the sample, and the sentence that inference was generated from.
    
    If not, return None for each tuple element.
    """
    sample_size_candidates = []
    
    for tok in doc:
        if tok.like_num and tok.head.lemma_ in SAMPLE_SIZE_NOUNS and tok.dep_ == "nummod":
            try:
                sample_size_int = text2int(tok.text.lower())
            except ValueError:
                continue
            sample_size_candidates.append((sample_size_int, tok.head.text, tok.sent.text))
    
    if len(sample_size_candidates) == 0:
        return (None, None, None)
    chosen_candidate = max(sample_size_candidates, key=lambda c: c[0])
    return chosen_candidate


# ### Study Type
# 
# There's a fairly complex set of definitions for study types -- we'll try to pull out some concepts and tokens that might indicate one study type over the other.  If we can't make a reasonable guess, we'll leave it blank.
# 
# Note we introduce an additional study type: "Observational".  In many cases, distinguishing between retrospective and prospective observational studies is very difficult, so we include a fallback umbrella term which encapsulates both.
# 
# We also didn't find enough instances of the following study types to develop rules for them:
# 
# - Cross-sectional study
# - Case series
# - Ecological regression

# In[ ]:


SYS_REV_CUIS = set((
    "C1955832",  # Review, Systematic
    "C0282458",  # Meta-Analysis (publications)
))

EXP_REV_CUIS = set((
    "C0282443",  # Review [Publication Type]
))

SIM_CUIS = set((
    "C0376284",  # Machine Learning
    "C0683579",  # scenario
))

RET_OBS_CUIS = set((
    "C0035363",  # Retrospective Studies
    "C2362543",  # Electronic Health Records
    "C2985505",  # Retrospective Cohort Study
))

OBS_CUIS = set((
    "C0030705",  # Patients
))

EDIT_WORDS = set((
    "editor",
    "opinion",
))

RET_OBS_WORDS = set((
    "retrospective",
    "retrospectively",
    "autopsy",
))

PROS_OBS_WORDS = set((
    "prospective",
    "prospectively",
    "enrolled",
))


class StudyType(Enum):
    PROS_OBS = "Prospective observational study"
    RET_OBS = "Retrospective observational study"
    OBS = "Observational study"
    SYS_REV = "Systematic review and meta-analysis"
    EXP_REV = "Expert review"
    SIM = "Simulation"
    EDIT = "Editorial"

def find_study_type(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to identify study type using words and concepts in the document.
    """
    study_type = None
    study_type_words = set()
    contexts = []
    # Fallback classification for studies which are clearly observational, but we can't
    # tell whether it's prospective or retrospective
    is_observational = False

    doc_annotations = {a["cui"]: a for a in annotations}
    
    def update_study_type(new_study_type: StudyType, word: str, context: str):
        nonlocal study_type
        study_type = new_study_type
        study_type_words.add(word)
        contexts.append(context)
        
    def check_cui_set(cui_set: Set[str], cui_set_study_type: StudyType):
        for cui in cui_set:
            if cui in doc_annotations:
                annotation = doc_annotations[cui]
                update_study_type(cui_set_study_type,
                                  annotation["canonical_name"],
                                  get_context(annotation, doc))
                
    def check_word_set(word_set: Set[str], word_set_study_type: StudyType):
        for sent in doc.sents:
            for tok in sent:
                if tok.lemma_.lower() in word_set:
                    update_study_type(word_set_study_type, tok.text, sent.text)
        
    # Check CUIs first, since those should be more reliable
    for cui_set, cui_set_study_type in (
        (SYS_REV_CUIS, StudyType.SYS_REV),
        (EXP_REV_CUIS, StudyType.EXP_REV),
        (SIM_CUIS, StudyType.SIM),
        (RET_OBS_CUIS, StudyType.RET_OBS),
    ):
        if study_type is None:
            check_cui_set(cui_set, cui_set_study_type)
    
    # Check word sets next
    for word_set, word_set_study_type in (
        (EDIT_WORDS, StudyType.EDIT),
        (RET_OBS_WORDS, StudyType.RET_OBS),
        (PROS_OBS_WORDS, StudyType.PROS_OBS)
    ):
        if study_type is None:
            check_word_set(word_set, word_set_study_type)
            
    # Finally, if we still don't have a study type, check the fallback umbrella study type
    # for all observational studies
    if study_type is None:
        check_cui_set(OBS_CUIS, StudyType.OBS)
        
    return (
        None if study_type is None else study_type.value,
        "; ".join(list(study_type_words)),
        "\n".join(contexts)
    )


# ### Severity
# 
# The severity is tougher to identify, since there are several general words which may be used to describe it.  We'll make a list of terms related to severity, pull them out, report all the unique ones, and report any context that could help a human reviewer make a quick decision.

# In[ ]:


SEVERITY_WORDS = set(("mild", "severe", "critical", "ICU", "intensive care unit"))

def find_severity(doc: Doc) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to find case severity in the document.
    """
    severity = set()
    contexts = []
    
    for tok in doc:
        if tok.lemma_ in SEVERITY_WORDS:
            severity.add(tok.text.lower())
            contexts.append(tok.sent.text)
            
    if len(severity) == 0:
        return None, None

    return "; ".join(list(severity)), "\n".join(contexts)


# ### Therapeutic Methods
# 
# Here, we'll use the annotated UMLS concepts to identify therapeutic methods and return their contexts.

# In[ ]:


THERAPEUTIC_METHOD_CUIS = set((
    "C0012582",  # dipyridamole
    "C1963724",  # Antiretroviral therapy
    "C0770546",  # heparin, porcine
))

def find_therapeutic_methods(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to find therapeutic methods by UMLS concepts in the document.
    """
    methods = set()
    contexts = []
    
    for annotation in annotations:
        if annotation["cui"] in THERAPEUTIC_METHOD_CUIS:
            methods.add(annotation["canonical_name"])
            contexts.append(get_context(annotation, doc))
            
    return (
        "; ".join(list(methods)),
        "\n".join(contexts),
    )


# ### Outcome/Conclusion Excerpt
# 
# Here, we'll search for various words that might indicate a sentence describes an outcome or conclusion and report all such sentences.

# In[ ]:


OUTCOME_CUIS = set((
    "C0332281",  # Associated with
    "C0392756",  # Reduced
    "C1260953",  # Suppressed
    "C0309872",  # PREVENT (product) [proxy for word "prevent"]
    "C0278252",  # Prognosis bad
    "C0035648",  # risk factors
    "C0184511",  # Improved
    "C0442805",  # Increased
    "C0205216",  # Decreased
))

def find_outcomes(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to find outcome excerpts by UMLS concepts in the document.
    """
    outcome_words = set()
    contexts = []
    
    for annotation in annotations:
        if annotation["cui"] in OUTCOME_CUIS:
            outcome_words.add(annotation["canonical_name"])
            contexts.append(get_context(annotation, doc))
            
    return (
        "; ".join(list(outcome_words)),
        "\n".join(contexts),
    )


# ### Primary Endpoint
# 
# We'll search for words that describe the primary endpoint(s) of the study and report all such sentences.

# In[ ]:


ENDPOINT_CUIS = set((
    "C0011065",  # Cessation of life
    "C0026565",  # Mortality Vital Statistics
))

def find_endpoints(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to find information related to primary endpoints by UMLS concepts in the document.
    """
    endpoint_words = set()
    contexts = []
    
    for annotation in annotations:
        if annotation["cui"] in ENDPOINT_CUIS:
            endpoint_words.add(annotation["canonical_name"])
            contexts.append(get_context(annotation, doc))
            
    return (
        "; ".join(list(endpoint_words)),
        "\n".join(contexts),
    )


# ### Clinical Improvement
# 
# This is a tricky one to identify with heuristics, since it's very general.  We're going to simplify it down to the problem of identifying a set of biological mechanisms/functions and tagging them as "good" or "bad" (from a clinical health perspective).  We can then identify all instances of these mechanisms in the text and use the spaCy model's dependency parser to identify whether those functions went "up" or "down".  We make the extremely naive assumption that a "good" mechanism going "up" or a "bad" mechanism going "down" leads to clinical improvement and vice versa.
# 
# We'll use CUIs when possible to ensure maximal coverage of synonyms and incorporating the biological knowledge in UMLS, but we'll use exact token searches in some cases where CUIs aren't tagged or otherwise don't exist for certain concepts.

# In[ ]:


GOOD_MECHANISM_CUIS = set((
    "C2247948",  # response to type I interferon
    "C0005821",  # Blood Platelets
    "C0200635",  # Lymphocyte Count measurement
    "C0301863",  # "U" lymphocyte
    "C1556326",  # Adverse Event Associated with Coagulation
    "C0019010",  # Hemodynamics
    "C1527144",  # Therapeutic Effect
    
))

BAD_MECHANISM_CUIS = set((
    "C1883725",  # Replicate
    "C0042774",  # Virus Replication
    "C0677042",  # Pathology processes
    "C0398623",  # Thrombophilia
    "C2826333",  # D-Dimer Measurement
    "C1861172",  # Venous Thromboembolism
))

UP_WORDS = set((
    "elicit",
    "good",
))

UP_CUIS = set((
    "C0442805",  # Increase
    "C0205250",  # High
    "C2986411",  # Improvement
))

DOWN_WORDS = set((
    "ameliorate",
))

DOWN_CUIS = set((
    "C1260953",  # Suppressed
    "C0205216",  # Decreased
    "C0392756",  # Reduced
    "C1550458",  # Abnormal
))

def find_clinical_improvement(
    doc: Doc,
    annotations: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Identify phrases corresponding to clinical improvement (or not).  Return a tuple with 3 strings:
    "y/n" based on whether more evidence for improvement or worsening is found, evidence for improvement, and
    evidence for worsening.
    """
    improvement_evidence = set()
    worsening_evidence = set()
    
    # Locate tokens corresponding to all categories (good/bad mechanisms, up/down)
    bad_spans = set()
    good_spans = set()
    up_spans = set()
    down_spans = set()
    
    for tok in doc:
        for word_set, span_set in (
            (UP_WORDS, up_spans),
            (DOWN_WORDS, down_spans),
        ):
            if tok.lemma_ in word_set:
                span_set.add(doc[tok.i:tok.i+1])
    
    for annotation in annotations:
        for cui_set, span_set in (
            (GOOD_MECHANISM_CUIS, good_spans),
            (BAD_MECHANISM_CUIS, bad_spans),
            (UP_CUIS, up_spans),
            (DOWN_CUIS, down_spans),
        ):
            if annotation["cui"] in cui_set:
                annotation_span = get_span(annotation, doc)
                span_set.add(annotation_span)
    
    # Check dependencies to see if any of the discovered tokens relate in ways that
    # might provide evidence one for improvement or worsening
    for mechanism_set, modifier_set, evidence_set in (
        (bad_spans, up_spans, worsening_evidence),
        (bad_spans, down_spans, improvement_evidence),
        (good_spans, up_spans, improvement_evidence),
        (good_spans, down_spans, worsening_evidence),
    ):
        for mechanism_span in mechanism_set:
            mechanism_tok = mechanism_span.root
            for modifier_span in modifier_set:
                modifier_tok = modifier_span.root
                if modifier_tok in mechanism_tok.children:
                    evidence_set.add((modifier_span, mechanism_span))
            
    improvement = None
    if len(improvement_evidence) > len(worsening_evidence):
        improvement = "y"
    elif len(improvement_evidence) < len(worsening_evidence):
        improvement = "n"
        
    def format_evidence(evidence_set):
        return "; ".join(
            " ".join((modifier_span.text, mechanism_span.text)) for modifier_span, mechanism_span in evidence_set
        )
        
    return (
        improvement,
        format_evidence(improvement_evidence),
        format_evidence(worsening_evidence),
    )


# ### Generate Final Output
# 
# We can apply all of the above heuristics to each document in turn to automatically generate a spreadsheet for human review.  Human reviewers can manually correct items which need correcting and provide feedback to improve the heuristics.

# In[ ]:


auto_data = []

for sha, paper_annotations in filtered_annotations.items():
    try:
        article_json = get_article_json(sha)
    except RuntimeError:
        warnings.warn(f"No article JSON found for SHA {sha}")
        continue
    
    doc_text = f"{get_title(article_json)}\n\n{get_abstract(article_json)}"
    doc = nlp(doc_text)
    
    publish_time = publish_times[sha]
    doi = dois[sha]
    sample_size, sample_unit, sample_size_context = find_sample_size(doc)
    severity, severity_context = find_severity(doc)
    therapeutic_methods, therapeutic_method_context = find_therapeutic_methods(doc, paper_annotations)
    study_type, study_type_words, study_type_context = find_study_type(doc, paper_annotations)
    outcome_words, outcome_context = find_outcomes(doc, paper_annotations)
    endpoints, endpoint_context = find_endpoints(doc, paper_annotations)
    improvement, improvement_evidence, worsening_evidence = find_clinical_improvement(doc, paper_annotations)

    auto_data.append({
        "Paper ID": get_paper_id(article_json),
        "Title": get_title(article_json),
        "Abstract": get_abstract(article_json),
        "DOI": doi,
        "Date": publish_time,
        "Sample Size": sample_size,
        "Sample Unit": sample_unit,
        "Sample Size Context": sample_size_context,
        "Severity": severity,
        "Severity Context": severity_context,
        "Therapeutic Methods": therapeutic_methods,
        "Therapeutic Method Context": therapeutic_method_context,
        "Study Type": study_type,
        "Study Type Words": study_type_words,
        "Study Type Context": study_type_context,
        "Outcome Words": outcome_words,
        "Outcome Context": outcome_context,
        "Endpoint Words": endpoints,
        "Endpoint Context": endpoint_context,
        "Clinical Improvement": improvement,
        "Clinical Improvement Evidence": improvement_evidence,
        "Clinical Worsening Evidence": worsening_evidence,
    })
    
auto_df = pd.DataFrame(auto_data).set_index("Paper ID")
auto_df.to_csv(OUTPUT_DIR / "What is the efficacy of novel therapeutics being tested currently_.csv")


# ## Conclusion
# 
# While we provide a semi-automated workflow to help with components of a rapid review (specifically, "the best method to combat the hypercoagulable state seen in COVID-19"), there are several areas for future work.  First, we restricted our submission to the CORD-19 literature set, which only contains articles from PMC, the WHO, bioRxiv, and medRxiv.  Though examining fewer databases and grey literature is a common strategy in rapid reviews, it may bias the findings towards articles that are more convenient to access. Additionally, we do not tackle other important aspects of rapid reviews such as a risk of bias assessment, reliablity checks by multiple reviewers, or a meta-analysis. Given we did not have a targeted [PICO](https://canberra.libguides.com/c.php?g=599346&p=4149722) statement defining our intended population, comparison groups, etc., as part of the challenge, we felt these steps would be better addressed in future work supported by domain experts and systematic review methodologists. 
# 
# Methodologically, our approaches relies heavily on heuristics, ontologies, and domain input. While data engineering folk wisdom [champions the use of heuristics](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_1_don%E2%80%99t_be_afraid_to_launch_a_product_without_machine_learning) as a strong baseline, ideally, machine learning algorithms could improve upon these heuristics in a more systematic way. Supervised machine learning was not strongly considered for information extract due to the limited number of articles and labeled data. Future work could explore the use of methods such as [data programming](https://dawn.cs.stanford.edu/pubs/snorkel-nips2016.pdf) to integrate heuristics as noisy labels for supervised learning.   
