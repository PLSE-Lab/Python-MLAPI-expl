#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from IPython.core.display import display, HTML, Javascript
from string import Template
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import json, random
import IPython.display

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Avoiding Search Term Insufficiency in CORD-19
# 
# ## Abstract
# The "COVID-19 Open Research Dataset Challenge" (CORD-19) is a call to action "to develop text and data mining tools that can help the medical community develop answers to high priority scientific questions" from a rapidly growing body of literature. Using "bag-of-word" text searches on large document sets is very popular, but often produces very large result sets. Coping with large result sets by using a one-size-fits-all ranking algorithm and paginated display of results is inferior to using more targeted search terms which are sufficient to limit results without missing relevant documents.  We introduce an improved methodology designed to provide searchers 
# - a means to search titles and abstracts of the CORD-19 dataset, (which avoids the risks and limitations of bag-of-words searches on high-context technical prose in most scientific literature), 
# - a means to view collated results displayed under MeSH subject headings, 
# - a means to deliberately and confidently choose to exclude documents under subject headings that are irrelevant to their purposes, 
# - a means to locate other documents at the relevant MeSH subject headings which can be used to suggest additional or more targeted search terms, and 
# - a means to consider documents that were were omitted.
# 
# The ability to produce a manageable, custom-tailored full result set without the risks of general-purpose ranking and pagination allows the quality of the search to be evaluated rather than assumed.
# 
# ## Background
# Mechanical algorithms that attempt to estimate relevance (such as TF-IDF, number of citations) are popular and well-tolerated, but they always come with the risk of over-weighting irrelevant results, marginalizing relevant results, and failing to show what similar results were omitted. 
# 
# I consider the CORD-19 success critiera to mean "locating a precise, trustworthy answer in the CORD-19 dataset."  It is not enough to produce "an answer", it must be a "correct answer" and there must be some way to be confident it is correct. Establishing confidence requires human evaluation, first in selecting which search results to access articles later, reading and understanding the article and evaluating it for suitability. In this notebook, we introduce tools to leverage the human activities which are already present that are not amenable to being easily automated:
# 
# - The author's effort to write careful titles and abstracts with descriptive terms used in low-context.
# - The publisher's effort to identify and provide accurate and appropriate MeSH subject headings and metadata to MEDLINE
# - The searcher's effort to select accurate and appropriate descriptive terms for searching.
# - The searcher's effort to review results for relevancy and consider what was missed and what was inadvertently included.
# 
# Methodology and a short example are provided in this notebook. Longer examples are provided in the separate notebooks being used to make CORD-19 task submissions using this methodology, in order to avoid duplicated content.
# 
# ## Description of Existing (unimproved) Search Methodology
# 
# A diagram of the existing search methodology can be presented as follows:
# 
# 

# ![existingmethodology.png](attachment:existingmethodology.png)

# While the author and publishing efforts are depicted in the same column in this diagram, they are certainly distinct in many respects. Prior to 2015, the MeSH tagging was performed by trained librarians at the NLM, which generally results in tighter control over vocabulary and subject usage. But the tagging done since then appears to maintain a high-level of consistency and accuracy.
# 
# The NLM still maintains the MeSH subject headings, and the current version (2020) was used for this work.
# 
# ## New Improvements to the Retrieval Process
# My approach is to focus on improving the efficiency and outcomes in the searcher's side of the retrieval process. (While future work could improve the author and publisher efforts, that is outside the scope of the Kaggle project, which must work with the existing dataset.) After considering the task and the state-of-the-art in natural language understanding, we tend to favor improvements which augment and enhance human capabilities rather than automate and eliminate them entirely. 
# 
# 
# I focus on the following aspects of the searcher's efforts:
# <ol><li> reduce risks that a non-trustworthy source is used to provide an answer. This might happen in three ways:
# 	<ol><li>A source can be non-trustworthy (e.g. unscientific, or even a work of complete fiction), or
# 	<li>A source can be trustworthy, but used in a way that is not trustworthy (e.g. taking a result out of context.)
# 	<li>Superior trustworthy sources are missed and inferior sources are located, (e.g. best source inadvertently ranked too low)
# 	</ol>
# <li> provide means to work with an entire result set, not a ranked, paginated result set
# <li> provide means to locate alternative resources which may have confirmation or alternative (potentially competing) answers and possibly suggest improved search terms 
# </ol>
# 
# 

# ## Improved Methodology
# ![methodology.png](attachment:methodology.png)

# The changes to search methodology compared to the previous diagram are an additional column for pre-processing the CORD-19 metadata in bulk, and augmentation of the human-controlled search process on the right side. The most important change is that instead of getting a paginated set of search results, results are collated and displayed within MeSH headings.  
# 
# 

# In[ ]:


htmlpros="""
<style>
 .l th { text-align:left;}
  .l td { text-align:left;}
   .l tr { text-align:left;}
</style>
<p>
<h2>Pros and cons of my approach</h2>
<table class=l border=1>
<tr><th>Risk</th><th>Example Failure</th><th>Design</th>
<tr><td>	1.1 Non-trustworthy sources<td>Unscientific or fictitious source<td>Use CORD-19 dataset.<br>Pro: which are primarily sources from a PUBMED search of documents that were
written and published for professionals (in medicine, government, etc.). They are not all peer-reviewed (not a panacaea anyway) but they are expected to be scientifically trustworthy.<br>Con: CORD-19 are "open" sources. Better sources may be excluded.
<tr><td rowspan=4>1.2 Inappropriate use of trustworthy sources<td rowspan=4>Result out of context. (E.g. incubation period from a paper about mice, or SARS, which is a different virus)
	<td>Use results from titles and abstracts, not full text. <br>Pro: Full-text search is likely to find many documents with the terms that were intended to be mentioned only in passing, and in a different context, unnecessarily multiplying the quantity, but not the quality of results.<br>Pro: Titles (and most abstracts) are most often written from a "low-context" perspective using wording that is precise and most carefully considered. This is in contrast to sentences from full text which are very high-context. We avoid excerpting high-context sentences for presentation in search results so that we avoid the risk of misinterpretation due to taking them out of the author's context and presenting them in a very different context.<br>Con: Important findings may not be described in the abstract using search terms.
<tr><td>Avoid use of secondary MeSH headings.<br>Pro: A source with trustworthy conclusions should be categorized under a primary MeSH heading. Secondary headings are inconsistently applied and using them decreases reliability.<br>Con: Secondary headings, if they are accurate, could be useful in limiting search results further than just primary Headings. (Due to the inconsistent application of secondary headings, there are better ways of doing this than secondary headings, though. Using the alternative primary heading is better, for example.)
<tr><td>Use Human-in-the-loop to show context and interactively limit sources (e.g. by MeSH header, date range, paper type.)
<br>Pro: Human can apply and review individual criteria for their circumstances
<br>Con: "Cluttered" user interface that requires some experience to become familiar and use well. 
<tr><td>Date range limit and sort by date range descending. [See note 1 below.]
<tr><td rowspan=7>1.3 Missed sources<td rowspan=4>Search terms too narrow, lacking synonyms
	<td>Show search results within MeSH subject context.
	<br>Pro: Sources with similar topics are grouped into a category. Finding one document finds the category, and that means other articles that are similar. Synonym terms will be in those documents.
	<br>Con: MeSH subjects are too broad.
<tr><td>Use synonyms in search terms Caveat: (Currently human-directed, as was done in CORD-19 query)
	<br>Pro: avoids creating a good synonym dictionary
	<br>Con: Extra work and likely to lead non-experts to searches that are too narrow.
<tr><td>Show links to nearby categories that were used, because MeSH tagging for articles is not 100% reliable.
	<br>Pro: Tolerates the categorization imprecision that exists in any subject index, by showing "adjacent" categories, parents, and children.
	<br>Con: MeSH trees are not without flaws, and may be unfamiliar to users. Makes searching harder because several pages need to be consulted, not just one.
<tr><td>Show articles at multiple MeSH tree locations, not just one.
	<br>Pro: Articles are multi-faceted and should not go into a subject tree at one heading.
	<br>Con: Articles will be seen more than once when searching several subject pages.
<tr><td rowspan=2>Sources outside CORD-19<td>Show concept keywords for making external searches
	<br>Pro: Allows consulting other systems
	<br>Con: Would be better to integrate the other sources into one system.
<tr><td>Deep link to full PubMed search so it can be edited and re-run
	<br>Pro: Allows consulting other systems
	<br>Con: Makes it look like that consulting other systems is usual and necessary, rather than for rare needs.
<tr><td>Missing because terms appear in body, not title and abstract<td>Provide link to PubMed with full-text search
	<br>Pro: Allows expanding to full-text search without needing a reverse index.
	<br>Con: Harder to use than clicking a checkbox and re-running
<tr><td>2. Too many results<td>Imposed ordering of paginated results<td>Present unranked, unpaginated results under categories. See Note 2
	<br>Pro: Makes it easy to use the full result set
	<br>Con: Requires using the full, unranked result set.

<tr><td rowspan=2>3. Unknown confidence in results<td>Accepting narrow search results that look good<td>Human-in-the-loop to explore "what could I be missing" using designs to avoid missed sources in 1.3
	<br>Pro: Human in the loop is good control	
	<br>Con: Training is necessary to get consistent results across people.
<tr><td>Lack of consensus on source quality<td>Show all results that match, without imposing filtering
	<br>Pro: This is not a "solvable" problem. Humans have individual opinions. There should be deep links from results to pages with mechanisms to encourage easy customization and re-resulting with different source filters.
	<br>Con: Non-experts will get different results and take extra work to compare.
<tr><td colspan=3>Note 1: The importance of considering date in evaluating results cannot be over-emphasized given two facts:<ol><li>In the abstract of the Review article <A href="https://pubmed.ncbi.nlm.nih.gov/32166607/">A Review of Coronavirus Disease-2019 (COVID-19)</A>", (2020/03) the authors report "[SARS-CoV-2] spreads faster than its two ancestors the SARS-CoV and Middle East respiratory syndrome coronavirus (MERS-CoV), but has lower fatality." <li>The CORD-19 dataset includes search terms for MERS and SARS and since SARS-CoV-2 is so new, there are likely to be a disproportionate majority of articles which could answer CORD-19 questions unreliably.
</ol>NLP and automated text processing that estimates consensus will be working with a very large data set of possibly inappropriate sources. SARS-CoV-2 must is distinct from other coronaviruses at least in several important aspects. General conclusions made in 2019 about "coronavirus" will need to be re-evaluated and not used indiscriminately. In a research field that is so new, a human would deem the consideration of date of publication as an essential part of identifying trustworthy sources. Using date-range descending order of presenting a list of results is a simple way to help with this task.
<tr><td colspan=3>Note 2: The problem of too many search results is so common that we have become comfortable with partial and risky coping strategies. The most common is that some form of automatic ranking is used to show results in a sorted list with pagination. This is bad because even if all users agreed on the ranking criteria, it is not within the current capabilities of automated text processing systems to reliably evaluate sources for useful criteria, such as "trustworthiness." There are many criteria that might seem to be proxies for trustworthiness (such as type of publication systematic review vs. study, number of citations) they are not reliable for many reasons, including: 
<ul><li>dependence on reliable metadata creation. (There are gaps and limitations in the MEDLINE metadata.)
<li>different users will have different ranking criteria they would prefer. What most people would expect would be a composite multi-faceted analysis of several base criteria that are not consistent across all the articles in a large data set.
<li>in "cutting edge" research, there has not been enough time to establish a body of research with scientific consensus so new and old articles need different criteria applied.
<li>it is not simple to indicate the valuation of trustworthiness or other ranking criteria to a user, even if it were reliable. 
</ul>
 In most systems, the ranking criteria is imposed, not selected by the individual. This is risky because sorted results and pagination train and encourage people to be satisfied with looking through some of the results and stopping when they find something "sufficient" rather than "better" and "optimal." When the answer is critical, all the relevant results should be reviewed, and if filtering and ranking is to be employed, it should be directed by the user.
<p>The CORD-19 dataset has about 50,000 items, which is certainly too many items to display if a stop-word (or word from the orginal search criteria) is used for filtering.  But if articles are segregated by MeSH subject heading, the MeSH heading with the largest number of articles in the CORD-19 data set is Coronavirus Infections, with 2,485 items. Most users will want to apply criteria or at least one additional text search term to limit the set. But even if they do not, the 9.6MB metadata file and use of modern desktop web browsers easily show 2,485 in one list.
</table>

"""

h = display(HTML(htmlpros))


# # Example Uses
# One notebook per CORD-19 task details the search results.  Here is one example from [CORD-19 Task](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=568)
# 
# > What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# 
# > Specifically, we want to know what the literature reports about:
# 
# >     Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery
# 
# 

# ## Example of Formatted search results

# In[ ]:


htmltask1a="""
<style>
 .l th { text-align:left;}
  .l td { text-align:left;}
   .l tr { text-align:left;}
</style>
Searching for (incubation period) in<br>[recent CORD-19 titles and abstracts]<br> finds documents at the following Medical Subject (MeSH) headings
<blockquote>
<p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a></span>
</p><p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a></span>
</p><p><span style="font-size:large"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a></span>
</p></blockquote><p>Click a MeSH heading to review these results in a WebApp where you can view full abstracts, modify the search terms, include documents published before 2019/12, etc.
<table class=l><tbody><tr><th>MeSH heading</th><th>Recent Titles and matching excerpts from Abstracts</th><th>Pub.Type</th><th>Date</th>
</tr><tr valign="top"><td rowspan="4"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/C/CO/Coronavirus Infections">Coronavirus Infections</a>
</td></tr><tr valign="top"><td>
 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32112886">
Characteristics of COVID-19 infection in Beijing.
</a>
<small>(PMID32112886</small>)
<br>...The median <b>incubation</b> <b>period</b> was 6.7 days, the interval time from between illness onset and seeing a doctor was 4.5 days.
</td><td>Journal Article; Research Support, Non-U.S. Gov't</td>
<td>2020/04</td>
</tr>
<tr valign="top"><td>
 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/31995857">
Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus-Infected Pneumonia.
</a>
<small>(PMID31995857</small>)
<br>...The mean <b>incubation</b> <b>period</b> was 5.2 days (95% confidence interval [CI], 4.1 to 7.0), with the 95th percentile of the distribution at 12.5 days.
</td><td>Journal Article; Research Support, N.I.H., Extramural; Research Support, Non-U.S. Gov't</td>
<td>2020/03</td>
</tr>
<tr valign="top"><td>
 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32046816">
Effectiveness of airport screening at detecting travellers infected with novel coronavirus (2019-nCoV).
</a>
<small>(PMID32046816</small>)
<br>...In our baseline scenario, we estimated that 46% (95% confidence interval: 36 to 58) of infected travellers would not be detected, depending on <b>incubation</b> <b>period</b>, sensitivity of exit and entry screening, and proportion of asymptomatic cases.
</td><td>Journal Article</td>
<td>2020/02</td>
</tr>
<tr valign="top"><td rowspan="2"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/B/BE/Betacoronavirus">Betacoronavirus</a>
</td></tr><tr valign="top"><td>
 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32046819">
Incubation period of 2019 novel coronavirus (2019-nCoV) infections among travellers from Wuhan, China, 20-28 January 2020.
</a>
<small>(PMID32046819</small>)
<br>...Using the travel history and symptom onset of 88 confirmed cases that were detected outside Wuhan in the early outbreak phase, we  estimate the mean <b>incubation</b> <b>period</b> to be 6.4 days (95% credible interval: 5.6-7.7), ranging from 2.1 to 11.1 days (2.5th to 97.5th percentile).
</td><td>Journal Article</td>
<td>2020/02</td>
</tr>
<tr valign="top"><td rowspan="2"><a href="http://www.softconcourse.com/CORD19/?filterText=incubation+period&amp;from=CORD19#/P/PN/Pneumonia, Viral">Pneumonia, Viral</a>
</td></tr><tr valign="top"><td>
 <a target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/32166607">
A Review of Coronavirus Disease-2019 (COVID-19).
</a>
<small>(PMID32166607</small>)
<br>...The disease is transmitted by inhalation or contact with infected droplets and the <b>incubation</b> <b>period</b> ranges from 2 to 14 d.
</td><td>Journal Article; Review</td>
<td>2020/04</td>
</tr>
</tbody></table>
</p><hr><p>There are also 131 matches before 2019/12
</p>
"""

h = display(HTML(htmltask1a)) 


# # Technical Summary: Dataset acquisition and preparation
# <h2>1. Obtaining MEDLINE metadata for CORD-19 data set</h2>
# The MEDLINE metadata which includes MH-type (MeSH heading) records was obtained as follows:
# <br>
# 1. By re-running the Pubmed <a href="https://www.ncbi.nlm.nih.gov/pmc/?term=%22COVID-19%22+OR+Coronavirus+OR+%22Corona+virus%22+OR+%222019-nCoV%22+OR+%22SARS-CoV%22+OR+%22MERS-CoV%22+OR+%E2%80%9CSevere+Acute+Respiratory+Syndrome%E2%80%9D+OR+%E2%80%9CMiddle+East+Respiratory+Syndrome%E2%80%9D">CORD-19</a>
# <p>
# 2. The details shown in the side bar are:
# </p><blockquote>
# "COVID-19"[All Fields] OR ("coronavirus"[MeSH Terms] OR "coronavirus"[All Fields]) OR "Corona virus"[All Fields] OR "2019-nCoV"[All Fields] OR "SARS-CoV"[All Fields] OR "MERS-CoV"[All Fields] OR "Severe Acute Respiratory Syndrome"[All Fields] OR "Middle East Respiratory Syndrome"[All Fields]
# </blockquote>
# <p>
# 
# 3. The sidebar has "Find Related Data". This truncates at 10000 items, so we need to use date ranges to limit the results.
# The date ranges used were: 1900-2006, 2007-2011, 2012-2014, 2015-2017, 2018-2020, and all are less than 10000 items.
# </p><p>
# 
# Clicking "Find Related Data" and asking for MEDLINE format download gets the format we want.
# </p><p>
# 
# 4. The files were concatenated to a 130MB file for use in later steps.
# </p>

# # Preparation of MeSH descriptor data
# As part of our approach we want to display responses to questions within the context of MeSH descriptors, alongside other articles from the dataset.
# 
# The MeSH subject headings were obtained by downloading the 2020 .XML file https://www.ncbi.nlm.nih.gov/mesh
# 
# XML file is a verbose file format and includes fields which are not useful. A javascript program (parseMesh.js) was run (using NodeJS) to converted the XML to a custom-format Tab-separated-value (TSV) file. The output has four different record types, and is saved to desc2020d.tsv
# 
# This is used a the MeSH index in later processing steps. 
# 
# The XML original is 290MB, and the TSV index is 20MB. 

# # Collation of CORD-19 MEDLINE data to separate files for each MeSH Descriptor
# 
# The results explorer is a javascript application that runs in a web browser. The full set of capabilities and requirements is discussed in a separate section, but some description is necessary here in order to explain the purpose of having a collated datafile per MeSH dscriptor.
# 
# The results explorer always shows results inside the context of one MeSH subject descriptor. This supports the ability to see what alternative results could have been provided. All of the MEDLINE data needed to display the results is collated to one datafile per MeSH descriptor name, and this datafile is fetched at the time the browser page is loaded and when the window.location is changed to show a different MeSH description.
# 
# Starting at the top of the explorer page, there are definitions and notes for the MeSH descriptors. These come from the NLM data. Below that are links to other MeSH descriptor pages that are related, narrower, and wider (more general). (These appear above the results section.)
# 
# It is important to keep in mind that a MeSH descriptor exists in more than one MeSH tree simultaneously. The user interface must support and indicate that there are multiple parents for many of the descriptors. (It is not just a simple "bread-crumb" path to arrive at a descriptor.)
# 
# The count of items at the target link is shown in parenthesis at the link.
# 
# So in addition to the MEDLINE article data, the metadata for the MeSH subject and related links must be available to display the page. All of this extra datails placed at the head of the data file which has all of the MEDLINE metadata. In short, everything that is needed to be displayed is in that one file, which are plain text following MEDLINE formats. I used a file extension of .pmidx.
# 
# The .files are created by a custom javascript program (parsePubmed2.js, run with NodeJS) which takes the source data (downloaded MEDLINE data, the processed MeSH descriptor data (desc2020d.tsv) and article count data (pubmed.tally from a previous run)) and produces individual files under a simple tree of folder-names (E.g. /C/CO/Cornavirus Infections.pmidx.)
# 
# The largest datafile for CORD-19 is "/C/CO/Coronavirus Infections.pmidx". It is 9.6MB and there are 2485 items in the file. Most index files are much smaller than this.
# 
# Writing .pmidx files is implemented a single-pass operation. The tallies of article counts and children are not available until the end of processing, and these are written to a small index file in tab-separated-value format. If the content of this output file has changed compared to previous runs, the file needs to be copied to pubmed.tally and the operation must be re-run. (If this is not done, the article counts displayed at MeSH subject links in the browser will be wrong.) 

# # Process to responding to CORD-19 Task Questions
# The specific responses are created by doing a term search on the AB (Abstract) and TI (Title) fields of the MEDLINE metadata. Devising search terms is still a manual process. Once the terms are determined, there is no web interface to do this search. (The web interface is for showing results is limited to one MeSH descriptor at a time.)
# 
# A javascript program (searchPubmed3.js, run using NodeJS) loads in the MEDLINE concatenated dataset and creates an array of objects in memory. The stdin stream is read for lines of search terms. As each line is read in, a linear scan of the objects is performed. Search results are written out to the console, grouping results by most common to least common primary MeSH headings. In this way, results are displayed next to adjacent results. As discussed in Pros and Cons, date range limits are very important.
# 
# The results are formatted with deep links to PubMed (for articles) and the MeSH results explorer (for MeSH descriptors.)
# 
# The process to confirm that a result is trustworthy requires that the user be diligent to review the formatted results in the context of other articles in the same subject area, spending some time with the results browser to verify that there was nothing missed in the subject, and considering what related subjects would also be candidate places to look. 

# 
