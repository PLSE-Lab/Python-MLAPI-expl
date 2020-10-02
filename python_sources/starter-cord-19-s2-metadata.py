#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This notebook demonstrates how to begin working with the the CORD-19 S2 metadata. 
# 
# I use dask.bag in deference to your local RAM, but dask.bag seems to be a bit more pokey in Kaggle kernels.

# In[ ]:


import dask.bag as db
import itertools
import json
import numpy as np
import os
import pandas as pd
from statistics import mean, median
from tqdm.auto import tqdm


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npaper_metadata = db.read_text(\n    "../input/s2_article_metadata-2020-04-18/*.json", files_per_partition=2000\n).map(\n    json.loads\n)')


# In[ ]:


list(paper_metadata.take(1)[0].keys())


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nauthor_metadata = db.read_text(\n    "../input/s2_author_metadata-2020-04-18/*.json", files_per_partition=2000\n).map(\n    json.loads\n)')


# In[ ]:


list(author_metadata.take(1)[0].keys())


# In[ ]:


def get_semantic_author_influence(author_semantic_metadata: dict) -> dict:
    author_id = author_semantic_metadata.get("author_uid")
    author_influence = {}
    if not author_semantic_metadata.get("error"):
        author_influence[author_id] = {
            author_semantic_metadata["name"]: author_semantic_metadata[
                "influentialCitationCount"
            ]
        }
    else:
        author_influence[author_id] = "error"
    return author_influence


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nauthor_influence_list = author_metadata.map(get_semantic_author_influence).compute()\nauthor_influence = {\n    author_id: influence_values\n    for author in author_influence_list\n    for author_id, influence_values in author.items()\n}')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npaper_to_author_ids = paper_metadata.map(\n    lambda paper_semantic_metadata: {\n        paper_semantic_metadata.get("cord_uid"): [\n            author["authorId"] for author in paper_semantic_metadata.get("authors", [])\n        ]\n    }\n).compute()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npaper_to_author_names = paper_metadata.map(\n    lambda paper_semantic_metadata: {\n        paper_semantic_metadata.get("cord_uid"): [\n            author["name"] for author in paper_semantic_metadata.get("authors", [])\n        ]\n    }\n).compute()')


# In[ ]:


paper_author_influence = []
for paper_authors in paper_to_author_ids:
    for cord_uid, author_list in paper_authors.items():
        author_influence_dict = {}
        for author_id in author_list:
            author_name_citations = author_influence.get(author_id)
            if author_name_citations != "error":
                author_influence_dict.update(author_name_citations)
            else:
                author_influence_dict.update({author_id: "error"})
        paper_author_influence.append(
            {
                "cord_uid": cord_uid,
                "author_influential_citations": author_influence_dict
            }
        )


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npaper_influence = paper_metadata.map(\n    lambda paper: {\n        "cord_uid": paper.get("cord_uid", np.nan),\n        "title": paper.get("title", np.nan),\n        "venue": paper.get("venue", np.nan),\n        "year": paper.get("year", np.nan),\n        "citation_velocity": paper.get("citationVelocity", np.nan),\n        "influential_citation_count": paper.get("influentialCitationCount", np.nan),\n    }\n).compute()')


# In[ ]:


semantic_influence_df = (
    pd.DataFrame(paper_influence)
    .set_index("cord_uid")
    .join(pd.DataFrame(paper_author_influence).set_index("cord_uid"))
)


# Question for the reader: is the sum of authors' influential citations the right metric to attend to? Or is the average of each author's influential citations more important?

# In[ ]:


semantic_influence_df["author_influential_citations_percentile"] = (
    semantic_influence_df["author_influential_citations"]
    .apply(lambda x: sum([i for i in x.values() if i != "error"]))
    .rank(pct=True)
)


# One assumes that papers from authors with a relatively higher number of influential citations have more face validity to the research community.

# In[ ]:


semantic_influence_df.sort_values(by="author_influential_citations_percentile", ascending=False).head(50)


# Let's take a look at topic frequencies-- nb that there is a very long tail here and S2 surfaces some topics that may or may not be relevant.

# In[ ]:


topic_frequencies = sorted(
    paper_metadata.pluck("topics", [{"topic": None}])
    .flatten()
    .pluck("topic")
    .frequencies()
    .compute(),
    key=lambda x: x[1],
    reverse=True,
)


# Can we identify study design from the topics that S2 has extracted? These are the categories:
# 
# * "Meta analysis"
# * "Randomized control trial"
# * "Non-randomized trial"
# * "Prospective cohort"
# * "Retrospective cohort"
# * "Case control"
# * "Cross-sectional"
# * "Case study"
# * "Other"

# In[ ]:


study_design_categories = [
    "Meta analysis",
    "Randomized control trial",
    "Non-randomized trial",
    "Prospective cohort",
    "Retrospective cohort",
    "Case control",
    "Cross-sectional",
    "Case study",
    "Other",
]


# In[ ]:


study_design_keywords = list(
    itertools.chain.from_iterable(
        [design.lower().split() for design in study_design_categories]
    )
)


# In[ ]:


study_design_keywords


# In[ ]:


study_design_topics = []
for topic in topic_frequencies:
    for keyword in study_design_keywords:
        if topic not in study_design_topics and topic[0] and keyword in topic[0].lower().split():
            study_design_topics.append(topic)

study_design_topics


# Looks like there are some promising leads-- "Phase I/II/III Trial", "Control Groups", "Cross-Sectional Studies", etc!
# 
# One might train a text categorizer by:
# 1. Identifying those S2-extracted topics most aligned to each category in study_design_categories
# 2. Pulling those out for annotation/confirmation
# 3. Training a custom textcat model against them
# 
# We will have to contend with unbalanced categories, but it's a start.

# ## Conclusion
# Feel free to post in the discussion forum if you have any questions. To go forward from here, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
