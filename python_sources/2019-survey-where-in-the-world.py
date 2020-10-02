#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to view where in the world people is using a programming language, or how the yearly compensation is distributed.
# 
# The values are relative to the number of respondents of that country.

# In[ ]:


# !pip install nb_black
# %load_ext nb_black
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.core.display import display, HTML
from ipywidgets import widgets

pio.templates.default = "plotly_white"

# import data
data_path = "../input/kaggle-survey-2019/"
multiple_choice_responses_19 = pd.read_csv(
    data_path + "multiple_choice_responses.csv", engine="c", low_memory=False
)
questions_only_19 = pd.read_csv(data_path + "questions_only.csv", engine="c")


# In[ ]:


# TABLE OF CONTENTS

# Iterate questions and add links
# The questions table is transposed and the first row discarted
questionnaire = "<h1>Table of contents</h1><br>"
for index, value in questions_only_19.T[1:].iterrows():
    questionnaire += f'<a href="#{index}" style="text-decoration: none;"><strong>{index}</strong>: {value[0]}</a><br>\n'
# Display the HTML content
display(HTML(questionnaire))


# # Number of respondents
# 
# 

# In[ ]:


# HTML tag to be linked in TOC
display(HTML('<span id="Q3"></span>'))

# Table with countries and respondents, ordered descending.
# It's necessary later to make the data proportional to the number of the respondents of the country.
countries = (
    multiple_choice_responses_19["Q3"][1:]
    .value_counts()
    .to_frame(name="count")
    .rename_axis("country")
    .reset_index()
)

# Map of respondents per country
fig = px.choropleth(
    countries,
    locations="country",
    locationmode="country names",
    color="count",
    hover_name="country",
    color_continuous_scale=px.colors.carto.Bluyl,
)
fig.show()


# In[ ]:


display(HTML(f'Five countries have more than the half of the respondents             ({countries[0:5].sum().values[1] / countries.sum().values[1]:.2%}):<br>             {countries[0:5].to_html()}'))


# # Questions

# In[ ]:


def where(question):
    """
    This function gets the data for the question, processes it and plot a Choropleth map
    
    :param question: Question to plot
    :return: None
    """    
    # Skip the Question 3 'In which country do you currently reside?'
    if question == "Q3":
        return

    # HTML tag to be linked in TOC
    display(HTML(f'<span id="{question}"></span>'))
    # Plot title
    title = f"<b>{question}</b>: {questions_only_19[question].values[0]}"
    
    # NORMAL QUESTION
    # ---------------    
    # If the question is single choice, it is in the multiple_choice_responses columns
    
    if question in multiple_choice_responses_19.columns:
        # Get the country and question #n columns of the multiple_choice_responses_19 table
        # and skip the first row
        m = multiple_choice_responses_19[["Q3", question]][1:]
        # Create a count column
        m["count"] = 1
        # Do the magic
        p = (
            m.groupby(["Q3", question]) # Group the country - response combination
            .agg(np.sum) # Get the sum
            .reset_index() # Plot.ly needs normal columns
            .rename(columns={"Q3": "country"}) # Rename country column
        )

    # MULTIPLE CHOICE QUESTION
    # ------------------------
    # If the question is multiple choice, the question is followed by _Part_
    # Get all columns with _Part_

    else:
        m = multiple_choice_responses_19[1:].filter(
            like=f"{question}_Part_", axis="columns"
        )
        # Get the relation of the part number and the most frequent value
        parts = {v + 1: k for v, k in enumerate(m.mode().values[0].tolist())}
        # Add the country column 
        m["country"] = multiple_choice_responses_19["Q3"][1:]
        # Group the table by country and do the counting
        g = m.groupby(["country"]).agg("count").reset_index()
        # Convert a wide table to a long table using _Part_ columns
        p = (
            pd.wide_to_long(g, stubnames=f"{question}_Part_", i="country", j="part")
            .reset_index()
            .rename(columns={f"{question}_Part_": "count"})
        )
        # Create a new column with the part most frequent value instead of the part number
        p[question] = p["part"].map(parts)

    # COMMON
    # ------
    # Instead of the value, get the percentage for each country
    
    p["percentage"] = p.apply(
        lambda x: x["count"]
        / countries[countries["country"] == x["country"]]["count"].values[0],
        axis="columns",
    )

    # Map of question #n per country
    fig = px.choropleth(
        p,
        title=title,
        locations="country",
        locationmode="country names",
        color="percentage",
        animation_frame=question,
        color_continuous_scale=px.colors.carto.Bluyl,
    )
    # Set the percentage format and the height
    fig.update_layout(
        coloraxis=dict(colorbar=dict(tickformat=",.0%")), height=600,
    )
    fig.show()


# In[ ]:


# Plot all questions
for i in questions_only_19.columns[1:].to_list():
    where(i)

