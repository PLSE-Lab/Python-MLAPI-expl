#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

pd.set_option("max_colwidth", 500)


# In[ ]:


train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
train = train.dropna()
train.head()


# In[ ]:


def highlight_selected_text(row):
    text = row["text"]
    selected_text = row["selected_text"]
    sentiment = row["sentiment"]

    color = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue",
    }[sentiment]

    highlighted = f'<span style="color: {color}; font-weight: bold">{selected_text}</span>'
    return text.replace(selected_text, highlighted)


# In[ ]:


train["highlighted"] = train.apply(highlight_selected_text, axis=1)
train.head()


# In[ ]:


from IPython.display import display, HTML

display(HTML(train.sample(30).to_html(escape=False)))


# # Positive

# In[ ]:


display(HTML(train[train["sentiment"] == "positive"][["highlighted"]].sample(30).to_html(escape=False)))


# # Negative

# In[ ]:


display(HTML(train[train["sentiment"] == "negative"][["highlighted"]].sample(30).to_html(escape=False)))


# # Neutral

# In[ ]:


display(HTML(train[train["sentiment"] == "neutral"][["highlighted"]].sample(30).to_html(escape=False)))


# In[ ]:




