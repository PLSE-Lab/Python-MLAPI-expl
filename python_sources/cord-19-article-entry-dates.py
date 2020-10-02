#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

from urllib.request import urlretrieve

import pandas as pd

# Create output directory
if not os.path.exists("/tmp/metadata"):
    os.mkdir("/tmp/metadata")

# Read list of dates from AI2 CORD-19 Releases page
dates = pd.read_html("https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html")[0]["Date"].tolist()

# Retrieve metadata files starting with 2020-03-27, when cord_uid was added
dates.remove("2020-03-13")
dates.remove("2020-03-20")

# Retrieve each file
for date in sorted(dates):
    url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/%s/metadata.csv" % date
    path = "/tmp/metadata/%s.csv" % date
    print("Retrieving %s to %s" % (url, path))
    urlretrieve(url, path)


# In[ ]:


import csv
import hashlib
import os
import re

import pandas as pd

def getId(row):
    # Use sha1 provided, if available
    uid = row["sha"].split("; ")[0] if row["sha"] else None

    if not uid:
        # Fallback to sha1 of title
        uid = hashlib.sha1(row["title"].encode("utf-8")).hexdigest()

    return uid

# Get sorted list of metadata csv files
directory = "/tmp/metadata"
files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and re.match(r"\d{4}-\d{2}-\d{2}\.csv", f)])

uids = {}

# Process each file, first time id is seen is considered entry date
for metadata in files:
    # Parse date from file name
    date = os.path.splitext(metadata)[0]
    with open(os.path.join(directory, metadata), mode="r") as csvfile:
        for row in csv.DictReader(csvfile):
            uid = getId(row)
            if uid not in uids:
                uids[uid] = (row["cord_uid"], uid, date)

# Build DataFrame
df = pd.DataFrame(uids.values(), columns=["cord_uid", "sha", "date"])
df.to_csv("entry-dates.csv", index=False)


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display

def plot(df, title, kind="bar", color="#bbddf5"):
    # Remove top and right border
    ax = plt.axes()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set axis color
    ax.spines['left'].set_color("#bdbdbd")
    ax.spines['bottom'].set_color("#bdbdbd")

    df.plot(ax=ax, title=title, kind=kind, color=color);

# Get count of records by day
day = df.date.value_counts().sort_index()

# Get count of reocrds by week
week = pd.to_datetime(df.date).value_counts().resample('W-Mon').sum().sort_index()

# Drop time component
week.index = week.index.strftime("%Y-%m-%d")

plot(week, None, color="#03a9f4")


# In[ ]:


day.to_frame(name="Articles")

