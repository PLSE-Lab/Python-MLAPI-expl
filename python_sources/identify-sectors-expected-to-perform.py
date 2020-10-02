#!/usr/bin/env python
# coding: utf-8

# # Identify sectors expected to perform well in near future
# > Find out beaten down sectors that are showing signs of reversal. 
# 
# Here I find out the sectors that are delivering diminishing returns i.e. returns are decreasing on lower time frames compared to higher time frames. The second criterion is to shortlist sectors that took a maximum beating recently.

# In[ ]:


from IPython.display import HTML
import pandas as pd

df = pd.read_csv("https://www1.nseindia.com/content/indices/mir.csv", header=None)

caption = df.iloc[0, 0]
df.columns = ["Sector", "1m", "3m", "6m", "12m"]
df = df[3:]
df.set_index("Sector", inplace=True)
df["1m"] = df["1m"].astype(float) / 100
df["3m"] = df["3m"].astype(float) / 100
df["6m"] = df["6m"].astype(float) / 100
df["12m"] = df["12m"].astype(float) / 100
df["diminishing_returns"] = False

mask_diminishing_returns = (
    (df["12m"] > df["6m"]) & (df["6m"] > df["3m"]) & (df["3m"] > df["1m"])
)
df.loc[mask_diminishing_returns, "diminishing_returns"] = True
df = df.sort_values(
    by=["diminishing_returns", "12m", "6m", "3m", "1m"], ascending=False
)


# In[ ]:


def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color


def hover(hover_color="#f0f0f0"):
    return dict(selector="tr:hover", props=[("background-color", "%s" % hover_color)])


styles = [
    hover(),
    dict(selector="th", props=[("font-size", "105%"), ("text-align", "left")]),
    dict(selector="caption", props=[("caption-side", "top")]),
]

format_dict = {
    "1m": "{:.2%}",
    "3m": "{:.2%}",
    "6m": "{:.2%}",
    "12m": "{:.2%}",
}

html = (
    df.style.format(format_dict)
    .set_table_styles(styles)
    .applymap(color_negative_red)
    .highlight_max(color="lightgreen")
    .set_caption(caption)
)


# In[ ]:


html


# Once you identify the beaten down sectors, you can check the stocks under those sectors. Both the sector and stocks should confirm the reversal.
# 
# As an investor, it is important to understand that there is a correlation between the economic cycle, stock market cycle and the performance of various sectors of the economy. 
# 
# During the early cycle, it is better to invest in interest-rate sensitive stocks like consumer discretionary, financials, real estate, industrial and transportation. You should avoid, communications, utilities, and energy sector stocks. 
# 
# During the middle of the cycle, you can invest in IT and capital goods stocks. Whereas you should avoid, metals and utilities during this phase. 
# 
# During the late cycle, you can invest in energy, metals, health care and the utilities and you can skip the IT and consumer discretionary stocks. 
# 
# Best sectors for investment during Economic Slowdown are FMCG, utilities and health care. Investment in Industrials, IT and Real Estate should be avoided during this time.  
# 
# ![Business cycle and relative stock performance](https://i.pinimg.com/originals/00/5c/bc/005cbc511e93c97318c4bfc95df4c38d.jpg)

# In[ ]:




