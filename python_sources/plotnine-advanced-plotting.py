#!/usr/bin/env python
# coding: utf-8

# Welcome to what is currently the last section of tutorials/notebooks on `Plotnine`, the Python port of R's `ggplot2`.
# 
# For more information on `Plotnine`, visit the API at https://plotnine.readthedocs.io/en/stable/api.html.
# 
# Since it is (essentially) a direct port of `ggplot2`, their API can also be very useful. It's at https://ggplot2.tidyverse.org/reference/.

# As I've said before, the power of a grammar of graphics system is the ease from which you can create beautifully complex graphs with rather straightforward, intuitive code. In this section, I'm going to examine the Olympic History data set to see how far I can push the abilities of Plotnine.

# In[ ]:


import pandas as pd
import numpy as np
import plotnine as pn
from itertools import chain
from matplotlib import pyplot as plt
data = pd.read_csv("../input/athlete_events.csv", index_col=0)
data.head(10)


# Have to love that tug-of-war used to be an Olympic sport!
# 
# I would like to create a graph that tracks the top 10 countries by total medals at each olympics since 1960, similar to this one (http://vis.berkeley.edu/courses/cs294-10-fa07/wiki/images/4/45/RowingbumpsSmall.jpg) that tracks the position of the college rowing clubs at Oxford.

# In[ ]:


#weight medals
data.Medal[data.Medal.isna()] = 0
data.Medal[data.Medal == "Gold"] = 1
data.Medal[data.Medal == "Silver"] = 0.667
data.Medal[data.Medal == "Bronze"] = 0.333
#To avoid confusions on countries (see row 4 Team/NOC above), I have eleceted too use the NOC only for identify countries
#groupby country and region
dataClean = (data
             .pipe(lambda df: df[df.Season == "Summer"])
             .pipe(lambda df: df[df.Year >=1960])
             #sum all medals won by each country each year
             .groupby(["NOC", "Year"])
             .Medal
             .sum()
             .reset_index()
             .set_index(["Year", "NOC"])
             .sort_index()
            )
"""
This is the most efficient way I could think of to retrieve the top 25 per year
For each year, it retrieves those not in the top 25 and drops them
"""
for year in dataClean.index.levels[0]:
    index = dataClean.loc[year, :].sort_values("Medal", ascending=False).tail(-10).index
    for row in index:
        dataClean = dataClean.drop(index=(year, row))
#Assign rank
newCol = list(chain.from_iterable([np.arange(10, 0, -1)] * int((len(dataClean.index) / 10))))
dataClean = (dataClean
             .reset_index()
             .sort_values(["Year", "Medal"])
             .assign(rank=newCol)
            )
#tidy index
dataClean.index = range(150)
dataClean.head(20)


# Clean data is always easier to work from! Now, we have a weighted sum of medals and a rank for the top 25 countries per year. I think the best way to track this likely going to be a line plot.

# In[ ]:


(pn.ggplot(dataClean)
 +pn.aes(x="Year", y="rank", color="NOC")
 +pn.geom_line()
)


# Well it's a start! Let's start by cleaning up the graph itself.

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC")
 + pn.geom_line()
 + pn.geom_point()
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0,1)
                        )
 + pn.scale_y_reverse(limits=(10, 1),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25)
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(figure_size=(12, 6))
)


# That's already much better! But now an interesting problems is visible...Note that in 1964, two countries seem to have rank 8! Is that an issue with the data or the graph?

# In[ ]:


dataClean[dataClean.Year.isin([1960, 1964, 1968])]


# A bit of both it seems! Yugoslavia was 8th in 1960, didn't rank in 1964, and was 8th in 1968, so the graph filled in the line between those data points. One option woud be to use `pn.geom_seqment` to specify every point, but that is far too wieldy. I think it would be better to go back to the data and add blanks from year to year.
# 

# In[ ]:


"""
This almost certainly isnt the best way to do this, but it was the solution I arrived
Remove first 10 rows.
Add rows for missing countries.
Add rows to bottom of original.
repeat 15 times as there are 15 olympics covered
"""
allNOC = list(dataClean.NOC.unique())
for i in range(15):
    hold = np.split(dataClean, [10], axis=0)
    for country in allNOC:
        if country not in list(hold[0].NOC):
            hold[0] = pd.concat([hold[0], pd.DataFrame(data={"Year": hold[0].iloc[0, 0],
                                                             "NOC": country,
                                                             "Medal": 0,
                                                             "rank": 11
                                                             }, 
                                                       index=range(1)
                                                      )
                                ])
    dataClean = pd.concat([hold[1], hold[0]])

dataClean = dataClean.sort_values(["Year", "rank"]).reset_index(drop=True)
dataClean.head(52)
    
    


# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC")
 + pn.geom_line()
 + pn.geom_point()
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0,1)
                        )
 + pn.scale_y_reverse(limits=(10, 1),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25)
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(figure_size=(12, 6))
)


# That's better! Now there's one point at each rank per year. But now lines that have values at 0 aren't shown. Also, the default color scale doesn't differentiate between colors enough, I think. Let's fix these...

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC")
 + pn.geom_line(size=1)
 + pn.geom_point()
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0,1)
                        )
 + pn.scale_y_reverse(limits=(11, 1),
                      breaks=range(11, 0, -1),
                      expand=(0, 0.25)
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="darkgray"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
)


# Hit and miss. By altering the colors slightly and moving onto a dark background with white grid lines, the colors can be made much more distinguishable. But actually including all of the lines makes the plot even more unreadable. Besides, given we are only interested in the top 10, 11 doesn't really work. Let's try adding some labels instead...

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC", label="NOC")
 + pn.geom_line(size=1)
 + pn.geom_point()
 + pn.geom_text(nudge_y=0.25)
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0, 2)
                        )
 + pn.scale_y_reverse(limits=(10, 0.5),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25)
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="darkgray"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
)


# A bit better, though too crowded. I think we need to reduce the labels to single points and end points only. Probably most easily achieved by selecting data and using multiple layers.

# In[ ]:


"""
I'm certain this could be done with some looped pandas commands to find all the endpoints
But I couldn't figure it out, so I manually selected
"""
dataCleanPoint = (dataClean
 .drop(range(10, 28, 1), axis=0)
 .drop(range(36, 56, 1), axis=0)
 .drop(range(62, 84, 1), axis=0)
 .drop(range(88, 106, 1), axis=0)
 .drop(range(114, 130, 1), axis=0)
 .drop(range(139, 156, 1), axis=0)
 .drop(range(166, 182, 1), axis=0)
 .drop(range(191, 208, 1), axis=0)
 .drop(range(218, 236, 1), axis=0)
 .drop(range(244, 266, 1), axis=0)
 .drop(range(270, 291, 1), axis=0)
 .drop(range(296, 317, 1), axis=0)
 .drop(range(322, 345, 1), axis=0)
 .drop(range(348, 364, 1), axis=0)
 .drop(range(374, 390, 1), axis=0)
 .drop([29, 30, 34, 108, 110, 111, 135, 158, 159, 164, 183, 189, 209, 213, 215, 237, 238, 240, 241, 267,  293, 318, 319],
       axis=0
      )
)
dataCleanPoint.head(26)


# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC", label="NOC")
 + pn.geom_line(size=1)
 + pn.geom_point()
 + pn.geom_text(data = dataCleanPoint,
                nudge_y=0.25)
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0, 2)
                        )
 + pn.scale_y_reverse(limits=(10, 0.5),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25)
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="darkgray"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
)


# We're getting there! Now that we have labels, we can hide the legend to reduce cluster. Also, bumping the text up doesn't seem to be the best option (it clashes with the lines). And I'd prefer the y-scale to be on both sides. Lets try something else.

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC", label="NOC")
 + pn.geom_line(size=1,
                show_legend=False
               )
 + pn.geom_point(show_legend=False)
 + pn.geom_label(data = dataCleanPoint,
                 fill="darkgray",
                 size=8,
                 show_legend=False
                )
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0, 2)
                        )
 + pn.scale_y_reverse(limits=(10, 1),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25),
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="darkgray"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
)


# Better! The `pn.geom_label()` adds the text in a little box. Thus, the box overlaps any lines prevents the text from being obscured. Frustratingly, there doesn't yet seem seem to be a good way to add a secondary y-axis with in `plotnine`. In ggplot2, it would look something like...

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC", label="NOC")
 + pn.geom_line(size=1,
                show_legend=False
               )
 + pn.geom_point(show_legend=False)
 + pn.geom_label(data = dataCleanPoint,
                 fill="darkgray",
                 size=8,
                 show_legend=False
                )
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0, 2)
                        )
 + pn.scale_y_reverse(limits=(10, 1),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25),
                      sec.axis=pn.dup_axis() # this would add second axis
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="darkgray"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
)


# Unfortunately, this produces an error. Reading the `plotnine` documentation, a `dup_axis()` function has not been implemented, and none of the y-axis scale funtions take a secondary axis command! At this point, that just leaves a few themables to tidy up - like text font and plot backgrounds and minor gridlines.

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC", label="NOC")
 + pn.geom_line(size=1,
                show_legend=False
               )
 + pn.geom_point(show_legend=False)
 + pn.geom_label(data = dataCleanPoint,
                 fill="#848482",
                 size=9,
                 label_size=0,
                 label_padding=0.1,
                 show_legend=False
                )
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0, 2),
                         minor_breaks=[]
                        )
 + pn.scale_y_reverse(limits=(10, 1),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25),
                      minor_breaks=[]
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="#848482"),
            panel_grid=pn.element_line(color="#E5E4E2",
                                       size=0.25
                                      ),
            figure_size=(12, 6),
            plot_background=pn.element_rect(fill="#E5E4E2"),
            text=pn.element_text(color="#0C090A"),
            axis_text=pn.element_text(style="oblique",
                                      size=9
                                     ),
            axis_title=pn.element_text(style="oblique",
                                      size=12
                                     ),
            plot_title=pn.element_text(size=16
                                      ),
            axis_ticks=pn.element_line(color="#848482"),
            dpi=300
           )
)


# The last feature I want to invoke is `pn.ggplot().save()` which lets us save figures!

# In[ ]:


(pn.ggplot(dataClean)
 + pn.aes(x="Year", y="rank", color="NOC", label="NOC")
 + pn.geom_line(size=1,
                show_legend=False
               )
 + pn.geom_point(show_legend=False)
 + pn.geom_label(data = dataCleanPoint,
                 fill="#848482",
                 size=9,
                 label_size=0,
                 label_padding=0.1,
                 show_legend=False
                )
 + pn.scale_x_continuous(limits=(1960, 2016),
                         breaks=range(1960, 2017, 4),
                         expand=(0, 2),
                         minor_breaks=[]
                        )
 + pn.scale_y_reverse(limits=(10, 1),
                      breaks=range(10, 0, -1),
                      expand=(0, 0.25),
                      minor_breaks=[]
                     )
 + pn.scale_color_hue(colorspace="husl",
                      h=.2,
                      l=.6,
                      s=1
                     )
 + pn.labs(x="Olympic Year", y="Medal Rank", color="Country")
 + pn.ggtitle("Ranking Countries by Total Olympic Medals")
 + pn.theme(panel_background=pn.element_rect(fill="#848482"),
            panel_grid=pn.element_line(color="#E5E4E2",
                                       size=0.25
                                      ),
            figure_size=(12, 6),
            plot_background=pn.element_rect(fill="#E5E4E2"),
            text=pn.element_text(color="#0C090A"),
            axis_text=pn.element_text(style="oblique",
                                      size=9
                                     ),
            axis_title=pn.element_text(style="oblique",
                                      size=12
                                     ),
            plot_title=pn.element_text(size=16
                                      ),
            axis_ticks=pn.element_line(color="#848482"),
            dpi=300
           )
).save(filename="OlympicMedals.png", verbose=True)


# That's all for now folks. As `plotnine` expands its features, and I get better at coding, there must some revisions or new sections. In any case, I hope you all found this as helpful as I did!

# In[ ]:




