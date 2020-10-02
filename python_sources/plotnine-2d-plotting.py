#!/usr/bin/env python
# coding: utf-8

# Larning about `Plotnine` (a Python 3 implementation of R's `ggplot2`) has been alot of fun! Though there are several good guides, I though I'd put a series of my own together  to walk through the different plot types available. This is section 2 on 2D plotting, and it covers scatter plots, 2D binning, and line plots.

# For more information on `Plotnine`, check out their API at https://plotnine.readthedocs.io/en/stable/api.html
# 
# As it's (essentially) a direct port of `ggplot2`, you can also check out the ggplot2 API at https://ggplot2.tidyverse.org/reference/index.html

# In[ ]:


import pandas as pd
import plotnine as pn #To avoid universal import
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.Generation = data.Generation.astype("object")
data.head()


# Let's start with a simple scatter plot. We will look at Attack and Defense.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense",)
 + pn.geom_point(size=0.9,
                  color="darkslateblue"
                 )
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      )
           )
)


# Not a bad first go. It's quite dense in the center - a major issue with using scatter plots on dense data sets is overplotting. This can be partly resolved by using a `pn.geom_jitter` plot. This moves the points a little when they overlap to try to prevent overplotting. 

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense", color="Generation")
 + pn.geom_jitter(size=0.9)
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8)
           )
)


# That's a bit better! But what if we wanted to label the points?

# In[ ]:


(pn.ggplot(data[data.Generation == 1]) #Downsampling
 + pn.aes(x="Attack", y="Defense", color="Type 1", label="Name")
 + pn.geom_text(size=8,
                #check_overlap=True
               )
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8)
           )
)


# Perhaps not the best because of how hard it is to read (NB: `ggplot geom_text` has a `check_overlap=True` parameter that won't plot overlapping text. This feature does not seem to work in `Plotnine` currently).

# Another use feature of scatter plots is the the ability to add lines of best fit! In `Plotnine` this simply means add another `geom`.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense", color="Generation")
 + pn.geom_jitter(size=0.9)
 + pn.geom_smooth(fullrange=False,
                  se=False, #hides confidence interval
                  method="lm" #to force linear fitting
                 )
 + pn.geom_rug(sides="tr")
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8)
           )
)


# A final nifty trick is the known as a `pn.geom_rug`. This feature adds tick marks to the ouside to give a sense of the 1D distributions and the 2D relations. Though these can get a little hard to read, so it's often better to either down sample or use them only on sparse sets.

# Let's move on to 2D binning, sometimes known as a heat map. This helps to counter overplotting by counting the number of occurences, rather than plotting each one. Let's look at the same data this way.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense")
 + pn.geom_bin2d(binwidth=10,
                 drop=False #to fill in all grey background
                )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 20),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 20),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
             panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ),
             figure_size=(8,8)
            )
 + pn.coord_fixed(ratio=1) #ensures square boxes
)


# These can also be used for categorical data. Let's look at the type breakdown...

# In[ ]:


(pn.ggplot(data.dropna()) #removes pokemon of only 1 type
 + pn.aes(x="Type 1", y="Type 2")
 + pn.geom_bin2d()
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(figure_size=(8,8),
            
           )
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8),
            axis_text=pn.element_text(rotation=45)
           )
 + pn.coord_fixed(ratio=1) #ensures square boxes
 + pn.scale_fill_cmap("RdPu")
 + pn.xlab("Primary Type")
 + pn.ylab("Secondary Type")
)


# The mature `ggplot2` library in `R` also contains a `geom_hex` that produces a hexplot. Unfortunately, `Plotnine` does not currently support this (to the best of my knowledge), though `Seaborn` definitely does.

# Another option to counter overplotting for 2D data is known as a 2D density, or countour, plot. It takes continuous variables on both axes, then calcultaes the density (number of occurences) at each point. The closer together the contours, the more closey clustered the data.
# 
# WARNING: Kaggle's notebook environment does not currently support the most recent version of `Plotnine` required for `pn.geom_density_2d`, so the following code will return an error. I'll update this text when Kaggle switches to `Plotnine` v0.4.0.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense")
 + pn.geom_jitter(size=1)
 + pn.geom_density_2d(color="darkslateblue",
                      size=1
                     )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=range(0, 201, 20),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 20),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
             panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ),
             figure_size=(8,8)
            )
 + pn.ggtitle("Attack and Defense Density Estimates")
)


# Another type of 2D plot is a line plot. These are particularly useful for plotting time dependent data! Let's see if the mean value for each stat has changed from generation to generation.

# In[ ]:


(pn.ggplot(pd.melt(data.groupby(["Generation"]).mean().reset_index(),
                   id_vars=["Generation"], value_vars=["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
                  )
          )
 + pn.aes(x="Generation", y="value", color="variable")
 + pn.geom_line()
 + pn.scale_x_discrete(limits=("1", "2", "3", "4", "5", "6"),
                       expand=(0,0))
 + pn.scale_y_continuous(limits=(60, 86),
                         breaks=range(60, 86, 5),
                         expand=(0, 0)
                        )
 + pn.labs(y="Base Stat Value",
           color="Stat"
          )
 + pn.ggtitle("Average Base Stats Across Generations")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8),
           )
)


# You probably noticed the data handling in the `pn.ggplot()`. Here's what that does...

# In[ ]:


pd.melt(data.groupby(["Generation"]).mean().reset_index(),
        id_vars=["Generation"], value_vars=["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
       )


# It converted the data to "long form." Effectively, after averaging agross generation, all the column heads were turned into variables along the same column. This allows us to plot them as different groups in the same layer! Neat, huh?

# Well that's it for this section! Join me next time as we explore 3D plotting and a few other nifty bits.

# In[ ]:




