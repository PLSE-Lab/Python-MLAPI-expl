#!/usr/bin/env python
# coding: utf-8

# Larning about `Plotnine` (a Python 3 implementation of R's `ggplot2`) has been alot of fun! Though there are several good guides, I though I'd put a series of my own together  to walk through the different plot types available. This is section is in 3D plotting and facetting, and it covers path plots, spoke plots, and various methods of facetting.

# For more information on `Plotnine`, check out their API at https://plotnine.readthedocs.io/en/stable/api.html
# 
# As it's (essentially) a direct port of `ggplot2`, you can also check out the ggplot2 API at https://ggplot2.tidyverse.org/reference/index.html

# In[ ]:


import numpy as np
import pandas as pd
import plotnine as pn #To avoid universal import
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.Generation = data.Generation.astype("object")
data.head()


# Frequently, 3D plotting involves the relationship between two variables over time. Perhaps the easiest way to do this a `pn.geom_path` plot. This plots a variable on each axis, then connects them based on their order in time, producing a path.

# In[ ]:


dataHold = data.groupby(["Generation"]).mean().reset_index()
(pn.ggplot(dataHold)
 + pn.aes(x="Sp. Atk", y="Sp. Def", color="Generation", label="Generation")
 + pn.geom_path(size=1)
 + pn.geom_point(color="black")
 + pn.geom_text(color="black",
                size=10,
                nudge_y=0.4
               )
 + pn.scale_x_continuous(limits=(65, 78),
                         breaks=np.arange(65, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(67.5, 78),
                         breaks=np.arange(67.5, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      )
           )
 + pn.labs(x="Base Sp. Attack",
           y="Base Sp. Defense",
           color="Gen."
          )
 + pn.ggtitle("Relation between Sp. Attack and Sp. Defense over Time")
)


# On this plot, I also added and labelled the points. But some of the labels are on top of lines or other points. One way to fix this is to specify subsets for labelling.

# In[ ]:


dataHold = data.groupby(["Generation"]).mean().reset_index()
dataHold = data.groupby(["Generation"]).mean().reset_index()
(pn.ggplot(dataHold)
 + pn.aes(x="Sp. Atk", y="Sp. Def", color="Generation", label="Generation")
 + pn.geom_path(size=1)
 + pn.geom_point(color="black")
 + pn.geom_text(data=dataHold.iloc[[0, 1, 3, 5], :],
                color="black",
                size=10,
                nudge_y=0.4
               )
  + pn.geom_text(data=dataHold.iloc[[2, 4], :],
                color="black",
                size=10,
                nudge_y=-0.4
               )
 + pn.scale_x_continuous(limits=(65, 78),
                         breaks=np.arange(65, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(67.5, 78),
                         breaks=np.arange(67.5, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      )
           )
 + pn.labs(x="Base Sp. Attack",
           y="Base Sp. Defense",
           color="Gen."
          )
 + pn.ggtitle("Relation between Sp. Attack and Sp. Defense over Time")
)


# This demonstrates another strength of `Plotnine`. While we can specify within `pn.ggplot()` and a single call of `pn.aes()` what data set and variables to use, at any time we call these inside any `pn.geom` to plot a unique set for that data.

# Another useful plot for 3D plotting is a `pn.geom_contour`, similar to `pn.geom_density_2d` . This uses contour curves (like a map) to illustrate the relations between 3 variables. Unfortunately, `Plotnine` has no plans to support this feature (see https://github.com/has2k1/plotnine/issues/110).

# Another type of three dimensional plot is known as a spoke plot. If you've studied pysics or engineering, this will look very familiar to you as they are essentially depictions of vector fields! They are good for plotting data where each position is associeated with a direction and magnitude.

# In[ ]:


"""
Data copied from the ggplot2 documentation (https://ggplot2.tidyverse.org/reference/geom_spoke.html)
The pokemon data isn't ideal for this, so lets create a dataframe!
"""
dataHold = pd.DataFrame({"angle": np.random.uniform(0, 2 * np.pi, 121),
                         "speed": np.random.uniform(0, 0.5, size=121)},
                        index=pd.MultiIndex.from_product([range(11), range(11)], names=["X", "Y"])
                       ).reset_index()
dataHold.head()


# In[ ]:


(pn.ggplot(dataHold)
 + pn.aes(x="X", y="Y", angle="angle", radius="speed")
 + pn.geom_point(size=0.5,
                 color="white"
                )
 + pn.geom_spoke(pn.aes(color="speed"), 
                 size=1
                )
 + pn.scale_x_continuous(limits=(0, 10),
                         breaks=np.arange(0, 11, 2),
                         expand=(0, 0.5)
                        )
 + pn.scale_y_continuous(limits=(0, 10),
                         breaks=np.arange(0, 11, 2),
                         expand=(0, 0.5)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(size=0.25,
                                       color="white"
                                      ),
            figure_size=(10, 10)
           ) 
 +pn.labs(color="Speed")
 + pn.scale_color_gradient2(low="yellow", mid="orange", high="red", midpoint=0.25)
 + pn.ggtitle("An Arbitrary Vector Space")
)


# This could be considered 4D space as each (x, y) point contains a direction and magnitude; however, it's also equally valid to think of this as each (x, y) point containing a single vector. Either works!

# Another way to break down relations between data is to facetting. In facetting, rather than having each variable be a different color(eg), you have each variable be a different graph!

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", color="Legendary")
 + pn.geom_density(size=1, 
                   trim=False
                  )
 + pn.facet_wrap("~ Generation")
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=np.arange(0, 201, 50),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 0.02),
                         breaks=np.arange(0, 0.021, 0.005),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      ),
            figure_size=(12, 6)
           )  
 + pn.labs(y="Density")
 + pn.ggtitle("Base Attack Stat for Legendary and Non-Legendary\n Pokemon Across Generations\n")
)


# The `pn.facet_wrap` allows us to create multiple graphs for a single stat. The "~" is important. It indicates which stat we are breaking the graphs by and MUST be included. But what if we wanted to break it up further? 

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack")
 + pn.geom_density(size=1, 
                   trim=False,
                   color="darkslateblue"
                  )
 + pn.facet_grid("Generation ~ Legendary")
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=np.arange(0, 201, 50),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 0.02),
                         breaks=np.arange(0, 0.021, 0.005),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      ),
            figure_size=(6, 12)
           )  
 + pn.labs(y="Density")
 + pn.ggtitle("Base Attack Stat for Legendary and Non-Legendary\n Pokemon Across Generations\n")
)


# In this case, use `pn.facet_grid`. Here, the data to be graphed along the row comes before the "~" and the data to be along the columns after. The power is that this type of facetting can be used for any type of plot!

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Defense", y="Sp. Def")
 + pn.geom_jitter(pn.aes(color="Type 1"),
                  size=1
                 )
 + pn.geom_smooth(size=0.5,
                  color="white",
                  method="lm",
                  fullrange=False,
                  se=False
                 )
 + pn.facet_grid("Generation ~ Legendary")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(size=0.25,
                                       color="white"
                                      ),
            figure_size=(8, 12)
           )  
 + pn.ggtitle("Correlation bettween Defense and Sp. Defense\n by Generation and LEgendary Status\n")
)


# That finish section 3! Stay tuned for section 4, Advanced Plotting, where I try an example to push what is possible with `Plotnine`.

# In[ ]:




