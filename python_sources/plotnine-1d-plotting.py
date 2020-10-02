#!/usr/bin/env python
# coding: utf-8

# Larning about `Plotnine` (a Python 3 implementation of R's `ggplot2`) has been alot of fun! Though there are several good guides, I though I'd put a series of my own together  to walk through the different plot types available. This is section 1, and it covers 1D plotting - bar/column plots, histograms(as columns, lines, or density estimates), violin plots, and box and whisker plots.

# For more information on `Plotnine`, check out their API at https://plotnine.readthedocs.io/en/stable/api.html
# 
# As it's (essentially) a direct port of `ggplot2`, you can also check out the ggplot2 API at https://ggplot2.tidyverse.org/reference/index.html

# In[ ]:


import pandas as pd
import plotnine as pn #To avoid universal import
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.Generation = data.Generation.astype("object")
data.sample(10)


# Let's move on to bar charts. `Plotnine` specifies two types: `pn.geom_col` and `pn.geom_bar`. `pn.geom_bar` is equivalent to a value count, while `pn.geom_col` use values in your data for bar height. Let's look at some examples to clarify. What is the most common type?

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Type 1")
 + pn.geom_bar(fill="darkslateblue")
 + pn.xlab("Primary Type")
 + pn.ylab("Number of Pokemon")
 + pn.coord_flip()
 + pn.ggtitle("A Count of Pokemon Type")
 + pn.theme(figure_size=(12, 6))
)


# How about a `pn.geom_col` chart?

# In[ ]:


(pn.ggplot(data.groupby(["Type 1"]).Attack.mean().round(2).reset_index())
 + pn.aes(x="Type 1", y="Attack", label="Attack")
 + pn.geom_col(fill="gold")
 + pn.geom_text(ha="center",
                nudge_y=3,
                size=8
               )
 + pn.scale_y_continuous(limits=(0, 130),
                         breaks=(range(0, 131, 10)),
                         expand=(0,0)
                        )
 + pn.theme(axis_text_x=pn.element_text(rotation=45),
            figure_size=(12, 6))
 + pn.ggtitle("Average Base Attack by Primary Type")
 + pn.xlab("Primary Type")
)


# Here, the values plotted come from the data frame and are not counts of the data itself. This leads `pn.geom_col` to be better for plotting stats (say, the median of a data set) and `pn.geom_bar` to be better for counting things. You will also note that labels can be added simply by plotting both text and bars at the same time. Finally, the `pn.scale_y_continuous` command let's us control the spacing of the y axis.

# A related plot is the histogram. It also counts the number of occurences, but over a continuous x interval. It lets us see how data are laid out. How does the base attack stat look?

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", fill="Generation")
 + pn.geom_histogram(binwidth=5,
                     position=pn.position_stack(reverse=True)
                    )
 + pn.xlab("Base Attack")
 + pn.ylab("Number of Pokemon")
 + pn.scale_y_continuous(limits=(0, 71),
                         breaks=(range(0, 71, 10)),
                         expand=(0,0)
                        )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=(range(0, 201, 10)),
                         expand=(0,0)
                        )
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(figure_size=(12, 6),
            panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      )
           )
)


# You'll notice here that for `fill` I specified a data rather than a color. This is a nifty feature of `Plotnine`: pretty much any aspect of the plot can be set to represent data! This sometime makes plots hard to read. Another alternative that can help de-cluster is the `pn.geom_freqplot`.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", color="Generation")
 + pn.geom_freqpoly(binwidth=5,
                    size=1
                   )
 + pn.xlab("Base Attack")
 + pn.ylab("Number of Pokemon")
 + pn.scale_y_continuous(limits=(0, 21),
                         breaks=(range(0, 21, 5)),
                         expand=(0,0)
                        )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=(range(0, 201, 10)),
                         expand=(0,0)
                        )
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
)


# Still a bit tough to read. Another option is a smooth density estimate. It effectively uses a moving average to get rid of all these peaks.
# 

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Attack", color="Generation")
 + pn.geom_density(adjust=0.5, 
                   size=1
                  )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=range(0, 201, 10),
                         expand=(0, 0)
                        )
 + pn.xlab("Base Attack")
 + pn.ylab("Number of Pokemon")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black", 
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
 )


# Much more readable! However, all those overlapping lines are a pain. One option, when looking at multiple distribution is to use either a box and whisker plot (though sadly Schrodinger's cat isn't involved!) or a violin plot.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Generation", y="Attack")
 + pn.geom_boxplot(notch=True, 
                   varwidth=True,
                   fill="slateblue",
                   size=1
                  )
 + pn.scale_y_continuous(limits=(0, 201),
                         breaks=range(0, 201, 25),
                        )
 + pn.xlab("Generation")
 + pn.ylab("Base Attack")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
)


# Here, the outliers are shown as points, the range by the vertical black line, and the middle 50% by the box. A violin plot is similar, but rather than showing the box, it shows a smoothed density curve, much like the earlier distribution.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Generation", y="Attack")
 + pn.geom_violin(adjust=0.5, 
                  scale="count",
                  draw_quantiles=[0.25, 0.5, 0.75],
                  size=1,
                  fill="Gold"
                 )
 + pn.geom_jitter(color="black",
                 size=1,
                  width=0.2
                 )
 + pn.scale_y_continuous(limits=(0, 201),
                         breaks=range(0, 201, 25),
                        )
 + pn.xlab("Generation")
 + pn.ylab("Base Attack")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
 + pn.coord_flip()
)


# Here, I've also elected to show the poins in the distribution. But what if we wanted to subdivide this further? Perhaps we also want to know how the legendary status affects the distribution.

# In[ ]:


(pn.ggplot(data)
 + pn.aes(x="Generation", y="Attack", fill="Legendary")
 + pn.geom_violin(adjust=0.5, 
                  scale="count",
                  draw_quantiles=[0.25, 0.5, 0.75],
                  size=1,
                 )
 + pn.geom_jitter(color="black",
                 size=1,
                  width=0.2
                 )
 + pn.scale_y_continuous(limits=(0, 201),
                         breaks=range(0, 201, 25),
                        )
 + pn.xlab("Generation")
 + pn.ylab("Base Attack")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
)


# The very skinny graph tells us there are not many legendary Pokemon, but they do seem to have an above average attack. 

# Box and whisker plots and violin plots can seem like 2D (or even 3D in the last case) plots, as they take an x and y set; however, they are merely convenient ways of comparing multiple 1D distributions. Another option is facetting, but that is for a later section.

# Hope you found this useful! Stay tuned for the second section on 2D plots - scatter plots and 2D binning (heat maps and hex plots)!
