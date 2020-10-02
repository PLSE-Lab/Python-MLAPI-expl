from ggplot import *

p = ggplot(aes(x='date', y='beef'), data=meat) +\
    geom_line() +\
    stat_smooth(colour='blue', span=0.2)

ggsave(p, "myplot.png")
