---
title: "Fishing Vessels Data from port landings in UK 2008-2015"
author: "Brennon Borbon"
date: "September 19, 2017"
output: html_document
---

## Introduction

This is an Exploratory Data Analysis on fishing vessels data from 2008 - 2015 port landings in UK.

A quick definition on fish species.

Pelagic fish live in the pelagic zone of ocean or lake waters – being neither close to the bottom nor near the shore – in contrast with demersal fish, which do live on or near the bottom, and reef fish, which are associated with coral reefs.

Shellfish examples are shells, oysters, clam and scallops. 

Updated : 2017-12-08. Updated with Blue theme.

Updated: 2018-06-10. Updated with two new graphs by Tim Cashion


```{r ,echo=FALSE, message=FALSE}
library(tidyverse)
library(magrittr)
library(data.table)
library(RColorBrewer)


```

```{r , message=FALSE}
df <- read.csv('../input/UK_fleet_landings.csv')
```

```{r, echo=FALSE}
# Function to define the theme use across all the plots in the file.

bluetheme <- theme(axis.text.x=element_text(angle =90, size=8, vjust = 0.4),
                  plot.title=element_text(size=16, vjust = 2,family = "Avenir Book", face = "bold", margin = margin(b = 20)),
                  axis.title.y = element_text(margin = margin(r = 20)),
                  axis.title.x =element_text(size=12, vjust = -0.35, margin = margin(t = 20)),
                  plot.background = element_rect(fill = "#DEEBF7"),
                  panel.background = element_rect(fill = "#DEEBF7" ),
                  legend.background = element_rect(fill = "#DEEBF7"),
                  legend.title = element_text(size = 10, family = "Avenir Book", face = "bold"),
                  legend.text = element_text(size = 8, family = "Avenir Book"),
                  panel.grid.major = element_line(size = 0.4, linetype = "solid", color = "#cccccc"),
                  panel.grid.minor = element_line(size = 0),
                  axis.ticks = element_blank(),
                  plot.margin = unit(c(0.5, 1, 1, 1), "cm"))

colors = c("#9E0142", "#D53E4F" ,"#F46D43", "#FDAE61", "#FEE08B" ,"#FFFFBF", "#E6F598",
"#ABDDA4", "#66C2A5", "#3288BD" ,"#5E4FA2", "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854" ,"#FFD92F")


```

```{r , echo=FALSE}
set.seed(88)

cols <- c("port_of_landing", "port_nationality", "vessel_nationality", "length_group",
          "gear_category", "species_code", "species_name", "species", "species_group")

df %<>% mutate_at(cols, funs(factor(.)))

```

```{r, echo = FALSE, message = FALSE}
df$species_group[df$species_group==""] <- "NA"
#summary(df)
```
```{r, echo=FALSE, message = FALSE}
df <- df[complete.cases(df), ]
#summary(df)
glimpse(df)
```


```{r, fig.width=8}
df %>%
  group_by(year) %>%
  summarise(Landed_WtMedian = median(landed_weight, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(year = year, Landed_WtMedian)%>%
  arrange(desc(Landed_WtMedian)) %>%
  
  ggplot(aes(x = year,y = Landed_WtMedian)) +
  geom_bar(stat='identity', fill = "#3288BD") +
  labs(x = '', 
       y = 'Median Weight Landed', 
       title = 'Record of Median Landed Weight Since 2008') + bluetheme
```

```{r, fig.width = 8}
df %>%
  group_by(year) %>%
  summarise(Live_WtMedian = median(live_weight, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(year = year, Live_WtMedian) %>%
  arrange(desc(Live_WtMedian)) %>%
  
  ggplot(aes(x = year,y = Live_WtMedian)) +
  geom_bar(stat='identity', fill = "#3288BD") +
  labs(x = '', 
       y = 'Median Live Weight Landed', 
       title = 'Record of Median Live Weight Since 2008') + bluetheme
```

```{r, fig.width=8}
df %>%
  group_by(year) %>%
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(year = year, Value_gbpMedian) %>%
  arrange(desc(Value_gbpMedian)) %>%
  
  ggplot(aes(x = year,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#3288BD") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Record of Median GBP Value Since 2008') + bluetheme
```

Value gbp of species 

```{r, fig.width=8}
df %>%
  group_by(species_group) %>% 
  filter(!species_group == "NA") %>% 
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(species_group = reorder(species_group, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>%
  
  ggplot(aes(x = species_group,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#3288BD") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Value GBP of Each Species') + bluetheme
```


```{r, message=FALSE, warning=FALSE, fig.width=8}

df %>%
  group_by(vessel_nationality, length_group) %>% 
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(vessel_nationality = reorder(vessel_nationality, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>% head(10) %>%
  
  ggplot(aes(x = vessel_nationality,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#D53E4F") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Vessel Nationalities Vs Value Gbp') + bluetheme + coord_flip() + facet_wrap(~length_group)

```

```{r, message=FALSE, warning=FALSE, fig.width=8}

# Top 20 vessel nationality who makes most valuegbp on their catches

df %>%
  group_by(vessel_nationality, gear_category) %>% filter(!gear_category == "Unknown") %>%
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(vessel_nationality = reorder(vessel_nationality, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>% head(10) %>%
  
  ggplot(aes(x = vessel_nationality,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#D53E4F") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Vessel Nationalities Vs Value Gbp') + bluetheme + coord_flip() + facet_wrap(~gear_category)

```

```{r, message=FALSE, warning=FALSE, fig.width=8}

df %>%
  group_by(port_of_landing) %>% 
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(port_of_landing = reorder(port_of_landing, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>% head(10) %>%
  
  ggplot(aes(x = port_of_landing,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#D53E4F") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Port of Landing Vs Value Gbp') + bluetheme + coord_flip()

```


```{r, message=FALSE, warning=FALSE, fig.width=8}

df %>%
  group_by(port_nationality) %>%
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(port_nationality = reorder(port_nationality, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>% head(10) %>%
  
  ggplot(aes(x = port_nationality,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#D53E4F") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Port Nationality Vs Value Gbp') + bluetheme + coord_flip()

```

```{r, message=FALSE, warning=FALSE, fig.width=8}

df %>%
  group_by(gear_category) %>% filter(!gear_category == "Unknown") %>%
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(gear_category = reorder(gear_category, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>% head(10) %>%
  
  ggplot(aes(x = gear_category,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#F46D43") +
  labs(x = '', 
       y = 'Median Value GBP', 
       title = 'Gear Category Vs Value Gbp') + bluetheme 

```

```{r, message=FALSE, warning=FALSE, fig.width=8}

df %>%
  group_by(species) %>%
  summarise(Value_gbpMedian = median(value_gbp, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(species = reorder(species, Value_gbpMedian)) %>%
  arrange(desc(Value_gbpMedian)) %>% head(10) %>%
  
  ggplot(aes(x = species ,y = Value_gbpMedian)) +
  geom_bar(stat='identity', fill = "#F46D43") +
  labs(x = '', 
       y = 'Median Value Gbp', 
       title = 'Species Vs Value Gbp') + bluetheme 

```
Worth looking into price by species and gear type as well

```{r, message=FALSE, warning=FALSE, fig.width=8}
df <- df %>% 
    mutate(price = value_gbp/landed_weight) %>%
    as.data.frame()

df %>%
  group_by(species) %>%
  summarise(Price_gbpMedian = median(price, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(species = reorder(species, Price_gbpMedian)) %>%
  arrange(desc(Price_gbpMedian)) %>% head(10) %>%
  ggplot(aes(x = species ,y = Price_gbpMedian)) +
  geom_bar(stat='identity', fill = "#F46D43") +
  labs(x = '', 
       y = 'Price (Gbp)', 
       title = 'Species Vs Price Gbp') + bluetheme 

df %>%
  group_by(gear_category) %>%
  summarise(Price_gbpMedian = median(price, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(gear_category = reorder(gear_category, Price_gbpMedian)) %>%
  arrange(desc(Price_gbpMedian)) %>% head(10) %>%
  ggplot(aes(x = gear_category ,y = Price_gbpMedian)) +
  geom_bar(stat='identity', fill = "#F46D43") +
  labs(x = '', 
       y = 'Price (Gbp)', 
       title = 'Gear category Vs Price Gbp') + bluetheme 

```

Still a work in-progress...

