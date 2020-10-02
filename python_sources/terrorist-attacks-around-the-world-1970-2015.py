---
title: "Terrorist attacks around the world"
author: "Pranav Pandya | s_pandya@stud.hwr-berlin.de"
date: "10 December 2016"
output: html_document
---

#####Update (3rd March 2017) : 
#####1. With an increased RAM, now I am able to plot 152253 (97.11 %) out of 156772  terrorist attacks on a map with street level zoom. The script runs fine and successfull on R studio and Kaggle IDE both but somehow it's not loading the map. Will try to fix the issue.  
#####2. The one below is still the old script (Total attacks plotted on the map are 45466 out of 156772)


```{r, message=FALSE, warning=FALSE}
library(leaflet)
library(dplyr)

GT <- read.csv("../input/globalterrorismdb_0616dist.csv")
GT01= GT[,c("iyear", "city", "country_txt", "latitude","longitude", "attacktype1_txt", "targtype1_txt", "targsubtype1_txt", 
               "target1", "weaptype1_txt","weapsubtype1_txt", "gname", "motive", "summary")]

```


```{r, message= FALSE, warning = FALSE}
# Ommiting blanks and NAs
GT01[GT01==""] <- NA
GT01 = na.omit(GT01)

mymap <- 
  leaflet() %>% 
  addTiles('http://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
  attribution='Map tiles by 
    <a href="http://stamen.com">Stamen Design</a>, 
    <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> 
    &mdash; 
    Map data &copy; 
    <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>') %>%
  setView(15, 40, zoom= 2)
```


####An interactive map to summarize Terrorist attacks around the world between 1970 to 2015.

####Instructions:
#####1.) Zoom in to view the exact location of terror attack (up to street level anywhere in the world)
#####2.) Hold left click and then drag (if screen freezes while dragging the map)
#####3.) Click on any red point for specific details about that terror attack
#####3.) If Chrome doesn't load this map properly or hangs then try to increase Chrome's cache memory
```{r, message=FALSE, warning=FALSE}

mymap %>% addCircles (data=GT01, lat= ~latitude, lng = ~longitude, 
              popup=paste(
                "<strong>Year: </strong>", GT01$iyear,
                "<br><strong>City: </strong>", GT01$city, 
                "<br><strong>Country: </strong>", GT01$country_txt, 
                "<br><strong>Attack type: </strong>", GT01$attacktype1_txt, 
                "<br><strong>Target: </strong>", GT01$targtype1_txt, 
                " | ", GT01$targsubtype1_txt, 
                " | ", GT01$target1, 
                "<br><strong>Weapon: </strong>", GT01$weaptype1_txt, 
                "<br><strong>Group: </strong>", GT01$gname, 
                "<br><strong>Motive: </strong>", GT01$motive, 
                "<br><strong>Summary: </strong>", GT01$summary),
              weight = 0.8, color="#8B1A1A", stroke = TRUE, fillOpacity = 0.6)
```


```{r, message=FALSE, warning=FALSE}
#
```