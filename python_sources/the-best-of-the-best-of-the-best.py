---
title: "The best of the best ... of the best"
author: "Jonathan Bouchet"
date: "`r Sys.Date()`"
output:
 html_document:
    fig_width: 10
    fig_height: 7
    toc: yes
    number_sections : yes
    code_folding: show
---

Let's settle this: who is the best player ?

<center><img src="https://i.ytimg.com/vi/bKyhc009cVU/maxresdefault.jpg"></center>

<hr>

<strong>History :</strong>

* _version 1 : initial commit_ 

<hr>

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE,message=FALSE,warning=FALSE)
```

```{r}
#load packages and csv file
library(ggplot2)
library(dplyr)
library(knitr)
library(DT)
library(kableExtra)
library(formattable)
```

# Code

Codes below is only to select the stats by `Primary Skill` and to style the `DT::datatable`

```{r}
df<-read.csv('../input/football.csv',sep=',',stringsAsFactors = F)
colnames(df) <- c('Primary Skill','Secondary Skill','Christiano Ronaldo',' Lionel Messi','Neymar')
```

```{r}
primSkills <- sort(unique(df[,1]))
skillList <- list()
cnt<-0
for(skill in primSkills){
  temp <- df %>% filter(`Primary Skill` == skill) %>% select(-starts_with("Primary"))
  rownames(temp)<- temp[,1]
  temp[,1]<-NULL
  cnt<-cnt+1
  skillList[[cnt]] <- temp
}
```

```{r,eval=T}
makeTable <- function(mydf, myskill){
  datatable(mydf, options = list(dom = 't', ordering=F), caption=paste0('Data for primary Skill: ', myskill)) %>% 
  formatStyle(names(mydf),
  background = styleColorBar(range(mydf), 'lightpink'),
  backgroundSize = '70% 50%',
  backgroundRepeat = 'no-repeat',
  backgroundPosition = 'right')
}
```

# Tables 

```{r,eval=T,echo=FALSE}
makeTable(skillList[[1]], primSkills[1])
```

```{r,eval=T,echo=FALSE}
makeTable(skillList[[2]], primSkills[2])
```

```{r,eval=T,echo=FALSE}
makeTable(skillList[[3]], primSkills[3])
```

```{r,eval=T,echo=FALSE}
makeTable(skillList[[4]], primSkills[4])
```

```{r,eval=T,echo=FALSE}
makeTable(skillList[[5]], primSkills[5])
```

```{r,eval=T,echo=FALSE}
makeTable(skillList[[6]], primSkills[6])
```

```{r,eval=T,echo=FALSE}
makeTable(skillList[[7]], primSkills[7])
```

# Summary

* Overall __Neymar__ seems the _less_ skilled player
* __Ronaldo__ is a true attacker, meaning he has strong _physical_ and _shooting_ skills needed to finish the action.
* __Messi__ is more of a midfield player where _passing_ skill as well as being able to navigate within the opponent's defence are needed.

# To do

We all know that the real question is to know who has the best hair cut though 
