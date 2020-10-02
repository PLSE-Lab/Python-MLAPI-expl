---
title: "Know the Developers in 2018"
author: "Bukun"
output:
  html_document:
    number_sections: true
    toc: true
    fig_width: 10
    code_folding: hide
    fig_height: 4.5
    theme: cosmo
    highlight: tango
---


#Introduction

* More than **80%** of the respondents consider Coding as a Hobby.  

The total analysis is divided into a couple of subject areas such as 

* Desire to Learn in Next Year    

* Lifestyle Analysis 

* Salary Analysis

* AI Viewpoint Analysis      

* Job Assessment Analysis

* Job Benefits Analysis       

* Developer mentalities       

* Ethics analysis       

* StackOverflow Inclusivity analysis        

* Hypothetical Tools Interest     

* **Persona** analysis of Engineering Managers, C- Suite Execs, Devops Specialists, Front End Developers, Back End Developers , Data Scientists and Business Analysts      


The reader may read the summary here and proceed in detail for the graphs. The graphs here are simple but pragmatic.       


<hr/>

<b> Desire to Learn in Next Year </b>

<hr/>

* Linux. Android and AWS are the top **Platforms** which respondents want to learn next year         

* Node,React  and Angular are the top **Frameworks** which respondents want to learn next year         

* MySQL, MongoDB and PostgreSQL are the top **Databases** which respondents want to learn next year       

* JavaScript,Python, HTML are the top **Languages** which respondents want to learn next year        

* Visual Studio Code, Visual Studio and Notepad++ are the top **IDE** which respondents want to learn next year          

* Git, Subversion and Team Foundation Version Control are the top **Version Control**  tools that the respondents use.         

* Slack, Jira, Office productivity tools are the top **Communication** tools that the respondents use.           

* Agile, Scrum , Kanban are the top **Methodology** tools that respondents are using.       


<hr/>
**Lifestyle Analysis**       
<hr/> 

* 9-12 hours is the most popular (38%) time interval that developers spend in front of the computer.          

* 1-2 hours is the most popular (28%) time interval that developers hours outside.          

* Most devs (46%) never skip meals        

* Ergonomic Keyboard or mouse , Standing desk , both 10% of the population are the most popular ergonomic devices used by the developers.        

<hr/>
**Salary Analysis**       
<hr/> 

* Engineering Manager, DevOps Specialist, C Suite executive are the top positions which have the highest salaries.        

* United States, Switzerland, Israel, Norway and Denmark are the countries with the highest median salaries with respondents of 100 or more.          

* The highest median salary for `25-34 years` old is **95K**. US, Switzerland , Israel , Denmark and Australia are the top Five countries which has the highest salaries in this group. 

* The highest median salary for `35-44 years` old is **495K** followed by more acceptable **120K**. Venezuala, US, Switzerland , Israel , Denmark and Norway are the top Six countries which has the highest salaries in this group.           

* The highest median salary for `45-54 years` old is **1000K** followed by more acceptable **130K**. Venezuala, HongKong, Switzerland , US, Ireland , Israel , Denmark and UAE are the top Eight countries which has the highest salaries in this group.          
       
* Marketing or Sales Professional, C Suite Executives, Academic Researchers , Engineering Managers and Product Managers get the highest salary in USA. 

 
<hr/>
**AI Viewpoint Analysis**       
<hr/> 

According to the StackOverflow respondents

* AI is **interesting**  because it would result in increasing the automation of jobs ( 27%) , algorithms would make interesting decisions(15%)       

* AI **responsibility**  would be in the hands of people developing or creating AI (32% ), government or regulatory body (18%)                

* AI **future** views are Excited more about the possibilities than about the dangers ( 51%) , Worried about the dangers more than excited about the possibilities (13%)        

<hr/>
**Job Assessment**
<hr/>

The questions have been framed on different topics for developers on how they assess the job. The framework of the questions have 1 with the most important and 10 is the least important.         

Developors provide the **Most Importance**  to               

* The languages, frameworks, and other technologies                 

* The compensation and benefits          

* The office environment or company culture          

* Opportunities for professional development               

* The opportunity to work from home/remotely     

<hr/>
**Job Benefits Assessment**
<hr/>

The developers have assessed a **job benefits package** and provided ratings. ( 1- Most Important, 11 - Least Important).

The most important factors while assessing the job benefits package are

*  Salary and/or bonuses         

*  Stock options or shares       

*  Health insurance        

* Retirement or pension savings matching  

* Computer/office equipment allowance      

* Conference or education budget               

<hr/>
**Developer mentalities**
<hr/>

We explore the Developer mentalities here. 

* Developers have sense of kinship or connection to other developers    

* Developers agree they have a competition among peers       

* They also agree they are as good as other developers        


<hr/>
**Ethics analysis**
<hr/>

* Most developers will not write unethical code             

* Regarding reporting unethical code, developers would report after assessing what it is         

* The developers feel the ethics repsonsibility stands with the upper management at the company        

* The developers also feel that they are obligated to consider the ethical implications of the code that they write

<hr/>
**StackOverflow Inclusivity analysis**
<hr/>

* Most Developers would recommend StackOverflow to a friend or colleague            

* Most Developers visit StackOverflow almost daily           

* Most Developers have a StackOverflow account         

* Most Developers participate in Q&A less than once a month/monthly in  on Stack Overflow           

* Most Developers have visited StackOverflow Jobs       

* Most Developers are unaware of the Stack Overflow Developer Story feature         

* Most Developers would recommend Stack Overflow Jobs to a friend or colleague                        

* Most Developers consider themselves as  a member of the Stack Overflow community    

* Mobile Developer, Educator or Academic Researcher, C Suite Executive, Product Manager and Engineering manager are the DevTypes which show the most Stack Overflow Inclusivity.               

* Bangladesh, Pakistan , India , Iran and Israel are the countries with the most Stack Overflow inclusivity with 500 or more respondents.           

* 27-29 Years , 24 -26 Years , 6-8 Years , 18-20 Years and 9-11 Years Coding Professionals have the most Stack Overflow Inclusivity.              



<hr/>
**Hypothetical Tools Interest**
<hr/>

Most Developers show the following interests for the Hypothetical Tools

* Not At all Interested for A private area for people new to programming

* Somewhat Interested for A programming-oriented blog platform

* **Very Interested** for `An employer or job review system`

* Somewhat Interested for An area for Q&A related to career growth

#Read Data

```{r,message=FALSE,warning=FALSE}

library(tidyverse)
library(stringr)
library(scales)

library(tidytext)
library(igraph)
library(ggraph)

library(treemap)

rm(list=ls())

fillColor = "#FFA07A"
fillColor2 = "#F1C40F"
fillColorLightCoral = "#F08080"

survey_results <- read_csv("../input/survey_results_public.csv")

```


#Participation by Country

```{r,message=FALSE,warning=FALSE}

survey_country = survey_results %>%
  group_by(Country) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(Country = reorder(Country,Count)) %>%
  head(20) 


treemap(survey_country, 
        index="Country", 
        vSize = "Count",  
        title="Participation by Country", 
        palette = "RdBu",
        fontsize.title = 14 
)

```


#Hobbies Analysis

```{r,message=FALSE,warning=FALSE}

TotalNoofRows = nrow(survey_results)

survey_results %>%
  group_by(Hobby) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(Hobby = reorder(Hobby,Count)) %>%
  
  ggplot(aes(x = Hobby,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = Hobby, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Hobby', 
       y = 'Percentage', 
       title = 'Hobby and Percentage') +
  coord_flip() +
  theme_bw()

```

More than **80%** of the respondents consider Coding as a Hobby.                 


#Countries with Hobby as Yes

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(Hobby == "Yes") %>%
  group_by(Country) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(Country = reorder(Country,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = Country,y = Count)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = Country,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Country', 
       y = 'Percentage', 
       title = 'Country with Hobby as Coding and Percentage') +
  coord_flip() +
  theme_bw()


```

United States, India, Germany, United Kingdom and Canada are the countries with people who consider coding as an Hobby.          


#Developer Type Analysis

```{r,message=FALSE,warning=FALSE}

DevType <- survey_results %>%
  mutate(DevType = str_split(DevType, pattern = ";")) %>%
  unnest(DevType) %>%
  select(DevType)

TotalNoofRows = nrow(DevType)

DevType %>%
  group_by(DevType) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(DevType = reorder(DevType,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = DevType,y = Count)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = DevType, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'DevType', 
       y = 'Percentage', 
       title = 'DevType and Percentage') +
  coord_flip() +
  theme_bw()

```

The most popular DevTypes are the Back-end , Full Stack and the Front end developers.        


#Top Dev Types with Years of Coding

The bar plot lists the most popular Developer Types along with the Years of Coding           


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  select(DevType,YearsCoding) %>%
  mutate(DevType = str_split(DevType, pattern = ";")) %>%
  unnest(DevType) %>%
  group_by(DevType,YearsCoding) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(YearsCoding = as.character(YearsCoding),
         DevType = as.character(DevType)) %>%
  mutate(DevType_YearsOfCoding = paste(DevType,YearsCoding)) %>%
  mutate(DevType_YearsOfCoding = reorder(DevType_YearsOfCoding,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = DevType_YearsOfCoding,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = DevType_YearsOfCoding, y = 1, label = paste0("( ",Count," )",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'DevType_YearsOfCoding', 
       y='Count', 
       title = 'DevType_YearsOfCoding and Count') +
  coord_flip() +
  theme_bw()


```

#Education Analysis

##Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}

TotalNoofRows = nrow(survey_results) 

plotFormalEducation <- function(survey_results,TotalNoofRows) {
  survey_results %>%
    filter(!is.na(FormalEducation )) %>%
    select(FormalEducation ) %>%
    group_by(FormalEducation ) %>%
    summarise(Count = n()/TotalNoofRows ) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(FormalEducation  = reorder(FormalEducation ,Count)) %>%
    head(10) %>%
    
    ggplot(aes(x = FormalEducation ,y = Count)) +
    geom_bar(stat='identity',fill= fillColorLightCoral) +
    geom_text(aes(x = FormalEducation , y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    scale_y_continuous(labels = percent_format()) +
    labs(x = 'FormalEducation ', 
         y = 'Percentage', 
         title = 'FormalEducation  and Percentage') +
    coord_flip() +
    theme_bw()
}

plotFormalEducation(survey_results,TotalNoofRows)

```


##UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree <- function(survey_results,TotalNoofRows) {
  survey_results %>%
    filter(!is.na(UndergradMajor)) %>%
    select(UndergradMajor) %>%
    group_by(UndergradMajor) %>%
    summarise(Count = n()/TotalNoofRows ) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(UndergradMajor = reorder(UndergradMajor,Count)) %>%
    head(10) %>%
    
    ggplot(aes(x = UndergradMajor,y = Count)) +
    geom_bar(stat='identity',fill= fillColor2) +
    geom_text(aes(x = UndergradMajor,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    scale_y_continuous(labels = percent_format()) +
    labs(x = 'UndergradMajor', 
         y = 'Percentage', 
         title = 'UndergradMajor and Percentage') +
    coord_flip() +
    theme_bw()
}

plotUnderGradDegree(survey_results,TotalNoofRows)

```


#Developer Desires Next Year

##Platform Desire Next Year

```{r,message=FALSE,warning=FALSE}

TotalNoofRows = nrow(survey_results)

plotPlatformDesire <- function(survey_results,TotalNoofRows) {
  survey_results %>%
    filter(!is.na(PlatformDesireNextYear)) %>%
    select(PlatformDesireNextYear) %>%
    mutate(PlatformDesireNextYear = str_split(PlatformDesireNextYear, pattern = ";")) %>%
    unnest(PlatformDesireNextYear) %>%
    group_by(PlatformDesireNextYear) %>%
    summarise(Count = n()) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(TotalCount =  sum(Count)) %>%
    mutate(Count =  Count/TotalCount) %>%
    mutate(PlatformDesireNextYear = reorder(PlatformDesireNextYear,Count)) %>%
    head(10) %>%
    
    ggplot(aes(x = PlatformDesireNextYear,y = Count)) +
    geom_bar(stat='identity',fill= fillColor) +
    geom_text(aes(x = PlatformDesireNextYear,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
    labs(x = 'PlatformDesireNextYear', 
         y='Percentage', 
         title = 'PlatformDesireNextYear and Percentage') +
    coord_flip() +
    theme_bw()
}

plotPlatformDesire(survey_results,TotalNoofRows)

```

Linux. Android and AWS are the top Platforms which respondents want to learn next year

##Framework Desire Next Year

```{r,message=FALSE,warning=FALSE}

plotFrameworkDesire <- function(survey_results,TotalNoofRows) {
  survey_results %>%
    filter(!is.na(FrameworkDesireNextYear)) %>%
    select(FrameworkDesireNextYear) %>%
    mutate(FrameworkDesireNextYear = str_split(FrameworkDesireNextYear, pattern = ";")) %>%
    unnest(FrameworkDesireNextYear) %>%
    group_by(FrameworkDesireNextYear) %>%
    summarise(Count = n()/TotalNoofRows) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(TotalCount =  sum(Count)) %>%
    mutate(Count =  Count/TotalCount) %>%
    mutate(FrameworkDesireNextYear = reorder(FrameworkDesireNextYear,Count)) %>%
    head(10) %>%
    
    ggplot(aes(x = FrameworkDesireNextYear,y = Count)) +
    geom_bar(stat='identity',fill= fillColorLightCoral) +
    geom_text(aes(x = FrameworkDesireNextYear,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
    labs(x = 'FrameworkDesireNextYear', 
         y='Percentage', 
         title = 'FrameworkDesireNextYear and Percentage') +
    coord_flip() +
    theme_bw()
}

plotFrameworkDesire(survey_results,TotalNoofRows)


```

Node,React  and Angular are the top Frameworks which respondents want to learn next year

##Database Desire Next Year

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire <- function(survey_results,TotalNoofRows) {
  survey_results %>%
    filter(!is.na(DatabaseDesireNextYear)) %>%
    select(DatabaseDesireNextYear) %>%
    mutate(DatabaseDesireNextYear = str_split(DatabaseDesireNextYear, pattern = ";")) %>%
    unnest(DatabaseDesireNextYear) %>%
    group_by(DatabaseDesireNextYear) %>%
    summarise(Count = n()/TotalNoofRows) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(TotalCount =  sum(Count)) %>%
    mutate(Count =  Count/TotalCount) %>%
    mutate(DatabaseDesireNextYear = reorder(DatabaseDesireNextYear,Count)) %>%
    head(10) %>%
    
    ggplot(aes(x = DatabaseDesireNextYear,y = Count)) +
    geom_bar(stat='identity',fill= fillColor) +
  scale_y_continuous(labels = percent_format()) +
    geom_text(aes(x = DatabaseDesireNextYear,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    labs(x = 'DatabaseDesireNextYear', 
         y='Percentage', 
         title = 'DatabaseDesireNextYear and Percentage') +
    coord_flip() +
    theme_bw()
}

plotDatabaseDesire(survey_results,TotalNoofRows)

```

MySQL, MongoDB and PostgreSQL are the top databases which respondents want to learn next year

##Language Desire Next Year

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire <- function(survey_results,TotalNoofRows) {
  survey_results %>%
    filter(!is.na(LanguageDesireNextYear)) %>%
    select(LanguageDesireNextYear) %>%
    mutate(LanguageDesireNextYear = str_split(LanguageDesireNextYear, pattern = ";")) %>%
    unnest(LanguageDesireNextYear) %>%
    group_by(LanguageDesireNextYear) %>%
    summarise(Count = n()/TotalNoofRows) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(TotalCount =  sum(Count)) %>%
    mutate(Count =  Count/TotalCount) %>%
    mutate(LanguageDesireNextYear = reorder(LanguageDesireNextYear,Count)) %>%
    head(10) %>%
    
    ggplot(aes(x = LanguageDesireNextYear,y = Count)) +
    geom_bar(stat='identity',fill= fillColor2) +
    geom_text(aes(x = LanguageDesireNextYear,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
    labs(x = 'LanguageDesireNextYear', 
         y='Percentage', 
         title = 'LanguageDesireNextYear and Percentage') +
    coord_flip() +
    theme_bw()
}

plotLanguageDesire(survey_results,TotalNoofRows)

```

JavaScript,Python, HTML are the top languages which respondents want to learn next year

#Language Worked With

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(LanguageWorkedWith)) %>%
  select(LanguageWorkedWith) %>%
  mutate(LanguageWorkedWith = str_split(LanguageWorkedWith, pattern = ";")) %>%
  unnest(LanguageWorkedWith) %>%
  group_by(LanguageWorkedWith) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(TotalCount =  sum(Count)) %>%
  mutate(Count =  Count/TotalCount) %>%
  mutate(LanguageWorkedWith = reorder(LanguageWorkedWith,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = LanguageWorkedWith,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = LanguageWorkedWith,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'LanguageWorkedWith', 
       y='Percentage', 
       title = 'LanguageWorkedWith and Percentage') +
  coord_flip() +
  theme_bw()

```

##Network Analysis -R           

The network graph shows the people associated with R

```{r,message=FALSE,warning=FALSE}

count_bigrams <- function(dataset) {
  dataset %>%
    unnest_tokens(bigram, LanguageWorkedWith, token = "ngrams", n = 2) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    count(word1, word2, sort = TRUE)
}


visualize_bigrams <- function(bigrams) {
  set.seed(2016)
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  
  bigrams %>%
    graph_from_data_frame() %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = a) +
    geom_node_point(color = "lightblue", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void()
  
}

visualize_bigrams_individual <- function(bigrams) {
  set.seed(2016)
  a <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  
  bigrams %>%
    graph_from_data_frame() %>%
    ggraph(layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = a,end_cap = circle(.07, 'inches')) +
    geom_node_point(color = "lightblue", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
    theme_void()
}


survey_results %>%
  count_bigrams() %>%
   filter(word1 == "r" | word2 == "r") %>%
  filter( n > 50) %>%
  visualize_bigrams()

```



##Network Analysis -Python

The network graph shows the people associated with python

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  count_bigrams() %>%
   filter(word1 == "python" | word2 == "python") %>%
  filter( n > 50) %>%
  visualize_bigrams()

```



#IDE Analysis

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(IDE)) %>%
  select(IDE) %>%
  mutate(IDE = str_split(IDE, pattern = ";")) %>%
  unnest(IDE) %>%
  group_by(IDE) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(TotalCount =  sum(Count)) %>%
  mutate(Count =  Count/TotalCount) %>%
  mutate(IDE = reorder(IDE,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = IDE,y = Count)) +
  geom_bar(stat='identity',fill= fillColorLightCoral) +
  geom_text(aes(x = IDE,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'IDE', 
       y='Percentage', 
       title = 'IDE and Percentage') +
  coord_flip() +
  theme_bw()

```

Visual Studio Code, Visual Studio and Notepad++ are the top IDE which respondents want to learn next year


#Version Control Analysis

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(VersionControl)) %>%
  select(VersionControl) %>%
  mutate(VersionControl = str_split(VersionControl, pattern = ";")) %>%
  unnest(VersionControl) %>%
  group_by(VersionControl) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(TotalCount =  sum(Count)) %>%
  mutate(Count =  Count/TotalCount) %>%
  mutate(VersionControl = reorder(VersionControl,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = VersionControl,y = Count)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = VersionControl,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'VersionControl',
       y='Percentage', 
       title = 'VersionControl and Percentage') +
  coord_flip() +
  theme_bw()

```

Git, Subversion and Team Foundation Version Control are the top Version Control tools that the respondents use.

#Communication Tool Analysis


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(CommunicationTools)) %>%
  select(CommunicationTools) %>%
  mutate(CommunicationTools = str_split(CommunicationTools, pattern = ";")) %>%
  unnest(CommunicationTools) %>%
  group_by(CommunicationTools) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(TotalCount =  sum(Count)) %>%
  mutate(Count =  Count/TotalCount) %>%
  mutate(CommunicationTools = reorder(CommunicationTools,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = CommunicationTools,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = CommunicationTools,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'CommunicationTools', 
       y='Percentage', 
       title = 'CommunicationTools and Percentage') +
  coord_flip() +
  theme_bw()

```

Slack, Jira, Office productivity tools are the top Communication tools that the respondents use.

#Methodology analysis

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(Methodology)) %>%
  select(Methodology) %>%
  mutate(Methodology = str_split(Methodology, pattern = ";")) %>%
  unnest(Methodology) %>%
  group_by(Methodology) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(TotalCount =  sum(Count)) %>%
  mutate(Count =  Count/TotalCount) %>%
  mutate(Methodology = reorder(Methodology,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = Methodology,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = Methodology,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Methodology', 
       y='Percentage', 
       title = 'Methodology and Percentage') +
  coord_flip() +
  theme_bw()

```


Agile, Scrum , Kanban are the top Methodology tools that respondents are using.      


#Lifestyle Analysis

* 9-12 hours is the most popular (38%) time interval that developers spend in front of the computer.          

* 1-2 hours is the most popular (28%) time interval that developers hours outside.          

* Most devs (46%) never skip meals        

* Ergonomic Keyboard or mouse , Standing desk , both 10% of the population are the most popular ergonomic devices used by the developers.           


##The Number of Hours in Computer



```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(HoursComputer)) %>%
  select(HoursComputer) %>%
  group_by(HoursComputer) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HoursComputer = reorder(HoursComputer,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = HoursComputer,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = HoursComputer,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'HoursComputer', 
       y='Percentage', 
       title = 'HoursComputer and Percentage') +
  coord_flip() +
  theme_bw()

```

9-12 hours is the most popular (38%) time interval that developers spend in front of the computer.

##Hours Outside

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(HoursOutside)) %>%
  select(HoursOutside) %>%
  group_by(HoursOutside) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HoursOutside = reorder(HoursOutside,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = HoursOutside,y = Count)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = HoursOutside,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'HoursOutside', 
       y='Percentage', 
       title = 'HoursOutside and Percentage') +
  coord_flip() +
  theme_bw()

```

1-2 hours is the most popular (28%) time interval that developers hours outside.

##SkipMeals

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(SkipMeals)) %>%
  select(SkipMeals) %>%
  group_by(SkipMeals) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(SkipMeals = reorder(SkipMeals,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = SkipMeals,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = SkipMeals,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'SkipMeals', 
       y='Percentage', 
       title = 'SkipMeals and Percentage') +
  coord_flip() +
  theme_bw()

```

Most devs (46%) never skip meals 

##Ergonomic Devices

Ergonomic Keyboard or mouse , Standing desk , both 10% of the population are the most popular ergonomic devices used by the developers.          


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(ErgonomicDevices)) %>%
  select(ErgonomicDevices) %>%
  group_by(ErgonomicDevices) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(ErgonomicDevices = reorder(ErgonomicDevices,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = ErgonomicDevices,y = Count)) +
  geom_bar(stat='identity',fill= fillColorLightCoral) +
  geom_text(aes(x = ErgonomicDevices,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'ErgonomicDevices', 
       y='Percentage', 
       title = 'ErgonomicDevices and Percentage') +
  coord_flip() +
  theme_bw()

```



#Salary Analysis

* Engineering Manager, DevOps Specialist, C Suite executive are the top positions which have the highest salaries.        

* United States, Switzerland, Israel, Norway and Denmark are the countries with the highest median salaries with respondents of 100 or more.          

* The highest median salary for `25-34 years` old is **95K**. US, Switzerland , Israel , Denmark and Australia are the top Five countries which has the highest salaries in this group. 

* The highest median salary for `35-44 years` old is **495K** followed by more acceptable **120K**. Venezuala, US, Switzerland , Israel , Denmark and Norway are the top Six countries which has the highest salaries in this group.           

* The highest median salary for `45-54 years` old is **1000K** followed by more acceptable **130K**. Venezuala, HongKong, Switzerland , US, Ireland , Israel , Denmark and UAE are the top Eight countries which has the highest salaries in this group.          
       
* Marketing or Sales Professional, C Suite Executives, Academic Researchers , Engineering Managers and Product Managers get the highest salary in USA.      

##Salary Analysis by DevType

Engineering Manager, DevOps Specialist, C Suite executive are the top positions which have the highest salaries.             


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(DevType)) %>%
  select(DevType,ConvertedSalary) %>%
  mutate(DevType = str_split(DevType, pattern = ";")) %>%
  unnest(DevType) %>%
  group_by(DevType) %>%
  summarise(MedianSalary = median(ConvertedSalary,na.rm = TRUE)) %>%
  arrange(desc(MedianSalary)) %>%
  ungroup() %>%
  mutate(DevType = reorder(DevType,MedianSalary)) %>%
  head(10) %>%
  
  ggplot(aes(x = DevType,y = MedianSalary)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = DevType, y = 1, label = paste0("( ",round(MedianSalary/1e3)," K)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'DevType', 
       y = 'Median Salary', 
       title = 'DevType and Median Salary') +
  coord_flip() +
  theme_bw()


```


##Salary Analysis with Countries 100 and more respondents

United States, Switzerland, Israel, Norway and Denmark are the countries with the highest median salaries with respondents of 100 or more.          


```{r,message=FALSE,warning=FALSE}

Countries <- survey_results %>%
  filter(!is.na(Country)) %>%
  group_by(Country) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(Country = reorder(Country,Count))

Countries100AndMore <- Countries %>%
  filter(Count >= 100)


survey_results %>%
  filter(Country %in% Countries100AndMore$Country) %>%
  group_by(Country) %>%
  summarise(MedianSalary = median(ConvertedSalary,na.rm = TRUE)) %>%
  arrange(desc(MedianSalary)) %>%
  ungroup() %>%
  mutate(Country = reorder(Country,MedianSalary)) %>%
  head(10) %>%
  
  ggplot(aes(x = Country,y = MedianSalary)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = Country, y = 1, label = paste0("( ",round(MedianSalary/1e3)," K)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Country', 
       y = 'Median Salary', 
       title = 'Country and Median Salary') +
  coord_flip() +
  theme_bw()

```



```{r,message=FALSE,warning=FALSE}

plotSalary <- function(datasetName,fillColorName) {
  
  datasetName %>%
    filter(Country %in% Countries100AndMore$Country) %>%
    group_by(Country) %>%
    summarise(MedianSalary = median(ConvertedSalary,na.rm = TRUE)) %>%
    arrange(desc(MedianSalary)) %>%
    ungroup() %>%
    mutate(Country = reorder(Country,MedianSalary)) %>%
    head(10) %>%
    
    ggplot(aes(x = Country,y = MedianSalary)) +
    geom_bar(stat='identity',fill= fillColorName) +
    geom_text(aes(x = Country, y = 1, label = paste0("( ",round(MedianSalary/1e3)," K)",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    labs(x = 'Country', 
         y = 'Median Salary', 
         title = 'Country and Median Salary') +
    coord_flip() +
    theme_bw()
  
}

```


##Salary Analysis for 25 - 34 years old

The highest median salary for 25-34 years old is **95K**. US, Switzerland , Israel , Denmark and Australia are the top Five countries which has the highest salaries in this group.        

```{r,message=FALSE,warning=FALSE}

survey_results_25_34 <- survey_results %>%
  filter(Age == "25 - 34 years old")

plotSalary(survey_results_25_34,fillColorName = fillColor)

```

##Salary Analysis for 35-44 year olds


The highest median salary for 35-44 years old is **495K** followed by more acceptable **120K**. Venezuala, US, Switzerland , Israel , Denmark and Norway are the top Six countries which has the highest salaries in this group.                 

```{r,message=FALSE,warning=FALSE}

survey_results_35_44 <- survey_results %>%
  filter(Age == "35 - 44 years old")

plotSalary(survey_results_35_44,fillColorName = fillColorLightCoral)

```


##Salary Analysis for 45-54 year olds

The highest median salary for 45-54 years old is **1000K** followed by more acceptable **130K**. Venezuala, HongKong, Switzerland , US, Ireland , Israel , Denmark and UAE are the top Eight countries which has the highest salaries in this group.         

```{r,message=FALSE,warning=FALSE}

survey_results_45_54 <- survey_results %>%
  filter(Age == "45 - 54 years old")

plotSalary(survey_results_45_54,fillColorName = fillColor2)

```

## Salary Analysis Dev Types- Years of Experience in USA

Marketing or Sales Professional, C Suite Executives, Academic Researchers , Engineering Managers and Product Managers get the highest salary in USA.            


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(Country == "United States") %>%
  select(DevType,YearsCoding,ConvertedSalary) %>%
  mutate(DevType = str_split(DevType, pattern = ";")) %>%
  unnest(DevType) %>%
  group_by(DevType,YearsCoding) %>%
  summarise(MedianSalary = median(ConvertedSalary,na.rm = TRUE)) %>%
  arrange(desc(MedianSalary)) %>%
  ungroup() %>%
  mutate(YearsCoding = as.character(YearsCoding),
         DevType = as.character(DevType)) %>%
  mutate(DevType_YearsOfCoding = paste(DevType,YearsCoding)) %>%
  mutate(DevType_YearsOfCoding =str_replace(DevType_YearsOfCoding,"NA","")) %>%
  mutate(DevType_YearsOfCoding = reorder(DevType_YearsOfCoding,MedianSalary)) %>%
  head(10) %>%
  
  ggplot(aes(x = DevType_YearsOfCoding,y = MedianSalary)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = DevType_YearsOfCoding, y = 1, label = paste0("( ",round(MedianSalary,2)/1e3," K)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'DevType_YearsOfCoding', 
       y='MedianSalary', 
       title = 'DevType_YearsOfCoding and MedianSalary') +
  coord_flip() +
  theme_bw()

```


#AI Analysis


##AI Interesting

Answers for `What do you think is the most exciting aspect of increasingly advanced AI technology?`

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AIInteresting)) %>%
  select(AIInteresting) %>%
  group_by(AIInteresting) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(AIInteresting = reorder(AIInteresting,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = AIInteresting,y = Count)) +
  geom_bar(stat='identity',fill= fillColorLightCoral) +
  geom_text(aes(x = AIInteresting,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'AIInteresting', 
       y = 'Percentage', 
       title = 'AIInteresting and Percentage') +
  coord_flip() +
  theme_bw()

```

##AI Responsible

Answers for `Whose responsibility is it, primarily, to consider the ramifications of increasingly advanced AI technology?`

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AIResponsible)) %>%
  select(AIResponsible) %>%
  group_by(AIResponsible) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(AIResponsible = reorder(AIResponsible,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = AIResponsible,y = Count)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = AIResponsible,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'AIResponsible', 
       y = 'Percentage', 
       title = 'AIResponsible and Percentage') +
  coord_flip() +
  theme_bw()

```


##AI Future

Answers for `Overall, what's your take on the future of artificial intelligence?`

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AIFuture)) %>%
  select(AIFuture) %>%
  group_by(AIFuture) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(AIFuture = reorder(AIFuture,Count)) %>%
  head(10) %>%
  
  ggplot(aes(x = AIFuture,y = Count)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = AIFuture,y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'AIFuture', 
       y = 'Percentage', 
       title = 'AIFuture and Percentage') +
  coord_flip() +
  theme_bw()

```


#Assess Job{.tabset .tabset-fade .tabset-pills}

The questions have been framed on different topics for developers on how they assess the job. The framework of the questions have 1 with the most important and 10 is the least important.         

Developors provide the **Most Importance**  to               

* The languages, frameworks, and other technologies                 

* The compensation and benefits          

* The office environment or company culture          

* Opportunities for professional development               

* The opportunity to work from home/remotely               

## Preference of Industry Rating

Answer to `The industry that I'd be working in`   

```{r,message=FALSE,warning=FALSE}

breaks = c(1:10)

TotalNoofRows = nrow(survey_results)

survey_results %>%
  filter(!is.na(AssessJob1)) %>%
  group_by(AssessJob1) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob1 = as.numeric(AssessJob1)) %>%
  
  ggplot(aes(x = AssessJob1,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Industry I would be working on' ,y = 'Percentage', title = "Industry I would be working on") +
  theme_bw()

```

The respondents are not concerned about the Industry that they would be working on.          

## Financial performance or funding status of the company or organization Rating

Answer to `The financial performance or funding status of the company or organization`           


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob2)) %>%
  group_by(AssessJob2) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob2 = as.numeric(AssessJob2)) %>%
  
  ggplot(aes(x = AssessJob2,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'The financial performance or funding status of the company or organization' ,y = 'Count', title = paste("The financial performance or funding status of the company or organization")) +
  theme_bw()

```

The respondents are not concerned about the financial performance or funding status of the company or organization.    


##Specific Department Rating

Answer to `The specific department or team I'd be working on`           


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob3)) %>%
  group_by(AssessJob3) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob3 = as.numeric(AssessJob3)) %>%
  
  ggplot(aes(x = AssessJob3,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColorLightCoral) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Specific Department Rating' ,y = 'Count', title = 'Specific Department Rating') +
  theme_bw()

```


##Technologies Rating

Answer to `The languages, frameworks, and other technologies I'd be working with`           


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob4)) %>%
  group_by(AssessJob4) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob4 = as.numeric(AssessJob4)) %>%
  
  ggplot(aes(x = AssessJob4,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'Technologies Rating' ,y = 'Count', title = 'Technologies Rating') +
  theme_bw()

```

##The compensation and benefits offered Rating

Answer to `The compensation and benefits offered`            

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob5)) %>%
  group_by(AssessJob5) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob5 = as.numeric(AssessJob5)) %>%
  
  ggplot(aes(x = AssessJob5,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'The compensation and benefits offered Rating' ,y = 'Count', title = 'The compensation and benefits offered Rating') +
  theme_bw()

```


##The office environment or company culture


Answer to `Answer to `The office environment or company culture`           


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob6)) %>%
  group_by(AssessJob6) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob6 = as.numeric(AssessJob6)) %>%
  
  ggplot(aes(x = AssessJob6,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'The office environment or company culture Rating' ,y = 'Count', title = 'The office environment or company culture Rating') +
  theme_bw()

```

##The opportunity to work from home/remotely

Answer to `The opportunity to work from home/remotely`           

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob7)) %>%
  group_by(AssessJob7) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob8 = as.numeric(AssessJob7)) %>%
  
  ggplot(aes(x = AssessJob7,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'The opportunity to work from home/remotely' ,y = 'Count', title = 'The opportunity to work from home/remotely Rating') +
  theme_bw()

```


##The Opportunities for professional development

Answer to `Opportunities for professional development`           

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob8)) %>%
  group_by(AssessJob8) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob8 = as.numeric(AssessJob8)) %>%
  
  ggplot(aes(x = AssessJob8,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'The Opportunities for professional development' ,y = 'Count', title = 'The Opportunities for professional development Rating') +
  theme_bw()

```

##The diversity of the company or organization


Answer to `The diversity of the company or organization`           


```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob9)) %>%
  group_by(AssessJob9) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob8 = as.numeric(AssessJob9)) %>%
  
  ggplot(aes(x = AssessJob9,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'The diversity of the company or organization' ,y = 'Count', title = 'The diversity of the company or organization Rating') +
  theme_bw()

```

##How widely used or impactful the product or service I'd be working on is Rating


Answer to `How widely used or impactful the product or service I'd be working on is`      

```{r,message=FALSE,warning=FALSE}

survey_results %>%
  filter(!is.na(AssessJob10)) %>%
  group_by(AssessJob10) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessJob8 = as.numeric(AssessJob10)) %>%
  
  ggplot(aes(x = AssessJob10,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_x_continuous(limits = c(0, 11),breaks=breaks ) +   scale_y_continuous(labels = percent_format()) +
  labs(x = 'How widely used or impactful the product or service' ,y = 'Count', title = 'How widely used or impactful the product or service') +
  theme_bw()

```


#Assess Benefits Analysis{.tabset .tabset-fade .tabset-pills}

The developers have assessed a **job benefits package** and provided ratings. ( 1- Most Important, 11 - Least Important).

The most important factors while assessing the job benefits package are

*  Salary and/or bonuses         

*  Stock options or shares       

*  Health insurance        

* Retirement or pension savings matching  

* Computer/office equipment allowance      

* Conference or education budget               


##Salary and/or bonuses Rating

        


```{r,message=FALSE,warning=FALSE}

breaks = c(1:11)

subjectName = 'Salary and/or bonuses Rating'

survey_results %>%
  filter(!is.na(AssessBenefits1)) %>%
  group_by(AssessBenefits1) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits1 = as.numeric(AssessBenefits1)) %>%
  
  ggplot(aes(x = AssessBenefits1,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()


```


##Stock options or shares Rating

     

```{r,message=FALSE,warning=FALSE}

subjectName = 'Stock options or shares Rating'

survey_results %>%
  filter(!is.na(AssessBenefits2)) %>%
  group_by(AssessBenefits2) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits2 = as.numeric(AssessBenefits2)) %>%
  
  ggplot(aes(x = AssessBenefits2,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()


```

##Health insurance

```{r,message=FALSE,warning=FALSE}

subjectName = 'Health insurance Rating'

survey_results %>%
  filter(!is.na(AssessBenefits3)) %>%
  group_by(AssessBenefits3) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits3 = as.numeric(AssessBenefits3)) %>%
  
  ggplot(aes(x = AssessBenefits3,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```

##Parental leave

```{r,message=FALSE,warning=FALSE}

subjectName = 'Parental leave Rating'

survey_results %>%
  filter(!is.na(AssessBenefits4)) %>%
  group_by(AssessBenefits4) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits4 = as.numeric(AssessBenefits4)) %>%
  
  ggplot(aes(x = AssessBenefits4,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```


##Fitness or wellness benefit

```{r,message=FALSE,warning=FALSE}

subjectName = 'Fitness or wellness benefit Rating'

survey_results %>%
  filter(!is.na(AssessBenefits5)) %>%
  group_by(AssessBenefits5) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits5 = as.numeric(AssessBenefits5)) %>%
  
  ggplot(aes(x = AssessBenefits5,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColorLightCoral) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```

##Retirement or pension savings matching

```{r,message=FALSE,warning=FALSE}

subjectName = 'Retirement or pension savings matching Rating'

survey_results %>%
  filter(!is.na(AssessBenefits6)) %>%
  group_by(AssessBenefits6) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits6 = as.numeric(AssessBenefits6)) %>%
  
  ggplot(aes(x = AssessBenefits6,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```


##Company-provided meals or snacks

```{r,message=FALSE,warning=FALSE}

subjectName = ' Company-provided meals or snacks Rating'

survey_results %>%
  filter(!is.na(AssessBenefits7)) %>%
  group_by(AssessBenefits7) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits7 = as.numeric(AssessBenefits7)) %>%
  
  ggplot(aes(x = AssessBenefits7,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```

##Computer/office equipment allowance

```{r,message=FALSE,warning=FALSE}

subjectName = ' Computer/office equipment allowance Rating'

survey_results %>%
  filter(!is.na(AssessBenefits8)) %>%
  group_by(AssessBenefits8) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits8 = as.numeric(AssessBenefits8)) %>%
  
  ggplot(aes(x = AssessBenefits8,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```


##Childcare benefit

```{r,message=FALSE,warning=FALSE}

subjectName = ' Childcare benefit Rating'

survey_results %>%
  filter(!is.na(AssessBenefits9)) %>%
  group_by(AssessBenefits9) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits9 = as.numeric(AssessBenefits9)) %>%
  
  ggplot(aes(x = AssessBenefits9,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```

##Transportation benefit

```{r,message=FALSE,warning=FALSE}

subjectName = ' Transportation benefit Rating'

survey_results %>%
  filter(!is.na(AssessBenefits10)) %>%
  group_by(AssessBenefits10) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits10 = as.numeric(AssessBenefits10)) %>%
  
  ggplot(aes(x = AssessBenefits10,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColorLightCoral) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()



```


##Conference or education budget

```{r,message=FALSE,warning=FALSE}

subjectName = ' Conference or education budget'

survey_results %>%
  filter(!is.na(AssessBenefits11)) %>%
  group_by(AssessBenefits11) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(AssessBenefits11 = as.numeric(AssessBenefits11)) %>%
  
  ggplot(aes(x = AssessBenefits11,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_x_continuous(limits = c(0, 12),breaks=breaks ) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  theme_bw()

```

#Developer Mentalities{.tabset .tabset-fade .tabset-pills}

We explore the Developer mentalities here. 

* Developers have sense of kinship or connection to other developers    

* Developers agree they have a competition among peers       

* They also agree they are as good as other developers            

##Sense of kinship or connection to other developers -  Agree / Disagree 

```{r,message=FALSE,warning=FALSE}

subjectName = "Sense of kinship or connection to other developers"

survey_results %>%
  filter(!is.na(AgreeDisagree1)) %>%
  group_by(AgreeDisagree1) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = AgreeDisagree1,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```
  

##Competing with my peers - Agree / Disagree 

```{r,message=FALSE,warning=FALSE}

subjectName = "Competing with my peers"

survey_results %>%
  filter(!is.na(AgreeDisagree2)) %>%
  group_by(AgreeDisagree2) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = AgreeDisagree2,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```


##Not as good at programming as most of my peers -  Agree / Disagree 

```{r,message=FALSE,warning=FALSE}

subjectName = "Not as good at programming as most of my peers"

survey_results %>%
  filter(!is.na(AgreeDisagree3)) %>%
  group_by(AgreeDisagree3) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  ggplot(aes(x = AgreeDisagree3,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```


#Ethics analysis{.tabset .tabset-fade .tabset-pills}

* Most developers will not write unethical code             

* Regarding reporting unethical code, developers would report after assessing what it is         

* The developers feel the ethics repsonsibility stands with the upper management at the company        

* The developers also feel that they are obligated to consider the ethical implications of the code that they write

##Will you write the unethical code if requested?

```{r,message=FALSE,warning=FALSE}

subjectName = "Will you write the unethical code if requested?"

TotalNoofRows = nrow(survey_results)

survey_results %>%
  filter(!is.na(EthicsChoice)) %>%
  group_by(EthicsChoice) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = EthicsChoice,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()

```

##Do you report or otherwise call out the unethical code in question?

```{r,message=FALSE,warning=FALSE}

subjectName = "Do you report or otherwise call out the unethical code in question?"

survey_results %>%
  filter(!is.na(EthicsReport)) %>%
  group_by(EthicsReport) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = EthicsReport,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()

```


##Most responsible for code that accomplishes something unethical?

```{r,message=FALSE,warning=FALSE}

subjectName = "Most responsible for code that accomplishes something unethical?"

survey_results %>%
  filter(!is.na(EthicsResponsible)) %>%
  group_by(EthicsResponsible) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = EthicsResponsible,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColorLightCoral) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()

```


##Obligation to consider the ethical implications of the code that you write

```{r,message=FALSE,warning=FALSE}

subjectName = "Obligation to consider the ethical implications of the code that you write"

survey_results %>%
  filter(!is.na(EthicalImplications)) %>%
  group_by(EthicalImplications) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = EthicalImplications,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()

```

#Stack Overflow Inclusivity analysis{.tabset .tabset-fade .tabset-pills}


* Most Developers would recommend StackOverflow to a friend or colleague         

* Most Developers visit StackOverflow almost daily          

* Most Developers have a StackOverflow account       

* Most Developers participate in Q&A less than once a month/monthly in  on Stack Overflow          

* Most Developers have visited StackOverflow Jobs

* Most Developers are unaware of the Stack Overflow Developer Story feature        

* Most Developers would recommend Stack Overflow Jobs to a friend or colleague              

* Most Developers consider themselves as  a member of the Stack Overflow community                



##Recommend Stack Overflow overall to a friend or colleague

This is the answer to the question "How likely is it that you would recommend Stack Overflow overall to a friend or colleague? "

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Recommend "

TotalNoofRows = nrow(survey_results)

survey_results %>%
  filter(!is.na(StackOverflowRecommend)) %>%
  group_by(StackOverflowRecommend) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = StackOverflowRecommend,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("red")) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```

##StackOverflow visit frequency


This is the answer to the question "How frequently would you say you visit Stack Overflow? "

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Visit "

survey_results %>%
  filter(!is.na(StackOverflowVisit)) %>%
  group_by(StackOverflowVisit) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  mutate(StackOverflowVisit = reorder(StackOverflowVisit,Count)) %>%
  
  ggplot(aes(x = StackOverflowVisit,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  coord_flip() +
  theme_bw()


```

##StackOverflow Account

This is the answer to the question "Do you have a Stack Overflow account? "

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Account "

survey_results %>%
  filter(!is.na(StackOverflowHasAccount)) %>%
  group_by(StackOverflowHasAccount) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  ggplot(aes(x = StackOverflowHasAccount,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()

```

##StackOverflow Participate

This is the answer to the question "How frequently would you say you participate in Q&A on Stack Overflow? By participate we mean ask, answer, vote for, or comment on questions."


```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Participate "

survey_results %>%
  filter(!is.na(StackOverflowParticipate)) %>%
  group_by(StackOverflowParticipate) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(StackOverflowParticipate = reorder(StackOverflowParticipate,Count)) %>%
  
  ggplot(aes(x = StackOverflowParticipate,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  coord_flip() +
  theme_bw()


```

##StackOverflow Jobs

This is the answer to the question "Have you ever used or visited Stack Overflow Jobs?"

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Jobs "

survey_results %>%
  filter(!is.na(StackOverflowJobs)) %>%
  group_by(StackOverflowJobs) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(StackOverflowJobs = reorder(StackOverflowJobs,Count)) %>%
  
  
  ggplot(aes(x = StackOverflowJobs,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColorLightCoral) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  coord_flip() +
  theme_bw()


```

##StackOverflow up-to-date Developer Story

This is the answer to the question "Do you have an up-to-date Developer Story on Stack Overflow?"

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow DevStory "

survey_results %>%
  filter(!is.na(StackOverflowDevStory)) %>%
  group_by(StackOverflowDevStory) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = StackOverflowDevStory,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```


##StackOverflow Jobs Recommend

This is the answer to the question "How likely is it that you would recommend Stack Overflow Jobs to a friend or colleague? Where 0 is not likely at all and 10 is very likely."

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Jobs Recommend "

survey_results %>%
  filter(!is.na(StackOverflowJobsRecommend)) %>%
  group_by(StackOverflowJobsRecommend) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = StackOverflowJobsRecommend,y=Count))+
  geom_bar(stat='identity',colour="white", fill = fillColor2) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```

##StackOverflow Consider Member

This is the answer to the question "Do you consider yourself a member of the Stack Overflow community?"

```{r,message=FALSE,warning=FALSE}

subjectName = "StackOverflow Consider Member "

survey_results %>%
  filter(!is.na(StackOverflowConsiderMember)) %>%
  group_by(StackOverflowConsiderMember) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  ungroup() %>%
  
  
  ggplot(aes(x = StackOverflowConsiderMember,y=Count))+
  geom_bar(stat='identity',colour="white", fill = c("blue")) +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Percentage', title = subjectName) +
  theme_bw()


```


#Hypothetical Tools Interest{.tabset .tabset-fade .tabset-pills}

Most Developers show the following interests for the Hypothetical Tools

* Not At all Interested for A private area for people new to programming

* Somewhat Interested for A programming-oriented blog platform

* **Very Interested** for `An employer or job review system`

* Somewhat Interested for An area for Q&A related to career growth

##Interest in Hypothetical Tools

```{r,message=FALSE,warning=FALSE}

subjectName = 'Hypothetical Tools Interest'

survey_results %>%
  filter(!is.na(HypotheticalTools1)) %>%
  group_by(HypotheticalTools1) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HypotheticalTools1 = reorder(HypotheticalTools1,Count)) %>%
 
  ggplot(aes(x = HypotheticalTools1,y=Count))+
  geom_bar(stat='identity',colour="white", fill =fillColor) +
  geom_text(aes(x = HypotheticalTools1, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  coord_flip() +
  theme_bw()

```

##Private Area for people new in Programming

```{r,message=FALSE,warning=FALSE}

subjectName = 'Private Area for people new in Programming'

survey_results %>%
  filter(!is.na(HypotheticalTools2)) %>%
  group_by(HypotheticalTools2) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HypotheticalTools2 = reorder(HypotheticalTools2,Count)) %>%
  
  ggplot(aes(x = HypotheticalTools2,y=Count))+
  geom_bar(stat='identity',colour="white", fill =fillColor2) +
  geom_text(aes(x = HypotheticalTools2, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  coord_flip() +
  theme_bw()

```

##A programming-oriented blog platform

```{r,message=FALSE,warning=FALSE}

subjectName = 'A programming-oriented blog platform'

survey_results %>%
  filter(!is.na(HypotheticalTools3)) %>%
  group_by(HypotheticalTools3) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HypotheticalTools3 = reorder(HypotheticalTools3,Count)) %>%
  
  ggplot(aes(x = HypotheticalTools3,y=Count))+
  geom_bar(stat='identity',colour="white", fill =fillColorLightCoral) +
  geom_text(aes(x = HypotheticalTools3, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  coord_flip() +
  theme_bw()

```


##An employer or job review system

```{r,message=FALSE,warning=FALSE}

subjectName = 'An employer or job review system'

survey_results %>%
  filter(!is.na(HypotheticalTools4)) %>%
  group_by(HypotheticalTools4) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HypotheticalTools4 = reorder(HypotheticalTools4,Count)) %>%
  
  ggplot(aes(x = HypotheticalTools4,y=Count))+
  geom_bar(stat='identity',colour="white", fill =fillColor) +
  geom_text(aes(x = HypotheticalTools4, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  coord_flip() +
  theme_bw()

```

##An area for Q&A related to career growth

```{r,message=FALSE,warning=FALSE}

subjectName = 'An area for Q&A related to career growth'

survey_results %>%
  filter(!is.na(HypotheticalTools5)) %>%
  group_by(HypotheticalTools5) %>%
  summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(HypotheticalTools5 = reorder(HypotheticalTools5,Count)) %>%
  
  ggplot(aes(x = HypotheticalTools5,y=Count))+
  geom_bar(stat='identity',colour="white", fill =fillColor2) +
  geom_text(aes(x = HypotheticalTools5, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = subjectName ,y = 'Count', title = subjectName) +
  coord_flip() +
  theme_bw()

```


#DevType StackOverflow Inclusivity Analysis

The following plot shows the Developer Types which show the most Stack Overflow Inclusivity. Mobile Developer, Educator or Academic Researcher, C Suite Executive, Product Manager and Engineering manager are the DevTypes which show the most Stack Overflow Inclusivity.                


```{r,message=FALSE,warning=FALSE}

DevType <- survey_results %>%
  mutate(DevType = str_split(DevType, pattern = ";")) %>%
  unnest(DevType) %>%
  select(DevType,StackOverflowConsiderMember)

DevType_StackOverflowConsiderMember <- DevType %>%
  filter(StackOverflowConsiderMember == "Yes") %>%
  group_by(DevType) %>%
  summarise(CountSOMember = n()) %>%
  arrange(DevType) 

DevType_All <- DevType %>%
  group_by(DevType) %>%
  summarise(CountSOMember2 = n()) %>%
  arrange(DevType) 

DevType_combined <- inner_join(DevType_StackOverflowConsiderMember,DevType_All)

DevType_combined <- DevType_combined %>%
  mutate(Percentage = CountSOMember/CountSOMember2)

DevType_combined %>%
  select(DevType,Percentage) %>%
  arrange(desc(Percentage)) %>%
  mutate(DevType = reorder(DevType,Percentage)) %>%
  head(10) %>%
  
  ggplot(aes(x = DevType,y=Percentage))+
  geom_bar(stat='identity',colour="white", fill =fillColor2) +
  geom_text(aes(x = DevType, y = .01, label = paste0("( ",round(Percentage*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = DevType ,y = 'Percentage', title = "DevType and Stackoverflow Inclusivity Percentage") +
  coord_flip() +
  theme_bw()

```

#Country and Stack Overflow Inclusivity

We consider Countries which have 500 or more respondents. Bangladesh, Pakistan , India , Iran and Israel are the countries with the most Stack Overflow inclusivity.              


```{r,message=FALSE,warning=FALSE}

Country_StackOverflowConsiderMember <- survey_results %>%
  filter(StackOverflowConsiderMember == "Yes") %>%
  group_by(Country) %>%
  summarise(CountSOMember = n()) %>%
  arrange(Country) 

Country_All <- survey_results %>%
  group_by(Country) %>%
  summarise(CountSOMember2 = n()) %>%
  arrange(Country) 

Country_combined <- inner_join(Country_StackOverflowConsiderMember,Country_All)

Country_combined <- Country_combined %>%
  mutate(Percentage = CountSOMember/CountSOMember2)

Country_combined %>%
  filter(CountSOMember2 >=500) %>%
  select(Country,Percentage) %>%
  
  arrange(desc(Percentage)) %>%
  mutate(Country = reorder(Country,Percentage)) %>%
  head(10) %>%
  
  ggplot(aes(x = Country,y=Percentage))+
  geom_bar(stat='identity',colour="white", fill =fillColor) +
  geom_text(aes(x = Country, y = .01, label = paste0("( ",round(Percentage*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Country',y = 'Percentage', title = "Country and Stackoverflow Inclusivity Percentage") +
  coord_flip() +
  theme_bw()


```


#Years Coding Professionally and Stack Overflow Inclusivity

27-29 Years , 24 -26 Years , 6-8 Years , 18-20 Years and 9-11 Years Coding Professionals have the most Stack Overflow Inclusivity.            


```{r,message=FALSE,warning=FALSE}

YearsCodingProf_StackOverflowConsiderMember <- survey_results %>%
  filter(StackOverflowConsiderMember == "Yes") %>%
  group_by(YearsCodingProf) %>%
  summarise(CountSOMember = n()) %>%
  arrange(YearsCodingProf) 

YearsCodingProf_All <- survey_results %>%
  group_by(YearsCodingProf) %>%
  summarise(CountSOMember2 = n()) %>%
  arrange(YearsCodingProf) 

YearsCodingProf_combined <- inner_join(YearsCodingProf_StackOverflowConsiderMember,YearsCodingProf_All)

YearsCodingProf_combined <- YearsCodingProf_combined %>%
  mutate(Percentage = CountSOMember/CountSOMember2)

YearsCodingProf_combined %>%
  select(YearsCodingProf,Percentage) %>%
  arrange(desc(Percentage)) %>%
  mutate(YearsCodingProf = reorder(YearsCodingProf,Percentage)) %>%
  head(10) %>%
  
  ggplot(aes(x = YearsCodingProf,y=Percentage))+
  geom_bar(stat='identity',colour="white", fill =fillColor2) +
  geom_text(aes(x = YearsCodingProf, y = .01, label = paste0("( ",round(Percentage*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'Years Coding Professionally' ,y = 'Percentage', title = "YearsCodingProf and Stackoverflow Inclusivity Percentage") +
  coord_flip() +
  theme_bw()



```


#Persona Engineering Manager{.tabset .tabset-fade .tabset-pills}

* Median Salary around 88K and Mean Salary of 129K            

* Most have a Bachelor's degree in Computer Science             
 
* Most want to learn Linux,AWS and Android in the Platform area                   

* Most want to learn Node,React and TensorFlow in the Framework area              

* Most want to learn PostgreSQL,MySQL and MongoDB in the Database area       

* Most want to learn Javascript,HTML and CSS in the Language area           

##Salary

```{r,message=FALSE,warning=FALSE}

eng_manager <- survey_results %>%
  filter(str_detect(DevType,"Engineering"))

TotalNoofRows = nrow(eng_manager) 

eng_manager %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor2) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(eng_manager$ConvertedSalary)

```


##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}

plotFormalEducation(eng_manager,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(eng_manager,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(eng_manager,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(eng_manager,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(eng_manager,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(eng_manager,TotalNoofRows)

```


#Persona C-Suite Executive{.tabset .tabset-fade .tabset-pills}

* Median Salary around 69K and Mean Salary of 117 K           

* Most have a Bachelor's degree in Computer Science             

* Most want to learn Linux,AWS and Android in the Platform area              

* Most want to learn Node,React and TensorFlow in the Framework area              

* Most want to learn PostgreSQL,MySQL and MongoDB in the Database area      

* Most want to learn Javascript,HTML and CSS in the Language area       

##Salary

```{r,message=FALSE,warning=FALSE}

c_suite <- survey_results %>%
  filter(str_detect(DevType,"C-suite"))

TotalNoofRows = nrow(c_suite) 

c_suite %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(c_suite$ConvertedSalary)

```

##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}



plotFormalEducation(c_suite,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(c_suite,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(c_suite,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(c_suite,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(c_suite,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(c_suite,TotalNoofRows)

```


#Persona DevOps specialist{.tabset .tabset-fade .tabset-pills}

* Median Salary around 72K and Mean Salary of 115 K           

* Most have a Bachelor's degree in Computer Science             

* Most want to learn Linux,AWS and Raspberry Pi in the Platform area              

* Most want to learn Node,React and Angular in the Framework area              

* Most want to learn PostgreSQL,Redis and ElasticSearch in the Database area      

* Most want to learn Javascript,Python and Bash/Shell in the Language area                 


##Salary

```{r,message=FALSE,warning=FALSE}

devops <- survey_results %>%
  filter(str_detect(DevType,"DevOps specialist"))

TotalNoofRows = nrow(devops) 

devops %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(devops$ConvertedSalary)

```

##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}



plotFormalEducation(devops,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(devops,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(devops,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(devops,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(devops,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(devops,TotalNoofRows)

```



#Persona Back-end developer{.tabset .tabset-fade .tabset-pills}

* Median Salary around 55K and Mean Salary of 96 K           

* Most have a Bachelor's degree in Computer Science             

* Most want to learn Linux,Android and AWS in the Platform area              

* Most want to learn Node,React and Angular in the Framework area              

* Most want to learn MySQL,PostgreSQL,MongoDB in the Database area      

* Most want to learn Javascript,HTML and SQL in the Language area                 

##Salary

```{r,message=FALSE,warning=FALSE}

backend <- survey_results %>%
  filter(str_detect(DevType,"Back-end developer"))

TotalNoofRows = nrow(backend) 

backend %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(backend$ConvertedSalary)

```

##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}



plotFormalEducation(backend,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(backend,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(backend,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(backend,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(backend,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(backend,TotalNoofRows)

```


#Persona Front-end developer{.tabset .tabset-fade .tabset-pills}

* Median Salary around 51K and Mean Salary of 95K           

* Most have a Bachelor's degree in Computer Science             

* Most want to learn Linux,Android and AWS in the Platform area              

* Most want to learn Node,React and Angular in the Framework area              

* Most want to learn MySQL,MongoDB,PostgreSQL in the Database area      

* Most want to learn Javascript,HTML and CSS in the Language area              

##Salary

```{r,message=FALSE,warning=FALSE}

frontend <- survey_results %>%
  filter(str_detect(DevType,"Front-end developer"))

TotalNoofRows = nrow(frontend) 

frontend %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(frontend$ConvertedSalary)

```

##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}



plotFormalEducation(frontend,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(frontend,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(frontend,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(frontend,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(frontend,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(frontend,TotalNoofRows)

```

#Persona Data scientist{.tabset .tabset-fade .tabset-pills}

* Median Salary around 60K and Mean Salary of 101 K           

* Most have a Bachelor's degree in Computer Science             

* Most want to learn Linux,AWS,Raspberry Pi in the Platform area              

* Most want to learn TensorFlow,Nodejs,Spark in the Framework area              

* Most want to learn PostgreSQL,MongoDB,MySQL in the Database area      

* Most want to learn Python,SQL and Javascript in the Language area        

##Salary

```{r,message=FALSE,warning=FALSE}

data_scientist <- survey_results %>%
  filter(str_detect(DevType,"Data scientist"))

TotalNoofRows = nrow(data_scientist) 

data_scientist %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(data_scientist$ConvertedSalary)

```

##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}



plotFormalEducation(data_scientist,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(data_scientist,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(data_scientist,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(data_scientist,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(data_scientist,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(data_scientist,TotalNoofRows)

```


#Persona Business Analyst{.tabset .tabset-fade .tabset-pills}

* Median Salary around 58K and Mean Salary of 106 K           

* Most have a Bachelor's degree in Computer Science             

* Most want to learn Linux,Windows Desktop/Server,Android in the Platform area              

* Most want to learn Nodejs,TensorFlow,Angular in the Framework area              

* Most want to learn SQL Server,MySQL,PostgreSQL in the Database area      

* Most want to learn SQL,Python, and Javascript in the Language area   

##Salary

```{r,message=FALSE,warning=FALSE}

business_analyst <- survey_results %>%
  filter(str_detect(DevType,"business analyst"))

TotalNoofRows = nrow(business_analyst) 

business_analyst %>%
    ggplot(aes(x = ConvertedSalary) )+
    geom_histogram(fill = fillColor) +
    scale_x_log10() +
    scale_y_log10() + 
    labs(x = 'Annual Salary' ,y = 'Count', title = paste("Distribution of", "Annual Salary")) +
    theme_bw()


summary(business_analyst$ConvertedSalary)

```

##Education Analysis

###Formal Education

Answers for `Which of the following best describes the highest level of formal education that you have completed?`

```{r,message=FALSE,warning=FALSE}



plotFormalEducation(business_analyst,TotalNoofRows)

```

###UnderGrad Major

Answers for `You previously indicated that you went to a college or university. Which of the following best describes your main field of study (aka 'major')?`

```{r,message=FALSE,warning=FALSE}

plotUnderGradDegree(business_analyst,TotalNoofRows)

```

##PlatformDesire 

```{r,message=FALSE,warning=FALSE}

plotPlatformDesire(business_analyst,TotalNoofRows)

```


##FrameworkDesire

```{r,message=FALSE,warning=FALSE}


plotFrameworkDesire(business_analyst,TotalNoofRows)

```

##DatabaseDesire

```{r,message=FALSE,warning=FALSE}

plotDatabaseDesire(business_analyst,TotalNoofRows)

```

##LanguageDesire

```{r,message=FALSE,warning=FALSE}

plotLanguageDesire(business_analyst,TotalNoofRows)

```