---
title: 'EDA : Detailed Analysis of Developers......'
author: "Nishant"
output:
  html_document:
    number_sections: false
    toc: true
    toc_depth: 6
    highlight: tango
    theme: cosmo
    smart: true
    code_folding: hide
    df_print: paged
editor_options: 
  chunk_output_type: console
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Introduction

> Each year, we at Stack Overflow ask the developer community about everything from their favorite technologies to their job preferences.
> This year marks the eighth year we’ve published our Annual Developer Survey results—with the largest number of respondents yet.
> Over 100,000 developers took the 30-minute survey in January 2018.


# Data Overview

## Loading Libraries
```{r Libraries, message=FALSE, warning=FALSE}

library(dplyr)
library(ggplot2)
library(highcharter)
library(tidyverse)
library(skimr)
library(qdap)
library(tm)
library(wordcloud)
library(Amelia)
library(mlbench)
library(plotrix)
library(plotly)

```


## Loading Data

```{r message=FALSE, warning=FALSE}

Survey_results = read_csv("../input/survey_results_public.csv")

total_rows = nrow(Survey_results)

as.data.frame(names(Survey_results))

```

## Analysing Missing Values

We will plot a Missigness map which will show values which are missing in the data.
Color red will denote value presence and blue will denote the missing patches.

```{r msissing_plot, message=FALSE, warning=FALSE, fig.height=10, fig.width= 10}

missmap(Survey_results, col = c("blue", "red"), legend = F)

```

As shown in the figure we can easily see there a lot of missing data only few colums from 
starting is showing continuous values. 


Lets look at the missing value in tabular form:

```{r, message=FALSE, warning=FALSE}

Survey_skim = Survey_results%>% skim()

Survey_skim%>%
  filter(stat == "missing")%>%
  select(variable, type, stat, value)

```

## Skiming the data

### Country Wise Distribution of developers

```{r country_tree, message=FALSE, warning=FALSE}

Survey_results$Country%>%
  skim()

Survey_results%>%
  group_by(Country)%>%
  summarise(n = n())%>%
  hchart("treemap", hcaes(x = Country, value = n, color = n))

```

### Developing A Hobby or Not


```{r hobby_col, message=FALSE, warning=FALSE}

Survey_results$Hobby%>%
  skim()

Survey_results%>%
filter(!is.na(Hobby))%>%
  count(Hobby)%>%
  hchart( "column", hcaes(x = Hobby, y = n))

```


### Developer Student Or Not

```{r Student_column, message=FALSE, warning=FALSE}

Survey_results$Student%>%
  skim()

Survey_results%>%
filter(!is.na(Student))%>%
  count(Student)%>%
  hchart( "column", hcaes(x = Student, y = n))

```
### Country Wise Salary

```{r country_sal, message=FALSE, warning=FALSE}

df = Survey_results%>%
  filter(!is.na(Country))%>%
  filter(!is.na(ConvertedSalary))%>%
  group_by(Country)%>%
  summarise( min = min(ConvertedSalary), max = max(ConvertedSalary), Count = n(), Average = mean(ConvertedSalary))%>%
  arrange(desc(Count))%>%
  head(15)

  highchart()%>%
  hc_xAxis(categories = df$Country)%>%
  hc_add_series(name = "Max Salary", data = df$max)%>%
    hc_add_series(name = "Average salary", data = df$Average)%>%
    hc_add_series(name = "No of Developers", data = df$Count)%>%
  hc_chart(type = "column",
           options3d = list(enabled = TRUE, beta = 15, alpha = 15))

```


### Job Satisfaction and career satisfaction of developers

```{r, message=FALSE, warning=FALSE}

hw_grid(Survey_results%>%
filter(!is.na(JobSatisfaction))%>%
  count(JobSatisfaction)%>%
    arrange(n)%>%
  hchart( "column", hcaes(x = JobSatisfaction, y = n))%>%
    hc_add_theme(hc_theme_flat()),
   
  Survey_results%>%
  filter(!is.na(JobSatisfaction))%>%
  count(CareerSatisfaction)%>%
    arrange(desc(n))%>%
  hchart( "column", hcaes(x = CareerSatisfaction, y = n))%>%
    hc_add_theme(hc_theme_flat()), 

  ncol = 2 , rowheight = 600
  
)

```



### Gender-Country distribution of developers 


```{r Gender-Country_bar,fig.height= 10, message=FALSE, warning=FALSE}

# t2 = Survey_results%>%
#   filter(Gender == "Female")%>%
#   summarise(Count = n()) 

Country_female = Survey_results%>%
  filter(Gender == "Female")%>%
  group_by(Country)%>%
  summarise(Count =n())%>%
   arrange(desc(Count))%>%
   head(10)

# t1 = Survey_results%>%
#   filter(Gender == "Male")%>%
#   filter(Country %in% Country_female$Country)%>%
#   summarise(Count = n())

Country_male = Survey_results%>%
  filter(Gender == "Male")%>%
  #filter(Country %in% Country_female$Country)%>%
  group_by(Country)%>%
  summarise(Count = n())%>%
  arrange(desc(Count))%>%
  head(10)

highchart()%>%
  hc_xAxis(categories = Country_male$Country)%>%
  hc_add_series(data =  Country_male$Count, name = "Male Count", type = "bar")%>%
  hc_add_series(data =  Country_female$Count, name = "Female Count", type = "bar")%>%
  hc_add_theme(hc_theme_flat())

  

```


### Age Wise Distribution of dvelopers

```{r age_bar, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(Age))%>%
  group_by(Age)%>%
  summarise(Count = n())%>%
  arrange(Count)%>%
  hchart("bar", hcaes(Age, Count))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Age Wise Distribution")
  

```

### Employment Wise Distribution

```{r employ_column, message=FALSE, warning=FALSE}

Survey_results$Employment%>%
  skim()

Survey_results%>%
filter(!is.na(Employment))%>%
  count(Employment)%>%
  hchart( "column", hcaes(x = Employment, y = n))

```

#### Poportions of developers 

```{r Employ_status, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(Employment))%>%
  count(Employment)%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = Employment, y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Proportion of developers In different Employment Status")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### Recommendation of StackOverflow By developer

```{r Recommendation, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(StackOverflowRecommend))%>%
  group_by(StackOverflowRecommend)%>%
  summarise(Count = n())%>%
  arrange(Count)%>%
  hchart("bar", hcaes(StackOverflowRecommend, Count))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Stack Overflow Recommendation")

```

#### Jobs Recommendations

```{r Job recommendation, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(StackOverflowJobsRecommend))%>%
  group_by(StackOverflowJobsRecommend)%>%
  summarise(Count = n())%>%
  arrange(Count)%>%
  hchart("bar", hcaes(StackOverflowJobsRecommend, Count))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Stack Overflow Job Recommendation")

```

## Most Used Language

#### Wordcloud
```{r word_lang1, message=FALSE, warning=FALSE}
word_freq_lang = Survey_results%>%
  filter(!is.na(LanguageWorkedWith))%>%
  select(LanguageWorkedWith)%>%
  mutate(LanguageWorkedWith = str_split(LanguageWorkedWith, pattern = ";"))%>%
  unnest(LanguageWorkedWith)%>%
  group_by(LanguageWorkedWith)%>%
  summarise(Count = n())%>%
  arrange(desc(Count))%>%
  ungroup()%>%
  mutate(LanguageWorkedWith = reorder(LanguageWorkedWith, Count))


wordcloud(word_freq_lang$LanguageWorkedWith, word_freq_lang$Count, colors = 'blue')

```

#### Bar Chart


```{r bar_lang1, message=FALSE, warning=FALSE}

highchart()%>%
  hc_title(text = paste("Bar Chart"))%>%
  hc_xAxis(categories = word_freq_lang$LanguageWorkedWith)%>%
  hc_add_series(data = word_freq_lang$Count, name = "Count", type = "bar")

```

### Most used Database

#### Wordcloud

```{r wordcloud_data, message=FALSE, warning=FALSE}

word_freq_database = Survey_results%>%
  filter(!is.na(DatabaseWorkedWith))%>%
  select(DatabaseWorkedWith)%>%
  mutate(DatabaseWorkedWith = str_split(DatabaseWorkedWith, pattern = ";"))%>%
  unnest(DatabaseWorkedWith)%>%
  group_by(DatabaseWorkedWith)%>%
  summarise(Count = n())%>%
  arrange(desc(Count))%>%
  ungroup()%>%
  mutate(DatabaseWorkedWith = reorder(DatabaseWorkedWith, Count))


wordcloud(word_freq_database$DatabaseWorkedWith, word_freq_database$Count, colors = 'blue')



```
#### Bar Chart


```{r bar_data,message=FALSE, warning=FALSE}

highchart()%>%
  hc_title(text = paste("Bar Chart"))%>%
  hc_xAxis(categories = word_freq_database$DatabaseWorkedWith)%>%
  hc_add_series(data = word_freq_database$Count, name = "Count", type = "bar")

```


### Langauge Desired Next Year

```{r bar_lang, message=FALSE, warning=FALSE}

word_freq_Lan_next = Survey_results%>%
  filter(!is.na(LanguageDesireNextYear))%>%
  select(LanguageDesireNextYear)%>%
  mutate(LanguageDesireNextYear = str_split(LanguageDesireNextYear, pattern = ";"))%>%
  unnest(LanguageDesireNextYear)%>%
  group_by(LanguageDesireNextYear)%>%
  summarise(Count = n())%>%
  arrange(desc(Count))%>%
  ungroup()%>%
  mutate(DatabaseWorkedWith = reorder(LanguageDesireNextYear, Count))


highchart()%>%
  hc_title(text = paste("Bar Chart"))%>%
  hc_xAxis(categories = word_freq_Lan_next$LanguageDesireNextYear)%>%
  hc_add_series(data = word_freq_Lan_next$Count, name = "Count", type = "bar")

```


#### Pyramid plot of Language worked with and Language worked with next year
```{r pyramid_lang, fig.height= 10, message=FALSE, warning=FALSE}

mcol<-color.gradient(c(0,0,0.5,1),c(0,0,0.5,1),c(1,1,0.5,1),18)
fcol<-color.gradient(c(1,1,0.5,1),c(0.5,0.5,0.5,1),c(0.5,0.5,0.5,1),18)


t1 = sum(word_freq_lang$Count)
t2 = sum(word_freq_Lan_next$Count)

word_freq_lang$Count = word_freq_lang$Count/t1 *100
word_freq_Lan_next$Count = word_freq_Lan_next$Count/t2*100

par(mar = pyramid.plot(word_freq_lang$Count, word_freq_Lan_next$Count, labels = word_freq_lang$LanguageWorkedWith,
top.labels=c("Langauage Worked With","","Language Desired Next Year"), main = "Language Worked with and will desire to work next", 
gap=5, show.values = T, rxcol = fcol, lxcol = mcol))

```


### DataBase Desired Next Year

```{r bar_database, fig.height= 8,message=FALSE, warning=FALSE}


word_freq_Database_next = Survey_results%>%
  filter(!is.na(DatabaseDesireNextYear))%>%
  select(DatabaseDesireNextYear)%>%
  mutate(DatabaseDesireNextYear = str_split(DatabaseDesireNextYear, pattern = ";"))%>%
  unnest(DatabaseDesireNextYear)%>%
  group_by(DatabaseDesireNextYear)%>%
  summarise(Count = n())%>%
  arrange(desc(Count))%>%
  ungroup()%>%
  mutate(DatabaseDesireNextYear = reorder(DatabaseDesireNextYear, Count))


highchart()%>%
  hc_title(text = paste("Bar Chart"))%>%
  hc_xAxis(categories = word_freq_Database_next$DatabaseDesireNextYear)%>%
  hc_add_series(data = word_freq_Database_next$Count, name = "Count", type = "bar")


```


#### Pyramid plot of Database worked with and Database worked with next year

```{r Pyramid_database, message=FALSE, warning=FALSE}


mcol<-color.gradient(c(0,0,0.5,1),c(0,0,0.5,1),c(1,1,0.5,1),18)
fcol<-color.gradient(c(1,1,0.5,1),c(0.5,0.5,0.5,1),c(0.5,0.5,0.5,1),18)


t1 = sum(word_freq_database$Count)
t2 = sum(word_freq_Database_next$Count)

word_freq_database$Count = word_freq_database$Count/t1 *100
word_freq_Database_next$Count = word_freq_Database_next$Count/t2*100

par(mar = pyramid.plot(word_freq_database$Count, word_freq_Database_next$Count, labels = word_freq_database$DatabaseWorkedWith,
top.labels=c("Database Worked","","Database Desired Next Year"), main = "Database Worked with and will desire to work next",
gap=5, show.values = T, rxcol = fcol, lxcol = mcol))


```


### AI Intereseting , Dangerous and Future Comments

```{r AI Comments, message=FALSE, warning=FALSE}

hw_grid(Survey_results%>%
filter(!is.na(AIInteresting))%>%
  count(AIInteresting)%>%
  hchart( "column", hcaes(x = AIInteresting, y = n))%>%
    hc_add_theme(hc_theme_flat()),
   
  Survey_results%>%
  filter(!is.na(AIDangerous))%>%
  count(AIDangerous)%>%
  hchart( "column", hcaes(x = AIDangerous, y = n))%>%
    hc_add_theme(hc_theme_flat()), 
  
  Survey_results%>%
  filter(!is.na(AIFuture))%>%
  count(AIFuture)%>%
  hchart( "column", hcaes(x = AIFuture, y = n))%>%
    hc_add_theme(hc_theme_flat()), 
  
  ncol = 3 , rowheight = 600
  
)

```

## Analysing Data Distribution of Stackoverflow Members (Participation , Salary)

### Salary Ditribution Vs Participation of different developers Type

We can se the trend of salary with participation:

* Developers with higher participation have higher Salary.
* The trend almost fit the linear model if we remove Students.

```{r, message=FALSE, warning=FALSE}
ggplotly(Survey_results%>%
           select(DevType, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           mutate(DevType = str_split(DevType, ";"))%>%
           unnest(DevType)%>%
           filter(!is.na(DevType))%>%
           group_by(DevType)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = DevType))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())
```

### Salary Ditribution Vs Participation of different Undergrad Major Type

> Here we can observe few interesting trend developers with Undergrad Major like humanities discipline, social science, 
> fine arts have higher salary than undergards of IT or Computer Science.
> Undergrad majors with math or stats and antural science have higher participation rate as well as salary.
> Web development major got lowest rank in terms of salary and participation.
> Graph shows a negative trend with participation and Salary

```{r, message=FALSE, warning=FALSE}

ggplotly(Survey_results%>%
           select(UndergradMajor, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           filter(!is.na(UndergradMajor))%>%
           group_by(UndergradMajor)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = UndergradMajor))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())

```

### Partcipation VS Membership labelling Education of Developers

The deveopers with higher eductaion tend to have higher participation 

```{r, message=FALSE, warning=FALSE}

ggplotly(Survey_results%>%
  select(FormalEducation, StackOverflowConsiderMember, StackOverflowParticipate)%>%
  filter(!is.na(FormalEducation))%>%
  mutate(FormalEducation = case_when(str_detect(FormalEducation, "Bachelor’s degree")~"Bachelor’s degree",
                                     str_detect(FormalEducation, "Master’s degree") ~ "Master’s degree" ,
                                     str_detect(FormalEducation, "I never completed any formal education") ~
                                       "No formal Education",
                                     str_detect(FormalEducation, "doctoral degree") ~ "Doctoral degree",
                                     str_detect(FormalEducation, "Professional degree") ~ "Professional degree",
                                     str_detect(FormalEducation, "Secondary school") ~ "Secondary school",
                                     str_detect(FormalEducation, "Secondary school") ~ "Secondary school",
                                     TRUE ~ FormalEducation))%>%
  group_by(FormalEducation)%>%
     summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
            Participation = mean(StackOverflowParticipate %in% 
                                   c("Multiple times per day",
                                     "Daily or almost daily",
                                     "A few times per week", 
                                     "A few times per month or weekly"),
                                 na.rm = T), n = n())%>%
  ggplot(aes(Participation, ConsiderMember, label = FormalEducation))+
   geom_smooth(method = "lm")+
  geom_point(aes(size = n))+
    xlab("People participating atleast weekly")+
    ylab("Consider themselves part of Stackoverflow")+
  theme_bw())

```


### Salary Ditribution Vs Participation of different developers working with different Langauges

The curve shows a negative trend with participation and few high paid langauages has been discovered 
with low number of developers and partcipation like Hack.

```{r, message=FALSE, warning=FALSE}
ggplotly(Survey_results%>%
           select(LanguageWorkedWith, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           mutate(LanguageWorkedWith = str_split(LanguageWorkedWith, ";"))%>%
           unnest(LanguageWorkedWith)%>%
           filter(!is.na(LanguageWorkedWith))%>%
           group_by(LanguageWorkedWith)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = LanguageWorkedWith))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())

```




### Salary Ditribution Vs Participation of different developers working with different Databases

* The curve shows a positive trend in partcipation and salary with few databases being the highest paid and participation like Google big query, Amazon dynamic DB etc..

```{r, message=FALSE, warning=FALSE}
ggplotly(Survey_results%>%
           select(DatabaseWorkedWith, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           mutate(DatabaseWorkedWith = str_split(DatabaseWorkedWith, ";"))%>%
           unnest(DatabaseWorkedWith)%>%
           filter(!is.na(DatabaseWorkedWith))%>%
           group_by(DatabaseWorkedWith)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = DatabaseWorkedWith))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())

```


### Salary Ditribution Vs Participation of different developers working with different Framework

```{r, message=FALSE, warning=FALSE}
ggplotly(Survey_results%>%
           select(FrameworkWorkedWith, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           mutate(FrameworkWorkedWith = str_split(FrameworkWorkedWith, ";"))%>%
           unnest(FrameworkWorkedWith)%>%
           filter(!is.na(FrameworkWorkedWith))%>%
           group_by(FrameworkWorkedWith)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = FrameworkWorkedWith))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())

```



### Salary Ditribution Vs Participation of different developers working with different Methodology

```{r, message=FALSE, warning=FALSE}
ggplotly(Survey_results%>%
           select(Methodology, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           mutate(Methodology = str_split(Methodology, ";"))%>%
           unnest(Methodology)%>%
           filter(!is.na(Methodology))%>%
           group_by(Methodology)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = Methodology))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())

```


### Salary Ditribution Vs Participation of different developers working with different IDE

```{r, message=FALSE, warning=FALSE}
ggplotly(Survey_results%>%
           select(IDE, StackOverflowConsiderMember, StackOverflowParticipate, ConvertedSalary)%>%
           mutate(IDE = str_split(IDE, ";"))%>%
           unnest(IDE)%>%
           filter(!is.na(IDE))%>%
           group_by(IDE)%>%
           summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = T),
                     Participation = mean(StackOverflowParticipate %in% 
                                            c("Multiple times per day",
                                              "Daily or almost daily",
                                              "A few times per week", 
                                              "A few times per month or weekly"),
                                          na.rm = T),
                     Salary = mean(ConvertedSalary, na.rm = T), 
                     n = n())%>%
           ggplot(aes(Participation, Salary, label = IDE))+
           geom_smooth(method = "lm")+
           geom_point(aes(size = n , alpha = 0.6))+
           xlab("% of Participation")+
           ylab("Mean Salary")+
           theme_classic())

```



## Accessing the Potential job opportunity ranking {.tabset .tabset-fade .tabset-pills}

As mentioned in the file, the developers rank the potential job opputunities by assessing these factors. 

```{r, message=FALSE, warning=FALSE}

Potential_job = as.data.frame( c("The industry that I'd be working in","The financial performance or funding status of the company or organization", 
"The specific department or team I'd be working on", "The languages, frameworks, and other technologies I'd be working with", 
"The compensation and benefits offered", "The office environment or company culture", "The opportunity to work from home/remotely", 
"Opportunities for professional development", "The diversity of the company or organization", "How widely used or impactful the product or service I'd be working on is" )) 

names(Potential_job) = c("Description Of Factors")

Potential_job

```

So lets Analyse the distibiution developers thoughts which they rank as a potential job opputunities.

** 1 Is highest and 10 is Lowest **

Analysing the overall population then the factor got highest priorityin following order:


* The compensation and benefits offered
* The languages, frameworks, and other technologies I'd be working with
* Opportunities for professional development
* The office environment or company culture
* The opportunity to work from home/remotely
* The industry that I'd be working in
* How widely used or impactful the product or service I'd be working on is
* The specific department or team I'd be working on
* The financial performance or funding status of the company or organization
* The diversity of the company or organization

So apparantely the Compensation got highest and Diversity got lowest.

### The industry that I'd be working in ????


```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessJob1))%>%
  group_by(AssessJob1)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob1), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The industry that I'd be working in")%>%
  hc_credits(style = list(fontSize = "12px"))

```


### The financial performance or funding status of the company or organization     ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob2))%>%
  group_by(AssessJob2)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob2), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The financial performance or funding status of the company or organization")%>%
  hc_credits(style = list(fontSize = "12px"))

```


### The specific department or team I'd be working on ????


```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessJob3))%>%
  group_by(AssessJob3)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob3), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The specific department or team I'd be working on")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### The languages, frameworks, and other technologies I'd be working with ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob4))%>%
  group_by(AssessJob4)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob4), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The languages, frameworks, and other technologies I'd be working with")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### The compensation and benefits offered ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob5))%>%
  group_by(AssessJob5)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob5), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The compensation and benefits offered")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### The office environment or company culture ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob6))%>%
  group_by(AssessJob6)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob6), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The office environment or company culture")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### The opportunity to work from home/remotely ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob7))%>%
  group_by(AssessJob7)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob7), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The opportunity to work from home/remotely")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### Opportunities for professional development ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob8))%>%
  group_by(AssessJob8)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob8), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Opportunities for professional development")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### The diversity of the company or organization ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessJob9))%>%
  group_by(AssessJob9)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob9), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The diversity of the company or organization")%>%
  hc_credits(style = list(fontSize = "12px"))


```

### How widely used or impactful the product or service I'd be working on is ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessJob10))%>%
  group_by(AssessJob10)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessJob10), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "The languages, frameworks, and other technologies I'd be working with")%>%
  hc_credits(style = list(fontSize = "12px"))

```

Analysing the overall population then the factor got highest priorityin following order:


* The compensation and benefits offered
* The languages, frameworks, and other technologies I'd be working with
* Opportunities for professional development
* The office environment or company culture
* The opportunity to work from home/remotely
* The industry that I'd be working in
* How widely used or impactful the product or service I'd be working on is
* The specific department or team I'd be working on
* The financial performance or funding status of the company or organization
* The diversity of the company or organization

So apparantely the Compensation got highest and Diversity got lowest.

## Assessing Job Benefits {.tabset .tabset-fade .tabset-pills}

Lets look at the ranking of Job benefit Factors that are considered highest by the developers.
```{r, message=FALSE, warning=FALSE}

  
Job_Benefits = as.data.frame( c("Salary and/or bonuses","Stock options or shares", " Health insurance", "Parental leave",
"Fitness or wellness benefit (ex. gym membership, nutritionist)", "Retirement or pension savings matching", "Company-provided meals or snacks",
"Computer/office equipment allowance", "Childcare benefit", " Transportation benefit (ex. company-provided transportation, public transit allowance)",
"Conference or education budget" )) 

names(Job_Benefits) = c("Description Of Factors")

Job_Benefits


```

** 1 Is highest and 11 is Lowest **

Analysing the overall population then the factor got highest to lowest priority of Benefits in following order:

* Salary and/or bonuses
* Health insurance
* Computer/office equipment allowance
* Conference or education budget
* Stock options or shares
* Retirement or pension savings matching
* Parental leave
* Transportation benefit (ex. company-provided transportation, public transit allowance)
* Fitness or wellness benefit (ex. gym membership, nutritionist)
* Company-provided meals or snacks
* Childcare benefit

### Salary and/or bonuses

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits1))%>%
  group_by(AssessBenefits1)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits1), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Salary and/or bonuses")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### Stock options or shares ???

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
filter(!is.na(AssessBenefits2))%>%
  group_by(AssessBenefits2)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits2), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Stock options or shares")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Health insurance ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits3))%>%
  group_by(AssessBenefits3)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits3), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Health insurance")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Parental leave ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits4))%>%
  group_by(AssessBenefits4)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits4), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Parental leave")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Fitness or wellness benefit (ex. gym membership, nutritionist) ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits5))%>%
  group_by(AssessBenefits5)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits5), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Fitness or wellness benefit (ex. gym membership, nutritionist)")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Retirement or pension savings matching ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits6))%>%
  group_by(AssessBenefits6)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits6), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Retirement or pension savings matching")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Company-provided meals or snacks ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits7))%>%
  group_by(AssessBenefits7)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits7), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Company-provided meals or snacks")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Computer/office equipment allowance ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits8))%>%
  group_by(AssessBenefits8)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits8), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Computer/office equipment allowance")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Childcare benefit ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits9))%>%
  group_by(AssessBenefits9)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits9), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Childcare benefit")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Transportation benefit (ex. company-provided transportation, public transit allowance) ???

```{r, message=FALSE, warning=FALSE}


Survey_results%>%
filter(!is.na(AssessBenefits10))%>%
  group_by(AssessBenefits10)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits10), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Transportation benefit (ex. company-provided transportation, public transit allowance)")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Conference or education budget

```{r, message=FALSE, warning=FALSE}
Survey_results%>%
filter(!is.na(AssessBenefits11))%>%
  group_by(AssessBenefits11)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(AssessBenefits11), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Conference or education budget")%>%
  hc_credits(style = list(fontSize = "12px"))
```



## Assessing priorities for Job Contact {.tabset .tabset-fade .tabset-pills}

The dataset along with us have ways in which contacts are made, So lets check which way of conatcting developers got highest priority.

Analysing the overall population, the mode of contact got highest to lowest priority are in following order:

* Email to my private address
* Telephone call
* Message on a job site
* Email to my work address
* Message on a social media site

```{r, message=FALSE, warning=FALSE}

  
Job_Contact = as.data.frame( c( "Telephone call",
                                "Email to my private address",
                                "Email to my work address",
                                "Message on a job site",
                                "Message on a social media site" )) 

names(Job_Contact) = c("Description Of Factors")

Job_Contact


```


###  Telephone call

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(JobContactPriorities1))%>%
  group_by(JobContactPriorities1)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(JobContactPriorities1), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = " Telephone call")%>%
  hc_credits(style = list(fontSize = "12px"))
```

### Email to my private address

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(JobContactPriorities2))%>%
  group_by(JobContactPriorities2)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(JobContactPriorities2), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Email to my private address")%>%
  hc_credits(style = list(fontSize = "12px"))

```

###  Email to my work address

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(JobContactPriorities3))%>%
  group_by(JobContactPriorities3)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(JobContactPriorities3), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = " Email to my work address")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### Message on a job site

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(JobContactPriorities4))%>%
  group_by(JobContactPriorities4)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(JobContactPriorities4), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Message on a job site")%>%
  hc_credits(style = list(fontSize = "12px"))

```

### Message on a social media site

```{r, message=FALSE, warning=FALSE}

Survey_results%>%
  filter(!is.na(JobContactPriorities5))%>%
  group_by(JobContactPriorities5)%>%
  summarise(n = n())%>%
  hchart("pie", innerSize = '50%', showInLegend = F,
         hcaes(x = as.factor(JobContactPriorities5), y = (n/total_rows * 100)))%>%
  hc_add_theme(hc_theme_flat())%>%
  hc_title(text = "Message on a social media site")%>%
  hc_credits(style = list(fontSize = "12px"))

```



## Stay Tuned for daily updates please do suggest and upvote..............


