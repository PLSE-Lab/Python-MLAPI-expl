---
title: "Stack Overflow Community"
output: 
    html_document:
        toc: yes
        theme: cosmo
        highlight: tango
        code_folding: hide
---

```{r }
knitr::opts_chunk$set(echo = FALSE,message=FALSE,warning=FALSE)
```

# Introduction
    This kernel gives detailed insights about  Stack Overflow Community , who is part of that community,participate and contribute to that community.
    Analyse the Stack overflow Community member along with their personal details, job etc.

```{r cars}
library(tidyverse)
library(scales)
library(ggridges)
library(ggrepel)
library(treemap)
library(grid)
library(gridExtra)
library(packcircles)
library(RColorBrewer)
library(viridis)

```

```{r  echo=FALSE}
so_survey<-read.csv("../input/survey_results_public.csv")
```


```{r}
colors=c("#2b2404","#f94842","#e59a49","#e5cf29","#ace528","#50e00d","#10ceb5","#077728","#43d3e0","#0b2ac4","#3c0bc4","#700bc4","#b10bc4","#c40b92","#c40b39","#c40b0b","#09172d","#0f042b","#042b20","#453fdd")
```

```{r}
percent <-function(col,tab=so_survey){
    tab %>% 
    filter_(!col=="")%>%
    group_by_(col)%>%
    summarise(tot=n())%>%
    mutate(percent=round(tot/sum(tot)*100))%>%
    arrange(desc(tot))
}

ccount <-function(col,tab=so_survey){
    tab %>% 
    filter_(!col=="")%>%
    group_by_(col)%>%
    summarise(cnt=n())%>%
    arrange(desc(cnt))
}
```
# Stackoverflow Community
```{r}
percent(col="StackOverflowConsiderMember")%>%filter(!is.na(StackOverflowConsiderMember))%>%ggplot(aes(x=reorder(StackOverflowConsiderMember,percent),y=percent,fill=StackOverflowConsiderMember))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowConsiderMember,0.5,label=paste(StackOverflowConsiderMember," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=viridis(3))+labs(title="Do you consider yourself a member of the Stack Overflow community?")
```

Around 43 % of respondents was feeling their belonging in SO Community, almost 23 % of didn't answer to this questions, Which infact impact our analysis.  
Will consider as well in the category of either **NO** or **I am not Sure**.  

Furtherly let's analyse what are the factors which effect the respondents to consider them as part of the SO Community.  

# Detailed Analysis- Who is part of Community 
 
## Top Countries 
```{r}
data<-so_survey %>%select(Country,StackOverflowConsiderMember)%>%filter(!is.na(Country),!is.na(StackOverflowConsiderMember))%>%filter(StackOverflowConsiderMember =="Yes")%>%group_by(Country)%>%summarise(cnt=n())%>%arrange(desc(cnt))%>%head(20)

data<-data.frame(grp=paste(data$Country,"\n ",data$cnt),value=data$cnt)

packing <- circleProgressiveLayout(data$value, sizetype='area')
data <- cbind(data, packing)
dat.gg <- circleLayoutVertices(packing, npoints=30)

ggplot() + 
  geom_polygon(data = dat.gg, aes(x, y, group = id, fill=id), colour = "black") +scale_fill_gradientn(colors=viridis(50))+
  geom_text(data = data, aes(x, y, size=value, label = grp),size=2,col="white")+scale_size_continuous(range = c(1,4))+ theme_void()+
  theme(legend.position = "none")+labs(title="Top Countries feeling their belonging")+coord_equal()
```

USA is the top most country, who is part of SO Community, next comes INDIA and United Kingdom.  
This is only based on the respondent count from each country who are feeling their membership.  


## Top Countries Participation
```{r}
data<-so_survey %>%select(Country,StackOverflowParticipate)%>%filter(!is.na(Country),!is.na(StackOverflowParticipate))%>%filter(StackOverflowParticipate %in% 
                                                                                                                                  c("Multiple times per day",
                                                                 "Daily or almost daily",
                                                                 "A few times per week",
                                                                 "A few times per month or weekly") )%>%group_by(Country)%>%summarise(cnt=n())%>%arrange(desc(cnt))%>%head(20)

data<-data.frame(grp=paste(data$Country,"\n ",data$cnt),value=data$cnt)

packing <- circleProgressiveLayout(data$value, sizetype='area')
data <- cbind(data, packing)
dat.gg <- circleLayoutVertices(packing, npoints=30)

ggplot() + 
  geom_polygon(data = dat.gg, aes(x, y, group = id, fill=id), colour = "black") +scale_fill_gradientn(colors= viridis::viridis(50))+geom_text(data = data, aes(x, y, size=value, label = grp),size=2,col="white")+scale_size_continuous(range = c(1,4))+
  theme_void()+theme(legend.position = "none")+labs(title="Top Countries participation in SO")+coord_equal()
```

Again the first three counties who feel their membership, particiapte more, USA,Inida,United Kingdom.  

## Top Countries frequent visits
```{r}
data<-so_survey %>%select(Country,StackOverflowVisit)%>%filter(!is.na(Country),!is.na(StackOverflowVisit))%>%filter(StackOverflowVisit %in% 
                                                                                                                                  c("Multiple times per day",
                                                                 "Daily or almost daily",
                                                                 "A few times per week",
                                                                 "A few times per month or weekly") )%>%group_by(Country)%>%summarise(cnt=n())%>%arrange(desc(cnt))%>%head(20)

data<-data.frame(grp=paste(data$Country,"\n ",data$cnt),value=data$cnt)

packing <- circleProgressiveLayout(data$value, sizetype='area')
data <- cbind(data, packing)
dat.gg <- circleLayoutVertices(packing, npoints=30)

ggplot() + 
  geom_polygon(data = dat.gg, aes(x, y, group = id, fill=id), colour = "black") +scale_fill_gradientn(colors=viridis::viridis(50))+
  geom_text(data = data, aes(x, y, size=value, label = grp),size=2,col="white")+scale_size_continuous(range = c(1,4))+ 
  theme_void()+theme(legend.position = "none")+labs(title="Top Countries Visit in SO")+coord_equal()
```

Here there is slight variation in top 3 countries , USA and India do participate and feel membership as usual, but now in frequent visits United Kindom goes to 4th place.  
Instead come to 3rd position in frequent visits.  

## Stack Overflow Jobs by SO Dev Story
```{r}

so_survey%>%select(StackOverflowJobs,StackOverflowDevStory)%>%filter(!is.na(StackOverflowJobs),!is.na(StackOverflowDevStory))%>%group_by(StackOverflowJobs,StackOverflowDevStory)%>%summarise(t=n())%>%ggplot(aes(x=reorder(StackOverflowJobs,t),y=t,fill=StackOverflowDevStory))+geom_bar(stat="identity",position="dodge")+scale_fill_manual(values =viridis(10))+theme_minimal()+labs(y="",x="Stackoverflow Jobs ")+
scale_y_continuous(labels = comma_format())+theme(axis.text.x = element_text(angle=0))+scale_x_discrete(labels = wrap_format(10))
```

## Stack Overflow Account by SO Member
```{r}
so_survey%>%select(StackOverflowHasAccount,StackOverflowConsiderMember)%>%filter(!is.na(StackOverflowHasAccount),!is.na(StackOverflowConsiderMember))%>%group_by(StackOverflowHasAccount,StackOverflowConsiderMember)%>%summarise(so=n())%>%ungroup()%>%ggplot(aes(StackOverflowHasAccount,so,fill=StackOverflowConsiderMember))+geom_bar(stat="identity",position="fill")+scale_fill_manual(values=viridis(10))+theme_minimal()+labs(y="")+scale_y_continuous(labels = percent_format())

```  
  
50% of respondents who has account was a member.  
Developers who has SO account consider them to be part of  their member, who doen't have account, doesn't most likely to be part of community.  

## Stack Overflow Recommend by SO Member
```{r}
so_survey%>%select(StackOverflowRecommend,StackOverflowConsiderMember)%>%filter(!is.na(StackOverflowRecommend),!is.na(StackOverflowConsiderMember))%>%group_by(StackOverflowRecommend,StackOverflowConsiderMember)%>%summarise(so=n())%>%ungroup()%>%ggplot(aes(reorder(StackOverflowRecommend,so),so,fill=StackOverflowConsiderMember))+geom_bar(stat="identity",position="dodge")+scale_fill_manual(values=viridis(10))+theme_minimal()+labs(y="",x="Stackoverflow Recommend")+scale_y_continuous(labels = comma_format())+theme(axis.text.x = element_text(angle=90))
```
## Stack Overflow Visits by SO member
```{r}

so_survey%>%select(StackOverflowVisit,StackOverflowConsiderMember)%>%filter(!is.na(StackOverflowVisit),!is.na(StackOverflowConsiderMember))%>%group_by(StackOverflowVisit,StackOverflowConsiderMember)%>%summarise(t=n())%>%ggplot(aes(x=reorder(StackOverflowConsiderMember,t),y=t,fill=StackOverflowVisit))+geom_bar(stat="identity",position="dodge")+scale_fill_manual(values=viridis(10))+theme_minimal()+labs(y="",x="Consider Member")+scale_y_continuous(labels = comma_format())
```

## Developers -who consider to be part of Community  
```{r}
so_survey %>%
  select(DevType, StackOverflowConsiderMember) %>%
  filter(!is.na(DevType),!is.na(StackOverflowConsiderMember)) %>%mutate(DevType=strsplit(as.character(DevType),";"))%>%unnest(DevType)%>%
  group_by(DevType) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),
            n = n()) %>%
    ggplot(aes(reorder(DevType,ConsiderMember), ConsiderMember, fill = ConsiderMember, label = DevType)) +geom_bar(stat="identity")+
      coord_flip() +
  theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+
  geom_text( aes(DevType,0.01,label=DevType), color="white", hjust=0,size=4) +scale_fill_gradientn(colors=viridis::viridis(50))+labs(title="Developers by Consider Member")+scale_y_continuous(label=percent_format())
```
## Under Graduates - Who consider to be part of SO Community
```{r}
so_survey %>%
  select(UndergradMajor, StackOverflowConsiderMember) %>%
  filter(!is.na(UndergradMajor),!is.na(StackOverflowConsiderMember))%>%mutate(UndergradMajor=strsplit(as.character(UndergradMajor),";"))%>%
                                                                                  unnest(UndergradMajor)%>%
  group_by(UndergradMajor) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),n = n()) %>%
    ggplot(aes(reorder(UndergradMajor,ConsiderMember), ConsiderMember, fill = ConsiderMember)) +geom_bar(stat="identity")+
      coord_flip() +
  theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+
  geom_text( aes(UndergradMajor,0.01,label=UndergradMajor), color="white", hjust=0,size=4) +scale_fill_gradientn(colors=viridis::viridis(50))+labs(title="Under Graduates by Consider Member")+scale_y_continuous(label=percent_format())
```

## Languages Worked With - Who consider to be part of SO Community
```{r fig.width=8,fig.height=8}
lang<-so_survey %>%
  select(LanguageWorkedWith, StackOverflowConsiderMember) %>%
  filter(!is.na(LanguageWorkedWith),!is.na(StackOverflowConsiderMember))%>%mutate(LanguageWorkedWith=strsplit(as.character(LanguageWorkedWith),";"))%>%
                                                                                  unnest(LanguageWorkedWith)%>%
  group_by(LanguageWorkedWith) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),n = n()) 



lang%>%ggplot(aes(x=reorder(LanguageWorkedWith,ConsiderMember),y=ConsiderMember))+geom_point(aes(color=n,size=n),alpha=0.7)   +theme_minimal()+coord_flip()+
  theme(
    axis.text.y= element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )  +labs(title="Languages Worked by Consier Member " ,subtitle="Avg Language Developers by Consider Member",size="Respondents Count")+scale_y_continuous(label=percent_format())+geom_text_repel(aes(label=LanguageWorkedWith))+scale_color_gradientn(colors = viridis::viridis(200))+scale_radius(range=c(1,20))

```
## Database Worked With- Who consider to be part of SO Community
```{r fig.width=8,fig.height=8}
lang<-so_survey %>%
  select(DatabaseWorkedWith, StackOverflowConsiderMember) %>%
  filter(!is.na(DatabaseWorkedWith),!is.na(StackOverflowConsiderMember))%>%mutate(DatabaseWorkedWith=strsplit(as.character(DatabaseWorkedWith),";"))%>%
                                                                                  unnest(DatabaseWorkedWith)%>%
  group_by(DatabaseWorkedWith) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),n = n()) 



lang%>%ggplot(aes(x=reorder(DatabaseWorkedWith,ConsiderMember),y=ConsiderMember,fill=ConsiderMember))+geom_bar(stat="identity")+
  scale_fill_gradientn(colors=viridis::viridis(50)) +theme_minimal()+coord_flip()+
  theme(
    
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )  +labs(title="Database- Respondens consider them as part of community" ,subtitle="Avg Database Developers by Consider Member")+scale_y_continuous(label=percent_format())

```

## Prof Years of Coding - Who consider to be part of SO Community
```{r fig.width=6,fig.height=6}


so_survey %>%
  select(YearsCodingProf, StackOverflowConsiderMember) %>%
  filter(!is.na(YearsCodingProf),!is.na(StackOverflowConsiderMember))%>%
  group_by(YearsCodingProf) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),n = n()) %>%



ggplot(aes(x=reorder(YearsCodingProf,ConsiderMember),y=ConsiderMember,group=1))+geom_line(col=viridis(1),size=1.5)+geom_point()+
 theme_minimal()+
  theme(
    
    
    panel.grid = element_blank(),
        legend.position = "none" ,axis.text.x = element_text(angle=90))  +labs(title="Prof Coding Exp by Consider Member" ,x="Prof Coding Exp", y="Avg  Respondents part of community")+scale_y_continuous(label=percent_format())

```

## Which Country respondents consider them as part of Community  
```{r}
so_survey %>%
  select(Country, StackOverflowConsiderMember, StackOverflowParticipate) %>%
    filter(!is.na(Country),!is.na(StackOverflowConsiderMember), !is.na(StackOverflowParticipate)) %>%
    group_by(Country) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = TRUE),
            Participation = mean(StackOverflowParticipate %in% c("Multiple times per day",
                                                                 "Daily or almost daily",
                                                                 "A few times per week",
                                                                 "A few times per month or weekly"),na.rm = TRUE),n = n()) %>%
  filter(n > 500) %>%
  ggplot(aes(Participation, ConsiderMember, label = Country,col=ConsiderMember)) +
   geom_text_repel(size = 3, point.padding = 0.25) +
  geom_point(aes(size = n), alpha = 1) +
  scale_y_continuous(labels = percent_format()) +
  scale_x_continuous(labels = percent_format()) +
  scale_size_continuous(labels = comma_format()) +theme_minimal()+scale_color_gradientn(colors=viridis::viridis(100))+theme(legend.position = "none")+
  labs(x = "% who participate at least weekly", 
       y = "% who consider themselves as SO Community",
       title = "Considering membership by Country and Particiaption",
              size = "Number of respondents")
```  

There exist correlation between who participate in SO Community are most likely to be a member.  
**INDIA** is the top country in both participation and membership.  

## SO Community member who has Account by Country  
```{r}
so_survey %>%
  select(Country, StackOverflowConsiderMember, StackOverflowHasAccount) %>%
    filter(!is.na(Country),!is.na(StackOverflowConsiderMember), !is.na(StackOverflowHasAccount)) %>%
    group_by(Country) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = TRUE),
            Account = mean(StackOverflowHasAccount =="Yes"),n = n()) %>%
  filter(n > 400) %>%
  ggplot(aes(Account, ConsiderMember, label = Country,col=ConsiderMember)) +
   geom_text_repel(size = 3, point.padding = 0.25) +
  geom_point(aes(size = n), alpha = 1) +
  scale_y_continuous(labels = percent_format()) +
  scale_x_continuous(labels = percent_format()) +
  scale_size_continuous(labels = comma_format()) +theme_minimal()+scale_color_gradientn(colors=viridis::viridis(50))+theme(legend.position = "none")+
  labs(x = "% who has SO Account", 
       y = "% who consider themselves as SO Community",
       title = "Considering Membership by having SO Account /Country "
              )

```
  
As we saw earlier, having an account in community makes them feel as a member, Bangladesh is at the top of having account and be part of membership.  
Next comes India.  




## Which Country developers consider them as SO community as per their visits
```{r}
so_survey %>%
  select(Country, StackOverflowConsiderMember, StackOverflowVisit) %>%
    filter(!is.na(Country),!is.na(StackOverflowConsiderMember), !is.na(StackOverflowVisit)) %>%
    group_by(Country) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = TRUE),
            Visit = mean(StackOverflowVisit %in% c("Multiple times per day",
                                                                 "Daily or almost daily",
                                                                 "A few times per week",
                                                                 "A few times per month or weekly"),na.rm = TRUE),n = n()) %>%
  filter(n > 700) %>%
  ggplot(aes(Visit, ConsiderMember, label = Country,col=ConsiderMember)) +
   geom_text_repel(size = 3, point.padding = 0.25) +
  geom_point(aes(size = n), alpha = 1) +
  scale_y_continuous(labels = percent_format()) +
  scale_x_continuous(labels = percent_format()) +
  scale_size_continuous(labels = comma_format()) +theme_minimal()+scale_color_gradientn(colors=viridis::viridis(50))+theme(legend.position = "none")+
  labs(x = "% who Visits at least weekly", 
       y = "% who consider themselves as SO Community ",
       title = "Community Memship by Country and visits")
              
```
## Developers/Coding Exp  by Community member  
```{r}
exp_lang<-so_survey%>%select(LanguageWorkedWith,StackOverflowConsiderMember,YearsCodingProf)%>%filter(!is.na(LanguageWorkedWith),!is.na(StackOverflowConsiderMember),!is.na(YearsCodingProf))%>%mutate(YearsCoding=parse_number(YearsCodingProf))%>%
  mutate(LanguageWorkedWith=str_split(LanguageWorkedWith,";"))%>%unnest(LanguageWorkedWith)%>%group_by(LanguageWorkedWith)%>%summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),n=n(),avg_years=mean(YearsCoding))%>%arrange(desc(n))%>%ungroup()%>%filter(n>1000)



exp_lang%>%ggplot(aes(avg_years,ConsiderMember,size=n,label=LanguageWorkedWith,col=ConsiderMember))+geom_point(alpha=1)+geom_text_repel(size = 3, point.padding = 0.25)+
  theme_minimal()+labs(title="Which Language Developers consider to be part of community ",x="Avg Coding Experience",y="% of Community member")+
  scale_color_gradientn(colors=viridis::viridis(100))+scale_y_continuous(label=percent_format())+scale_size_continuous(labels = comma_format())+theme(legend.position = "none")
```

## Age/Gender - Who is part of Community member   
```{r}
age_gender<-so_survey%>%select(Gender,StackOverflowConsiderMember,Age)%>%filter(!is.na(Gender),!is.na(StackOverflowConsiderMember),!is.na(Age))%>%mutate(age=parse_number(Age),Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%
  filter(Gender %in% c("Male","Female"))%>%group_by(Age,Gender)%>%summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes"),n=n(),avg_age=mean(age))%>%arrange(desc(n))%>%ungroup()
age_gender%>%ggplot(aes(avg_age,ConsiderMember,group=Gender,col=Gender))+geom_line(alpha=1,size=1.5)+
  theme_minimal()+labs(title="Gender by Age - Who is part of SO Community",x="Avg Age",y="% Community Member")+scale_color_manual(values=viridis(2))+scale_y_continuous(label=percent_format())+theme(legend.position = "bottom")

```

Male repondents are feeling their belonging to the community than female.


## Coding as Hobby / Years of coding - will impact their belonging to community
```{r}
so_survey %>%select(Hobby,StackOverflowConsiderMember,YearsCodingProf)%>%filter(!is.na(StackOverflowConsiderMember),!is.na(YearsCodingProf))%>%mutate(YearsCoding=parse_number(YearsCodingProf))%>%group_by(Hobby,YearsCoding)%>%mutate(ConsiderMember=mean(StackOverflowConsiderMember=="Yes"),AvgYears=mean(YearsCoding),n())%>%ungroup()%>%
  ggplot(aes(AvgYears,ConsiderMember,group=Hobby,color=Hobby))+geom_line(size=1.5)+theme_minimal()+scale_color_manual(values=viridis(2))+
  scale_y_continuous(label=percent_format())+
  scale_x_continuous(label=comma_format())+labs(title="consideration as Community by Avg Years of Coding Exp and Coding as Hobby ",x="Avg Years of Coding Exp",y="% Considering as Community")
```
Yes, definitely who has taken coding as hobby considering them as community member when comparing to those who don't code for Hobby.  


## SO Community Member by Developer Type and Job Search status  
```{r}
so_survey %>%
  select(DevType, StackOverflowConsiderMember, JobSearchStatus) %>%
  mutate(DevType = str_split(DevType, pattern = ";")) %>%
  unnest(DevType) %>%
  mutate(JobSearchStatus= str_split(JobSearchStatus,pattern=";"))%>%
  unnest(JobSearchStatus)%>%
  filter(!is.na(DevType)) %>%
  mutate(DevType = case_when(str_detect(DevType, "Data scientist") ~ "Data scientist",
                             str_detect(DevType, "academic") ~ "Academic researcher",
                             TRUE ~ DevType)) %>%
  group_by(DevType) %>%
  summarise(ConsiderMember = mean(StackOverflowConsiderMember == "Yes", na.rm = TRUE),
            Jobsearch = mean(JobSearchStatus %in% c("I am actively looking for a job")),
            n = n()) %>%
  filter(n > 1500) %>%
  ggplot(aes(Jobsearch, ConsiderMember, label = DevType,col=ConsiderMember)) +
    geom_text_repel(size = 3, point.padding = 0.25) +
  geom_point(aes(size = n), alpha = 1) +
  scale_y_continuous(labels = percent_format()) + scale_x_continuous(labels = percent_format())+
    scale_size_continuous(labels = comma_format()) +
  scale_color_gradientn(colors=viridis::viridis(50))+theme_minimal()+
  labs(x = "% Job Search status", 
       y = "% Consider as SO Community",
       title = "SO Community Member by Developer Type and Job Search status"
              )+theme(legend.position = "none")
```


## Connection to other Developers
Find out which developers Strongly agree towards kinship with co-developers.
```{r fig.height=12,fig.width=10}
kins<-so_survey%>%select(Gender,DevType,AgreeDisagree1)%>%filter(!is.na(Gender),!is.na(DevType),!is.na(AgreeDisagree1))%>%mutate(Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%
  filter(Gender %in% c("Male","Female"))%>% filter(AgreeDisagree1 %in% c("Agree","Strongly agree"))%>%
  mutate(DevType=strsplit(as.character(DevType),";"))%>%unnest(DevType)%>%group_by(Gender,DevType)%>%summarise(n=n())%>%ungroup()

kins%>%ggplot(aes(reorder(DevType,n),n,fill=Gender))+geom_bar(stat="identity",position="dodge")+coord_flip()+theme_minimal()+labs(title="Kinship by Developer Type and Gender",y="Count",x="Developer") +scale_fill_manual(values=c(colors))+scale_y_continuous(label=comma_format())+theme(legend.position = "bottom")
```

## To be Continued..