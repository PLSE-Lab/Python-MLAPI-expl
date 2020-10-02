---
title: "Stack Overflow Developer Survey"
output: 
    html_document:
        toc: yes
        theme: cosmo
        highlight: tango
        code_folding: hide
---

```{r}
knitr::opts_chunk$set(warning=FALSE,message=FALSE)
```
# Loading libraries  

```{r}
library(tidyverse)
library(RColorBrewer)
library(packcircles)
library(grid)
library(gridExtra)
library(plotrix)
library(scales)
library(ggridges)
library(knitr)
library(treemap)
library(ggrepel)
```
# Sourcing Data  

Dataset has 98855 observations and   129 variables.
```{r}
so_survey<-read.csv("../input/survey_results_public.csv")

```

```{r}
colors=c("#f94842","#e59a49","#e5cf29","#ace528","#50e00d","#10ceb5","#077728","#43d3e0","#0b2ac4","#3c0bc4","#700bc4","#b10bc4","#c40b92","#c40b39","#c40b0b","#09172d","#0f042b",
"#042b20","#2b2404","#0c2d3a","#0b173a")
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

# Respondent Details  {.tabset .tabset-fade}

## Country  
Around 183 countries participated in this survey, let us find out which country has got more respondents and their percentage contribution to this survey.  
```{r}
data<-ccount(col="Country")%>%filter(!Country=="")
data<-data.frame(grp=paste(data$Country,"\n ",data$cnt),value=data$cnt)

packing <- circleProgressiveLayout(data$value, sizetype='area')
data <- cbind(data, packing)
dat.gg <- circleLayoutVertices(packing, npoints=30)

ggplot() + 
  geom_polygon(data = dat.gg, aes(x, y, group = id, fill=as.factor(id)), colour = "black") +scale_fill_manual(values= c(colors,colors,colors,colors,colors,colors,colors,colors,colors,colors))+geom_text(data = data, aes(x, y, size=value, label = grp),size=2,col="white")+scale_size_continuous(range = c(1,4))+ theme_void()+theme(legend.position = "none")+labs(title="Countrywise Respondents")+coord_equal()
```


## Gender  
```{r}
so_survey %>%filter(!is.na(Gender))%>%mutate(Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%group_by(Gender)%>%summarise(gcount=n())%>%mutate(p=round(gcount/sum(gcount)*100,2))%>%arrange(desc(p))%>%
ggplot(aes(x=Gender,y=p,fill=Gender)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=0,vjust=0.5),legend.position='none',plot.title = element_text(size=10)) +scale_fill_manual(values=colors)+labs(title="Which of the following do you currently identify as?")+scale_x_discrete(labels = wrap_format(10))+geom_text( aes(Gender,p,label=paste(p,"%")))
```

## Age
```{r}
percent(col="Age")%>%filter(!is.na(Age))%>%ggplot(aes(x=reorder(Age,percent),y=percent,fill=Age))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(Age,0.5,label=paste(Age," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="What is your age?",subtitle="Age - Percentage Contribution to survey")
```

### Education Perents
```{r}
percent(col="EducationParents")%>%filter(!is.na(EducationParents))%>%ggplot(aes(x=reorder(EducationParents,percent),y=percent,fill=EducationParents)) + geom_bar(stat='identity')+coord_flip()+theme_minimal()+
  theme(axis.text = element_blank(),  axis.text.x = element_blank(),axis.text.y = element_blank(),   axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(x=EducationParents, 0.5, label=paste(EducationParents," ",percent,"%")), color="black", fontface="bold" ,hjust=0,size=3) +scale_fill_manual(values=colors)+labs(title="What is the highest level of education received by either of your parents?")
```

## Race
```{r}
so_survey%>%filter(!is.na(RaceEthnicity))%>%mutate(RaceEthnicity=strsplit(as.character(RaceEthnicity),";"))%>%unnest(RaceEthnicity)%>%percent(col="RaceEthnicity")%>%ggplot(aes(x=reorder(RaceEthnicity,percent),y=percent,fill=RaceEthnicity)) + geom_bar(stat='identity') +coord_flip()+theme_minimal()+
  theme(axis.text = element_blank(),  axis.text.x = element_blank(),axis.text.y = element_blank(),   axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(x=RaceEthnicity, 0.5, label=paste(RaceEthnicity," ",percent,"%")), color="black", fontface="bold" ,hjust=0,size=3) +scale_fill_manual(values=colors)+labs(title="RaceEthnicity")
```

## Dependents
```{r}
percent(col="Dependents")%>%filter(!is.na(Dependents))%>%ggplot(aes(x=Dependents,y=percent,fill=Dependents)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=12)) +scale_fill_manual(values=colors)+labs(title="Do you have any children or other dependents that you care for?")
```


# Education {.tabset .tabset-fade} 

## How many are Student
```{r}

levels(so_survey$Student)<-c(levels(so_survey$Student),"None")
so_survey$Student[is.na(so_survey$Student)] <- "None" 
percent("Student")%>%
ggplot(aes(x=Student,y=percent,fill=Student)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=10)) +scale_fill_manual(values=colors)+labs(title="Are you currently enrolled in a formal, degree-granting college or university program?")
```

## Employment Status
```{r}
  percent("Employment")%>%filter(!is.na(Employment))%>%
ggplot(aes(x=reorder(Employment,percent),y=percent,fill=Employment,label=Employment)) + geom_bar(stat='identity')  +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(Employment,0.5,label=paste(Employment," ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Which of the following best describes your current employment status?",subtitle="Employment Status Percentage")
```

## Formal Education
```{r}
  percent("FormalEducation")%>%filter(!is.na(FormalEducation))%>%
ggplot(aes(x=reorder(FormalEducation,percent),y=percent,fill=FormalEducation,label=FormalEducation)) + geom_bar(stat='identity')  +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(FormalEducation,0.5,label=paste(FormalEducation," ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Which of the following best describes the highest level of formal education that you’ve completed?
",subtitle="Formal Education in Percentage")
```

## Undergraduate Major
```{r}
  percent(col="UndergradMajor")%>%filter(!is.na(UndergradMajor))%>%ggplot(aes(x=reorder(UndergradMajor,percent),y=percent,fill=UndergradMajor))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(UndergradMajor,0.5,label=paste(UndergradMajor," ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="You previously indicated that you went to a college or university.\n Which of the following best describes your main field of study (aka 'major')?",subtitle="Under Graduates")
```

# Employer {.tabset .tabset-fade}
## Company Size
```{r}
percent(col="CompanySize")%>%filter(!is.na(CompanySize))%>%ggplot(aes(x=reorder(CompanySize,percent),y=percent,fill=CompanySize))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(CompanySize,0.5,label=paste(CompanySize," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Approximately how many people are employed by the company or organization you work for?",subtitle="Company Size")
```

## Developer Type
```{r}
so_survey %>%filter(!is.na(DevType))%>%mutate(DevType=strsplit(as.character(DevType),";"))%>%unnest(DevType)%>%percent(col="DevType")%>%ggplot(aes(x=reorder(DevType,percent),y=percent,fill=DevType))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(DevType,0.5,label=paste(DevType," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="",subtitle="Developer Type- Percentage Contribution")
```

# Career {.tabset .tabset-fade}
## Hobby 
```{r}
percent(col="Hobby")%>%ggplot(aes(x=Hobby,y=percent,fill=Hobby)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=12)) +scale_fill_manual(values=colors)+labs(title="Coding as Hobby")
```

## Coding Experience
```{r}
ccount(col="YearsCoding")%>%filter(!is.na(YearsCoding))%>%ggplot(aes(x=reorder(YearsCoding,-cnt),y=cnt,fill=YearsCoding))+geom_bar(stat="identity") +theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   , axis.text.x = element_text(angle=90), legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(YearsCoding,cnt,label=cnt), color="black", hjust=0.4,size=4,vjust=0.1) +scale_fill_manual(values=colors)+labs(title="Including any education, for how many years have you been coding?",subtitle="Years Spent in Coding -Count")

ccount(col="YearsCodingProf")%>%filter(!is.na(YearsCodingProf))%>%ggplot(aes(x=reorder(YearsCodingProf,-cnt),y=cnt,fill=YearsCodingProf))+geom_bar(stat="identity") +theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   , axis.text.x = element_text(angle=90), legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(YearsCodingProf,cnt,label=cnt), color="black", hjust=0.4,size=4,vjust=0.1) +scale_fill_manual(values=colors)+labs(title="For how many years have you coded professionally (as a part of your work)?",subtitle="Professional Coding Experience -Count")
```

## Job Satisfaction
```{r}
js<-percent(col="JobSatisfaction")%>%filter(!is.na(JobSatisfaction))
pie3D(js$percent,labels = paste(js$JobSatisfaction," ",js$percent,"%"),explode = 0.1,labelcex=0.7, main = "How satisfied are you with your current job? ",radius=1.5,col=colors)
```

## Career Satisfaction
```{r}
cs<-percent(col="CareerSatisfaction")%>%filter(!is.na(CareerSatisfaction))

pie3D(cs$percent,labels = paste(cs$CareerSatisfaction," ",cs$percent,"%"),explode = 0.1,labelcex=0.7, main = "Overall, how satisfied are you with your career thus far?",radius=1.8,col=colors)
```

## Open Source Contribution
```{r}
percent(col="OpenSource")%>%ggplot(aes(x=OpenSource,y=percent,fill=OpenSource)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=12)) +scale_fill_manual(values=colors)+labs(title="Contribution to Open Source Projects")
```

## Next Five Years Plan
```{r}
percent(col="HopeFiveYears")%>%filter(!is.na(HopeFiveYears))%>%ggplot(aes(x=reorder(HopeFiveYears,percent),y=percent,fill=HopeFiveYears))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(HopeFiveYears,0.5,label=paste(HopeFiveYears," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Which of the following best describes what you hope to be doing in five years?")
```

## Job Search Status {.tabset .tabset-fade}

### Which Age group is looking for a Job
```{r}
so_survey %>%select(Age,JobSearchStatus,Gender)%>%filter(!is.na(Age),!is.na(JobSearchStatus),!is.na(Gender))%>%mutate(Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%group_by(Age,JobSearchStatus,Gender)%>%filter(Gender %in% c("Male","Female"))%>%summarise(js=n())%>%ungroup()%>%ggplot(aes(Gender,js,fill=Age))+geom_bar(stat="identity",position="fill")+facet_wrap(~JobSearchStatus, labeller = labeller(JobSearchStatus= label_wrap_gen(15)))+scale_fill_manual(values=colors)+theme_minimal()+labs(y="",title="Which Age group and Gender is looking for new Job")+scale_y_continuous(labels = percent_format())
```

### Job Search by Country
```{r}
so_survey%>%select(Country,JobSearchStatus)%>%filter(!is.na(JobSearchStatus))%>%filter(Country %in% c("United States","India","Germany","United Kingdom","Russian Federation"))%>%group_by(Country,JobSearchStatus)%>%summarise(t=n())%>%mutate(p=round(t/sum(t)*100),2)%>%ungroup()%>%ggplot(aes(x=reorder(Country,p),p,fill=JobSearchStatus))+geom_bar(stat="identity",position="dodge")+coord_flip()+theme_minimal()+
  theme(    axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "bottom",plot.title = element_text(size=10) ) +scale_fill_manual(values=colors)+labs(title="Job Search Status by Country")

```
  
# Programming Tools/Languages {.tabset .tabset-fade}

In this section, lets find out the which programming languages, tools, database etc developers are working, and which one they want to work next year.  

## Language Worked
```{r fig.width=6,fig.height=6}

lang_cur<-so_survey %>% select(LanguageWorkedWith) %>%mutate(LanguageWorkedWith=strsplit(as.character(LanguageWorkedWith),";"))%>%unnest(LanguageWorkedWith)%>%percent(col="LanguageWorkedWith")%>%filter(!is.na(LanguageWorkedWith))%>%filter(percent>0)

label_data<-data.frame(id=seq(1:nrow(lang_cur)),
                       lbl=paste(lang_cur$LanguageWorkedWith,lang_cur$percent,"%"),
                       value=lang_cur$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

lang_cur%>%ggplot(aes(x=reorder(LanguageWorkedWith,-percent),y=percent,fill=LanguageWorkedWith,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-20,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Programming/scripting/Markup Languages\n Working currently" )

```

## Language for Next Year
```{r fig.width=6,fig.height=6}
lang_ny<-so_survey %>% select(LanguageDesireNextYear) %>%mutate(LanguageDesireNextYear=strsplit(as.character(LanguageDesireNextYear),";"))%>%unnest(LanguageDesireNextYear)%>%percent(col="LanguageDesireNextYear")%>%filter(!is.na(LanguageDesireNextYear))%>%filter(percent>0)




label_data<-data.frame(id=seq(1:nrow(lang_ny)),
                       lbl=paste(lang_ny$LanguageDesireNextYear,lang_ny$percent,"%"),
                       value=lang_ny$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

lang_ny%>%ggplot(aes(x=reorder(LanguageDesireNextYear,-percent),y=percent,fill=LanguageDesireNextYear,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-20,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Programming/scripting/Markup Languages\n Desired to Work in Future" )
```

## Database Working
```{r fig.width=6,fig.height=6}
db_cur<-so_survey %>% select(DatabaseWorkedWith) %>%mutate(DatabaseWorkedWith=strsplit(as.character(DatabaseWorkedWith),";"))%>%unnest(DatabaseWorkedWith)%>%percent(col="DatabaseWorkedWith")%>%filter(!is.na(DatabaseWorkedWith))%>%filter(percent>0)


label_data<-data.frame(id=seq(1:nrow(db_cur)),
                       lbl=paste(db_cur$DatabaseWorkedWith,db_cur$percent,"%"),
                       value=db_cur$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

db_cur%>%ggplot(aes(x=reorder(DatabaseWorkedWith,-percent),y=percent,fill=DatabaseWorkedWith,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-18,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Database Environment - Working Currently" )
```

## Database for Next Year
```{r fig.width=6,fig.height=6}

db_ny<-so_survey %>% select(DatabaseDesireNextYear) %>%mutate(DatabaseDesireNextYear=strsplit(as.character(DatabaseDesireNextYear),";"))%>%unnest(DatabaseDesireNextYear)%>%percent(col="DatabaseDesireNextYear")%>%filter(!is.na(DatabaseDesireNextYear))%>%filter(percent>0)


label_data<-data.frame(id=seq(1:nrow(db_ny)),
                       lbl=paste(db_ny$DatabaseDesireNextYear,db_ny$percent,"%"),
                       value=db_ny$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

db_ny%>%ggplot(aes(x=reorder(DatabaseDesireNextYear,-percent),y=percent,fill=DatabaseDesireNextYear,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-18,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Database Environment - Desired to work" )

```

## Framework Working
```{r fig.width=6,fig.height=6}
fm_cur<-so_survey %>% select(FrameworkWorkedWith) %>%mutate(FrameworkWorkedWith=strsplit(as.character(FrameworkWorkedWith),";"))%>%unnest(FrameworkWorkedWith)%>%percent(col="FrameworkWorkedWith")%>%filter(!is.na(FrameworkWorkedWith))%>%filter(percent>0)


label_data<-data.frame(id=seq(1:nrow(fm_cur)),
                       lbl=paste(fm_cur$FrameworkWorkedWith,fm_cur$percent,"%"),
                       value=fm_cur$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

fm_cur%>%ggplot(aes(x=reorder(FrameworkWorkedWith,-percent),y=percent,fill=FrameworkWorkedWith,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-18,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Framework - Working Currently" )

```

## Framework for Next Year
```{r fig.width=6,fig.height=6}
fm_ny<-so_survey %>% select(FrameworkDesireNextYear) %>%mutate(FrameworkDesireNextYear=strsplit(as.character(FrameworkDesireNextYear),";"))%>%unnest(FrameworkDesireNextYear)%>%percent(col="FrameworkDesireNextYear")%>%filter(!is.na(FrameworkDesireNextYear))%>%filter(percent>0)


label_data<-data.frame(id=seq(1:nrow(fm_ny)),
                       lbl=paste(fm_ny$FrameworkDesireNextYear,fm_cur$percent,"%"),
                       value=fm_ny$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

fm_ny%>%ggplot(aes(x=reorder(FrameworkDesireNextYear,-percent),y=percent,fill=FrameworkDesireNextYear,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-18,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Framework - Desire to work" )
```

## IDE
```{r fig.width=6,fig.height=6}
ide<-so_survey %>% select(IDE) %>%mutate(IDE=strsplit(as.character(IDE),";"))%>%unnest(IDE)%>%percent(col="IDE")%>%filter(!is.na(IDE))%>%filter(percent>0)


label_data<-data.frame(id=seq(1:nrow(ide)),
                       lbl=paste(ide$IDE,ide$percent,"%"),
                       value=ide$percent)

no_bars<-nrow(label_data)
angle=90-360*(label_data$id-0.6)/no_bars
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

ide%>%ggplot(aes(x=reorder(IDE,-percent),y=percent,fill=IDE,size=tot*100))+geom_bar(stat="identity")+
  scale_fill_manual(values=c(colors,colors))+ylim(-18,20)+coord_polar(start=0) +theme_minimal()+
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
        legend.position = "none" )+
  geom_text(data=label_data, aes(x=id, y=value, label=lbl, hjust=hjust), color="black", fontface="bold",alpha=0.7, size=3, angle= label_data$angle, inherit.aes = FALSE)+labs(title="Development Environment" )
```

## Operating System
```{r fig.width=5,fig.height=5}
percent(col="OperatingSystem")%>%filter(!is.na(OperatingSystem))%>%filter(percent>0)%>%ggplot(aes(x=reorder(OperatingSystem,percent),y=percent,fill=OperatingSystem))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(OperatingSystem,0.5,label=paste(OperatingSystem," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Primary OS -Working Currently")
```

# Stack Overflow Community {.tabset .tabset-fade}
In this section, we get know what percent of respondents visits SO site, frequency,their contribution, and how they use it for jobs search etc.

## Stack Overflow Vist
```{r fig.width=6,fig.height=6}
percent(col="StackOverflowVisit")%>%filter(!is.na(StackOverflowVisit))%>%ggplot(aes(x=reorder(StackOverflowVisit,percent),y=percent,fill=StackOverflowVisit))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowVisit,0.5,label=paste(StackOverflowVisit," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="How frequently would you say you visit Stack Overflow?",subtitle="Frequent Visits")
```

## Stack Overflow -Has Account
```{r fig.width=6,fig.height=6}
percent(col="StackOverflowHasAccount")%>%filter(!is.na(StackOverflowHasAccount))%>%ggplot(aes(x=reorder(StackOverflowHasAccount,percent),y=percent,fill=StackOverflowHasAccount))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowHasAccount,0.5,label=paste(StackOverflowHasAccount," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Do you have a Stack Overflow account?
")
```

## Stack Overflow Participate
```{r fig.width=6,fig.height=6}
percent(col="StackOverflowParticipate")%>%filter(!is.na(StackOverflowParticipate))%>%ggplot(aes(x=reorder(StackOverflowParticipate,percent),y=percent,fill=StackOverflowParticipate))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowParticipate,0.5,label=paste(StackOverflowParticipate," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="How frequently would you say you participate in Q&A on Stack Overflow?.")
```

## Stack Overflow Jobs
```{r fig.width=6,fig.height=6}
percent(col="StackOverflowJobs")%>%filter(!is.na(StackOverflowJobs))%>%ggplot(aes(x=reorder(StackOverflowJobs,percent),y=percent,fill=StackOverflowJobs))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowJobs,0.5,label=paste(StackOverflowJobs," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Have you ever used or visited Stack Overflow Jobs?")
```

## Stack Overflow DevStory
```{r fig.width=6,fig.height=6}
percent(col="StackOverflowDevStory")%>%filter(!is.na(StackOverflowDevStory))%>%ggplot(aes(x=reorder(StackOverflowDevStory,percent),y=percent,fill=StackOverflowDevStory))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowDevStory,0.5,label=paste(StackOverflowDevStory," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Do you have an up-to-date Developer Story on Stack Overflow?")
```

## Stack Overflow Jobs Recommend
```{r fig.width=7,fig.height=5}
percent(col="StackOverflowJobsRecommend")%>%filter(!is.na(StackOverflowJobsRecommend))%>%ggplot(aes(x=reorder(StackOverflowJobsRecommend,percent),y=percent,fill=StackOverflowJobsRecommend))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowJobsRecommend,0.5,label=paste(StackOverflowJobsRecommend," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="How likely is it that you would recommend Stack Overflow Jobs to a friend or colleague?")
```

## Stack Overflow Consider Member
```{r fig.width=7,fig.height=5}
percent(col="StackOverflowConsiderMember")%>%filter(!is.na(StackOverflowConsiderMember))%>%ggplot(aes(x=reorder(StackOverflowConsiderMember,percent),y=percent,fill=StackOverflowConsiderMember))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=8) )+ geom_text( aes(StackOverflowConsiderMember,0.5,label=paste(StackOverflowConsiderMember," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="Do you consider yourself a member of the Stack Overflow community?")
```






# Learning  {.tabset .tabset-fade}
This section has got answers to the questions realted to learning aspects of a Developer, their means of learning, self taught,Hackathon etc.

## Communication Tools
```{r}
com_tool<-so_survey %>%mutate(ctools=strsplit(as.character(CommunicationTools),";"))%>%unnest(ctools)%>%group_by(ctools)%>%summarize(ct_count=n())%>%arrange(desc(ct_count))%>%filter(!is.na(ctools))
treemap(com_tool,
            index=c("ctools","ct_count"),
            vSize="ct_count",
            type="index",
        fontsize.labels=c(12,8),                
    fontcolor.labels=c("white","black"),   
    fontface.labels=c(2,1),                  
    bg.labels=c("transparent"),             
    align.labels=list(
        c("center", "center"), 
        c("right", "bottom")                ),                  
                palette=colors,
            title="Which of the following tools do you use to communicate,\n coordinate, or share knowledge with your coworkers?\n",fontsize.title = 10
            )
```

## Education Types
```{r}
edu_type<-so_survey %>%mutate(et=strsplit(as.character(EducationTypes),";"))%>%unnest(et)%>%group_by(et)%>%summarize(et_count=n())%>%arrange(desc(et_count))%>%filter(!is.na(et))
treemap(edu_type,
            index=c("et","et_count"),
            vSize="et_count",
            type="index",
        fontsize.labels=c(12,8),                
    fontcolor.labels=c("white","black"),   
    fontface.labels=c(2,1),                  
    bg.labels=c("transparent"),             
    align.labels=list(
        c("center", "center"), 
        c("right", "bottom")                ),                  
                palette=colors,
            title="Which of the following types of non-degree education have you used or participated in?",fontsize.title = 10
            )
```

## Self Taught
```{r}
selftaught_type<-so_survey %>%mutate(selftaught=strsplit(as.character(SelfTaughtTypes),";"))%>%unnest(selftaught)%>%group_by(selftaught)%>%summarize(st_count=n())%>%arrange(desc(st_count))%>%filter(!is.na(selftaught))

treemap(selftaught_type,
            index=c("selftaught","st_count"),
            vSize="st_count",
            type="index",
        fontsize.labels=c(12,8),                
    fontcolor.labels=c("white","black"),   
    fontface.labels=c(2,1),                  
    bg.labels=c("transparent"),             
    align.labels=list(
        c("center", "center"), 
        c("right", "bottom")                ),                  
                palette=colors,
            title="You indicated that you had taught yourself a programming technology without taking a course.\n What resources did you use to do that?\n",fontsize.title = 10
            )
```

## Bootcamp
```{r}
percent(col="TimeAfterBootcamp")%>%filter(!is.na(TimeAfterBootcamp))%>%ggplot(aes(x=reorder(TimeAfterBootcamp,percent),y=percent,fill=TimeAfterBootcamp))+geom_bar(stat="identity") +coord_flip()+theme_minimal()+
  theme(  axis.text.y= element_blank() ,  axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(TimeAfterBootcamp,0.5,label=paste(TimeAfterBootcamp," - ",percent,"%")), color="black", hjust=0,size=4) +scale_fill_manual(values=colors)+labs(title="You indicated previously that you went through a developer training program or bootcamp.\n How long did it take you to get a full-time job as a developer after graduating?"
,subtitle="TimeAfter Bootcamp to get a job")

hackathon<-so_survey %>%mutate(hackath=strsplit(as.character(HackathonReasons),";"))%>%unnest(hackath)%>%group_by(hackath)%>%summarize(hk_count=n())%>%arrange(desc(hk_count))%>%filter(!is.na(hackath))

treemap(hackathon,
            index=c("hackath","hk_count"),
            vSize="hk_count",
            type="index",
        fontsize.labels=c(12,8),                
    fontcolor.labels=c("white","black"),   
    fontface.labels=c(2,1),                  
    bg.labels=c("transparent"),             
    align.labels=list(
        c("center", "center"), 
        c("right", "bottom")                ),                  
                palette=colors,
            title="You indicated previously that you had participated in an online coding competition or hackathon.  Which of the following best describe your reasons for doing so?",fontsize.title = 10
            )
```

# Work Life {.tabset .tabset-fade}
One should be healthy and fit inorder to persue their career growth and fulfill their dreams. Questions related to their day-day activities tell us about basic habits/lifestyle and health.

## Wake Time
```{r}
percent(col="WakeTime")%>%filter(!is.na(WakeTime))%>%ggplot(aes(x=WakeTime,y=percent,fill=WakeTime)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=12)) +scale_fill_manual(values=colors)+labs(title="Wake up Time")
```

## Work/Life
```{r}
so_survey %>%select(HoursComputer,HoursOutside)%>%filter(!is.na(HoursComputer),!is.na(HoursOutside))%>%group_by(HoursComputer,HoursOutside)%>%summarise(hrs=n())%>%ungroup()%>%ggplot(aes(HoursComputer,hrs,fill=HoursOutside))+geom_bar(stat="identity",position="fill")+scale_fill_manual(values=colors)+theme_minimal()+labs(y="",title="Work-Life Balance")+scale_y_continuous(labels = percent_format())
```

## Skip Meals
```{r}
percent(col="SkipMeals")%>%filter(!is.na(SkipMeals))%>%ggplot(aes(x=SkipMeals,y=percent,fill=SkipMeals)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=12)) +scale_fill_manual(values=colors)+labs(title="Skip Meal for Productivity")
```

## ErgonomicDevices
```{r}
so_survey%>%filter(!is.na(ErgonomicDevices))%>%mutate(ErgonomicDevices=strsplit(as.character(ErgonomicDevices),";"))%>%unnest(ErgonomicDevices)%>%percent(col="ErgonomicDevices")%>%ggplot(aes(x=reorder(ErgonomicDevices,percent),y=percent,fill=ErgonomicDevices)) + geom_bar(stat='identity') +coord_flip()+theme_minimal()+
  theme(axis.text = element_blank(),  axis.text.x = element_blank(),axis.text.y = element_blank(),   axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(x=ErgonomicDevices, 0.5, label=paste(ErgonomicDevices," ",percent,"%")), color="black", fontface="bold" ,hjust=0,size=3) +scale_fill_manual(values=colors)+labs(title="Ergonomic furniture/Devices")
```

## Exercise
```{r}
percent(col="Exercise")%>%filter(!is.na(Exercise))%>%ggplot(aes(x=Exercise,y=percent,fill=Exercise)) + geom_bar(stat='identity') + theme_minimal()+
   theme(axis.text.x=element_text(angle=45,vjust=0.5),legend.position='none',plot.title = element_text(size=12)) +scale_fill_manual(values=colors)+labs(title="Exercise")
```

## Sexual Orientation
```{r}
so_survey%>%filter(!is.na(SexualOrientation))%>%mutate(SexualOrientation=strsplit(as.character(SexualOrientation),";"))%>%unnest(SexualOrientation)%>%percent(col="SexualOrientation")%>%ggplot(aes(x=reorder(SexualOrientation,percent),y=percent,fill=SexualOrientation)) + geom_bar(stat='identity') +coord_flip()+theme_minimal()+
  theme(axis.text = element_blank(),  axis.text.x = element_blank(),axis.text.y = element_blank(),   axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(x=SexualOrientation, 0.5, label=paste(SexualOrientation," ",percent,"%")), color="black", fontface="bold" ,hjust=0,size=3) +scale_fill_manual(values=colors)+labs(title="SexualOrientation")
```





# Salary  {.tabset .tabset-fade}
## 1.Salary Distribution
```{r}


so_survey$salary<-as.double(so_survey$Salary)
so_survey %>%select(salary)%>%filter(!is.na(salary))%>%ggplot(aes(x=salary))+geom_histogram(fill=colors[2])+labs(title="Distribution of Salary")

#so_survey %>%select(JobSatisfaction,salary)%>%filter(!is.na(salary),!is.na(JobSatisfaction))%>%ggplot(aes(x=salary,y=JobSatisfaction,fill=JobSatisfaction))+geom_density_ridges()+theme_ridges()+theme(legend.position = "none")+scale_fill_manual(values=colors)+labs(title="Does Salary and Job Satisfaction is realted?")
```

## 2.Avg Salary Per Developers
```{r}
so_survey%>%select(DevType,ConvertedSalary)%>%filter(!is.na(DevType),!is.na(ConvertedSalary))%>%
  mutate(DevType=str_split(DevType,";"))%>%unnest(DevType)%>%group_by(DevType)%>%summarise(avg_salary=round(median(ConvertedSalary),0))%>%arrange(desc(avg_salary))%>%ungroup()%>%
   ggplot(aes(x=reorder(DevType,avg_salary),y=avg_salary,col=DevType,size=avg_salary*10))+geom_point()+geom_text_repel(aes(label=avg_salary),size=3)+geom_segment(aes(x=DevType, 
                   xend=DevType, 
                   y=0, 
                   yend=avg_salary),size=1)+
#+geom_jitter(size=0.3)+
  theme_minimal()+theme(legend.position = "none",axis.text.x = element_text(angle=0), )+labs(title=" Avg Salary per Developers",x="Developer Type",y="Avg Salary")+scale_color_manual(values=colors)+coord_flip()+scale_y_continuous(label=dollar_format())
```

## 3.Countrywise Developers Salary
```{r fig.height=10,fig.width=10}
so_survey%>%select(DevType,ConvertedSalary,Country)%>%filter(!is.na(DevType),!is.na(ConvertedSalary),!is.na(Country))%>%filter(Country %in% c("United States","India","Germany","United Kingdom","Russian Federation"))%>%
  mutate(DevType=str_split(DevType,";"))%>%unnest(DevType)%>%group_by(Country,DevType)%>%summarise(avg_salary=round(median(ConvertedSalary),0))%>%arrange(desc(avg_salary))%>%ungroup()%>%
   ggplot(aes(x=reorder(DevType,avg_salary),y=avg_salary,col=Country,group=Country))+geom_line(size=1.5)+
  theme_minimal()+theme(legend.position = "bottom",axis.text.x = element_text(angle=90), )+labs(title="Respondent Top Country wise Developers Salary",x="Developer Type",y=" Salary")+scale_color_manual(values=colors)+scale_y_continuous(label=dollar_format())
```

## 4.Agewise  Developers Salary
```{r fig.height=10,fig.width=10}
so_survey%>%select(DevType,ConvertedSalary,Age)%>%filter(!is.na(DevType),!is.na(ConvertedSalary),!is.na(Age))%>%
  mutate(DevType=str_split(DevType,";"))%>%unnest(DevType)%>%group_by(Age,DevType)%>%summarise(avg_salary=round(median(ConvertedSalary),0))%>%arrange(desc(avg_salary))%>%ungroup()%>%
   ggplot(aes(x=reorder(DevType,avg_salary),y=avg_salary,col=Age,group=Age))+geom_line(size=1.5)+
  theme_minimal()+theme(legend.position = "bottom",axis.text.x = element_text(angle=90), )+labs(title="Age wise Developers Salary",x="Developer Type",y=" Salary")+scale_color_manual(values=colors)+scale_y_continuous(label=dollar_format())
```

## 5.Salary by Age and Gender
```{r}
age_gender<-so_survey%>%select(Gender,ConvertedSalary,Age)%>%filter(!is.na(Gender),!is.na(ConvertedSalary),!is.na(Age))%>%mutate(age=parse_number(Age),Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%
  filter(Gender %in% c("Male","Female"))%>%group_by(Age,Gender)%>%summarise(avg_salary=round(median(ConvertedSalary),0),n=n(),avg_age=mean(age))%>%arrange(desc(n))%>%ungroup()

age_gender%>%ggplot(aes(avg_age,avg_salary,group=Gender,col=Gender))+geom_line(alpha=1,size=1.5)+
  theme_minimal()+labs(title="Salary by Age and Gender",x="Avg Age",y="Avg Salary(USD)")+scale_color_manual(values=c(colors))+scale_y_continuous(label=dollar_format())+theme(legend.position = "bottom")
```

## 6.Coding Exp vs Developers Salary
```{r}
exp<-so_survey%>%select(DevType,ConvertedSalary,YearsCodingProf)%>%filter(!is.na(DevType),!is.na(ConvertedSalary),!is.na(YearsCodingProf))%>%mutate(YearsCoding=parse_number(YearsCodingProf))%>%
  mutate(DevType=str_split(DevType,";"))%>%unnest(DevType)%>%group_by(DevType)%>%summarise(avg_salary=round(median(ConvertedSalary),0),n=n(),avg_years=mean(YearsCoding))%>%arrange(desc(n))%>%ungroup()%>%filter(n>1000)


exp%>%ggplot(aes(avg_years,avg_salary,size=n,label=DevType,col=DevType))+geom_point(alpha=1)+geom_text_repel(size = 3, point.padding = 0.25)+
  theme_minimal()+labs(title="Coding Experience vs Developers Salary",x="Avg Coding Experience",y=" Salary")+scale_color_manual(values=colors)+scale_y_continuous(label=dollar_format())+scale_size_continuous(labels = comma_format())+theme(legend.position = "none")
```

## 7.Coding Exp and Salary by Prog Language
```{r}
exp_lang<-so_survey%>%select(LanguageWorkedWith,ConvertedSalary,YearsCodingProf)%>%filter(!is.na(LanguageWorkedWith),!is.na(ConvertedSalary),!is.na(YearsCodingProf))%>%mutate(YearsCoding=parse_number(YearsCodingProf))%>%
  mutate(LanguageWorkedWith=str_split(LanguageWorkedWith,";"))%>%unnest(LanguageWorkedWith)%>%group_by(LanguageWorkedWith)%>%summarise(avg_salary=round(median(ConvertedSalary),0),n=n(),avg_years=mean(YearsCoding))%>%arrange(desc(n))%>%ungroup()%>%filter(n>1000)

exp_lang%>%ggplot(aes(avg_years,avg_salary,size=n,label=LanguageWorkedWith,col=LanguageWorkedWith))+geom_point(alpha=1)+geom_text_repel(size = 3, point.padding = 0.25)+
  theme_minimal()+labs(title="Coding Experience & Salary by Language",x="Avg Coding Experience",y="Avg Salary(USD)")+scale_color_manual(values=c(colors,colors))+scale_y_continuous(label=dollar_format())+scale_size_continuous(labels = comma_format())+theme(legend.position = "none")
``` 

## 8.Coding Exp and Salary by Gender
```{r}
exp_gender<-so_survey%>%select(Gender,ConvertedSalary,YearsCodingProf)%>%filter(!is.na(Gender),!is.na(ConvertedSalary),!is.na(YearsCodingProf))%>%mutate(YearsCoding=parse_number(YearsCodingProf),Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%
  filter(Gender %in% c("Male","Female"))%>%group_by(YearsCoding,Gender)%>%summarise(avg_salary=round(median(ConvertedSalary),0),n=n(),avg_years=mean(YearsCoding))%>%arrange(desc(n))%>%ungroup()

exp_gender%>%ggplot(aes(avg_years,avg_salary,group=Gender,col=Gender))+geom_line(alpha=1,size=1.5)+
  theme_minimal()+labs(title="Coding Experience & Salary by Gender",x="Avg Coding Experience",y="Avg Salary(USD)")+scale_color_manual(values=c(colors,colors))+scale_y_continuous(label=dollar_format())+theme(legend.position = "bottom")
```

# Detailed Analysis {.tabset .tabset-fade}
## Top 20 Countries
```{r}
so_survey %>%filter(Gender %in%c("Male","Female"))%>%group_by(Country,Gender)%>%summarise(gcnt=n())%>%ungroup()%>%spread(Gender,gcnt)%>%arrange(desc(Female))%>%head(20)%>%gather(key="Gender",value="gcnt",2:3)%>%ggplot(aes(x=reorder(Country,gcnt),y=gcnt,fill=Gender))+geom_bar(stat="identity",position="dodge")+coord_flip()+theme_minimal()+
  theme(    axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "bottom",plot.title = element_text(size=12) )+ geom_text( aes(Country,gcnt,label=gcnt), color="black", hjust=0,size=2) +scale_fill_manual(values=colors)+labs(title="Comparing Male Vs Female respondents"
,subtitle="Top 20 Countries- Female Count")
```

## Age per Gender
```{r}
so_survey %>%filter(!is.na(Age))%>%filter(Gender %in%c("Male","Female"))%>%group_by(Gender,Age)%>%summarise(gcnt=n())%>%ungroup()%>%spread(Gender,gcnt)%>%arrange(desc(Female))%>%gather(key="Gender",value="gcnt",2:3)%>%ggplot(aes(x=reorder(Age,gcnt),y=gcnt,fill=Gender))+geom_bar(stat="identity",position="dodge")+coord_flip()+theme_minimal()+
  theme(    axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "bottom",plot.title = element_text(size=12) )+ geom_text( aes(Age,gcnt,label=gcnt), color="black", hjust=0,vjust=0.7,size=2) +scale_fill_manual(values=colors)+labs(title="Gender vs Age Groups"
,subtitle="Gender segregation between age groups")
```

## Hobby -Open Source Contribution
```{r}
so_survey %>%select(Age,Hobby,OpenSource)%>%filter(!is.na(Age))%>%group_by(Age,Hobby,OpenSource)%>%summarise(hb=n())%>%ungroup()%>%ggplot(aes(Age,hb,group=OpenSource,col=OpenSource))+geom_line(stat="identity")+facet_wrap(~Hobby)+scale_color_manual(values=colors)+theme_minimal()+labs(y="",title="How many of them contributing to Opensource, consider Coding as Hobby")+theme(axis.text.x = element_text(angle=90))
```

## Race Ethnicity vs Developer Type
```{r fig.width=10,fig.height=13}
so_survey%>%filter(!is.na(RaceEthnicity),!is.na(DevType))%>%mutate(RaceEthnicity=strsplit(as.character(RaceEthnicity),";"))%>%unnest(RaceEthnicity)%>%mutate(DevType=strsplit(as.character(DevType),";"))%>%unnest(DevType)%>%group_by(RaceEthnicity,DevType)%>%summarise(t=n()) %>%mutate(percent=round(t/sum(t)*100),0)%>%ggplot(aes(x=reorder(DevType,percent),y=percent,fill=DevType)) + geom_bar(stat='identity') +facet_wrap(~RaceEthnicity)+coord_flip()+theme_minimal()+
  theme(axis.text = element_blank(),  axis.text.x = element_blank(),axis.text.y = element_blank(),   axis.title.x = element_blank(),axis.title.y = element_blank()   ,  legend.position = "none",plot.title = element_text(size=12) )+ geom_text( aes(x=DevType, 0.5, label=paste(DevType," ",percent,"%")), color="black", fontface="bold" ,hjust=0,size=3) +scale_fill_manual(values=colors)+labs(title="RaceEthnicity vs Developer Type")
```


# Part of SO Community {.tabset .tabset-fade}
## Age groups vs SO Member
```{r fig.width=6,fig.height=6}
so_survey %>%select(Gender,Age,StackOverflowConsiderMember)%>%filter(!is.na(StackOverflowConsiderMember),!is.na(Age))%>%mutate(Gender=strsplit(as.character(Gender),";"))%>%unnest(Gender)%>%filter(Gender %in% c("Male","Female"))%>%group_by(Gender,StackOverflowConsiderMember,Age)%>%summarise(gs=n())%>%mutate(p=gs/sum(gs))%>%ungroup()%>%ggplot(aes(Age,p,group=Gender,col=Gender))+geom_line()+geom_point()+facet_wrap(~StackOverflowConsiderMember)+theme_minimal()+theme(axis.text.x = element_text(angle=90))+scale_y_continuous(labels = percent_format() )+scale_color_manual(values=colors)+labs(title="Which Age group consider them as SO Community",y="percentage")
```

## Job Search status Vs SO member
```{r}
so_survey %>%select(Hobby,JobSearchStatus,StackOverflowConsiderMember)%>%filter(!is.na(Hobby),!is.na(JobSearchStatus),!is.na(StackOverflowConsiderMember))%>%group_by(Hobby,JobSearchStatus,StackOverflowConsiderMember)%>%summarise(js=n())%>%mutate(p=js/sum(js))%>%ungroup()%>%ggplot(aes(StackOverflowConsiderMember,p,color=Hobby,group=Hobby))+geom_line(stat="identity")+facet_wrap(~JobSearchStatus, labeller = labeller(JobSearchStatus= label_wrap_gen(15)))+scale_color_manual(values=colors)+theme_minimal()+labs(y="",title="JobSearch Status /Coding Hobby - Who consider to be member of SO")+scale_y_continuous(labels = percent_format())
```

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
  ggplot(aes(Jobsearch, ConsiderMember, label = DevType,col=DevType)) +
    geom_text_repel(size = 3, point.padding = 0.25) +
  geom_point(aes(size = n), alpha = 1) +
  scale_y_continuous(labels = percent_format()) + scale_x_continuous(labels = percent_format())+
    scale_size_continuous(labels = comma_format()) +
  scale_color_manual(values=colors)+theme_minimal()+
  labs(x = "% Job Search status", 
       y = "% Consider as SO Community",
       title = "Who considers themselves part of the Stack Overflow community?",
       subtitle = "Job Search Status vs Developer Type",
       size = "Number of respondents")+theme(legend.position = "none")
```