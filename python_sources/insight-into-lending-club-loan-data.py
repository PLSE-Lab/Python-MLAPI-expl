---

title: "Insight Into Loan Data"

author: "Sibi Joseph"

date: "11 February 2018"

output: html_document

---

```{r setup, include=FALSE}

knitr::opts_chunk$set(echo = TRUE)

```

## R Markdown



This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.



When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:



```{r cars}

summary(cars)

```

## Including Plots



You can also embed plots, for example:

```{r pressure, echo=FALSE}

plot(pressure)

```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.

```{r, echo=FALSE,include=FALSE}

# 1. Remove objects from R memory and collect garbage

rm(list=ls()) ; gc()

```



## Lending club data analysis by Sibi

```{r, include = FALSE}

library(dplyr)

library(lubridate)

library(ggplot2)

library(readr)

library(lubridate)

library(gridExtra)

library(ggthemes)

options(scipen=999) 

options(max.print=999999)

```

```{r, include = FALSE,echo=FALSE}

ld<-read.csv("../input/loan.csv")

ld %>% mutate_if(is.character, as.factor) -> ld

ld$inq_fi<-as.factor(ld$inq_fi)

ld$no_of_credit_lines<-cut(ld$total_acc,5,labels =c("v_low_acc_nos","low_acc_nos","med_acc_nos","high_acc_nos","v_high_acc_nos"))

no_of_inq<-ld%>%filter(inq_fi!="0.0")%>%filter(inq_fi!="NA")

```

```{r,include=FALSE, echo=FALSE}

ld$issue_dd<-paste0("01-",ld$issue_d)

ld$issue_dd<-dmy(paste0("01-",ld$issue_d))

funded_amnt_by_months<-ld%>%group_by(issue_dd)%>%summarise(funded_amnt_by_months=sum(funded_amnt))

fnd_amnt_state<-ld%>%group_by(addr_state)%>%summarise(fnd_amnt_state=sum(funded_amnt))%>%arrange(fnd_amnt_state)

```



Buckets showing Annual Income:



```{r echo=FALSE,warning=FALSE}

ggplot(ld,aes(x=annual_inc))+geom_histogram(bins=20)+xlim(0,100000)

```



Customers with very low number of credit lines and sending a high number of inquiries, are in need of a financial product. These are potential customers to be considered for marketing calls.



```{r,warning=FALSE,echo=FALSE,message=FALSE}

ld$inq_fi<-as.factor(ld$inq_fi)

ld$no_of_credit_lines<-cut(ld$total_acc,5,labels =c("v_low_acc_nos","low_acc_nos","med_acc_nos","high_acc_nos","v_high_acc_nos"))

no_of_inq<-ld%>%filter(inq_fi!="0.0")%>%filter(inq_fi!="NA")

ggplot(no_of_inq, aes(x=no_of_credit_lines, fill=inq_fi)) + geom_bar()

```



Rising trend of Loan disbursed per month.



```{r,warning=FALSE,echo=FALSE,message=FALSE}

ggplot(funded_amnt_by_months, aes(issue_dd, funded_amnt_by_months)) + geom_bar(stat = "identity")+geom_smooth()

#limldl<-limld%>%select(funded_amnt,issue_dd)  

  

#ggplot(limldl, aes(x=issue_dd, y=funded_amnt)) + stat_summary(fun.y="sum",  geom="bar") +   labs(y ="Total funded amnt")

```



Loan distribution per status of loan



```{r,warning=FALSE,echo=FALSE,message=FALSE}

#ggplot(loan_amnt_by_status,aes(loan_status,loan_amnt_by_status,fill=loan_status))+geom_bar(stat = "identity")+ scale_x_discrete(breaks=NULL)



ggplot(ld, aes(x=loan_status, y=funded_amnt,fill=loan_status)) +

  stat_summary(fun.y="sum", geom="bar") +

  labs(y ="Total Charges")+ scale_x_discrete(breaks=NULL)



```



State_wise Loan Distribution



```{r,warning=FALSE,echo=FALSE,message=FALSE}

#ggplot(fnd_amnt_state,aes(addr_state,fnd_amnt_state,fill=addr_state))+geom_bar(stat = "identity")



ggplot(fnd_amnt_state, aes(x=addr_state, y=fnd_amnt_state, fill=addr_state)) + stat_summary(fun.y="sum",  geom="bar") +   labs(x="States", y ="Total funded amnt",title="State_wise Loan Distribution")

```