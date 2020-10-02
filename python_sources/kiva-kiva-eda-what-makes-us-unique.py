
---
title: 'Data Science for Good: Kiva Crowdfunding'
author: "Kheirallah Samaha"
date: "May 21, 2018"
output:
  html_document:
    code_folding: hide
    fig_height: 7
    fig_width: 7
    number_sections: yes
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


### About Kiva {-}

  Kiva is an international nonprofit, founded in 2005 and based in San Francisco, with a mission to connect people through lending to alleviate poverty. We celebrate and support people looking to create a better future for themselves, their families and their communities.
  
  By lending as little as $25 on Kiva, anyone can help a borrower start or grow a business, go to school, access clean energy or realize their potential. For some, it’s a matter of survival, for others it’s the fuel for a life-long ambition.
  
  100% of every dollar you lend on Kiva goes to funding loans. Kiva covers costs primarily through optional donations, as well as through support from grants and sponsors.

**It's a loan, not a donation**

  We believe lending alongside thousands of others is one of the most powerful and sustainable ways to create economic and social good. Lending on Kiva creates a partnership of mutual dignity and makes it easy to touch more lives with the same dollar. Fund a loan, get repaid, fund another.




### Objective of the kernel{-}

  To get a better idea and understanding of the data provided by Kiva.

  so Let us first import the data using <read.csv funtion


### Import Data {-}


```{r Kiva}
Kiva_imp_df <- read.csv("../input/kiva_loans.csv")
loan_theme_imp_df <- read.csv("../input/loan_theme_ids.csv")
```
### Loading Libraries {-}

Then Let us import the necessary libraries

- tidyverse
- ggplot2
- gridExtra
- DT
- tm
- wordcloud
- wordcloud2
- quanteda

```{r libraries, echo=FALSE}
suppressMessages(library(tidyverse))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))
suppressMessages(library(DT))
suppressMessages(library(tm))
suppressMessages(library(wordcloud))
suppressMessages(library(wordcloud2))
suppressMessages(library(quanteda))

```


let us check is any dublicates in ID Feature, and then check the missing data

```{r diff}
#summary(Kiva_imp_df)
anyDuplicated(Kiva_imp_df$id)

```

No duplicates

Now im goint add a new feature to see the diffrence between loan_amount and funded_amoun, just curiosity

```{r new feature}


Kiva_imp_df <- Kiva_imp_df %>%
  mutate(diff = Kiva_imp_df$loan_amount-Kiva_imp_df$funded_amount)
head(Kiva_imp_df)

```
## What makes us unique{-}


### It's a loan, not a donation{-}

We believe lending alongside thousands of others is one of the most powerful and sustainable ways to create economic and social good. Lending on Kiva creates a partnership of mutual dignity and makes it easy to touch more lives with the same dollar. Fund a loan, get repaid, fund another.



```{r}

loan_fea <- Kiva_imp_df$loan_amount

summary(loan_fea)

sd(loan_fea)

boxplot(log(loan_fea) ~ Kiva_imp_df$sector,las=2,,cex=0.5,font=2,font.lab=2,cex.lab=2,cex.axis=0.7)


Kiva_imp_df%>%
  filter(loan_amount < 1000
           ) %>% 
  ggplot(aes(sample = loan_amount))+
  geom_qq()+
  theme_minimal()+
  theme(axis.text = element_text(size = 7),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 0.5))+
  ggtitle("Q-Q Plot")



ggplot(Kiva_imp_df%>%
  filter(loan_amount < 1000
           ), aes(sample = loan_amount, colour = sector)) +
   stat_qq()+
  theme_minimal()+
  theme(axis.text = element_text(size = 7),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 0.5))+
  ggtitle("Q-Q Plot")

```



### You choose where to make an impact{-}

Whether you lend to friends in your community, or people halfway around the world (and for many, it’s both), Kiva creates the opportunity to play a special part in someone else's story. At Kiva, loans aren’t just about money—they’re a way to create connection and relationships.


```{r}

Kiva_imp_df %>% 
  select(country,loan_amount)%>%
  group_by(country)%>% 
  summarise( total = sum(loan_amount))%>%
  top_n(10,wt=total)%>%
  arrange(desc(total))%>%
  ggplot(aes(x = reorder(country,-total), y = total))+
  geom_bar(stat="identity",fill = "steelblue")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution by countries")+
  labs(x="Country",
       y="Loan Sum")


```


### Pushing the boundaries of a loan {-}

Kiva started as a pioneer in crowdfunding in 2005, and is constantly innovating to meet people’s diverse lending needs. Whether it’s reinventing microfinance with more flexible terms, supporting community-wide projects or lowering costs to borrowers, we are always testing and learning.


```{r}
head(loan_theme_imp_df)

loan_theme_imp_df %>% 
  select(Loan.Theme.Type)%>%
  filter(Loan.Theme.Type !="")%>%
  group_by(Loan.Theme.Type)%>% 
  summarise( total_count = n())%>%
  top_n(15,wt=total_count)%>%
  arrange(desc(total_count))%>%
  ggplot(aes(x = reorder(Loan.Theme.Type,-total_count), y = total_count))+
  geom_bar(stat="identity",fill = "steelblue")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loan Theme Type")+
  labs(x="Loan Theme Type",
       y="Count")



```

### Lifting one, to lift many {-}

When a Kiva loan enables someone to grow a business and create opportunity for themselves, it creates opportunities for others as well. That ripple effect can shape the future for a family or an entire community.


```{r}
Kiva_imp_df$use <- tolower(Kiva_imp_df$use)
Kiva_imp_df$use <- removePunctuation(Kiva_imp_df$use)


kiva_use_lift <- Kiva_imp_df %>%
  select(loan_amount,use)%>%
  filter(use != "")%>% 
  group_by(use)%>% 
  summarise( total_count = n())%>%
  top_n(10,wt=total_count)%>%
  arrange(desc(total_count))

datatable(
 kiva_use_lift,
  options = list(pageLength = 10, 
                 dom = "tip"), rownames = FALSE
) %>% formatStyle("use",  color = "white", backgroundColor = "darkgreen", fontWeight = "bold")%>%
formatStyle("total_count",  color = "darkgreen", backgroundColor = "white")

  
```

We should appriciate Kiva's idea and efforts.


OK! Let us see the Distribution of loans by Sector:

```{r}
sec_loan <- Kiva_imp_df %>% select(sector,loan_amount)%>%
  group_by(sector)%>%
  summarise(sum_ta = sum(loan_amount))%>%
  mutate(perct = sum_ta/sum(Kiva_imp_df$loan_amount)*100)%>%
  ggplot(aes(x= reorder(sector, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(perct,2)), size = 3, y= 2.5, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution %")+
  labs(x="Sectors",
       y="Percentage")
sec_loan

```


Agriculture 25.3 % has the highest number of loans followed by food 21.51% and retail 17.35%. Now let us look at the loan details at activity level.



```{r}
act_loan <- Kiva_imp_df %>% select(activity,loan_amount)%>%
  group_by(activity)%>%
  summarise(sum_ta = sum(loan_amount))%>%
  mutate(perct = sum_ta/sum(Kiva_imp_df$loan_amount)*100)%>%
  filter(perct>=2.5)%>%
  ggplot(aes(x= reorder(activity, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(perct,2)), size = 3, y= 2, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution VS Activities %")+
  labs(x="Activities",
       y="Percentage")
act_loan


```

Farming 9.14% has the highest number of loans followed by General Store 6.8% and again Agriculture 4.85%. Now let us look at the loan details at countries level.



```{r}
country_loan <- Kiva_imp_df %>% select(country,loan_amount,sector)%>%
  group_by(country,sector)%>%
  summarise(sum_ta = sum(loan_amount))%>%
  mutate(perct = sum_ta/sum(Kiva_imp_df$loan_amount)*100)%>%
  filter(perct >= 1)%>%
  ggplot(aes(x= reorder(country, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  geom_text(aes(label = round(perct,2)), size = 3.5, y= 0.5, col = "white", angle = 90, fontface="bold")+
  facet_wrap(~sector)+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution by countries VS Sectors %")+
  labs(x="Country",
       y="Percentage")

country_loan
```


Philippines - Retail 3.4%
Kenya - Agriculture 3.11%
Philippines Food 2.61%
Philippines Agriculture 2.11


**So let me create a new data frame for loan distribution by countries**

```{r}
kiva_plot_country <- Kiva_imp_df %>% select(country,loan_amount,sector)%>%
  group_by(country)%>%
  summarise(sum_ta = sum(loan_amount))%>%
  mutate(perct = sum_ta/sum(Kiva_imp_df$loan_amount)*100)

```


#### Loans disribution by countries {.tabset .tabset-fade .unnumbered} 
##### Greater than or equal to 2.5% {-}

```{r}
kiva_plot_country%>%
  filter(perct >= 2.5)%>%
  ggplot(aes(x= reorder(country, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(perct,2)), size = 3, y= 2, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution by countries\n Grt than or equal to 2.5%")+
  labs(x="Country",
       y="Percentage")

```


##### less than 2.5 and greater or equal to 0.5% {-}

```{r}
kiva_plot_country %>%
  filter(perct < 2.5 & perct >= 0.5) %>%
  ggplot(aes(x= reorder(country, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(perct,2)), size = 1.5, y= 0.4, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution by countries\n less than 2.5 and grt or equal to 0.5%")+
  labs(x="Country",
       y="Percentage")

```


##### less than 0.5 greater than or equal to 0.1% {-}

```{r}

kiva_plot_country %>%
  filter(perct < 0.5 & perct >= 0.1) %>%
  ggplot(aes(x= reorder(country, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(perct,2)), size = 2, y = 0.25, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution by countries\n less than 0.5 grt than or equal to 0.1%")+
  labs(x="Country",
       y="Percentage")

```


##### less than 0.1% {-}

```{r}
kiva_plot_country %>%
  filter(perct < 0.1) %>%
  ggplot(aes(x= reorder(country, perct),y = perct))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(perct,2)), size = 2, y= 0.025, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loans disribution by countries\n less than 0.1%")+
  labs(x="Country",
       y="Percentage")

```

-----------

Table using DT package, and note that you may find different ranking ! it is just rounding effect..


```{r}
kiva_plot_country$perct <- round(kiva_plot_country$perct,2)

datatable(
  kiva_plot_country %>%
         arrange(desc(round(perct,2))),
  options = list(pageLength = 10, 
                 dom = "tip"), rownames = FALSE
) %>% formatStyle(c("country","perct"),  color = "white", backgroundColor = "Steelblue", fontWeight = "bold")%>%
formatStyle(c("sum_ta"),  color = "darkgreen", backgroundColor = "white")
```

Philippines 9.79% has the highest number of loans followed by Kenya 6.11% and USA 5.57. Now let us look at the loan details at countries VS Sector

I am sure there is onther way to manage the labels issue in the 3rd plot, let me worry about it later..geom_text is not an option for now :) split the the table again?, maybe?


#### Loan amount Grouping and distributions {-}


```{r}
summary(Kiva_imp_df$loan_amount)


amtbreaks <- c(25,100,200,500,1000,1250,2000,2250,3000,3250,4000,4250,5000,10000)
amtlabels <- c("25-100","101-200","201-500","501-1000","1001-1250","1251-2000","2001-2250",
               "2251-3000","3001-3250","3251-4000","4001-4250","4251-5000","5001-10000")

Kiva_imp_df$loan_amount_group <- cut(Kiva_imp_df$loan_amount, 
                                breaks = amtbreaks, 
                                right = FALSE, 
                                labels = amtlabels)

```

```{r}
loan_amt_group <- Kiva_imp_df %>% select(country,loan_amount,sector,loan_amount_group)%>%
  filter (!is.na(loan_amount_group))%>%
  group_by(loan_amount_group)%>%
  summarise(count_grp = n())%>%
  arrange(desc(count_grp))%>%
  ggplot(aes(x= reorder(loan_amount_group, count_grp),y = count_grp))+
  geom_bar(stat="identity",fill = "steelblue")+
  coord_flip()+
  geom_label(aes(label = round(count_grp,2)), size = 3, y= 30000, col = "darkgreen")+
  theme_minimal()+
  theme(axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loan Amount Groups Distributions - Count")+
  labs(x="Loan amount groups",
       y="Count")
loan_amt_group
```


##### Top 4 Countries Loan groups distributions {-}

- Philippines
- Kenya
- United States
- Peru

```{r}
Kiva_imp_df %>% select(country,loan_amount,sector,loan_amount_group)%>%
  filter (!is.na(loan_amount_group) &
            country %in% c( "Philippines","Kenya","United States","Peru"))%>%
  group_by(loan_amount_group , country)%>%
  summarise(sum_grp = sum(loan_amount))%>%
  arrange(desc(sum_grp))%>%
  ggplot(aes(x= reorder(loan_amount_group, sum_grp),y = sum_grp))+
  geom_bar(stat="identity",fill = "steelblue")+
  facet_wrap(~country)+
  coord_flip()+
  theme_minimal()+
  theme(axis.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Loan Amount Groups Distributions - Sum ")+
  labs(x="Loan amount groups",
       y="Sum")
```

let's see these distributions base on the sectores.

we knew by the above plots that the top sectores the loans uesed for are Agriculture, food, retail, Services, Education, Housing and Clothing.

I will stick on those Sectors and let's see what we will get.

```{r}

Kiva_imp_df %>% select(country,loan_amount,sector,loan_amount_group)%>%
  filter (!is.na(loan_amount_group) &
            country == "Philippines" & 
            sector %in% c("Agriculture", "Food", "Retail", "Services", "Education", "Housing","Clothing"))%>%
  group_by(loan_amount_group ,country,sector)%>%
  summarise(sum_grp = sum(loan_amount))%>%
  arrange(desc(sum_grp))%>%
  ggplot(aes(x= reorder(loan_amount_group, sum_grp),y = sum_grp))+
  geom_bar(stat="identity",fill = "darkgreen")+
  facet_wrap(~sector)+
  theme_minimal()+
  theme(axis.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Philippines\n Loan Amount Groups Distributions\n VS Sector")+
  labs(x="Loan amount groups",
       y="Sum")


Kiva_imp_df %>% select(country,loan_amount,sector,loan_amount_group)%>%
  filter (!is.na(loan_amount_group) &
            country == "Kenya"&
          sector %in% c("Agriculture", "Food", "Retail", "Services", "Education", "Housing","Clothing"))%>%
  group_by(loan_amount_group ,country,sector)%>%
  summarise(sum_grp = sum(loan_amount))%>%
  arrange(desc(sum_grp))%>%
  ggplot(aes(x= reorder(loan_amount_group, sum_grp),y = sum_grp))+
  geom_bar(stat="identity",fill = "darkgreen")+
  facet_wrap(~sector)+
  theme_minimal()+
  theme(axis.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("Kenya\n Loan Amount Groups Distributions\n VS Sector")+
  labs(x="Loan amount groups",
       y="Sum")


Kiva_imp_df %>% select(country,loan_amount,sector,loan_amount_group)%>%
  filter (!is.na(loan_amount_group) &
            country == "United States" &
            sector %in% c("Agriculture", "Food", "Retail", "Services", "Education", "Housing","Clothing"))%>%
  group_by(loan_amount_group ,country,sector)%>%
  summarise(sum_grp = sum(loan_amount))%>%
  arrange(desc(sum_grp))%>%
  ggplot(aes(x= reorder(loan_amount_group, sum_grp),y = sum_grp))+
  geom_bar(stat="identity",fill = "darkgreen")+
  facet_wrap(~sector)+
  theme_minimal()+
  theme(axis.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.text.y = element_text(angle = 0, vjust = 0.5, hjust = 1))+
  ggtitle("United States\n Loan Amount Groups Distributions\n VS Sector")+
  labs(x="Loan amount groups",
       y="Sum")

```




**Philippines:**
Loan group 201-500, Mainly for Agriculture, Food and Retail Sector, Actually we can say that the majority of loans are going to **FOOD** witch includes (Agriculture as a product for Food).

**Kenya:**
Loan group 501-100, Mainly for Agriculture, Food and Retail Sector (Retail Sector is less than Retail Sector for Philippines)

**United States:**
Loan group **5000 - 10000** going to Services, Food, Retail, Clothing and Agriculture Sectors. it is telling !!

I think the use Variable is essential to know the bound between those three Sectors (Agriculture, Food and Retail)



### Wordcloud {-}

I wanted to use quanteda package, but there was a problem with Kernel, did not work well, So i moved to wordcloud. 

just to text cloud the most common words so to see the relation between the 4 obbs, Agriculture, Food, Retail and Services.


```{r}

Kiva_use_text <- Kiva_imp_df %>% select(use,country,sector)%>%
  filter (country %in% c( "Philippines","Kenya","United States") & 
            sector %in% c("Agriculture", "Food", "Retail", "Services")) 

str(Kiva_use_text)


```


**Creating DF for Text analysis.**


```{r}

Kiva_use_text$use <- as.character(Kiva_use_text$use)

Kiva_use_text$use <- tolower(Kiva_use_text$use)
Kiva_use_text$use <- removePunctuation(Kiva_use_text$use)
Kiva_use_text$use <- removeNumbers(Kiva_use_text$use)

# Adding "canned","etc","additional","like","sell","buy", "purchase","items","business" to the list of stopwords

r_buy_pur_stopw <- c("canned","etc","additional","like","sell","buy", "purchase","items","business", stopwords("en"))

Kiva_use_text$use <- removeWords(Kiva_use_text$use, r_buy_pur_stopw)


```

since i cleand the use var now i can move to the next step,

Split the Kiva_use_text DF to 3 data frames based on 3 country, PHL, KEN and USA. filter them by 4 obbs. "Agriculture", "Food", "Retail", "Services"  


**Corpus for PHL, KEN**

       **and USA**

- WordCloud

```{r}

corpus_phi <- Corpus(VectorSource(Kiva_use_text %>%
  filter (country == "Philippines"&
          sector %in% c("Agriculture", "Food", "Retail", "Services"))))

tdm_phi <- TermDocumentMatrix(corpus_phi)
mat_phi <- as.matrix(tdm_phi)
pre_frq_phi <- sort(rowSums(mat_phi),decreasing = TRUE)
doc_cloud_phi <- data.frame(word=names(pre_frq_phi),freq =pre_frq_phi)


corpus_ken <- Corpus(VectorSource(Kiva_use_text %>%
  filter (country == "Kenya" &
          sector %in% c("Agriculture", "Food", "Retail", "Services"))))

tdm_ken <- TermDocumentMatrix(corpus_ken)
mat_ken <- as.matrix(tdm_ken)
pre_frq_ken <- sort(rowSums(mat_ken),decreasing = TRUE)
doc_cloud_ken <- data.frame(word=names(pre_frq_ken),freq =pre_frq_ken)


corpus_usa <- Corpus(VectorSource(Kiva_use_text %>%
  filter (country == "United States"&
          sector %in% c("Agriculture", "Food", "Retail", "Services"))))

tdm_usa <- TermDocumentMatrix(corpus_usa)
mat_usa <- as.matrix(tdm_usa)
pre_frq_usa <- sort(rowSums(mat_usa),decreasing = TRUE)
doc_cloud_usa <- data.frame(word=names(pre_frq_usa),freq =pre_frq_usa)


```

**WordCloud**


```{r}

par(mfrow = c(2,2))

wordcloud(corpus_phi,min.freq = 500,max.words = 50, random.order = F, scale = c(3,.5),
          colors = c("darkred","steelblue","darkgreen"))
wordcloud(corpus_ken,min.freq = 500,max.words = 50, random.order = F, scale = c(3,.5),
          colors = c("darkred","steelblue","darkgreen"))
wordcloud(corpus_usa,min.freq = 1000,max.words = 80, random.order = F, scale = c(3,.5),
          colors = c("darkred","steelblue","darkgreen"))

```


**DataTable**


```{r}

datatable(
 doc_cloud_phi[doc_cloud_phi$freq > 1000,],
  options = list(pageLength = 7, 
                 dom = "tip"), rownames = FALSE
) %>% formatStyle("word",  color = "white", backgroundColor = "darkgreen", fontWeight = "bold")%>%
formatStyle("freq",  color = "darkgreen", backgroundColor = "white")


datatable(
 doc_cloud_ken[doc_cloud_ken$freq > 1000,],
  options = list(pageLength = 7, 
                 dom = "tip"), rownames = FALSE
) %>% formatStyle("word",  color = "white", backgroundColor = "darkgreen", fontWeight = "bold")%>%
formatStyle("freq",  color = "darkgreen", backgroundColor = "white")


datatable(
 doc_cloud_usa[doc_cloud_usa$freq > 7,],
  options = list(pageLength = 10, 
                 dom = "tip"), rownames = FALSE
) %>% formatStyle("word",  color = "white", backgroundColor = "darkgreen", fontWeight = "bold")%>%
formatStyle("freq",  color = "darkgreen", backgroundColor = "white")


```


Regardless of the proportion issue, within the country, witch i will go through soon,

**Philippines:**

Although supplies has high number, but the rest is strongly related to Food

**Kenya**

Same like Philippines mainly Food Issue

**USA**
Not surprisingly; equipment, Marketing, Expand ...etc.. all related to Business enhancement... and quality of service improvement. 

I'll try to dive deeper!