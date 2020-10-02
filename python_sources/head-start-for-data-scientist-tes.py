
---

title: "Head Start for Data Scientist"

output:

  html_document:

    number_sections: TRUE

    toc: TRUE

    fig_height: 4

    fig_width: 7

    code_folding: show

---



```{r setup, include=FALSE, echo=FALSE}

knitr::opts_chunk$set(echo=TRUE)



```



#   IT IS WHAT IT IS - Data Science ��



Early days, When i was a newbie to the world of Machine Learning. 

I used to get overwhelmed by small decisions, like choosing the language to code with, choosing the right online courses, or choosing the correct algorithms.





So, I have planned to make it easier for folks to get into Machine Learning.



I�ll assume that many of us are starting from scratch on our Machine Learning journey. Let�s find out how current professionals in the field reached their destination, and how we can emulate them on our journey.



##  HOW IT CAN BE DONE !!



**Stage 1 - Commit Yourself**



For anyone starting out in Machine Learning, it�s important to surround yourselves with people who are also learning, teaching and practicing Machine Learning.



Learning the ropes is not easy if you do it alone. So, commit yourselves to learning Machine Learning�and find data science communities to help make your entry less painful.



**Stage 2 - Learn the Ecosystem**



**Discover the Machine Learning ecosystem**

Data Science is a field which has embraced and made full use of open source platforms. While data analysis can be conducted in a number of languages, using the right tools can make or break projects.





Data Science libraries are flourishing in the Python and R ecosystems. See [here for an infographic](https://www.datacamp.com/community/tutorials/r-or-python-for-data-analysis) on Python vs R for data analysis.



Whichever language you choose, Jupyter Notebook and RStudio makes our life much easier. They allow us to visualize data while manipulating it. Follow this [link](http://blog.kaggle.com/2015/12/07/three-things-i-love-about-jupyter-notebooks/) to read more on the features of Jupyter Notebook.



Kaggle, Analytics Vidhya, MachineLearningMastery and KD Nuggets are some of the active communityies where data scientists all over the world enrich each other�s learning.



Machine Learning has been democratized by online courses or MOOCs from Coursera, EdX and others, where we learn from amazing professors at world class universities. Here�s a [list of the top MOOCs](https://medium.freecodecamp.org/i-ranked-all-the-best-data-science-intro-courses-based-on-thousands-of-data-points-db5dc7e3eb8e) on data science available right now.



**Stage 3 - Cement the Foundation**



**Learn to manipulate data**



**Data scientists, according to interviews and expert estimates, spend 50 percent to 80 percent of their time mired in the mundane labor of collecting and preparing unruly digital data, before it can be explored for useful nuggets. - Steve Lohr of New York Times**



�Data Crunching� is the soul of the whole Machine Learning workflow. To help with this process, the [Pandas](https://pandas.pydata.org/pandas-docs/stable/) library in python or R�s DataFrames allow you to manipulate and conduct analysis. They provide data structures for relational or labeled data.





Data science is more than just building machine learning models. It�s also about explaining the models and using them to drive data-driven decisions. In the journey from analysis to data-driven outcomes, data visualization plays a very important role of presenting data in a powerful and credible way.



[Matplotlib](https://matplotlib.org/) library in Python or [ggplot](http://ggplot2.org/) in R offer complete 2D graphic support with very high flexibility to create high quality data visualizations.



These are some of the libraries you will be spending most of your time on when conducting the analysis.





**Stage 4 -  Practice day in and day out**





**Learn Machine Learning algorithms and practice them**



After the foundation is set, you get to implement the Machine Learning algorithms to predict and do all the cool stuff.



The Scikit-learn library in Python or the caret, e1071 libraries in R provide a range of supervised and unsupervised learning algorithms via a consistent interface.



These let you implement an algorithm without worrying about the inner workings or nitty-gritty details.



Apply these machine learning algorithms in the use cases you find all around you. This could either be in your work, or you can practice in Kaggle competitions. In these, data scientists all around the world compete at building models to solve problems.



Simultaneously, understand the inner workings of one algorithm after another. Starting with �Hello World!� of Machine Learning, Linear Regression then move to Logistic Regression, Decision Trees to Support Vector Machines. This will require you to brush up your statistics and linear algebra.



Coursera Founder Andrew Ng, a pioneer in AI has developed a [Machine Learning course](https://www.coursera.org/learn/machine-learning) which gives you a good starting point to understanding inner workings of Machine Learning algorithms.



**Stage 5 - Learn the advanced skills**





**Learn complex Machine Learning Algorithms and Deep Learning architectures**

While Machine Learning as a field was established long back, the recent hype and media attention is primarily due to Machine Learning applications in AI fields like Computer Vision, Speech Recognition, Language Processing. Many of these have been pioneered by the tech giants like Google, Facebook, Microsoft.



These recent advances can be credited to the progress made in cheap computation, the availability of large scale data, and the development of novel Deep Learning architectures.



To work in Deep Learning, you will need to learn how to process unstructured data � be it free text, images, 

You will learn to use platforms like TensorFlow or Torch, which lets us apply Deep Learning without worrying about low level hardware requirements. You will learn Reinforcement learning, which has made possible modern AI wonders like AlphaGo Zero







# WHAT IT IS NOW,...

 

I see many new learners at Kaggle, though of making one kernal for them to have a head start.

This kernal for basic learners, is an attempt to get a quick understanding of Data Science, i picked regular conversation approch.

In kernal we will come accross two characters 'MARK' and 'JAMES' where MARK is new to Data Science (Laymen) and JAMES makes him understand concepts

 

 

For easy start i took,

dataset - Titanic: Machine Learning from Disaster.





## Introduction.



On 14 April 1912, the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) struck a large iceberg and took approximately 1,500 of its passengers and crew below the icy depths of the Atlantic Ocean. Considered one of the worst peacetime disasters at sea, this tragic event led to the creation of numerous [safety regulations and policies](http://www.gc.noaa.gov/gcil_titanic-history.html) to prevent such a catastrophe from happening again. Some critics, however, argue that circumstances other than luck resulted in a disproportionate number of deaths. The purpose of this analysis is to explore factors that influenced a person�s likelihood to survive.



 



### Software.

The following analysis was conducted in the [R software environment for statistical computing](https://www.r-project.org/).



 

**MARK**    - JAMES, What is that i am learning today?                                         

**JAMES**   - Basics of Data science.

 

**MARK**    - What is Data science?                                                                                  

**JAMES**   - Data science is a multidisciplinary blend of data inference, algorithmm development, and technology in order to solve analytically complex problems.

 

**MARK**    - How do data scientists mine out insights?                                          

**JAMES**   - It starts with                                                                             

        1.Collect the raw data needed to solve the problem.                                                                            

        2.Process the data (data wrangling).                                                                           

        3.Explore the data (data visualization).                                                                            

        4.Perform in-depth analysis (machine learning, statistical models, algorithms).                                                                            

        5.Communicate results of the analysis.                                      

 



**MARK**    - James, could you please explain in elaborate.                                                                               

**JAMES**   - Yes, Here in 'Titanic: Machine Learning from Disaster�'  we have a raw data set given. In general data is fetched from database



##Import library's, data

                                         

**MARK**    - How to import data set into Rstudio?                                         

**JAMES**   - Prior to loading data, library should be called for functions and specific algorithms





**MARK**    - Oh what if librarys are not called.                                         

**JAMES**   - Error message are thrown in R Console, on running functions/Algorithms



**MARK**    - Oh Gowd, i will mind calling library.                                                   

**JAMES**   - lets import library's.



```{r dependencies, message = FALSE, warning = FALSE}



# data wrangling

library(tidyverse)

library(forcats)

library(stringr)

library(caTools)



# data assessment/visualizations

library(DT)

library(data.table)

library(pander)

library(ggplot2)

library(scales)

library(grid)

library(gridExtra)

library(corrplot)

library(VIM) 

library(knitr)

library(vcd)

library(caret)





# model

library(xgboost)

library(MLmetrics)

library('randomForest') 

library('rpart')

library('rpart.plot')

library('car')

library('e1071')

library(vcd)

library(ROCR)

library(pROC)

library(VIM)

library(glmnet) 



```



**MARK**    - Now we can import data set.                                                                                                                               

**JAMES**   - Yes,



```{r, message=FALSE, warning=FALSE, results='hide'}



train <- read_csv('../input/train.csv')

test  <- read_csv('../input/test.csv')







```





**JAMES**   - For studing the complete data set lets join test and train data set.                                          

Before that we will add a new coloum "set" and give name as 'test' for test dataset                                          

and 'train' for train dataset to have an idea about which record it is.                                         



```{r , message=FALSE, warning=FALSE, results='hide'}



train$set <- "train"

test$set  <- "test"

test$Survived <- NA

full <- rbind(train, test)



```



**MARK**    - Are we done with Collecting the raw data needed to solve the problem?                                         

**JAMES**   - Yes, NextProcess the data.  



                                                                                  

                                         



**MARK**    - Why do we need to Process the data (data wrangling)?                                          

**JAMES**   - The data you have collected is still �raw data�,                                         

which is very likely to contain mistakes, missing and corrupt values.                                          

Before you draw any conclusions from the data you need to subject it to some data wrangling,                                          

which is the subject of our next section.We select the data what we want to do operation                                         







**MARK**    - What operations performed under data science?                                         

**JAMES**   - This is over all view .                                                                                  

<center><img src="https://doubleclix.files.wordpress.com/2012/12/data-science-02.jpg"></center>



**JAMES**   - This give a clear idea .                                                                                  

<center><img src="https://cdn-images-1.medium.com/max/1600/1*2T5rbjOBGVFdSvtlhCqlNg.png"></center>





                                                                                  

                                         

**MARK**    -This look great, it shows even tools/programming language.                                          

**JAMES**   -In this kernal we will be using R programming language.                                                                                  

                                                                                  

                                                                                  

                                         

**JAMES**   -Before that we will have a look-into data - Exploratory Analysis                                                                                                                           

1. Check for data how it is scattered                                                                                   

2.dataset dimensions                                                                                  

3.columns names                                           

4.How many unique values are there in each row                                         

4. Missing values  etc                                                                                   

lets do str, names, summary, glimpse,                                                                                                                             



```{r , message=FALSE, warning=FALSE, results='hide'}



# check data

str(full)



# dataset dimensions

dim(full)



# Unique values per column

lapply(full, function(x) length(unique(x))) 



#Check for Missing values



missing_values <- full %>% summarize_all(funs(sum(is.na(.))/n()))



missing_values <- gather(missing_values, key="feature", value="missing_pct")

missing_values %>% 

  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +

  geom_bar(stat="identity",fill="red")+

  coord_flip()+theme_bw()





#Useful data quality function for missing values



checkColumn = function(df,colname){

  

  testData = df[[colname]]

  numMissing = max(sum(is.na(testData)|is.nan(testData)|testData==''),0)



  

  if (class(testData) == 'numeric' | class(testData) == 'Date' | class(testData) == 'difftime' | class(testData) == 'integer'){

    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = sum(is.infinite(testData)), 'avgVal' = mean(testData,na.rm=TRUE), 'minVal' = round(min(testData,na.rm = TRUE)), 'maxVal' = round(max(testData,na.rm = TRUE)))

  } else{

    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = NA,  'avgVal' = NA, 'minVal' = NA, 'maxVal' = NA)

  }

  

}

checkAllCols = function(df){

  resDF = data.frame()

  for (colName in names(df)){

    resDF = rbind(resDF,as.data.frame(checkColumn(df=df,colname=colName)))

  }

  resDF

}





datatable(checkAllCols(full), style="bootstrap", class="table-condensed", options = list(dom = 'tp',scrollX = TRUE))





```



```{r , message=FALSE, warning=FALSE, results='hide'}



miss_pct <- map_dbl(full, function(x) { round((sum(is.na(x)) / length(x)) * 100, 1) })



miss_pct <- miss_pct[miss_pct > 0]



data.frame(miss=miss_pct, var=names(miss_pct), row.names=NULL) %>%

    ggplot(aes(x=reorder(var, -miss), y=miss)) + 

    geom_bar(stat='identity', fill='red') +

    labs(x='', y='% missing', title='Percent missing data by feature') +

    theme(axis.text.x=element_text(angle=90, hjust=1))









```



## Feature engineering.



**MARK**    -  What is  feature engineering?                                                                                

**JAMES**   -  This process attempts to create additional relevant features from the existing raw features in the data,

and to increase the predictive power of the learning algorithm. 



To get an idea on feature enginering technique reffer - https://github.com/bobbbbbi/Machine-learning-Feature-engineering-techniques



## Data manupulation



**MARK**    - So we know how our dataset?                                                                               

**JAMES**   - Then we start doing with data manupulation.                                                                                    





**MARK**    -  What is data manupulation?                                                                                 

**JAMES**   -  Data manipulation is the process of changing data in an effort to make it easier to read or be more organized.                                            



                                                                                  

 The following section focuses on preparing the data so that it can be used for study, such as exploratory data analysis and modeling fitting.



### Age



Replace missing Age cells with the mean Age of all passengers on the Titanic.



```{r age, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



full <- full %>%

    mutate(

      Age = ifelse(is.na(Age), mean(full$Age, na.rm=TRUE), Age),

      `Age Group` = case_when(Age < 13 ~ "Age.0012", 

                                 Age >= 13 & Age < 18 ~ "Age.1317",

                                 Age >= 18 & Age < 60 ~ "Age.1859",

                                 Age >= 60 ~ "Age.60Ov"))



```



### Embarked



Use the most common code to replace NAs in the *Embarked* feature.



```{r pp_embarked, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), 'S')



```



### Titles



Extract an individual's title from the *Name* feature.



```{r pp_titles, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}







names <- full$Name

title <-  gsub("^.*, (.*?)\\..*$", "\\1", names)



full$title <- title



table(title)



  

###MISS, Mrs, Master and Mr are taking more numbers



###Better to group Other titles into bigger basket by checking gender and survival rate to aviod any overfitting





full$title[full$title == 'Mlle']        <- 'Miss' 

full$title[full$title == 'Ms']          <- 'Miss'

full$title[full$title == 'Mme']         <- 'Mrs' 

full$title[full$title == 'Lady']          <- 'Miss'

full$title[full$title == 'Dona']          <- 'Miss'



## I am afraid creating a new varible with small data can causes a overfit

## However, My thinking is that combining below feauter into original variable may loss some predictive power as they are all army folks, doctor and nobel peoples 



full$title[full$title == 'Capt']        <- 'Officer' 

full$title[full$title == 'Col']        <- 'Officer' 

full$title[full$title == 'Major']   <- 'Officer'

full$title[full$title == 'Dr']   <- 'Officer'

full$title[full$title == 'Rev']   <- 'Officer'

full$title[full$title == 'Don']   <- 'Officer'

full$title[full$title == 'Sir']   <- 'Officer'

full$title[full$title == 'the Countess']   <- 'Officer'

full$title[full$title == 'Jonkheer']   <- 'Officer'  



```



### Family Groups



Families are binned into a discretized feature based on family member count.



```{r pp_familygrp, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



full$FamilySize <-full$SibSp + full$Parch + 1 

full$FamilySized[full$FamilySize == 1] <- 'Single' 

full$FamilySized[full$FamilySize < 5 & full$FamilySize >= 2] <- 'Small' 

full$FamilySized[full$FamilySize >= 5] <- 'Big' 

full$FamilySized=as.factor(full$FamilySized)



         

```



###Tickets

Engineer features based on all the passengers with the same ticket.



```{r, message=FALSE, warning=FALSE}



##Engineer features based on all the passengers with the same ticket

ticket.unique <- rep(0, nrow(full))

tickets <- unique(full$Ticket)



for (i in 1:length(tickets)) {

  current.ticket <- tickets[i]

  party.indexes <- which(full$Ticket == current.ticket)

  

  

  for (k in 1:length(party.indexes)) {

    ticket.unique[party.indexes[k]] <- length(party.indexes)

  }

}



full$ticket.unique <- ticket.unique





full$ticket.size[full$ticket.unique == 1]   <- 'Single'

full$ticket.size[full$ticket.unique < 5 & full$ticket.unique>= 2]   <- 'Small'

full$ticket.size[full$ticket.unique >= 5]   <- 'Big'



```







### Independent Variable/Target



### Survival



The independent variable, *Survived*, is labeled as a *Bernoulli trial* where a passenger or crew member surviving is encoded with the value of 1. Among observations in the train set, approximately 38% of passengers and crew survived.



```{r iv, message=FALSE, warning=FALSE}



full <- full %>%

  mutate(Survived = case_when(Survived==1 ~ "Yes", 

                              Survived==0 ~ "No"))



crude_summary <- full %>%

  filter(set=="train") %>%

  select(PassengerId, Survived) %>%

  group_by(Survived) %>%

  summarise(n = n()) %>%

  mutate(freq = n / sum(n))



crude_survrate <- crude_summary$freq[crude_summary$Survived=="Yes"]



kable(crude_summary, caption="2x2 Contingency Table on Survival.", format="markdown")



```



##Exploratory data analysis

**MARK**  - What is exploratory data analysis?                                                                                                       

**JAMES** - Data science is a multidisciplinary blend of data inference, algorithmm development, and technology in order to solve analytically complex problems.                                                   





In statistics, exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.



For more ways of Data Visualization[refer data visualization handbook](https://www.kaggle.com/hiteshp/visualization-handbook)





### Relationship Between Dependent and Independent Variables



### Dependent Variables/Predictors {.tabset}



**Note - Go through each tab for different Variables**



### Relationship to Survival Rate {.tabset}



#### Pclass {-}



```{r rate_pclass, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Pclass, fill=Survived)) +

  geom_bar(position = "fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Class") + 

  theme_minimal()

```



#### Sex {-}



```{r rate_sex, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Sex, fill=Survived)) +

  geom_bar(position = "fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Sex") + 

  theme_minimal()

```



#### Age {-}



```{r rate_age, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



tbl_age <- full %>%

  filter(set=="train") %>%

  select(Age, Survived) %>%

  group_by(Survived) %>%

  summarise(mean.age = mean(Age, na.rm=TRUE))



ggplot(full %>% filter(set=="train"), aes(Age, fill=Survived)) +

  geom_histogram(aes(y=..density..), alpha=0.5) +

  geom_density(alpha=.2, aes(colour=Survived)) +

  geom_vline(data=tbl_age, aes(xintercept=mean.age, colour=Survived), lty=2, size=1) +

  scale_fill_brewer(palette="Set1") +

  scale_colour_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Density") +

  ggtitle("Survival Rate by Age") + 

  theme_minimal()

```



#### Age Groups {-}



```{r rate_age_group, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train" & !is.na(Age)), aes(`Age Group`, fill=Survived)) +

  geom_bar(position = "fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Age Group") + 

  theme_minimal()

```



#### SibSp {-}



```{r rate_sibsp, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(SibSp, fill=Survived)) +

  geom_bar(position = "fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by SibSp") + 

  theme_minimal()

```



#### Parch {-}



```{r rate_parch, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Parch, fill=Survived)) +

  geom_bar(position = "fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Parch") + 

  theme_minimal()

```



#### Embarked {-}



```{r rate_embarked, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Embarked, fill=Survived)) +

  geom_bar(position = "fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Embarked") + 

  theme_minimal()

```



#### Title {-}



```{r rate_title, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train") %>% na.omit, aes(title, fill=Survived)) +

  geom_bar(position="fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Title") + 

  theme_minimal() +

  theme(axis.text.x = element_text(angle = 90, hjust = 1))



```



#### Family {-}



```{r rate_family, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train") %>% na.omit, aes(`FamilySize`, fill=Survived)) +

  geom_bar(position="fill") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  ylab("Survival Rate") +

  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +

  ggtitle("Survival Rate by Family Group") + 

  theme_minimal() +

  theme(axis.text.x = element_text(angle = 90, hjust = 1))

```

                                                                                                                                                         

                                                                                                                                                         

                                                                                                                                                         

**Note - Go through each tab for different Relationship to Frequency of Variables**

### Relationship to Frequency {.tabset}



#### Pclass {-}



```{r freq_pclass, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Pclass, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Class") + 

  theme_minimal()

```



#### Sex {-}



```{r freq_sex, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Sex, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Sex") + 

  theme_minimal()

```



#### Age {-}



```{r freq_age, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Age, fill=Survived)) +

  geom_histogram(aes(y=..count..), alpha=0.5) +

  geom_vline(data=tbl_age, aes(xintercept=mean.age, colour=Survived), lty=2, size=1) +

  scale_fill_brewer(palette="Set1") +

  scale_colour_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Density") +

  ggtitle("Survived by Age") + 

  theme_minimal()

```



#### Age Groups {-}



```{r freq_age_group, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train" & !is.na(Age)), aes(`Age Group`, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Age Group") + 

  theme_minimal()

```



#### SibSp {-}



```{r freq_sibsp, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(SibSp, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=percent) +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by SibSp") + 

  theme_minimal()

```



#### Parch {-}



```{r freq_parch, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Parch, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Parch") + 

  theme_minimal()

```



#### Embarked {-}



```{r freq_embarked, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train"), aes(Embarked, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Embarked") + 

  theme_minimal()

```



#### Title {-}



```{r freq_title, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train") %>% na.omit, aes(title, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Title") + 

  theme_minimal() +

  theme(axis.text.x = element_text(angle = 90, hjust = 1))



```



#### Family {-}



```{r freq_family, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4.5, fig.width=9}



ggplot(full %>% filter(set=="train") %>% na.omit, aes(`FamilySize`, fill=Survived)) +

  geom_bar(position="stack") +

  scale_fill_brewer(palette="Set1") +

  scale_y_continuous(labels=comma) +

  ylab("Passengers") +

  ggtitle("Survived by Family Group") + 

  theme_minimal() +

  theme(axis.text.x = element_text(angle = 90, hjust = 1))

```



### Interactive Relationships Between Variables



#### Correlation Plot



**MARK**    -  What is Correlation Plot?                                                                                 

**JAMES**   -  The corrplot package is a graphical display of a correlation matrix, confidence interval. It also contains some algorithms to do matrix reordering. In addition, corrplot is good at details, including choosing color, text labels, color labels, layout, etc.





Correlation measures between numeric features suggest redundant information such as *Fare* with *Pclass*. This relationship, however, may be distorted due to passengers who boarded as a family where *Fare* represents the sum of a family's total cost.



```{r corrplot, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4, fig.width=9}



tbl_corr <- full %>%

  filter(set=="train") %>%

  select(-PassengerId, -SibSp, -Parch) %>%

  select_if(is.numeric) %>%

  cor(use="complete.obs") %>%

  corrplot.mixed(tl.cex=0.85)



```



#### Mosaic Plot



**MARK**    -  What is Mosaic Plot?                                                                                 

**JAMES**   -  Mosaic plot (also known as Marimekko diagrams) is a graphical method for visualizing data from two or more qualitative variables. It is the multidimensional extension of spineplots, which graphically display the same information for only one variable.





```{r mosaicplot, message=FALSE, warning=FALSE, echo=TRUE, fig.height=4, fig.width=9}



tbl_mosaic <- full %>%

  filter(set=="train") %>%

  select(Survived, Pclass, Sex, AgeGroup=`Age Group`, title, Embarked, `FamilySize`) %>%

  mutate_all(as.factor)



mosaic(~Pclass+Sex+Survived, data=tbl_mosaic, shade=TRUE, legend=TRUE)



```



#### Alluvial Diagram



**MARK**    -  What is Alluvial Diagram?                                                                                 

**JAMES**   -  Alluvial diagrams are a type of flow diagram originally developed to represent changes in network structure over time. In allusion to both their visual appearance and their emphasis on flow, alluvial diagrams are named after alluvial fans that are naturally formed by the soil deposited from streaming water.



Likelihood to survive was lowest among third class passengers; however, their chances for survival improved when *Sex* was female. Surprisingly, half of toddlers and adolescents perished. A plausible explanation for this could be that many of these children who perished came from larger families as suggested in the conditional inference tree model below.



```{r alluvial, message=FALSE, warning=FALSE, echo=TRUE, fig.height=6, fig.width=9}

library(alluvial)



tbl_summary <- full %>%

  filter(set=="train") %>%

  group_by(Survived, Sex, Pclass, `Age Group`, title) %>%

  summarise(N = n()) %>% 

  ungroup %>%

  na.omit

  

alluvial(tbl_summary[, c(1:4)],

         freq=tbl_summary$N, border=NA,

         col=ifelse(tbl_summary$Survived == "Yes", "blue", "gray"),

         cex=0.65,

         ordering = list(

           order(tbl_summary$Survived, tbl_summary$Pclass==1),

           order(tbl_summary$Sex, tbl_summary$Pclass==1),

           NULL,

           NULL))

```



## Machine learning algorithm                                                                                v



**MARK**    -  What is Machine learning?                                                                                 

**JAMES**   -  Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.



The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.



<center><img src="https://www.mathworks.com/help/stats/machinelearningtypes.jpg"></center>



**MARK**    -  What is Supervised and Unsupervised machine learning?                                                                                 

**JAMES**   -                                                                                                                                                                  

**Supervised Learning** - Supervised learning is a data mining task of inferring a function from labeled training data.The training data consist of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and the desired output value (also called the supervisory signal).

    A supervised learning algorithm analyzes the training data and produces an inferred function, which can used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a �reasonable� way.                                                                                                                                                                                                                                                

**Unsupervised Learning** - In data mining or even in data science world, the problem of an unsupervised learning task is trying to find hidden structure in unlabeled data. Since the examples given to the learner are unlabeled, there is no error or reward signal to evaluate a potential solution.



<center><img src="https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAtrAAAAJDc2ZmQ4NDE0LTI0ODAtNDdmYi1hNDI0LThhN2M4MTFjNmYzYw.png"></center>







**MARK**    -  So, now we will be doing Supervised machine learning?                                                                                 

**JAMES**   -  Yes, it is Supervised machine learning as we want to predict survival of passanger.

    

**MARK**    -  What is algorithms are we going to use in Supervisedmachine learning?                                                                                

**JAMES**   -  For easy understanding look in at this pic for more information of algorithm statistical function [reffer](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

     







<center><img src="https://blogs.sas.com/content/subconsciousmusings/files/2017/04/machine-learning-cheet-sheet.png"></center>









                                                 

**JAMES**   - Let�s prepare the training data set with "Pclass", "title","Sex","Embarked","FamilySized","ticket.size"

 Variables and splitting our data set into 70% as training dataset and 30 % as testing data set                                                                                

**MARK**    - What is training and test data set?                                                                                                    





**JAMES**   -  

                                                                                                  

**Training Set** - In machine learning, a training set is a dataset used to train a model.  In training the model, specific features are picked out from the training set.  These features are then incorporated into the model. 



**Test Set** - The test set is a dataset used to measure how well the model performs at making predictions on that test set







###Prepare and keep data set.



```{r, message=FALSE, warning=FALSE}







###lets prepare and keep data in the proper format



feauter1<-full[1:891, c("Pclass", "title","Sex","Embarked","FamilySized","ticket.size")]

response <- as.factor(train$Survived)

feauter1$Survived=as.factor(train$Survived)





###For Cross validation purpose will keep 20% of data aside from my orginal train set

##This is just to check how well my data works for unseen data

set.seed(500)

ind=createDataPartition(feauter1$Survived,times=1,p=0.8,list=FALSE)

train_val=feauter1[ind,]

test_val=feauter1[-ind,]



```





```{r, message=FALSE, warning=FALSE}



####check the proprtion of Survival rate in orginal training data, current traing and testing data

round(prop.table(table(train$Survived)*100),digits = 1)

round(prop.table(table(train_val$Survived)*100),digits = 1)

round(prop.table(table(test_val$Survived)*100),digits = 1)





```







                                      

**JAMES**   - Let�s do training with algorithms.                                                                                                                  





**MARK**    - After training with algorithms,  what�s next ?                                                                                                                                                                   

**JAMES**   - We have to validate our trained algorithms with test data set.                                                                                





**MARK**    - How do we measure we our algorithms performance?                                                   

**JAMES**   - With [goodness of fit](https://en.wikipedia.org/wiki/Goodness_of_fit),  lets go with Confusion Matrix for validation.



                                                                                                                                                         

                                                                                                                                                         

                                                                                                                                                         

                                                                                                                                                         



**Note - Go through each tab for different algorithms**



### Predictive Analysis and Cross Validation {.tabset}



#### Decison tree {-}



```{r, message=FALSE, warning=FALSE}



##Random forest is for more better than Single tree however single tree is very easy to use and illustrate

set.seed(1234)

Model_DT=rpart(Survived~.,data=train_val,method="class")





rpart.plot(Model_DT,extra =  3,fallen.leaves = T)



###Surprise, Check out the plot,  our Single tree model is using only Title, Pclass and Ticket.size and vomited rest

###Lets Predict train data and check the accuracy of single tree

```





```{r, message=FALSE, warning=FALSE}

PRE_TDT=predict(Model_DT,data=train_val,type="class")

confusionMatrix(PRE_TDT,train_val$Survived)



#####Accuracy is 0.8375

####Not at all bad using Single tree and just 3 feauters



##There is chance of overfitting in Single tree, So I will go for cross validation using '10 fold techinque'



set.seed(1234)

cv.10 <- createMultiFolds(train_val$Survived, k = 10, times = 10)



# Control

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,

                       index = cv.10)



                     



train_val <- as.data.frame(train_val)



##Train the data

Model_CDT <- train(x = train_val[,-7], y = train_val[,7], method = "rpart", tuneLength = 30,

                   trControl = ctrl)



##Check the accurcay

##Accurcay using 10 fold cross validation of Single tree is 0.8139 

##Seems Overfitted earlier using Single tree, there our accurcay rate is 0.83



# check the variable imporatnce, is it the same as in Single tree?

rpart.plot(Model_CDT$finalModel,extra =  3,fallen.leaves = T)



##Yes, there is no change in the imporatnce of variable









###Lets cross validate the accurcay using data that kept aside for testing purpose

PRE_VDTS=predict(Model_CDT$finalModel,newdata=test_val,type="class")

confusionMatrix(PRE_VDTS,test_val$Survived)



###There it is, How exactly our train data and test data matches in accuracy (0.8192)





col_names <- names(train_val)



train_val[col_names] <- lapply(train_val[col_names] , factor)

test_val[col_names] <- lapply(test_val[col_names] , factor)



```





#### Random Forest {-}



```{r, message=FALSE, warning=FALSE}



set.seed(1234)





rf.1 <- randomForest(x = train_val[,-7],y=train_val[,7], importance = TRUE, ntree = 1000)

rf.1

varImpPlot(rf.1)







####Random Forest accurcay rate is 82.91 which is 1% better than the decison  tree

####Lets remove 2 redaundant varibles and do the modeling again

train_val1=train_val[,-4:-5]

test_val1=test_val[,-4:-5]





set.seed(1234)

rf.2 <- randomForest(x = train_val1[,-5],y=train_val1[,5], importance = TRUE, ntree = 1000)

rf.2

varImpPlot(rf.2)



###Can see the Magic now, increase in accuracy by just removing 2 varibles, accuracy now is 84.03 



##Even though random forest is so power full we accept the model only after cross validation





set.seed(2348)

cv10_1 <- createMultiFolds(train_val1[,5], k = 10, times = 10)



# Set up caret's trainControl object per above.

ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,

                      index = cv10_1)







set.seed(1234)

rf.5<- train(x = train_val1[,-5], y = train_val1[,5], method = "rf", tuneLength = 3,

              ntree = 1000, trControl =ctrl_1)



rf.5



##Cross validation give us the accurcay rate of .8393



###Lets Predict the test data 



pr.rf=predict(rf.5,newdata = test_val1)



confusionMatrix(pr.rf,test_val1$Survived)



####accuracy rate is 0.8192, lower than what we have expected  



```



#### lasso-ridge regression {-}





```{r, message=FALSE, warning=FALSE}



train_val <- train_val %>%

  mutate(Survived = case_when(Survived==1 ~ "Yes", 

                              Survived==0 ~ "No"))







train_val<- as.data.frame(train_val)

train_val$title<-as.factor(train_val$title)

train_val$Embarked<-as.factor(train_val$Embarked)

train_val$ticket.size<-as.factor(train_val$ticket.size)



table(train_val$Survived)



test_val<- as.data.frame(test_val)

test_val$title<-as.factor(test_val$title)

test_val$Embarked<-as.factor(test_val$Embarked)

test_val$ticket.size<-as.factor(test_val$ticket.size)

test_val$Survived<-as.factor(test_val$Survived)







train.male = subset(train_val, train_val$Sex == "male")

train.female = subset(train_val, train_val$Sex == "female")

test.male = subset(test_val, test_val$Sex == "male")

test.female = subset(test_val, test_val$Sex == "female")





train.male$Sex = NULL



train.male$title = droplevels(train.male$title)



train.female$Sex = NULL

train.female$title = droplevels(train.female$title)



test.male$Sex = NULL





test.male$title = droplevels(test.male$title)





test.female$Sex = NULL

test.female$title = droplevels(test.female$title)



set.seed(101) 

train_ind <- sample.split(train.male$Survived, SplitRatio = .75)





# MALE



## set the seed to make your partition reproductible





cv.train.m <- train.male[train_ind, ]

cv.test.m  <- train.male[-train_ind, ]



# FEMALE

set.seed(100)



## set the seed to make your partition reproductible

set.seed(123)

train_ind <- sample.split(train.female$Survived, SplitRatio = .75)



cv.train.f <- train.male[train_ind, ]

cv.test.f  <- train.male[-train_ind, ]





x.m = data.matrix(cv.train.m[,1:5])

y.m = cv.train.m$Survived





set.seed(356)

# 10 fold cross validation

cvfit.m.ridge = cv.glmnet(x.m, y.m, 

                  family = "binomial", 

                  alpha = 0,

                  type.measure = "class")



cvfit.m.lasso = cv.glmnet(x.m, y.m, 

                  family = "binomial", 

                  alpha = 1,

                  type.measure = "class")

par(mfrow=c(1,2))

plot(cvfit.m.ridge, main = "Ridge")

plot(cvfit.m.lasso, main = "Lasso")



coef(cvfit.m.ridge, s = "lambda.min")



# Prediction on training set

PredTrain.M = predict(cvfit.m.ridge, newx=x.m, type="class")





table(cv.train.m$Survived, PredTrain.M, cv.train.m$title)



# Prediction on validation set

PredTest.M = predict(cvfit.m.ridge, newx=data.matrix(cv.test.m[,1:5]), type="class")

table(cv.test.m$Survived, PredTest.M, cv.test.m$title)





# Prediction on test set

PredTest.M = predict(cvfit.m.ridge, newx=data.matrix(test.male[,1:5]), type="class")

table(PredTest.M, test.male$title)





#female

x.f = data.matrix(cv.train.f[,1:5])

y.f = cv.train.f$Survived



set.seed(356)

cvfit.f.ridge = cv.glmnet(x.f, y.f, 

                  family = "binomial", 

                  alpha = 0,

                  type.measure = "class")

cvfit.f.lasso = cv.glmnet(x.f, y.f, 

                  family = "binomial", 

                  alpha = 1,

                  type.measure = "class")

par(mfrow=c(1,2))

plot(cvfit.f.ridge, main = "Ridge")

plot(cvfit.f.lasso, main = "Lasso")



coef(cvfit.f.ridge, s = "lambda.min")



# Ridge Model

# Prediction on training set

PredTrain.F = predict(cvfit.f.ridge, newx=x.f, type="class")

table(cv.train.f$Survived, PredTrain.F, cv.train.f$title)



confusionMatrix(cv.train.f$Survived, PredTrain.F)





# Prediction on validation set

PredTest.F = predict(cvfit.f.ridge, newx=data.matrix(cv.test.f[,1:5]), type="class")

table(cv.test.f$Survived, PredTest.F, cv.test.f$title)



confusionMatrix(cv.test.f$Survived, PredTest.F)





# Ridge Model

# Prediction on training set

PredTrain.F = predict(cvfit.f.lasso, newx=x.f, type="class")

table(cv.train.f$Survived, PredTrain.F, cv.train.f$title)



confusionMatrix(cv.train.f$Survived, PredTrain.F)



# Prediction on validation set

PredTest.F = predict(cvfit.f.lasso, newx=data.matrix(cv.test.f[,1:5]), type="class")

table(cv.test.f$Survived, PredTest.F, cv.test.f$title)



confusionMatrix(cv.test.f$Survived, PredTest.F)





# Prediction on test set

PredTest.F = predict(cvfit.f.ridge, newx=data.matrix(test.female[,1:5]), type="class")

table(PredTest.F, test.female$title)





MySubmission.F<-cbind(cv.train.m$Survived, PredTrain.M)

MySubmission.M<-cbind(cv.train.f$Survived, PredTrain.F)





MySubmission<-rbind(MySubmission.M,MySubmission.F)



colnames(MySubmission) <- c('Actual_Survived', 'predict')

MySubmission<- as.data.frame(MySubmission)



confusionMatrix(MySubmission$Actual_Survived, MySubmission$predict)



```









#### Support Vector Machine - Linear Support vector Machine {-}

```{r, message=FALSE, warning=FALSE}



###Before going to model lets tune the cost Parameter



set.seed(1274)

liner.tune=tune.svm(Survived~.,data=train_val1,kernel="linear",cost=c(0.01,0.1,0.2,0.5,0.7,1,2,3,5,10,15,20,50,100))



liner.tune



###best perforamnce when cost=3 and accuracy rate is 82.7





###Lets get a best.liner model  

best.linear=liner.tune$best.model



##Predict Survival rate using test data



best.test=predict(best.linear,newdata=test_val1,type="class")

confusionMatrix(best.test,test_val1$Survived)



###Linear model accuracy is 0.8136

```



#### XGBoost {-}

```{r, message=FALSE, warning=FALSE}





library(xgboost)

library(MLmetrics)



train <- read_csv('../input/train.csv')

test  <- read_csv('../input/test.csv')



train$set <- "train"

test$set  <- "test"

test$Survived <- NA

full <- rbind(train, test)



full <- full %>%

    mutate(

      Age = ifelse(is.na(Age), mean(full$Age, na.rm=TRUE), Age),

      `Age Group` = case_when(Age < 13 ~ "Age.0012", 

                                 Age >= 13 & Age < 18 ~ "Age.1317",

                                 Age >= 18 & Age < 60 ~ "Age.1859",

                                 Age >= 60 ~ "Age.60Ov"))

                                 

full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), 'S')





full <- full %>%

  mutate(Title = as.factor(str_sub(Name, str_locate(Name, ",")[, 1] + 2, str_locate(Name, "\\.")[, 1]- 1)))







full <- full %>%

  mutate(`Family Size`  = as.numeric(SibSp) + as.numeric(Parch) + 1,

         `Family Group` = case_when(

           `Family Size`==1 ~ "single",

           `Family Size`>1 & `Family Size` <=3 ~ "small",

           `Family Size`>= 4 ~ "large"

         ))

         

full <- full %>%

  mutate(Survived = case_when(Survived==1 ~ "Yes", 

                              Survived==0 ~ "No"))





full_2 <- full %>% 

  select(-Name, -Ticket, -Cabin, -set) %>%

  mutate(

    Survived = ifelse(Survived=="Yes", 1, 0)

  ) %>% 

  rename(AgeGroup=`Age Group`, FamilySize=`Family Size`, FamilyGroup=`Family Group`)





# OHE

ohe_cols <- c("Pclass", "Sex", "Embarked", "Title", "AgeGroup", "FamilyGroup")

num_cols <- setdiff(colnames(full_2), ohe_cols)



full_final <- subset(full_2, select=num_cols)



for(var in ohe_cols) {

  values <- unique(full_2[[var]])

  for(j in 1:length(values)) {

    full_final[[paste0(var,"_",values[j])]] <- (full_2[[var]] == values[j]) * 1

  }

}





submission <- TRUE



data_train <- full_final %>%

  filter(!is.na(Survived)) 



data_test  <- full_final %>% 

  filter(is.na(Survived))



set.seed(777)

ids <- sample(nrow(data_train))



# create folds for cv

n_folds <- ifelse(submission, 1, 5)



score <- data.table()

result <- data.table()







for(i in 1:n_folds) {

  

  if(submission) {

    x_train <- data_train %>% select(-PassengerId, -Survived)

    x_test  <- data_test %>% select(-PassengerId, -Survived)

    y_train <- data_train$Survived

    

  } else {

    train.ids <- ids[-seq(i, length(ids), by=n_folds)]

    test.ids  <- ids[seq(i, length(ids), by=n_folds)]

    

    x_train <- data_train %>% select(-PassengerId, -Survived)

    x_train <- x_train[train.ids,]

    

    x_test  <- data_train %>% select(-PassengerId, -Survived)

    x_test  <- x_test[test.ids,]

    

    y_train <- data_train$Survived[train.ids]

    y_test  <- data_train$Survived[test.ids]

  }

  

  x_train <- apply(x_train, 2, as.numeric)

  x_test <- apply(x_test, 2, as.numeric)

  

  if(submission) {

    nrounds <- 12

    early_stopping_round <- NULL

    dtrain <- xgb.DMatrix(data=as.matrix(x_train), label=y_train)

    dtest <- xgb.DMatrix(data=as.matrix(x_test))

    watchlist <- list(train=dtrain)

  } else {

    nrounds <- 3000

    early_stopping_round <- 100

    dtrain <- xgb.DMatrix(data=as.matrix(x_train), label=y_train)

    dtest <- xgb.DMatrix(data=as.matrix(x_test), label=y_test)

    watchlist <- list(train=dtrain, test=dtest)

  }

  

  params <- list("eta"=0.01,

                 "max_depth"=8,

                 "colsample_bytree"=0.3528,

                 "min_child_weight"=1,

                 "subsample"=1,

                 "objective"="reg:logistic",

                 "eval_metric"="auc")

  

  model_xgb <- xgb.train(params=params,

                         data=dtrain,

                         maximize=TRUE,

                         nrounds=nrounds,

                         watchlist=watchlist,

                         early_stopping_round=early_stopping_round,

                         print_every_n=2)

  

  pred <- predict(model_xgb, dtest)

  

  if(submission) {

    result <- cbind(data_test %>% select(PassengerId), Survived=round(pred, 0))

  } else {

    score <- rbind(score, 

                   data.frame(accuracy=Accuracy(round(pred, 0), y_test), best_iteration=model_xgb$best_iteration))

    temp   <- cbind(data_train[test.ids,], pred=pred)

    result <- rbind(result, temp)

  }

}





head(result)



```

















#### bRadial Support vector Machine {-}



```{r, message=FALSE, warning=FALSE}





######Lets go to non liner SVM, Radial Kerenl

set.seed(1274)



rd.poly=tune.svm(Survived~.,data=train_val1,kernel="radial",gamma=seq(0.1,5))



summary(rd.poly)

best.rd=rd.poly$best.model



###Non Linear Kerenel giving us a better accuray 



##Lets Predict test data

pre.rd=predict(best.rd,newdata = test_val1)



confusionMatrix(pre.rd,test_val1$Survived)



####Accurcay of test data using Non Liner model is 0.81

####it could be due to we are using smaller set of sample for testing data

```



#### Logistic Regression {-}



```{r, message=FALSE, warning=FALSE}



contrasts(train_val1$Sex)

contrasts(train_val1$Pclass)



##The above shows how the varible coded among themself



##Lets run Logistic regression model

log.mod <- glm(Survived ~ ., family = binomial(link=logit), 

               data = train_val1)

###Check the summary

summary(log.mod)

confint(log.mod)



###Predict train data

train.probs <- predict(log.mod, data=train_val1,type =  "response")

table(train_val1$Survived,train.probs>0.5)



(395+204)/(395+204+70+45)



###Logistic regression predicted train data with accuracy rate of 0.83 



test.probs <- predict(log.mod, newdata=test_val1,type =  "response")

table(test_val1$Survived,test.probs>0.5)



(97+47)/(97+12+21+47)



###Accuracy rate of test data is 0.8135



```

                                                                                                                                                              

                                                                                                                                                              

## Evaluate Machine Learning Algorithms



                                                                                                                                                              

Accuracy with **Random Forest**                  - 84.03%                                                                                                                                                                

Accuracy with **Dession trees**                  - 83.75%                                                                                                                                                                                                                                                                                                                                  

Accuracy with **Radial Support vector Machine**  - 81.92%                                                                                             

Accuracy with **lasso-ridge regression**         - 81.90%                                                                                                                                                                                                                                                                                                                                  

Accuracy with **Linear Support vector Machine**  - 81.36%                                                                                                                                                                

Accuracy with **Logistic Regression**            - 81.36%                                                                                                                                                                





**MARK**    - Oh Gowd, Random Forest works good with  84.03%  accuracy                                                                                                                                                             

**JAMES**   - Yes!!                                                                                                                                                               

                                                                                                                                                                                                        

                                                                                                    

**MARK**    - Thanks James.                                                   

                                                                                                                                                      

**Please excuse any typos.**                                                                                                    

**Thanks for reading. If you have any feedback,suggestions I'd love to hear! .**                                                                                                    

**Please like the kernel. Your likes are my motivation. ;) **                                                                                                                                                              





***You May also like this kernels*



--> [Data visualization handbook](https://www.kaggle.com/hiteshp/data-visualization-handbook)

--> ['R' to 'Python' tutorial](https://www.kaggle.com/hiteshp/data-visualization-handbook)

--> ['Python' to 'R' tutorial](https://www.kaggle.com/hiteshp/python-to-r-tutorial)

/**
 * Calculates and displays a walking route from the St Paul's Cathedral in London
 * to the Tate Modern on the south bank of the River Thames
 *
 * A full list of available request parameters can be found in the Routing API documentation.
 * see:  http://developer.here.com/rest-apis/documentation/routing/topics/resource-calculate-route.html
 *
 * @param   {H.service.Platform} platform    A stub class to access HERE services
 */
function calculateRouteFromAtoB (platform) {
  var router = platform.getRoutingService(),
    routeRequestParams = {
      mode: 'shortest;pedestrian',
      representation: 'display',
      waypoint0: '51.51326,-0.0968752', // St Paul's Cathedral
      waypoint1: '51.5081,-0.0985',  // Tate Modern
      routeattributes: 'waypoints,summary,shape,legs',
      maneuverattributes: 'direction,action'
    };


  router.calculateRoute(
    routeRequestParams,
    onSuccess,
    onError
  );
}
/**
 * This function will be called once the Routing REST API provides a response
 * @param  {Object} result          A JSONP object representing the calculated route
 *
 * see: http://developer.here.com/rest-apis/documentation/routing/topics/resource-type-calculate-route.html
 */
function onSuccess(result) {
  var route = result.response.route[0];
 /*
  * The styling of the route response on the map is entirely under the developer's control.
  * A representitive styling can be found the full JS + HTML code of this example
  * in the functions below:
  */
  addRouteShapeToMap(route);
  addManueversToMap(route);

  addWaypointsToPanel(route.waypoint);
  addManueversToPanel(route);
  addSummaryToPanel(route.summary);
  // ... etc.
}

/**
 * This function will be called if a communication error occurs during the JSON-P request
 * @param  {Object} error  The error message received.
 */
function onError(error) {
  alert('Can\'t reach the remote server');
}

/**
 * Boilerplate map initialization code starts below:
 */

// set up containers for the map  + panel
var mapContainer = document.getElementById('map'),
  routeInstructionsContainer = document.getElementById('panel');

//Step 1: initialize communication with the platform
// In your own code, replace variable window.apikey with your own apikey
var platform = new H.service.Platform({
  apikey: window.apikey
});
var defaultLayers = platform.createDefaultLayers();

//Step 2: initialize a map - this map is centered over Berlin
var map = new H.Map(mapContainer,
  defaultLayers.vector.normal.map,{
  center: {lat:52.5160, lng:13.3779},
  zoom: 13,
  pixelRatio: window.devicePixelRatio || 1
});
// add a resize listener to make sure that the map occupies the whole container
window.addEventListener('resize', () => map.getViewPort().resize());

//Step 3: make the map interactive
// MapEvents enables the event system
// Behavior implements default interactions for pan/zoom (also on mobile touch environments)
var behavior = new H.mapevents.Behavior(new H.mapevents.MapEvents(map));

// Create the default UI components
var ui = H.ui.UI.createDefault(map, defaultLayers);

// Hold a reference to any infobubble opened
var bubble;

/**
 * Opens/Closes a infobubble
 * @param  {H.geo.Point} position     The location on the map.
 * @param  {String} text              The contents of the infobubble.
 */
function openBubble(position, text){
 if(!bubble){
    bubble =  new H.ui.InfoBubble(
      position,
      // The FO property holds the province name.
      {content: text});
    ui.addBubble(bubble);
  } else {
    bubble.setPosition(position);
    bubble.setContent(text);
    bubble.open();
  }
}


/**
 * Creates a H.map.Polyline from the shape of the route and adds it to the map.
 * @param {Object} route A route as received from the H.service.RoutingService
 */
function addRouteShapeToMap(route){
  var lineString = new H.geo.LineString(),
    routeShape = route.shape,
    polyline;

  routeShape.forEach(function(point) {
    var parts = point.split(',');
    lineString.pushLatLngAlt(parts[0], parts[1]);
  });

  polyline = new H.map.Polyline(lineString, {
    style: {
      lineWidth: 4,
      strokeColor: 'rgba(0, 128, 255, 0.7)'
    }
  });
  // Add the polyline to the map
  map.addObject(polyline);
  // And zoom to its bounding rectangle
  map.getViewModel().setLookAtData({
    bounds: polyline.getBoundingBox()
  });
}


/**
 * Creates a series of H.map.Marker points from the route and adds them to the map.
 * @param {Object} route  A route as received from the H.service.RoutingService
 */
function addManueversToMap(route){
  var svgMarkup = '<svg width="18" height="18" ' +
    'xmlns="http://www.w3.org/2000/svg">' +
    '<circle cx="8" cy="8" r="8" ' +
      'fill="#1b468d" stroke="white" stroke-width="1"  />' +
    '</svg>',
    dotIcon = new H.map.Icon(svgMarkup, {anchor: {x:8, y:8}}),
    group = new  H.map.Group(),
    i,
    j;

  // Add a marker for each maneuver
  for (i = 0;  i < route.leg.length; i += 1) {
    for (j = 0;  j < route.leg[i].maneuver.length; j += 1) {
      // Get the next maneuver.
      maneuver = route.leg[i].maneuver[j];
      // Add a marker to the maneuvers group
      var marker =  new H.map.Marker({
        lat: maneuver.position.latitude,
        lng: maneuver.position.longitude} ,
        {icon: dotIcon});
      marker.instruction = maneuver.instruction;
      group.addObject(marker);
    }
  }

  group.addEventListener('tap', function (evt) {
    map.setCenter(evt.target.getGeometry());
    openBubble(
       evt.target.getGeometry(), evt.target.instruction);
  }, false);

  // Add the maneuvers group to the map
  map.addObject(group);
}


/**
 * Creates a series of H.map.Marker points from the route and adds them to the map.
 * @param {Object} route  A route as received from the H.service.RoutingService
 */
function addWaypointsToPanel(waypoints){



  var nodeH3 = document.createElement('h3'),
    waypointLabels = [],
    i;


   for (i = 0;  i < waypoints.length; i += 1) {
    waypointLabels.push(waypoints[i].label)
   }

   nodeH3.textContent = waypointLabels.join(' - ');

  routeInstructionsContainer.innerHTML = '';
  routeInstructionsContainer.appendChild(nodeH3);
}

/**
 * Creates a series of H.map.Marker points from the route and adds them to the map.
 * @param {Object} route  A route as received from the H.service.RoutingService
 */
function addSummaryToPanel(summary){
  var summaryDiv = document.createElement('div'),
   content = '';
   content += '<b>Total distance</b>: ' + summary.distance  + 'm. <br/>';
   content += '<b>Travel Time</b>: ' + summary.travelTime.toMMSS() + ' (in current traffic)';


  summaryDiv.style.fontSize = 'small';
  summaryDiv.style.marginLeft ='5%';
  summaryDiv.style.marginRight ='5%';
  summaryDiv.innerHTML = content;
  routeInstructionsContainer.appendChild(summaryDiv);
}

/**
 * Creates a series of H.map.Marker points from the route and adds them to the map.
 * @param {Object} route  A route as received from the H.service.RoutingService
 */
function addManueversToPanel(route){



  var nodeOL = document.createElement('ol'),
    i,
    j;

  nodeOL.style.fontSize = 'small';
  nodeOL.style.marginLeft ='5%';
  nodeOL.style.marginRight ='5%';
  nodeOL.className = 'directions';

     // Add a marker for each maneuver
  for (i = 0;  i < route.leg.length; i += 1) {
    for (j = 0;  j < route.leg[i].maneuver.length; j += 1) {
      // Get the next maneuver.
      maneuver = route.leg[i].maneuver[j];

      var li = document.createElement('li'),
        spanArrow = document.createElement('span'),
        spanInstruction = document.createElement('span');

      spanArrow.className = 'arrow '  + maneuver.action;
      spanInstruction.innerHTML = maneuver.instruction;
      li.appendChild(spanArrow);
      li.appendChild(spanInstruction);

      nodeOL.appendChild(li);
    }
  }

  routeInstructionsContainer.appendChild(nodeOL);
}


Number.prototype.toMMSS = function () {
  return  Math.floor(this / 60)  +' minutes '+ (this % 60)  + ' seconds.';
}

// Now use the map as required...
calculateRouteFromAtoB (platform);