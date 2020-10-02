---
title: "DonorsChoose (Motivations to Donate Again)"
author: "Marco Marchetti"
date: "27 maggio 2018"
output: 
  html_document:
    toc: true
    fig_width: 10
    code_folding: hide
---

```{r library,message=FALSE,warning=FALSE, results="hide"}
library(tidyr)
library(dplyr)
library(ggplot2)

library(tidyverse)
library(stringr)

library(DT)

```
# 1 Introduction

Founded in 2000 by a Bronx history teacher, DonorsChoose.org has raised $685 million for America’s classrooms. Teachers at three-quarters of all the public schools in the U.S. have come to DonorsChoose.org to request what their students need, making DonorsChoose.org the leading platform for supporting public education. To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.  

In this analysis we compare single dorors/donations with donors with more than one donations to understand if there are some **pattern and characteristics that will help us to individuate motivations to donate more than one time**. In this context the Kernel is not a full descriptive analysis but tries to reply at some basic questions maybe usefull to choose the recommendation algorithm.  

The basic main questions are:  

- Have Teachers/Non teacher differents behavior in  donations?  

- Is there any geographical correspondence between donors donations and schools?  

- Is there any direct or personal relationship between Donors and Schools? 

- When the donors decide to Donate?   

- Is there any pattern in the donations sequence of the same donor? 

# 2 Data Clean and Merge

First of all we remove outliers and donors with many donations (perhaps "match" donations). We consider donors with up to 10 donations because it seems reasonable that those who donated once will donate two or three times at most. 

We split the datasets in two:  

- Donation/Donors with 1 donation 

- Donation/Donors with 2 donation up to 10 donations 

NB: Due to memory limitation on Kaggle we consider only data from 2016-01-01.

```{r DatasetLoad ,message=FALSE,warning=FALSE, results="hide"}
Donations <- read_csv("../input/Donations.csv")
Donors <- read_csv("../input/Donors.csv")
Schools <- read_csv("../input/Schools.csv")
Projects <- read_csv("../input/Projects.csv")

Donations$`Donation Received Date` <- as.Date(as.character(Donations$`Donation Received Date`),"%Y-%m-%d")
Donations <- Donations[Donations$`Donation Received Date`> as.POSIXct("2016-01-01"),]
Projects <- Projects[Projects$`Project Posted Date`> as.POSIXct("2016-01-01"),]
Projects$`Project Essay` <- NULL

```


```{r DatasetSplit ,message=FALSE,warning=FALSE, results="hide"}
#---------------------------------------------------------------------
# Split donors and donations in two datasets
# 1 donation (Testing)
# 2-10 donations (Training and Validation)
#---------------------------------------------------------------------

# Remove Donors Outliers (max 10 donations)
donorsDonNum <- Donations %>% group_by(`Donor ID`) %>% summarise(Count = n())
nrow(donorsDonNum[donorsDonNum$Count > 1,]) # 552941
nrow(donorsDonNum[donorsDonNum$Count == 1,]) # 1471613
donorsDonNum <- donorsDonNum[donorsDonNum$Count < 11,]

# donors who donated 1 time
donorsDon1 <- donorsDonNum[donorsDonNum$Count == 1,] 
donorsDon1List <- unique(donorsDon1$`Donor ID`)
# donors who donated more than 1 time
donorsDon10 <- donorsDonNum[donorsDonNum$Count > 1,] 
donorsDon10List <- unique(donorsDon10$`Donor ID`)

rm(donorsDonNum)
rm(donorsDon10)
rm(donorsDon1)

# Split Donations and Donors in two datasets: 1 donation / 2-10 donations
Donations10 <- Donations[Donations$`Donor ID` %in% donorsDon10List, ]
Donations1 <- Donations[Donations$`Donor ID` %in% donorsDon1List, ]
rm(Donations)

# Split donors in two tables
# Donors with 1 donation
Donors10 <- Donors[Donors$`Donor ID` %in% donorsDon10List, ]
# Donord with 2-10 donations 
Donors1 <- Donors[Donors$`Donor ID` %in% donorsDon1List, ]

rm(Donors)
rm(donorsDon1List)
rm(donorsDon10List)
```

To execute our analysis we have to join Donations-Donators-Projects-Schools datasets in a unique dataset.

```{r DatasetJoin ,message=FALSE,warning=FALSE, results="hide"}

#---------------------------------------------------------
# Join Donations-Donators-Projects-Schools
# 2-10 donations
#---------------------------------------------------------
DonDon10 <- left_join(Donations10, Donors10, by = c("Donor ID"))
DonDonPrj10 <- left_join(DonDon10, Projects, by = c("Project ID"))
DonDonPrjSch10 <- left_join(DonDonPrj10, Schools, c("School ID"))
rm(DonDon10)
rm(DonDonPrj10)

#---------------------------------------------------------
# Join Donations-Donators-Projects-Schools
# 1 donations
#---------------------------------------------------------
DonDon1 <- left_join(Donations1, Donors1, by = c("Donor ID"))
DonDonPrj1 <- left_join(DonDon1, Projects, by = c("Project ID"))
DonDonPrjSch1 <- left_join(DonDonPrj1, Schools, c("School ID"))
rm(DonDon1)
rm(DonDonPrj1)

rm(Donations10)
rm(Donations1)

DonDonPrjSch10$`Donation Included Optional Donation` <- NULL
DonDonPrjSch10$`Donor Cart Sequence`<- NULL
DonDonPrjSch10$`Donor City`<- NULL
DonDonPrjSch10$`Donor State`<- NULL
DonDonPrjSch10$`Teacher Project Posted Sequence`<- NULL
DonDonPrjSch10$`Project Type`<- NULL
DonDonPrjSch10$`Project Title`<- NULL
DonDonPrjSch10$`Project Posted Date`<- NULL
DonDonPrjSch10$`Project Current Status`<- NULL
DonDonPrjSch10$`School Name`<- NULL
DonDonPrjSch10$`School Metro Type`<- NULL
DonDonPrjSch10$`School Percentage Free Lunch`<- NULL
DonDonPrjSch10$`School State`<- NULL
DonDonPrjSch10$`School City`<- NULL
DonDonPrjSch10$`School State`<- NULL
DonDonPrjSch10$`School County`<- NULL
DonDonPrjSch10$`School District`<- NULL

DonDonPrjSch1$`Donation Included Optional Donation` <- NULL
DonDonPrjSch1$`Donor Cart Sequence`<- NULL
DonDonPrjSch1$`Donor City`<- NULL
DonDonPrjSch1$`Donor State`<- NULL
DonDonPrjSch1$`Teacher Project Posted Sequence`<- NULL
DonDonPrjSch1$`Project Type`<- NULL
DonDonPrjSch1$`Project Title`<- NULL
DonDonPrjSch1$`Project Posted Date`<- NULL
DonDonPrjSch1$`Project Current Status`<- NULL
DonDonPrjSch1$`School Name`<- NULL
DonDonPrjSch1$`School Metro Type`<- NULL
DonDonPrjSch1$`School Percentage Free Lunch`<- NULL
DonDonPrjSch1$`School State`<- NULL
DonDonPrjSch1$`School City`<- NULL
DonDonPrjSch1$`School State`<- NULL
DonDonPrjSch1$`School County`<- NULL
DonDonPrjSch1$`School District`<- NULL
```

# 3 Comparative Analysis

## 3.1 Teachers/Non Teachers

The first question is quite sinmple; considering the two dataset (one donation, 2-10 Donations)  

**"How many Teachers/Non Teachers there are ?"**  


**Table 1: Teacher/Non Teacher Donation Percentage**
```{r teachNonTeach ,message=FALSE,warning=FALSE}

Don10Rows <- nrow(DonDonPrjSch10)
Don1Rows <- nrow(DonDonPrjSch1)

Don10Teach <- nrow(DonDonPrjSch10[DonDonPrjSch10$`Donor Is Teacher` == "Yes",]) 
Don10PTeach <- round(Don10Teach * 100 / Don10Rows,2) 
Don10NoTeach <- nrow(DonDonPrjSch10[DonDonPrjSch10$`Donor Is Teacher` == "No",])  
Don10PNoTeach <- round(Don10NoTeach * 100 / Don10Rows,2)

Don1Teach <- nrow(DonDonPrjSch1[DonDonPrjSch1$`Donor Is Teacher` == "Yes",]) 
Don1PTeach <- round(Don1Teach * 100 / Don1Rows,2) 
Don1NoTeach <- nrow(DonDonPrjSch1[DonDonPrjSch1$`Donor Is Teacher` == "No",]) 
Don1PNoTeach <- round(Don1NoTeach * 100 / Don1Rows,2) 

DonationsTeachers <- data.frame(matrix(ncol = 2, nrow = 2))
DonationsTeachers[1,1] <- Don1PTeach
DonationsTeachers[2,1] <- Don1PNoTeach
DonationsTeachers[1,2] <- Don10PTeach
DonationsTeachers[2,2] <- Don10PNoTeach
colnames(DonationsTeachers) <- c("1Donation", "2-10Donation")
rownames(DonationsTeachers) <- c("Teachers", "NoTeachers")

datatable(DonationsTeachers)
```

The tables shows that the percentage of donors that are teacher increase a lot from single donation to 2-10 donations. This is an important information for a recommender algorithm.  

## 3.2 Geographical corrispondence

With the second question we investigate if there is any geographical corrispondence between donors and schools; so the question is:   

**"Are Donors and Schools in the same ZIP area?"**  


**Table 2: Zip Donations Percentage (both 2 and 3 zip numbers)**

```{r GeoComparative ,message=FALSE,warning=FALSE}
# ZIP corrispondence (first 3 numbers)
DonDonPrjSch10$`School Zip3` <- substr(DonDonPrjSch10$`School Zip`, 1, 3)
DonDonPrjSch10$Diff <- as.integer(DonDonPrjSch10$`School Zip3`) - as.integer(DonDonPrjSch10$`Donor Zip`)
Don10Zip3 <- nrow(DonDonPrjSch10[DonDonPrjSch10$Diff == 0,])
Don10PZip3 <- round(Don10Zip3 * 100 / Don10Rows,2) 

DonDonPrjSch1$`School Zip3` <- substr(DonDonPrjSch1$`School Zip`, 1, 3)
DonDonPrjSch1$Diff <- as.integer(DonDonPrjSch1$`School Zip3`) - as.integer(DonDonPrjSch1$`Donor Zip`)
Don1Zip3 <- nrow(DonDonPrjSch1[DonDonPrjSch1$Diff == 0,])
Don1PZip3 <- round(Don1Zip3 * 100 / Don1Rows,2) 

# ZIP corrispondence (first 2 numbers)
DonDonPrjSch10$`School Zip2` <- substr(DonDonPrjSch10$`School Zip`, 1, 2)
DonDonPrjSch10$`Donor Zip2` <- substr(DonDonPrjSch10$`Donor Zip`, 1, 2)
DonDonPrjSch10$Diff2 <- as.integer(DonDonPrjSch10$`School Zip2`) - as.integer(DonDonPrjSch10$`Donor Zip2`)
Don10Zip2 <- nrow(DonDonPrjSch10[DonDonPrjSch10$Diff2 == 0,]) 
Don10PZip2 <- round(Don10Zip2 * 100 / Don10Rows,2) 

DonDonPrjSch1$`School Zip2` <- substr(DonDonPrjSch1$`School Zip`, 1, 2)
DonDonPrjSch1$`Donor Zip2` <- substr(DonDonPrjSch1$`Donor Zip`, 1, 2)
DonDonPrjSch1$Diff2 <- as.integer(DonDonPrjSch1$`School Zip2`) - as.integer(DonDonPrjSch1$`Donor Zip2`)
Don1Zip2 <- nrow(DonDonPrjSch1[DonDonPrjSch1$Diff2 == 0,]) 
Don1PZip2 <- round(Don1Zip2 * 100 / Don1Rows,2)

#  Donors and Schools in the same ZIP area? (with Teacher/non Teacher distinction)

Don10Zip3Teach <- nrow(DonDonPrjSch10[DonDonPrjSch10$Diff == 0 & DonDonPrjSch10$`Donor Is Teacher`== "Yes",]) # 
Don10PZip3Teach <- round(Don10Zip3Teach * 100 / Don10Rows,2) 
Don10Zip3NoTeach <- nrow(DonDonPrjSch10[DonDonPrjSch10$Diff == 0 & DonDonPrjSch10$`Donor Is Teacher`== "No",])
Don10PZip3NoTeach <- round(Don10Zip3NoTeach * 100 / Don10Rows,2)
Don10Zip2Teach <- nrow(DonDonPrjSch10[DonDonPrjSch10$Diff2 == 0 & DonDonPrjSch10$`Donor Is Teacher`== "Yes",]) 
Don10PZip2Teach <- round(Don10Zip2Teach * 100 / Don10Rows,2)
Don10Zip2NoTeach <- nrow(DonDonPrjSch10[DonDonPrjSch10$Diff2 == 0 & DonDonPrjSch10$`Donor Is Teacher`== "No",])
Don10PZip2NoTeach <- round(Don10Zip2NoTeach * 100 / Don10Rows,2)

Don1Zip3Teach <- nrow(DonDonPrjSch1[DonDonPrjSch1$Diff == 0 & DonDonPrjSch1$`Donor Is Teacher`== "Yes",]) 
Don1PZip3Teach <- round(Don1Zip3Teach * 100 / Don1Rows,2)
Don1Zip3NoTeach <-nrow(DonDonPrjSch1[DonDonPrjSch1$Diff == 0 & DonDonPrjSch1$`Donor Is Teacher`== "No",]) 
Don1PZip3NoTeach <- round(Don1Zip3NoTeach * 100 / Don1Rows,2)
Don1Zip2Teach <-nrow(DonDonPrjSch1[DonDonPrjSch1$Diff2 == 0 & DonDonPrjSch1$`Donor Is Teacher`== "Yes",]) 
Don1PZip2Teach <- round(Don1Zip2Teach * 100 / Don1Rows,2) 
Don1Zip2NoTeach <-nrow(DonDonPrjSch1[DonDonPrjSch1$Diff2 == 0 & DonDonPrjSch1$`Donor Is Teacher`== "No",])
Don1PZip2NoTeach <- round(Don1Zip2NoTeach * 100 / Don1Rows,2)


DonationsSchool <- data.frame(matrix(ncol = 2, nrow = 6))
DonationsSchool[1,1] <- Don1PZip2
DonationsSchool[2,1] <- Don1PZip3
DonationsSchool[3,1] <- Don1PZip2Teach
DonationsSchool[4,1] <- Don1PZip2NoTeach
DonationsSchool[5,1] <- Don1PZip3Teach
DonationsSchool[6,1] <- Don1PZip3NoTeach

DonationsSchool[1,2] <- Don10PZip2
DonationsSchool[2,2] <- Don10PZip3
DonationsSchool[3,2] <- Don10PZip2Teach
DonationsSchool[4,2] <- Don10PZip2NoTeach
DonationsSchool[5,2] <- Don10PZip3Teach
DonationsSchool[6,2] <- Don10PZip3NoTeach

colnames(DonationsSchool) <- c("1 Donation", "2-10 Donation")
rownames(DonationsSchool) <- c("Zip2", "Zip3",
                               "Zip2 Teachers", "Zip2 NoTeachers",
                               "Zip3 Teachers", "Zip3 NoTeachers")

datatable(DonationsSchool)
```

We notice that more or less half of donors donate in the same residence area, and obviously we find again that the teachers donations increase in 2-10 Donations. 

## 3.3 Donors Schools corrispondence

The third important evidence says that that half people donate in the same residence area maybe because their chidren or people that they knows attending local schools. In this context it is intresting investigate if:   

**Is there any direct or personal relationship between Donors and Schools?**  

**Table 3: Donations number that occur n times grouped by Donors and schools.**
```{r DonorsSchool ,message=FALSE,warning=FALSE}
DonorsSchool10 <- DonDonPrjSch10 %>% group_by(`Donor ID`,`School ID`) %>% summarise(Count = n())
DonorsSchool10Table <- table(DonorsSchool10$Count) # Donor-School occurences
DonorsSchool10Table # same donors-school donations frequency
```

The corrispondence decrease if the "n" increases so maybe if a donors donate many times he/she donates not to the same school; donors that donates 2/3 times maybe tend to donate in the same school. This second group is interesting for us because if we want to convince single donators to donate again maybe they will donate 1 o 2 times.

## 3.4 Project expiration - Donation Received comparison

In this section we investigate if there is any pattern in donation date; is close to the deadline or long before? in other words:  

**When the donors decide to Donate?** .

```{r DateCompare10 ,message=FALSE,warning=FALSE, results="hide"}
DonDonPrjSch10$DateDiff <- DonDonPrjSch10$`Project Fully Funded Date`- DonDonPrjSch10$`Donation Received Date`

# most donations in the last day! 
ggplot(data = DonDonPrjSch10, aes(x = DateDiff)) +
  geom_histogram(fill = "#00BFC4", color = "black", binwidth = 1) +
  labs(x = "Date to finish",
       y = "Donation Num.",
       title = "Donation to finish Distribution (2/10 Donations)")+
  coord_cartesian(xlim = c(0, 50))

```

```{r DateCompare1 ,message=FALSE,warning=FALSE, results="hide"}
DonDonPrjSch1$DateDiff <- DonDonPrjSch1$`Project Fully Funded Date`- DonDonPrjSch1$`Donation Received Date`

# most donations in the last day! 
ggplot(data = DonDonPrjSch1, aes(x = DateDiff)) +
  geom_histogram(fill = "#00BFC4", color = "black", binwidth = 1) +
  labs(x = "Date to finish",
       y = "Donation Num.",
       title = "Donation to finish Distribution (1 Donation)")+
  coord_cartesian(xlim = c(0, 50))

```

The two plots shows that both "occasional donors" or "serial donors" donate close to the deadline.
This is an important information for the recommender algorithm because identify **when to recommend**.

## 3.5 Pattern in first-last donations

The last question is quite complex and try to identify some patterns in donations sequence; so the question is:  

**"Is there any pattern in the donations sequence of the same donor?"**  

In this analysis we will keep only first and last donation because we want to identify some "transition patterns".  


```{r PattDonDF ,message=FALSE,warning=FALSE, results="hide"}
# Aggregate first and last donation date for each donors
DonorsMin <- aggregate(`Donation Received Date` ~ `Donor ID`, data = DonDonPrjSch10, min )
DonorsMax <- aggregate(`Donation Received Date` ~ `Donor ID`, data = DonDonPrjSch10, max )
names(DonorsMin) <- c("Donor ID", "First Donation")
names(DonorsMax) <- c("Donor ID", "Last Donation")
DonorsMinMax <- left_join(DonorsMin, DonorsMax, by = c("Donor ID"))
rm(DonorsMin)
rm(DonorsMax)

# Subset Donations: keep only first and last donations
DonDonPrjSch10 <- left_join(DonDonPrjSch10, DonorsMinMax, by = c("Donor ID"))
rm(DonorsMinMax)
# keep only first and last donations
DonDonPrjSch10FL <- DonDonPrjSch10[DonDonPrjSch10$`Donation Received Date`== DonDonPrjSch10$`First Donation` |
                             DonDonPrjSch10$`Donation Received Date`== DonDonPrjSch10$`Last Donation`, ]
DonDonPrjSch10FL$DonSeq <- ifelse(DonDonPrjSch10FL$`Donation Received Date`== DonDonPrjSch10FL$`First Donation`, "First", "Last")

# reshape dataframe to have last and first project of a single donors row
# Doors id + first project + Last Project
Pairvar <- c("Donor ID", "DonSeq", "Project ID")
DonDonPrjSch10FL <- as.data.frame(DonDonPrjSch10FL[,Pairvar])
DonDonPrjSch10FL <- reshape(DonDonPrjSch10FL,timevar="DonSeq",idvar="Donor ID",direction="wide")
DonDonPrjSch10FL <- DonDonPrjSch10FL[!is.na(DonDonPrjSch10FL$`Project ID.Last`),]

# Add projects informations to the first-last dataframe
Pairvar <- c("Project ID", "School ID", "Teacher ID", 
             "Project Subject Category Tree", "Project Subject Subcategory Tree",
             "Project Grade Level Category", "Project Resource Category",
             "Project Cost" )
ProjectCat <- Projects[,Pairvar]

# lookup first project info
DonDonPrjSch10FL <- left_join(DonDonPrjSch10FL, ProjectCat, by = c("Project ID.First" = "Project ID"))
names(DonDonPrjSch10FL) <- c("Donor ID" , "Project ID.Last", "Project ID.First", 
                     "School ID.First", "Teacher ID.First",
                     "Project Subject Category Tree.First",
                     "Project Subject Subcategory Tree.First",
                     "Project Grade Level Category.First", 
                     "Project Resource Category.First",
                     "Project Cost.First" )

DonDonPrjSch10FL <- left_join(DonDonPrjSch10FL, ProjectCat, by = c("Project ID.Last" = "Project ID"))
names(DonDonPrjSch10FL) <- c("Donor ID" , "Project ID.Last", "Project ID.First", 
                     "School ID.First", "Teacher ID.First",
                     "Project Subject Category Tree.First",
                     "Project Subject Subcategory Tree.First",
                     "Project Grade Level Category.First", 
                     "Project Resource Category.First",
                     "Project Cost.First",
                     "School ID.Last", "Teacher ID.Last",
                     "Project Subject Category Tree.Last",
                     "Project Subject Subcategory Tree.Last",
                     "Project Grade Level Category.Last", 
                     "Project Resource Category.Last",
                     "Project Cost.Last")

rm(ProjectCat)
```

**Table 4: Pattern Donation Percentage (first > last)**
```{r PattDonAnalysis ,message=FALSE,warning=FALSE}
Don10Rows <- nrow(DonDonPrjSch10FL)
# 2 donations to same category
DonCat <- nrow(DonDonPrjSch10FL[DonDonPrjSch10FL$`Project Subject Category Tree.First` == DonDonPrjSch10FL$`Project Subject Category Tree.Last`,]) # 123841 su 436980
DonPCat <- round(DonCat * 100 / Don10Rows,2)

# 2 donations to same sub category
DonSubCat <- nrow(DonDonPrjSch10FL[DonDonPrjSch10FL$`Project Subject Subcategory Tree.First` == DonDonPrjSch10FL$`Project Subject Subcategory Tree.Last`,]) # 77273 su 436980
DonPSubCat <- round(DonSubCat * 100 / Don10Rows,2)

# 2 donations to same grade level
DonGrade <- nrow(DonDonPrjSch10FL[DonDonPrjSch10FL$`Project Grade Level Category.First` == DonDonPrjSch10FL$`Project Grade Level Category.Last`,]) # 264000 su 436980
DonPGrade <- round(DonGrade * 100 / Don10Rows,2)

# 2 donations to same teacher
DonTeach <- nrow(DonDonPrjSch10FL[DonDonPrjSch10FL$`Teacher ID.First` == DonDonPrjSch10FL$`Teacher ID.Last`,]) # 179993 su 436980
DonPTeach <- round(DonTeach * 100 / Don10Rows,2)

# 2 donations to same resources
DonRes <- nrow(DonDonPrjSch10FL[DonDonPrjSch10FL$`Project Resource Category.First` == DonDonPrjSch10FL$`Project Resource Category.Last`,]) # 161196 su 436980
DonPRes <- round(DonRes * 100 / Don10Rows,2)

DonationsFL <- data.frame(matrix(ncol = 1, nrow = 4))
DonationsFL[1,1] <- DonPCat
DonationsFL[2,1] <- DonPSubCat
DonationsFL[3,1] <- DonPTeach
DonationsFL[4,1] <- DonPRes
colnames(DonationsFL) <- c("First-Last Donations")
rownames(DonationsFL) <- c("same category", "same sub category",
                               "same grade level", "same resources")
datatable(DonationsFL)

```

The results shows that there is no prevalent project characteristic in donate again; maybe only grade level is a stimulus to donate again.

# 4 Results

based on this analysis it seems that the main characteristics of people who donate again are: 

- Most of teachers donate again.

- Donors tend to donate in the same residence area/state.

- Who donates 2 o 3 times tend to donate at the same School.

- Donors tipically donates close to the deadline.

- Seems that there is no prevalent project characteristic in "donate again"" behavior; maybe only grade level is relevant.
