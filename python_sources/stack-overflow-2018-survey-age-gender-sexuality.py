---
title: 'Stack Overflow 2018 survey: impact of age, gender, and sexuality on inclusion, interest in new tools, and ethics'
output:
  html_document:
    number_sections: true
    toc: true
    fig_width: 7
    fig_height: 4.5
    theme: cosmo
    highlight: tango
    code_folding: hide
---

# Introduction

This visual exploration of the Stack Overflow 2018 survey data aims for depth rather than breadth, with a focus on understanding how age, gender, and sexuality influence the following aspects of the data: 

1) Inclusion in the Stack Overflow community
2) Interest in hypothetical tools on Stack Overflow
3) Attitudes towards ethical responsibility

In trying to understand how survey respondents differ in these dimensions, I also consider the impact of potential mediating variables such as years of coding experience, undergraduate major, and salary. 

## Key findings

### Inclusion 

- Users under the age of 18 are the least likely to feel like part of the community.
- Women are greatly underrepresented on Stack Overflow and much more likely to feel excluded than men, and this is true across levels of coding experience, across different sexual orientations, and across different undergraduate majors. 
- Women on Stack Overflow are much more likely to be LGBTQ than men. 
- Women and LGBTQ individuals are more likely to major in a less technical discipline in college, while asexual people are the most likely to major in a technical discipline.
- Gay and bisexual/queer people are less likely to feel like part of the community, and this is true across different undergraduate majors.
- Asexual women are similar to straight women, but asexual men are more likely to feel like part of the community than straight men.

### Support for hypothetical tools

- Users under the age of 18 strongly support a special area for beginners.
- Users between 18-34 years old strongly support adding a career growth Q & A to Stack Overflow.
- Women show stronger support for all of the hypothetical tools as compared to men, and women show the strongest support for a career growth Q & A and employer/job reviews.
- LGBTQ individuals did not differ much from the overall community in their support for hypothetical tools.

### Ethics

- Early-career individuals and elderly individuals are more likely to say "yes" to an unethical job.
- People with lower salaries are more likely to say "yes" to an unethical job.
- People with more technical undergraduate major are more likely to say "yes" to an unethical job.
- Asexual people are more likely to say "yes" to an unethical job, and this is likely mediated by income (asexuals tended to be younger and have lower salaries) and academic background (asexual people are more likely to come from a highly technical discipline).
- Women are less likely to say "yes" to an unethical job.

# Preparations

Load packages and data.

```{r, message=FALSE}
# Load packages
library("tidyverse")
library("gridExtra")
library("knitr")

# Load data
data <- read.csv("../input/survey_results_public.csv")

# Options
opts_chunk$set(warning=FALSE, message=FALSE)
```

Check and modify features of interest, and create some new features that will be useful later on. 

```{r}
# Define membership feeling as factor
data$ConsiderMember <- factor(data$StackOverflowConsiderMember, levels=c("Yes","No"))
data$ConsiderMemberBi <- if_else(data$StackOverflowConsiderMember=="Yes", 1, 0)

# Make binary version of ethical choice (Yes, No/depends)
data$EthicsChoice <- factor(data$EthicsChoice, levels=c("No","Depends on what it is","Yes"),
                                                           labels=c("No","Depends","Yes"))
data$EthicsChoiceBi <- if_else(data$EthicsChoice=="Yes", 1,0) 

# Define age as an ordered factor
data$Age <- factor(data$Age, levels=c("Under 18 years old",
                                      "18 - 24 years old",
                                      "25 - 34 years old",
                                      "35 - 44 years old",
                                      "45 - 54 years old",
                                      "55 - 64 years old",
                                      "65 years or older"),
                   labels=c("<18","18-24","25-34","35-44","45-54","55-64",">65"))

# Define years of coding experience as an ordered factor
data$YearsCoding <- factor(data$YearsCoding, levels=c("0-2 years",
                                                      "3-5 years",
                                                      "6-8 years",
                                                      "9-11 years",
                                                      "12-14 years",
                                                      "15-17 years",
                                                      "18-20 years",
                                                      "21-23 years",
                                                      "24-26 years",
                                                      "27-29 years",
                                                      "30 or more years"),
                           labels=c("0-2","3-4","6-8","9-11","12-14","15-17","18-20","21-23","24-26","27-29","30+"))

# Simplify gender categories (Male, female)
data[!data$Gender%in%c("Male","Female"),"Gender"] <- NA
data$Gender <- factor(data$Gender, levels=c("Male","Female"))

# Simplify sexuality categories (Straight, Gay, Asexual, Bi/queer)
data[!data$SexualOrientation%in%c("Gay or Lesbian","Asexual","Bisexual or Queer","Straight or heterosexual"),"SexualOrientation"] <- NA
data$Sexuality <- factor(data$SexualOrientation, levels=c("Straight or heterosexual",
                                                                  "Gay or Lesbian",
                                                                  "Bisexual or Queer",
                                                                  "Asexual"),
                                 labels = c("Straight","Gay","Bi/queer","Asexual"))

# Fix levels of undergraduate majors
levs <- data %>% 
  filter(!is.na(UndergradMajor),!is.na(ConsiderMemberBi)) %>%
  group_by(UndergradMajor) %>%
  summarize(Prop = sum(ConsiderMemberBi)/n()) %>%
  arrange(Prop) %>% .$UndergradMajor
labs <- rev(c("Comp sci","Health sci","Engineering","Web dev", "Info tech", "Business","Math/stats","Natural sci", "None","Social sci","Humanities", "Arts"))
data$UndergradMajor <- factor(data$UndergradMajor, levels = levs, labels = labs)

# Create new salary variable with outliers removed
SalaryKeep <- data$ConvertedSalary %>% log1p 
salary.mean <- mean(SalaryKeep, na.rm=TRUE)
salary.sd <- sd(SalaryKeep,na.rm=TRUE)
SalaryKeep <- ifelse(is.na(SalaryKeep), NA, (SalaryKeep - salary.mean)/salary.sd)
SalaryKeep[SalaryKeep < -3 | SalaryKeep > 3] <- NA
data$SalaryNew <- ifelse(is.na(SalaryKeep), NA, log(data$ConvertedSalary))

# Create binary variables for hypothetical tools (1 = very/extremely interested)
data$HypotheticalTools1 <- ifelse(data$HypotheticalTools1 %in% c("Very interested","Extremely interested"), 1, 0)
data$HypotheticalTools2 <- ifelse(data$HypotheticalTools2 %in% c("Very interested","Extremely interested"), 1, 0)
data$HypotheticalTools3 <- ifelse(data$HypotheticalTools3 %in% c("Very interested","Extremely interested"), 1, 0)
data$HypotheticalTools4 <- ifelse(data$HypotheticalTools4 %in% c("Very interested","Extremely interested"), 1, 0)
data$HypotheticalTools5 <- ifelse(data$HypotheticalTools5 %in% c("Very interested","Extremely interested"), 1, 0)
```

# Inclusion in the Stack Overflow community and support for hypothetical tools

A simple yes/no question on the survey indicates whether a respondent feels included in the Stack Overflow community, while support for 5 different hypothetical tools was evaluated using a 5-point scale. 

In this section, I try to understand what characteristics are most predictive of a person feeling like part of the community, and what hypothetical tools are most desired by users, particularly those that do not feel like part of the community.

### Age

The following plot shows that the vast majority of Stack Overflow users are between 18 and 44, and nearly half are between 25 and 34. The age distribution is similar for men and women.

```{r}
# Proportion of users in each age category
table(data$Age,data$Gender) %>% 
  prop.table(2) %>% 
  data.frame() %>%
  rename(Age=Var1, Sex=Var2, Proportion=Freq) %>%
  ggplot(aes(x=Age, y=Proportion, fill=Sex)) + 
  geom_col(position="dodge") +
  scale_fill_manual(values=c("navy","darkred")) +
  ggtitle("Proportion of users in each age category") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12))
```

The next plot shows that there is a fairly weak relationship between age and feeling like part of the community.  

```{r}
data %>% filter(!is.na(Age),!is.na(ConsiderMember)) %>%
  group_by(Age) %>%
  summarize(Proportion=sum(ConsiderMember=="Yes",na.rm=TRUE)/sum(!is.na(ConsiderMember))) %>%
  ggplot(aes(x=Age, y=Proportion)) +
  geom_col(fill="darkorchid4") +
  ggtitle("Proportion that feels like part of the community") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12))
```

It is noteworthy that users under the age of 18 are the least likely to consider themselves part of the community (about 65%, compared to 72.5% for users 25-34). Although this may not seem like a very large different, the under 18 age group is especially important since this is an age when people may make life altering decisions about whether to continue with coding or do something else.

In light of this, it might be a good idea to identify which hypothetical tools younger users are especially interested in seeing added to Stack Overflow. Below is a heatmap showing the proportion of users in each age group that were "Very interested" or "Extremely interested" in each of the 5 hypothetical tools included in the survey.

```{r}
# Heatmap of support (interest) level for each hypothetical tool for each age group
data %>% 
  filter(!is.na(Age),!is.na(HypotheticalTools1),!is.na(HypotheticalTools2),!is.na(HypotheticalTools3),!is.na(HypotheticalTools4),!is.na(HypotheticalTools5)) %>%
  select(Age, HypotheticalTools1,HypotheticalTools2,HypotheticalTools3,HypotheticalTools4,HypotheticalTools5) %>%
  gather("Tool","Level",HypotheticalTools1,HypotheticalTools2,HypotheticalTools3,HypotheticalTools4,HypotheticalTools5) %>%
  group_by(Age,Tool) %>% 
  summarize(Proportion=sum(Level)/n()) %>%
  ggplot(aes(Age, Tool)) +
  geom_tile(aes(fill = Proportion), color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ylab("Hypothetical tool") + xlab("Age group") +
  ggtitle("Proportion supporting hypothetical tools") +
  theme(legend.title = element_text(size = 10),
        legend.text = element_text(size = 12),
        plot.title = element_text(size=16),
        axis.title=element_text(size=14,face="bold"),
        axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_discrete(labels=c("Peer mentoring","Newbie area","Blog platform","Employer/job review","Career growth Q&A")) +
  labs(fill = "Proportion supporting") +
  theme(axis.text.x = element_text(angle = 45))
```

In general, younger users seem more interested in all of the hypothetical tools compared to older users. By far, the most popular hypothetical tool among under 18 users is a private area for beginners (supported by 55.6% of under 18 users), followed by a programming-oriented blog platform (supported by 45% of under 18 users). In fact, the support for a private beginner area among users under 18 was the most strongly supported hypothetical tool among any of the different age categories.

The heatmap also shows strong interest in a career growth Q & A among users aged 18-24 and to a slightly lesser extent, 25-34. Since these age ranges represent the majority of Stack Overflow users, adding a career growth Q & A is likely to be one of the most widely appreciated features that could be added to Stack Overflow.

### Gender

Survey respondents were permitted to check multiple boxes for their gender, leading to many possible combinations of responses. I simplified this data to include only three categories: male, female, and other. 

As many have noted, there is a strong gender bias on Stack Overflow, with only 6.3% of users being female. There is also a strong bias towards females feeling less a part of the Stack Overflow community, with 40.7% of women feeling excluded verses 27.7% of men. 

Perhaps the relationship between gender and feeling accepted in the Stack Overflow community is mediated by another variable, such as coding experience. To investigate this possibility, the following plot shows the proportion of men and women who feel that they are part of the community, stratified by years of coding experience. 

```{r}
data %>% filter(!is.na(YearsCoding),!is.na(ConsiderMember),!is.na(Gender)) %>%
  group_by(YearsCoding, Gender) %>%
  summarize(Proportion=sum(ConsiderMemberBi)/n()) %>%
  ggplot(aes(x=YearsCoding, y=Proportion, group=Gender, fill=Gender)) +
  geom_col(position="dodge") +
  scale_fill_manual(values=c("navy","darkred")) +
  ggtitle("Proportion that feels like part of the community") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12)) +
  theme(axis.text.x = element_text(angle = 45)) +
  xlab("Years coding")
```

The evidence from this plot is very clear: experience level has nothing to do with women's feelings of being left out of the Stack Overflow community. Even men who have only been coding for 0-2 years are more likely to feel a part of the Stack Overflow community than women who have been coding for 24-26 years. Interestingly, women seem to feel equally a part of the community as men when they reach a whopping 27+ years of coding. 

So what hypothetical tools on Stack Overflow do women feel the most strongly about, and how do they compare to men?

```{r}
# Heatmap of support (interest) level for each hypothetical tool for each gender
data %>% 
  filter(!is.na(Gender),!is.na(HypotheticalTools1),!is.na(HypotheticalTools2),!is.na(HypotheticalTools3),!is.na(HypotheticalTools4),!is.na(HypotheticalTools5)) %>%
  select(Gender, HypotheticalTools1,HypotheticalTools2,HypotheticalTools3,HypotheticalTools4,HypotheticalTools5) %>%
  gather("Tool","Level",HypotheticalTools1,HypotheticalTools2,HypotheticalTools3,HypotheticalTools4,HypotheticalTools5) %>%
  group_by(Gender,Tool) %>% 
  summarize(Proportion=sum(Level)/n()) %>%
  ggplot(aes(Gender, Tool)) +
  geom_tile(aes(fill = Proportion), color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ylab("Hypothetical tool") + xlab("Gender") +
  ggtitle("Proportion supporting hypothetical tools") +
  theme(legend.title = element_text(size = 10),
        legend.text = element_text(size = 12),
        plot.title = element_text(size=16),
        axis.title=element_text(size=14,face="bold"),
        axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_discrete(labels=c("Peer mentoring","Newbie area","Blog platform","Employer/job review","Career growth Q&A")) +
  labs(fill = "Proportion supporting") +
  theme(axis.text.x = element_text(angle = 45))
```

It seems that women are more interested in all 5 of the hypothetical tools in the Stack Overflow survey, and they are most interested in a career growth Q & A and an employer/job review tool. We can only speculate as to why this is. Perhaps women are very interested in a career growth Q & A because they are having a more difficult time climbing the career ladder as programmers and are seeking support. Women might also be more interested in an employer/job review tool because they would like to know whether an organization provides a female-friendly work place. Men and women are similarly interested in a programming-oriented blogging platform, but there is a drastic difference in female support for a newbie area and peer mentoring on Stack Exchange. Men showed scarcely any support for these newbie-oriented tools, while 30-35% of women supported these tools. Since we know there is not much of a gender difference in the age distribution on Stack Overflow, this could be due to women in general being more enthusiastic about providing a support network for beginners and early-career coders. 

Taken together, these results suggest that on average, men are more satisfied with the current services offered by Stack Overflow, while women are more interested in new features, particularly those that support beginner and early-career coders. Adding these features to Stack Overflow could thus be an important way to help women feel like part of the community.

### Sexual orientation

Like the gender data, I simplified the sexual orientation data to contain 5 cagetories: straight, gay, bi/queer, asexual, and other (this includes anyone who checked multiple boxes). Here is the number of men and women in each category.

```{r}
# Counts of different genders and sexual orientations
table(data$Sexuality, data$Gender) %>% kable
```

Overall, 93.5% of Stack Overflow users identify as straight, 2% identify as gay, 3.3% identify as bisexual or queer, and 1.2% identify as asexual. However, these percentages mask some interesting variation across the genders. Specifically, women within this community are proportionally much more likely to identify as non-straight than men.

```{r}
# Proportion of men and women in each sexuality category
table(data$Sexuality,data$Gender) %>% 
  prop.table(2) %>% 
  data.frame() %>%
  rename(Sexuality=Var1, Sex=Var2, Proportion=Freq) %>%
  ggplot(aes(x=Sexuality, y=Proportion, fill=Sex)) + 
  geom_col(position="dodge") +
  scale_fill_manual(values=c("navy","darkred")) +
  ggtitle("Proportion of users in each sexuality category") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12))
```

Why is this? Computer programming is generally a male-dominated activity with a male-dominated culture, so perhaps women who are less gender conforming are more likely to enjoy (or at least tolerate) such an environment, while less gender conforming men are less likely to enjoy it. Of course, gender conformity and sexual orientation are not the same thing, but there is an association (that's why [gaydar](https://en.wikipedia.org/wiki/Gaydar) exists!). 

Does sexual orientation have any association with feeling like part of the community? It seems that it does.

```{r}
# Proportion of different sexualities who feel like part of the community 
data %>% filter(!is.na(Sexuality),!is.na(ConsiderMember),!is.na(Gender)) %>%
  group_by(Sexuality, Gender) %>%
  summarize(Proportion=sum(ConsiderMemberBi)/n()) %>%
  ggplot(aes(x=Sexuality, y=Proportion, group=Gender, fill=Gender)) +
  geom_col(position="dodge") +
  scale_fill_manual(values=c("navy","darkred")) +
  ggtitle("Proportion that feels like part of the community") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12)) +
  theme(axis.text.x = element_text(angle = 45)) +
  xlab("Sexuality")
```

The chart shows that gay and bisexual/queer people feel less like part of the community than their straight counterparts, and the gender gap persists across all categories of sexuality. Interestingly, asexual individuals do feel like part of the community, and asexual men are even more likely than straight men to feel like part of the community!

My first thought was that these feelings of exclusion might lead gay and bisexual/queer to favor the addition of new tools on Stack Overflow that help individuals connect with others, similar to how women tended to support the new tools more than men. However, this is not the case.

```{r}
# Heatmap of support (interest) level for each hypothetical tool for different sexual orientations
data %>% 
  filter(!is.na(Sexuality),!is.na(HypotheticalTools1),!is.na(HypotheticalTools2),!is.na(HypotheticalTools3),!is.na(HypotheticalTools4),!is.na(HypotheticalTools5)) %>%
  select(Sexuality, HypotheticalTools1,HypotheticalTools2,HypotheticalTools3,HypotheticalTools4,HypotheticalTools5) %>%
  gather("Tool","Level",HypotheticalTools1,HypotheticalTools2,HypotheticalTools3,HypotheticalTools4,HypotheticalTools5) %>%
  group_by(Sexuality,Tool) %>% 
  summarize(Proportion=sum(Level)/n()) %>%
  ggplot(aes(Sexuality, Tool)) +
  geom_tile(aes(fill = Proportion), color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ylab("Hypothetical tool") + xlab("Sexuality") +
  ggtitle("Proportion supporting hypothetical tools") +
  theme(legend.title = element_text(size = 10),
        legend.text = element_text(size = 12),
        plot.title = element_text(size=16),
        axis.title=element_text(size=14,face="bold"),
        axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_discrete(labels=c("Peer mentoring","Newbie area","Blog platform","Employer/job review","Career growth Q&A")) +
  labs(fill = "Proportion supporting") +
  theme(axis.text.x = element_text(angle = 45))
```

If anything, individuals identifying as gay are *less* interested that straight people in the addition of the new tools on Stack Overflow. Straight and bisexual/queer people show a very similar pattern of support for the new tools, while asexual people mainly differ in that they are more interested in a blog platform and area for beginners. These differences are difficult to explain, and it seems unlikely that there is a causal connection here. 

This led me to wonder whether people of different sexual orientations on Stack Overflow differ in some other characteristics that might help explain these patterns. Gender differences are unlikely to explain these patterns since men greatly outnumber women in every category of sexuality. However, age could contribute to the pattern, as the following chart shows that asexuals are skewed towards a younger crowd. 

```{r}
# Proportion of users in each age category, stratified by sexual orientation
table(data$Age,data$Sexuality) %>% 
  prop.table(2) %>% 
  data.frame() %>%
  rename(Age=Var1, Sexuality=Var2, Proportion=Freq) %>%
  ggplot(aes(x=Age, y=Proportion, fill=Sexuality)) + 
  geom_col(position="dodge") +
  scale_fill_manual(values=c("red","yellow","blue","purple")) +
  ggtitle("Proportion of users in each age category") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12))
```

The chart shows that that individuals identifying as bisexual/queer and asexual tend to be younger than individuals identifying as gay or straight, and this is especially true for asexuals. This is likely due to the fact that the labels of "bisexual", "queer", and "asexual" have become much more widely adopted among younger people, whereas older people grew up in a culture where the two simple categories of "gay" and "straight" were largely considered the only options. Regardless of the reasons for the pattern, it could help explain why there is stronger support for a blog platform and a newbie area among asexuals, since both of these tools are favored by younger users. 

### Undergraduate major

I has already been noted on the [Stack Overflow blog](https://stackoverflow.blog/2018/05/30/public-data-release-of-stack-overflows-2018-developer-survey/) and in other kernels that there is a link between people's undergraduate major and whether they feel left out of the Stack Overflow community. In general, people from more technical disciplines (e.g., computer science) tend to feel like part of the community more often than people from less technical disciplines (e.g., humanities). 

These patterns make good sense. Users from more technical disciplines are more likely to feel comfortable and make valuable contributions to a community with a highly techincal orientation. In contrast, users from less technical disciplines may often feel confused or discouraged by their lack of familiarity with terminology or culture on Stack Overflow. For example, it is common to see a new user being chastised by long-term community members for failing to write a sufficiently clear question or providing a minimal example that reproduces their problem.

In light of the importance of academic background and feeling like part of the Stack Overflow community, it is interesting to consider whether undergradute major is associated with any demographic characteristics of the survey respondents, such as gender and sexual orientation. If there is some association, then it is possible that difference academic backgrounds can explain associations between feeling like part of the community and these characteristics.

The following plots show how undergraduate major is related to different demographic characteristics. The bars represent the proportion of individuals in each undergraduate major that are a member of each demographic category. Majors that are associated with feeling like part of the Stack Overflow community are positioned towards the top of the plots.  

```{r}
# Create special dataframe for these plots
data.um <- data %>% filter(!is.na(Gender),!is.na(Sexuality),!is.na(UndergradMajor)) %>%
  mutate(Female = ifelse(Gender=="Female",1,0),
         GayBi = ifelse(Sexuality%in%c("Gay","Bi/queer"),1,0),
         Asexual = ifelse(Sexuality=="Asexual",1,0),
         StraightMale = ifelse(Sexuality=="Straight" & Gender=="Male",1,0)) %>%
  select(UndergradMajor,Female,GayBi,Asexual,StraightMale) %>%
  group_by(UndergradMajor) %>%
  summarize(Female = sum(Female)/n(),
            GayBi = sum(GayBi)/n(),
            Asexual = sum(Asexual)/n(),
            StraightMale = sum(StraightMale)/n())

# Proportion of different categories across undergraduate majors
# Female
um.female <- ggplot(data.um, aes(x=UndergradMajor, y=Female)) + 
  geom_col(fill="darkred") +
  ggtitle("Female") +
  ylab("") + xlab("Undergraduate major") +
  theme(plot.title = element_text(hjust = 0.5, size=12),
        axis.title = element_text(size=10),
        axis.text.y = element_text(size=9),
        axis.text.x = element_text(size=6)) +
  coord_flip() +
  ylim(c(0,0.25))
# GayBi
um.gaybi <- ggplot(data.um, aes(x=UndergradMajor, y=GayBi)) + 
  geom_col(fill="darkorchid4") +
  ggtitle("Gay/bi/queer") +
  ylab("") + xlab("") +
  theme(plot.title = element_text(hjust = 0.5, size=12),
        axis.title = element_text(size=10),
        axis.text.y = element_text(size=9),
        axis.text.x = element_text(size=6)) +
  coord_flip() +
  ylim(c(0,0.15))
# Asexual
um.a <- ggplot(data.um, aes(x=UndergradMajor, y=Asexual)) + 
  geom_col(fill="goldenrod") +
  ggtitle("Asexual") +
  ylab("Proportion") + xlab("Undergraduate major") +
  theme(plot.title = element_text(hjust = 0.5, size=12),
        axis.title = element_text(size=10),
        axis.text.y = element_text(size=9),
        axis.text.x = element_text(size=6)) +
  coord_flip() +
  ylim(c(0,0.03))
# Straight male
um.sm <- ggplot(data.um, aes(x=UndergradMajor, y=StraightMale)) + 
  geom_col(fill="darkgray") +
  ggtitle("Straight male") +
  ylab("Proportion") + xlab("") +
  theme(plot.title = element_text(hjust = 0.5, size=12),
        axis.title = element_text(size=10),
        axis.text.y = element_text(size=9),
        axis.text.x = element_text(size=6)) +
  coord_flip() +
  ylim(c(0,1))

# plot together
grid.arrange(um.female, um.gaybi, um.a, um.sm, nrow=2)
```

This plot shows that women and LGBTQ individuals represent a higher proportion of the less technical fields compared to the more technical fields (e.g., about 5% of computer science majors are female, while about 17% of social science majors are female). In contrast, asexual individuals represent a relatively higher proportion of the more technical majors (I wouldn't take the results for health science very seriously since there are only 169 respondents in this category).

This leads to the question: are the relationships between gender, sexual orientation, and feeling like part of the Stack Overflow community driven by the fact that LGBTQ people and women are more likely to have a background in a less technical academic discipline? 

To explore this possibility, we can look at the proportion of people in the different gender and sexuality categories that feel like part of the Stack Overflow community.

```{r}
# Proportion feeling like part of the community within computer scientists
data %>% filter(!is.na(UndergradMajor),!is.na(ConsiderMember),!is.na(Gender)) %>%
  group_by(UndergradMajor, Gender) %>%
  summarize(Proportion=sum(ConsiderMemberBi)/n()) %>%
  ggplot(aes(x=UndergradMajor, y=Proportion, group=Gender, fill=Gender)) +
  geom_col(position="dodge") +
  scale_fill_manual(values=c("navy","darkred")) +
  ggtitle("Proportion that feels like part of the community") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12)) +
  theme(axis.text.x = element_text(angle = 45)) +
  xlab("Undergraduate Major") +
  coord_flip()
```

Clearly, women are less likely to feel like part of the community regardless of their undergraduate major, and the discrepancy between men and women increases in the less technical fields. 

For the plots looking at sexuality, I limited the data to only include majors with at least 1000 representatives on Stack Overflow, because LGBTQ individuals are such a minority that they appear too infrequently among the less common majors to provide a reasonable sample size. 

```{r}
data %>% filter(!is.na(UndergradMajor),!is.na(ConsiderMember),!is.na(Sexuality),
                UndergradMajor %in% c("Humanities","Natural sci","Math/stats","Business","Info tech","Web dev","Engineering","Comp sci")) %>%
  group_by(UndergradMajor, Sexuality) %>%
  summarize(Proportion=sum(ConsiderMemberBi)/n()) %>%
  ggplot(aes(x=UndergradMajor, y=Proportion, group=Sexuality, fill=Sexuality)) +
  geom_col(position="dodge") +
  scale_fill_manual(values=c("red","yellow","blue","purple")) +
  ggtitle("Proportion that feels like part of the community") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12)) +
  xlab("Undergraduate Major") +
  coord_flip()
```

This plot shows that straight people are more likely feel like part of the community than gay/bisexual/queer people across every undergraduate major. Asexual people are also doing well, and are even more likely to say they are part of the community than straight people for some majors (computer science, engineering, information technology, and business). 

Taken together, these results show that the tendancy for women and LGBTQ individuals to feel left out of Stack Overflow cannot be entirely explained by the tendancy for women and LGBTQ people to come from a less technical background. 

# Attitude towards ethical responsibility

The Stack Overflow survey included several questions about ethics. In this section, I will focus on respondents' answer to the question: "Imagine that you were asked to write code for a purpose or product that you consider extremely unethical. Do you write the code anyway?" Possible answers included "yes", "no", and "it depends". 

Overall, people on Stack Overflow tended to answer this question in the "ethical" way. That is to say, most respondents said they would NOT write code for an unethical project. Here is a breakdown of answers to the first question, where "no" is the ethical answer.

```{r}
table(data$EthicsChoice) %>% kable
```

I am mainly interested in knowing what kind of people choose the unethical answer, "yes". Now let's explore how answers vary across age, gender, sexuality, and undergraduate major.

### Age

A finding that I found quite surprising and interesting is that there is a relationship between age and people's answers to the ethical questions.  

```{r}
data %>% filter(!is.na(Age),!is.na(EthicsChoiceBi)) %>%
  group_by(Age) %>%
  summarize(Proportion=sum(EthicsChoiceBi)/n()*100) %>%
  ggplot(aes(x=Age, y=Proportion)) +
  geom_col(fill="black") +
  ggtitle("Percentage who would do an unethical project") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12)) +
  xlab("Age group") + ylab("Percentage")
```

What can explain this???

I'm not so sure. It is possible that people between the ages of 35 and 65 are just "more ethical" for some reason. However, and alternative explanation could stem from the possibility that very young people at the beginning of their careers and very old people who are past their prime may have less career stability and financial security compared to people in the prime of their careers. This may explain why the very young and the very old are more likely to be willing to do a job that serves an unethical end. In other words, people in the prime of their career may simply have the *luxury* of saying "no" to a job that displeases them.

Providing some support for this idea, there is in fact a relationship between salary and answers to the ethical questions, with people who have a lower salary being more likely to be willing to do a job that serves an unethical end, and more likely to answer "no" or that they are "unsure" of whether a programmer is obligated to consider the ethical implications of their work. Note that I've cleaned up the salary data by removing outliers, which I defined as all salaries falling outside 3 standard deviations of the log-transformed salary data.

```{r}
# Boxplot
data %>% filter(!is.na(SalaryNew),!is.na(EthicsChoice)) %>%
  ggplot(aes(EthicsChoice, SalaryNew)) +
    geom_boxplot(fill="steelblue") +
  xlab("Would you code a bad thing?") +
  ylab("Log salary")
```

These differences may seem small on a log scale, but the median salary of people who answered "yes" to the question of whether they would write code for an unethical project is \$34,611, while the median salary of people who answered "no" to this question is \$62,880, making it much easier for them to say no. So perhaps people who gave ethical answers shouldn't be smug about it. Imagine you are making \$34K while trying to support a family of 4, and you are asked to do an unethical project or risk losing your job. On a salary like that, it is difficult to save up money, so unemployment would pose an enormous burden on the family. Doesn't one also have an ethical obligation to care for one's family? 

I'm not saying this explanation can excuse everyone. There are probably some genuinely crappy people in the sample. But we shouldn't be too quick to judge. The ethical choice is not always black and white. 

### Gender

Gender seems to be important as well, with women being slighly more likely to say "no" and less likely to say "yes" to an unethical project.

```{r}
table(data$EthicsChoice,data$Gender) %>% 
  prop.table(2) %>% 
  data.frame() %>%
  rename(EthicsChoice=Var1, Gender=Var2, Proportion=Freq) %>%
  ggplot(aes(x=EthicsChoice, y=Proportion, fill=Gender)) + 
  geom_col(position="dodge") +
  scale_fill_manual(values=c("navy","darkred")) +
  ggtitle("Gender differences in answer to ethical question") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12)) +
  xlab("Would you do an unethical project?")
```

The difference is not very large: 3.4% of men said "yes" compared to 4.6% of women. However, this is especially surprising considering that women tend to make less money than men (this has been shown decisively in other kernels), so based on the reasoning I used to explain the pattern for age, one would expect women to be more willing to do an unethical job.

Why would this be the case? Perhaps women just tend to be more ethically conscientous than men. Perhaps men are more likely to be the breadwinners of their family and are therefore more likely to consider alternative ethical considerations, such as the imperative to make money to care for a family. Perhaps there are other mediating variables at play, such as academic background (I will consider the relationship between undergraduate major and ethical choice later).

### Sexual orientation

As with gender, the results for sexual orientation are a bit puzzling. 

```{r}
table(data$EthicsChoice,data$Sexuality) %>% 
  prop.table(2) %>% 
  data.frame() %>%
  rename(EthicsChoice=Var1, Sexality=Var2, Proportion=Freq) %>%
  ggplot(aes(x=EthicsChoice, y=Proportion, fill=Sexality)) + 
  geom_col(position="dodge") +
  scale_fill_manual(values=c("red","yellow","purple","blue")) +
  ggtitle("Sexual orientation and answers to ethical question") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12),
        legend.title = element_text(size=15),
        legend.text = element_text(size=12)) +
  xlab("Would you do an unethical project?")
```

Here, the asexual people stand out from the crowd. Straights, gays, and bisexual/queers answered the question similarly, but asexual people seem much more willing to do an unethical job. I don't think this is just a small sample size issue: there are 717 asexual people in the dataset, which is not that small. 

What on earth is going on here? I suspect there are some confounding factors at play. We saw earlier that asexual individuals are more likely to be young and more likely to have majored in a technical discipline in college. The association with age, and thus career stage and salary, could explain part of this relationship. Supporting that idea, it appears that asexual individuals tend to have a lower salary on average than the other groups.

```{r}
# Boxplot
data %>% filter(!is.na(SalaryNew),!is.na(Sexuality)) %>%
  ggplot(aes(Sexuality, SalaryNew)) +
    geom_boxplot(fill="steelblue") +
  xlab("Sexual orientation") +
  ylab("Log salary")
```

This is a pretty large difference: the median salary for the asexual individuals is \$26,670, compared to \$57,324 for straight people.

What about the influence of undergraduate major? I will consider that next.

### Undergraduate major

```{r}
data %>% filter(!is.na(UndergradMajor),!is.na(EthicsChoiceBi)) %>%
  group_by(UndergradMajor) %>%
  summarize(Proportion=sum(EthicsChoiceBi)/n()*100) %>%
  ggplot(aes(x=UndergradMajor, y=Proportion)) +
  geom_col(fill="black") +
  ggtitle("Percentage who would do an unethical project") +
  theme(plot.title = element_text(hjust = 0.5, size=20),
        axis.title = element_text(size=15),
        axis.text = element_text(size=12)) +
  xlab("Undergraduate major") + ylab("Percentage") +
  coord_flip()
```

Wow, there is a pretty strong relationship here. Web development seem to be the "least ethical" discipline, with web development majors being nearly 4 times as likely to say "yes" to an unethical job as compared to individuals with a background in the social sciences, humanities, or arts. These results raise the possibility that including more people with a background in the humanities could improve the ethical culture of the programming community. In light of this, it is especially troubling that people with these backgrounds feel the least included in the Stack Overflow community. 

Returning to the question of why asexual people are more willing to do an unethical job, it seems that academic background could contribute to that relationship. Not including health science (which has a very small sample size... 169 individuals and just 4 that identify as asexual), the undergraduate majors with the highest proportion of asexual individuals are computer science, web development, and information technology. These also happen to be the three disciplines which are most strongly associated with saying "yes" to an unethical job. 

# Summary and conclusion

Who feels left out of the community?

- Users <18 years old
- Women
- LGBTQ people

What tools are people most interested in?

- Young users want a special area for beginners.
- Users aged 18-34 years want a career growth Q & A.
- Women are more interested than men in all of the tools, but especially a career growth Q&A and employer/job reviews.

Who is less willing to do an unethical job?

- People with higher salaries
- People who studied arts/humanities in college
- Women

While none of the patterns identified above can prove causality, combining these patterns with a bit of common sense can be very suggestive. If I had more time, I would be interested in actually modeling some of these relationships to quantify the effects of different variables and try to disentangle the complex relationships among them.  