---
title: 'StackOverflow 2018 Developer Survey (Updating)'
output:
  html_document:
    df_print: paged
    toc: yes
    number_sections: true
    code_folding: hide
    fig_width: 10
    fig_asp: 1.2
    highlights: tango
    theme: spacelab
  html_notebook:
    code_folding: hide
    df_print: paged
    fig_width: 10
    fig_asp: 1.2
    highlight: tango
    number_sections: yes
    theme: spacelab
    toc: yes
---
# Stack Overflow 2018 Developer Survey: A Newbie's EDA

Dear Kagglers,

Greetings from a new player on Kaggle! This is my first kernel published on Kaggle. 

Hope my work can interest you. You are very welcome to give comments. 

Thanks and happy kaggling!

## Data Description

Each year, we at Stack Overflow ask the developer community about everything from their favorite technologies to their job preferences. This year marks the eighth year we’ve published our Annual Developer Survey results—with the largest number of respondents yet. Over 100,000 developers took the 30-minute survey in January 2018.

This year, we covered a few new topics ranging from artificial intelligence to ethics in coding. We also found that underrepresented groups in tech responded to our survey at even lower rates than we would expect from their participation in the workforce. Want to dive into the results yourself and see what you can learn about salaries or machine learning or diversity in tech? We look forward to seeing what you find!

More details can be found [here](https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey)

## Load Needed Packages

```{r load-packages, message=FALSE, warning=FALSE}

knitr::opts_chunk$set(echo = TRUE, message = FALSE, warning = FALSE)
library(data.table)
# library(ggvis)
library(tidyverse)
library(knitr)
library(highcharter)
library(plotly)
library(viridisLite)
library(kableExtra)
library(ggthemes)
library(scales)
colors <- c('#b35806','#f1a340','#fee0b6','#d8daeb','#998ec3','#542788')

blank_theme <- theme_minimal() +
  theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.border = element_blank(),
  panel.grid = element_blank(),
  axis.ticks = element_blank(),
  plot.title = element_text(size=14, face="bold")
  )

```

## Function for Semicolon Separated String
```{r}

split_fn <- function(x) {
  fn <- function(y) str_split(y, pattern = ";")[[1]][[1]]
  plyr::ldply(x, fn)[[1]]
}

```

## Load Data {.tabset .tabset-fade .tabset-pills}

```{r load-data}
survey <- fread("../input/survey_results_public.csv") %>% 
  mutate(Salary = parse_number(Salary),
         DevType2 = split_fn(DevType))
```


### Take a Peek at the Variables in the Data

The items incorporated in the survey are:

```{r data-vars}

variables <- names(survey)
kable(variables %>% matrix(ncol = 4)) %>% 
  kable_styling()

```


# Distribution of Participants by Countries {.tabset .tabset-fade .tabset-pills}

```{r country-distribution, include=FALSE}

country <- survey %>% 
  filter(!is.na(Country)) %>% 
  group_by(Country) %>% 
  mutate(n = n()) %>% 
  ungroup() %>% 
  select(Country, n) %>% 
  filter(n >= 500) %>% 
  drop_na() %>% 
  distinct() 

col_pal <- viridis_pal(alpha = 1, begin = 0, end = 1, 
                       direction = 1,  option = "D")  

  tm <- country %>% 
  treemap::treemap(index = "Country", vSize = "n", 
          vColor = "n", type = "value", 
          palette = viridis(38, begin = 0.85, end = 0, 
                            alpha = 1, option = "D", direction = -1),
        title = "Distribution of Respondents by Countries") 
  
```
  
```{r fig.width=12}
  hctreemap(tm) %>% 
  hc_title(text = 'Distribution of Respondents by Countries')
```

# Working Envorinment {.tabset .tabset-fade .tabset-pills}

> Pie chart produced with help from [here](https://bit.ly/2xAB63X) 

## Operating System

Basically, about 50% of the developers use Windows, and the users using Linux an MaxOS are almost the same. The group using BSD/Unix is relatively small.

```{r os}

os <- survey %>% 
  select(Country, DevType, Age, OperatingSystem, Gender) %>% 
  mutate(DevType2 = split_fn(DevType),
         gener2 = split_fn(Gender))  %>% 
  group_by(OperatingSystem) %>% 
    mutate(os_n = n()) %>% 
  ungroup() %>% 
  filter(!is.na(OperatingSystem))

os1 <- os %>% 
  select(OperatingSystem, os_n) %>% 
  distinct() %>% 
  mutate(value = os_n / dim(os)[1] * 100) 

os1 %>% 
  ggplot(aes(x = "", y = value, fill = factor(OperatingSystem))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar(theta = "y", start = pi/3) +
  scale_fill_manual(values = colors[c(2,1,5,6)])  +
  geom_text(aes(y = c(90, 25, 65, 0), 
                label = percent(value/100)), size = 5, color = "white") +
  blank_theme +
  theme(axis.text.x = element_blank()) +
  labs(fill = "OS")

```

# Salary  {.tabset .tabset-fade .tabset-pills}

People may still be interested in how much each type of developer earn.

```{r salary-dist}
# survey <- survey  %>% 
#   mutate(DevType2 = fn(DevType))

salary <- survey %>% 
  select(Country, DevType2, ConvertedSalary, 
         Employment, Student, Gender, Age) %>% 
  group_by(DevType2) %>% 
  mutate(dev_n = n()) %>% 
  ungroup() %>% 
  group_by(Employment) %>% 
  mutate(emp_n = n()) %>% 
  ungroup() %>% 
  drop_na()

dev_type <- salary$DevType2 %>% unique()

```

## How many people are employed full-time?

```{r}

employ <- salary %>% select(Employment, emp_n) %>% 
  distinct() 

employ %>% 
  mutate(count = emp_n / sum(emp_n) ) %>% 
  ggplot(aes(x = reorder(Employment, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5], width = 0.5) +
  geom_label(aes(x = Employment, y = 0.1, 
                label = str_c("(", round(count*100, 1), "%)")),
            fontface = "bold", hjust = 0.5, vjust = 0.5, 
            size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "Number of Participants",
       title = "Employment Status") +
  theme(axis.text.x = element_text(size = rel(1.25)),
        axis.text.y = element_text(size = rel(1.25)),
        plot.title = element_text(size = rel(1.25), face = "bold"))

```

Among all the pariticipants, `r round(70495/dim(survey)[1], 3) * 100`% of them are full-time employed.

## Who did not report salaries?

```{r}

## The maximum salary is from a back-end developer
## The wealthest person currently is Jeff Bezons, who has 134.3 billion US dollors. 134.3 billion USD = 1.343e+11
## If the salary in the data is higher than this number, it was filtered. But the first thing to do is to unify the currencies, let's do it in US dollars.

fn_student <- function(x) {
  split_fn <- function(y) str_split(y, pattern = ",")[[1]][[1]]
  plyr::ldply(x, split_fn)[[1]]
}

no_salary <- salary %>% 
  filter(ConvertedSalary < 1) %>%
  mutate(Student2 = fn_student(Student),
    stdt_grp = if_else(Student2 == "No", "N", "Y")) %>% 
  group_by(stdt_grp) %>% 
  mutate(student_n = n()) %>% 
  ungroup() %>% 
  group_by(Country) %>% 
  mutate(country_n2 = n()) %>% 
  ungroup()

```

In fact, only about `r round(360/736, 2)*100`% of the participants who reported zero salary are student.

```{r no-salary-bar, fig.height=10}
## To get a more compact view, the countries with 
## less than five participants were filtered out.
no_salary %>% 
  select(Country, country_n2) %>% 
  filter(country_n2 > 5) %>% 
  distinct() %>% 
  mutate(count = country_n2 / dim(survey)[1]) %>% 
  ggplot(aes(x = reorder(Country, country_n2), country_n2)) +
  geom_bar(stat = "identity", fill = colors[5]) + 
   # geom_label(aes(x = Country, y = 1e-4, 
   #              label = str_c("(", round(count*100, 2), "%)")),
   #          fontface = "bold", hjust = 0.5, vjust = 0.5, 
   #          size = 4, color = "black") +
   # scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(y = "", x = "", title = "Who Did not Provide the Salary?") +
  theme(axis.text.y = element_text(size = rel(1.15), angle = 15),
        axis.text.x = element_text(size = rel(1.15), angle = 15),
        plot.title = element_text(size = rel(1.25), face = "bold"))

```

## Salary distribution by developer types

```{r salary-devtype}

employed <- "Employed full-time"
salary1 <- salary %>% 
  filter(Employment == employed, ConvertedSalary > 1000) %>%
  mutate(Student2 = fn_student(Student),
    stdt_grp = if_else(Student2 == "No", "N", "Y")) %>% 
  group_by(stdt_grp) %>% 
  mutate(student_n = n()) %>% 
  ungroup() %>% 
  group_by(Country) %>% 
  mutate(country_n = n()) %>% 
  ungroup() 

```

```{r fig.height=10}

salary_median <- median(salary1$ConvertedSalary)

options(scipen = 10000)
# salary1 %>%
#   filter(Gender %in% c("Male", "Female")) %>% 
#   ggplot(aes(x = DevType2, y = ConvertedSalary, fill = Gender)) +
#   geom_boxplot(fill = colors[5], outlier.shape = 21, outlier.size = 0.5) +
#   scale_y_log10(labels = scales::comma,
#                 breaks = c(1e4, 1e5/2, 1e5, 1e6)) +
#   coord_flip() +
#   labs(y = "Annual Salary (USD)", x = "",
#        title = "How Much Were Different Types of Developers Paid?") +
#   geom_hline(yintercept = salary_median,
#              color = colors[1], linetype = "dashed", size = 1) +
#   scale_x_discrete(limits = salary1$Country) +
#     theme(axis.text.x = element_text(size = rel(1.15), angle = 30),
#         axis.text.y = element_text(size = rel(1.15)),
#         axis.title.x = element_text(size = rel(1.15), face = "bold"),
#         plot.title = element_text(size = rel(1.25), face = "bold"))

vline <- function(x = salary_median, color = colors[1]) {
  list(
    type = "line", 
    y0 = 0, 
    y1 = 1, 
    yref = "paper",
    x0 = x, 
    x1 = x, 
    line = list(color = color)
  )
}

salary1 %>% plot_ly(y = ~DevType2, x = ~ConvertedSalary, 
                    line = list(color = colors[5]),
                    marker = list(size = 3),
                  type = "box") %>% 
  layout(title = "Salary Distribution by Developer Type",
         xaxis = list(title = "Annual Salary (USD)", type = "log"),
         yaxis = list(title = ""),
         shapes = list(vline(salary_median)))

```

There are a total of `r dim(salary1)[1]` participants answered this question. The median salary is `r salary_median` USD. The median salary of Engineering managers, DevOPs specialists, C-suite executives are the highest.

## How much does data scientists earn?

The figure is a salary distribution by countries and gender. Only countires with more than 200 participants were included. Similarly, the orange veritical line is the overall median salary (`r salary_median` USD).

```{r ds-salary, fig.height=10, fig.width=12, include=TRUE}

salary2 <- salary1 %>% 
  mutate(gender2 = split_fn(Gender)) %>% 
  group_by(gender2) %>% 
  mutate(gender_n = n())

gender_type <- unique(salary2$gender2)


data_dev <- unique(salary2$DevType2)[c(2, 12)]

sal2 <- salary2 %>% 
  filter(DevType2 %in% data_dev, 
         gender2 != gender_type[3],
         country_n > 200, 
         Employment == employed)

sal2 %>% plot_ly(y = ~Country, x = ~ConvertedSalary, 
                    color = ~gender2,
                    line = list(color = colors[c(1, 2, 5)]),
                    marker = list(size = 3),
                 type = "box") %>% 
  layout(boxmode = "group",
         title = "Salary Distribution by Developer Type",
         xaxis = list(title = "Annual Salary (USD)", type = "log"),
         yaxis = list(title = ""),
         shapes = list(vline(salary_median)))


# sal2 %>% 
#   ggplot(aes(x = Country, y = ConvertedSalary, fill = gender2)) +
#   geom_boxplot(outlier.size = 0.75, outlier.shape = 21, width = 0.75) +
#   geom_hline(yintercept = salary_median, color = colors[6], 
#              size = 1.5, linetype = "dashed") +
#   scale_y_log10(breaks = c(1e4/2, 1e4, 1e5, 1e6/2, 1e6)) +
#   scale_fill_manual(values = colors[c(1, 3, 5)]) +
#   coord_flip() +
#   labs(x = "", y = "Annual Salary (USD)", 
#        fill = "", title="Annual Salary of Data Scientists/Analysts") +
#   theme(axis.text.x = element_text(size=rel(1.15), angle = 30),
#         axis.text.y = element_text(size = rel(1.15)),
#         plot.title = element_text(size = rel(1.25), face="bold"))

```

```{r include=FALSE}

hc <- hcboxplot(var = sal2$Country, x = sal2$ConvertedSalary, 
                var2 = sal2$gender2, 
          outliers = FALSE)

hc %>% hc_add_theme(hc_theme_smpl()) %>% 
  hc_colors(colors[c(1, 5, 6)]) %>% 
  hc_title(text = "Annual Salary of Data Scientists/Analysts") %>% 
  hc_yAxis(title = list(text = "Annual Salary (USD)"))

```

## Data sicence salary distribution by gender

Only countries with more than 200 participants were included.

```{r ds-gender-salary, fig.height=10, include=TRUE}

kable(matrix(gender_type, ncol = 1)) %>% kable_styling()

salary2 %>% 
  filter(country_n > 200, 
         DevType2 %in% dev_type[c(3, 8)],
         Employment == employed,
         ConvertedSalary > 1000,
         gender2 %in% c("Male", "Female")) %>%
  plot_ly(y = ~Country, x = ~ConvertedSalary, 
                    color = ~gender2,
                    line = list(color = colors[c(1, 2, 5)]),
                    marker = list(size = 3),
                 type = "box") %>% 
  layout(boxmode = "group",
         title = "Data Scientists/Analysts Salary Distribution by Countries",
         xaxis = list(title = "Annual Salary (USD)", type = "log"),
         yaxis = list(title = ""),
         shapes = list(vline(salary_median)))
  
  
  # ggplot(aes(x = gender2, y = ConvertedSalary, fill = gender2)) +
  # geom_boxplot(outlier.size = 1, outlier.shape = 21, width = 0.35) +
  #   geom_hline(yintercept = salary_median, 
  #            linetype="dashed", color=colors[6], size = 1.5) +
  # scale_y_log10(breaks = c(1e4/2, 1e4, 1e5/2, 1e5, 1e6)) +
  # scale_fill_manual(values = colors[c(1,3,5)]) +
  # # scale_color_manual(values = colors[c(1,3,5)]) +
  # # coord_flip() +
  # labs(y = "Annual Salary (USD)", x = "", 
  #      fill = "", title = "Data Scientist Annual Salary Distribution by Gender") +
  # theme(axis.text.y = element_text(size = rel(1.15)),
  #       axis.text.x = element_text(size=rel(1.15), angle = 25),
  #       axis.title = element_text(size = rel(1.15), face = "bold"),
  #       plot.title = element_text(size = rel(1.25), face = "bold")
  #       )

```

```{r include=FALSE}

sal3 <- salary2 %>% 
  filter(country_n > 200, 
         Employment == employed,
         ConvertedSalary > 1000,
         gender2 %in% c("Female", "Male"))

hc <- hcboxplot(var = sal3$Country, x = sal3$ConvertedSalary, 
                var2 = sal3$gender2, outliers = FALSE)
hc %>% hc_add_theme(hc_theme_smpl()) %>% 
hc_colors(colors[c(1, 5, 6)]) %>% 
  hc_title(text = "Annual Salary of Data Scientists/Analysts") %>% 
  hc_yAxis(title = list(text = "Annual Salary (USD)"))

```

Overall, female developers seem to slightly better paied than male developers

## Salary distribution by countries and gender

- By Country Only 

```{r salary-country, fig.height=10, include=TRUE}

sal4 <- salary2 %>% 
  filter(country_n > 100, Employment == employed)

sal4 %>% plot_ly(y = ~Country, x = ~ConvertedSalary, 
                    # color = ~gender2,
                    line = list(color = colors[c(6)]),
                    marker = list(size = 3),
                 type = "box") %>% 
  layout(boxmode = "group",
         title = "Data Scientists/Analysts Salary Distribution by Countries",
         xaxis = list(title = "Annual Salary (USD)", type = "log"),
         yaxis = list(title = ""),
         shapes = list(vline(salary_median)))
  
  
  # ggplot(aes(x = Country, y = ConvertedSalary)) +
  # geom_boxplot(fill = colors[5], 
  #              outlier.shape = 21, outlier.size = 0.5 ) +
  # scale_y_log10(breaks = c(1e4, 1e5/2, 1e5, 1e6/2, 1e6)) + 
  # coord_flip() +
  # geom_hline(yintercept = salary_median, 
  #            color = colors[1], linetype = "dashed", size = 1.5) +
  #   labs(y = "Annual Salary (USD)", x = "", 
  #        title = "Salary Distribution by Countires") +
  # theme(axis.text.y = element_text(angle = 10, size = rel(1.15)),
  #       axis.text.x = element_text(size = rel(1.15), angle = 30),
  #       axis.title = element_text(face = "bold", size = rel(1.15)),
  #       plot.title = element_text(face = "bold", size = rel(1.15))) 

# ggsave("~/Downloads/Images/salary-country.png", dpi=300,
#         width = 12, height = 10, unit = "in")

```

```{r include=FALSE}

hc <- hcboxplot(var = sal4$Country, x = sal4$ConvertedSalary, 
          outliers = FALSE)
hc %>% hc_add_theme(hc_theme_smpl()) %>% 
hc_colors(colors[c(6)]) %>% 
  hc_title(text = "Annual Salary of Data Scientists/Analysts") %>% 
  hc_yAxis(title = list(text = "Annual Salary (USD)"))

```


It appears that most of the millionare developers are from the US.

- Included Gender

```{r salary-country-gender, fig.height=10, include=TRUE}

salary2 %>% 
  filter(country_n > 100, gender2 == gender_type[c(1, 2)]) %>% 
  group_by(Country) %>% 
  mutate(med_salary = median(ConvertedSalary)) %>% 
  ungroup() %>% 
  arrange(desc(med_salary)) %>% 
  plot_ly(y = ~Country, x = ~ConvertedSalary, 
                    color = ~gender2,
                    line = list(color = colors[c(1, 6)]),
                    marker = list(size = 3),
                 type = "box") %>% 
  layout(boxmode = "group",
         title = "Data Scientists/Analysts Salary Distribution by Countries",
         xaxis = list(title = "Annual Salary (USD)", type = "log"),
         yaxis = list(title = ""),
         shapes = list(vline(salary_median)))
  
  
  # ggplot(aes(x = Country, y = ConvertedSalary, fill = gender2)) +
  # geom_boxplot(outlier.shape = 21, outlier.size = 0.5 ) +
  # scale_y_log10(breaks = c(1e4, 1e5/2, 1e5, 1e6/2, 1e6)) + 
  # scale_fill_manual(values = colors[c(1, 2, 5)]) +
  # coord_flip() +
  # geom_hline(yintercept = salary_median, 
  #            color = colors[6], linetype = "dashed", size = 1.5) +
  #   labs(x = "", y = "", fill = "", 
  #      title = "Salary Distribution by Countries and Gender") +
  # theme(axis.text.y = element_text(angle = 10, size = rel(1.15)),
  #       axis.text.x = element_text(size = rel(1.15), angle = 30),
  #       plot.title = element_text(size = rel(1.15), face = "bold"))

# ggsave("~/Downloads/Images/salary-country-gender.png", dpi=300,
#         width = 12, height = 10, unit = "in")
```

## Does salary increase with length of experience?

```{r salayr-age-coding, fig.height=10, fig.asp=1.2}
# get_coding_year <- function(x) {
#    split_fn <- function(y) str_split(y, pattern = " ")[[1]][[1]]
# plyr::ldply(x, split_fn)[[1]] 
# }
# 
# year_up <- function(x) {
#    split_fn <- function(y) str_split(y, pattern = "-")[[1]][[2]]
# plyr::ldply(x, split_fn)[[1]] %>% as.numeric()
# }
# 
# year_low <- function(x) {
#    split_fn <- function(y) str_split(y, pattern = "-")[[1]][[1]]
# plyr::ldply(x, split_fn)[[1]]  %>% as.numeric()
# }

salary3 <- survey %>% 
  select(Country, ConvertedSalary, Age, Gender, Employment,
         YearsCoding, YearsCodingProf, DevType2) %>% 
  drop_na() %>% 
  mutate(gender2 = split_fn(Gender)) %>% 
  group_by(Country) %>% 
  mutate(country_n = n()) %>% 
  ungroup() %>% 
  as_tibble() 

age_sal <- salary3 %>% filter(Employment == employed, 
                   country_n > 200, ConvertedSalary > 1000)

## reorder age groups by theirs age
age_factor <- factor(age_sal$Age)
age_levels <- levels(age_factor)
age_factor <- factor(age_factor, levels = c(age_levels[7], age_levels[1], 
                                             age_levels[2], age_levels[3],
                                             age_levels[4], age_levels[5],
                                             age_levels[6]))
age_sal$age_factor <- age_factor
age_sal %>% 
  ggplot(aes(x = age_factor, y = ConvertedSalary)) +
  geom_boxplot(width = 0.25, fill = colors[5],
               outlier.size = 0.5, outlier.shape = 21) +
  scale_y_log10(breaks = c(1e4, 1e5/2, 1e5, 1e6/2, 1e6)) +
  coord_flip() +
  labs(x = "Age", y = "Annual Salary (USD)", 
       title = "Annual Salary Distribution by Age") +
  theme(axis.text.x = element_text(size = rel(1.15), angle = 15),
        axis.text.y = element_text(size = rel(1.15)),
        axis.title = element_text(size = rel(1.15), face = "bold"),
        plot.title = element_text(size = rel(1.15), face = "bold"))

```

Clearly, there is a trend of salary increase with developers' ages until 65 years and older.  

```{r sal-lang}

df_lang <- survey %>% 
  select(Country, ConvertedSalary, Age, 
         LanguageWorkedWith, FrameworkWorkedWith, DevType2, Employment) %>% 
  as_tibble() %>% 
  filter(!is.na(LanguageWorkedWith))

langs <- str_split(df_lang$LanguageWorkedWith, ";") %>% 
  unlist()  %>% 
  unique() 

```

# Programming Languages Worked With {.tabset .tabset-fade .tabset-pills}

```{r}
kable(matrix(langs, ncol = 2)) %>% kable_styling()
```

```{r code-lang, fig.height=10, include=TRUE}

languages <- df_lang$LanguageWorkedWith
mat <- matrix(ncol = length(langs), nrow = length(languages))

for (i in 1:length(languages)) {
  for (j in 1:length(langs)) {    
    mat[i, j] = sum(unlist(str_split(languages[i], ";")) %in% langs[j])
    }  
}

lang_mat <- as_tibble(mat)
colnames(lang_mat) <- langs

lang_mat %>% summarise_all(sum)

lang_sum <- apply(lang_mat, 2, sum)
lang_sum2 <- tibble(languages = langs, counts = lang_sum)

lang_sum2 %>% 
  mutate(count = counts / sum(counts)) %>% 
  ggplot(aes(x = reorder(languages, count), y = count))  +
  geom_bar(stat = "identity", fill = colors[5]) +
  geom_label(aes(x = languages, y = 1e-2, 
                label = str_c("(", round(count*100, 2), "%)")),
            fontface = "bold", hjust = 0.5, vjust = 0.5, 
            size = 4, color = "black", fill = colors[3] ) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = " ", y = " ", 
       title = "Programming Lanugages Worked With") + 
  theme(plot.title = element_text(size = rel(1.25), face = "bold"))

```


```{r lang-salary, fig.asp=1.2, fig.width=10, include=TRUE}
lang_sal <- cbind(lang_mat, df_lang$ConvertedSalary, 
                  df_lang$Employment, df_lang$DevType2, df_lang$Country) %>% 
  as_tibble() %>% 
  rename(ConvertedSalary = `df_lang$ConvertedSalary`,
         Employment = `df_lang$Employment`,
         DevType = `df_lang$DevType2`,
         Country = `df_lang$Country`)

lang_sal2 <- lang_sal %>% 
  gather(JavaScript:Ocaml, key = "languages", value = "IS_LANG") %>% 
  filter(IS_LANG == 1, 
         !is.na(ConvertedSalary), 
         Employment == employed,
         ConvertedSalary > 1000)

```

## Salary distribution by languages

```{r lang-sal-box, fig.height=10}

median_sal <- median(lang_sal2$ConvertedSalary)

lang_sal2 %>% 
  group_by(languages) %>% 
  mutate(median_salary = median(ConvertedSalary)) %>% 
  ungroup() %>% 
  ggplot(aes(x = reorder(languages, median_salary), 
                         y = ConvertedSalary)) +
  geom_boxplot(fill = colors[5], outlier.shape = 21, 
               outlier.size = 0.25, width = 0.5) +
  scale_y_log10(breaks = c(1e4, 1e5/2, 1e5, 1e6/2, 1e6)) +
  coord_flip() +
  geom_hline(yintercept = median_sal, color = colors[1], 
             size = 1, linetype = "dashed") +
  labs(x = "", y = "Annual Salary (USD)",
       title = "Salary Distribution by Languages Used (USD)") +
  theme(axis.text.x = element_text(size = rel(1.15), angle = 15),
        axis.title.x = element_text(size = rel(1.15)),
        axis.text.y = element_text(size = rel(1.15)),
        plot.title = element_text(size = rel(1.15), face = "bold"))

```

The overall median salary is $61,000. It should be noted that since a developer could have used more than one type of languages, while only the salary is reported as a whole contribution of all languages used. Thus, the salary may be (for sure) mutiplely counted. A ideal case is that a developer has reported the portion of contribution to the salary of a language, but this is impossible. 

```{r lang-median-sal, fig.height=8}

lang_median_salary <- lang_sal2 %>% 
  group_by(languages) %>% 
  summarise(median_salary = median(ConvertedSalary),
            max_salary = max(ConvertedSalary),
            min_salary = min(ConvertedSalary)) %>% 
  arrange(by = median_salary) %>% 
  ungroup()

kable(lang_median_salary) %>% kable_styling()

lang_median_salary %>% ggplot(aes(x = reorder(languages, median_salary), 
                                  y = median_salary)) +
  geom_bar(stat = "identity", fill = colors[5]) +
  geom_hline(yintercept = median_sal, size = 1.5, 
             color = colors[6], linetype = "dashed") +
  geom_label(aes(y = 3e4, x = languages, label = str_c("(", median_salary, " USD)")),
             hjust = 0, vjust = 0.5, size = 4, fontface = "bold", 
             color = "black", fill = colors[3]) +
    coord_flip() +
    labs(x = "", y = "Annual Salary (USD)", 
         title = "Median Salary by Languages Used (USD)") + 
    theme(axis.text.x = element_text(size = rel(1.15)),
          axis.text.y = element_text(size = rel(1.15)),
          plot.title = element_text(size = rel(1.25), face = "bold")) +
    annotate("segment", y = 6e4+2000, yend = 7e4, x=1, xend=4, 
             colour=colors[2], size = 1, arrow = arrow(length = unit(0.15, "inches"))) +
    annotate("label", y = 7e4+7000, x = 5.5, 
             label = "Overall Median Salary\n($61K)",
             color = colors[6], size = 4, fontface = "bold") 

```

## Languages Used by Data Scientists/Analysts?

```{r language-devtype, fig.height=10}

ds_dev <- unique(lang_sal$DevType)[c(3, 8)]

ds_lang <- lang_sal %>% select(JavaScript:Ocaml, DevType) %>% 
  filter(DevType %in% ds_dev)

ds_lang2 <- ds_lang %>% 
  gather(JavaScript:Ocaml, key = "language", value = "IS_LANG") %>% 
  filter(IS_LANG == 1) %>% 
  group_by(language) %>% 
  mutate(n_lang = n()) %>% 
  ungroup() %>% 
  select(language, n_lang) %>% 
  distinct() %>% 
  mutate(count =  n_lang / sum(n_lang))
  
ds_lang2 %>% 
  ggplot(aes(x = reorder(language, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5]) +
  geom_label(aes(x = language, y = 1e-2, 
               label = str_c("(", round(count*100, 2), "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "",
       title = " Languages Used by Data Scientists/Analysts") +
  theme(axis.text = element_text(size = rel(1.15)),
        plot.title = element_text(size = rel(1.15), face = "bold"))

```

Undebatablely, Python has now been the dominating language used in the field data science. Interestingly, `C` is ranked number 2 in the DS's language list. Also, SQL is a must have tool for a data scientist/analyst.

## Languages Desire Next Year

# The Battle of IDEs  {.tabset .tabset-fade .tabset-pills}

## IDEs included in they survey are:

```{r ide-dist}

df_ide <- survey %>% 
  select(Country, DevType2, IDE, ConvertedSalary, Gender, Age, 
         YearsCoding, YearsCodingProf, LanguageWorkedWith, 
         FrameworkWorkedWith) %>% 
   filter(!is.na(IDE)) %>% 
  group_by(IDE) %>% 
  mutate(ide_n = n()) %>% 
  ungroup()
 

ide_type <- str_split(df_ide$IDE, ";") %>% 
  unlist()  %>% 
  unique() 

# print("IDEs Included in the survey are")
kable(matrix(ide_type, ncol = 3)) %>% kable_styling()

num_ide <- length(ide_type)
ide_mat <- matrix(ncol = num_ide, nrow = dim(df_ide)[1])
ide_all <- df_ide$IDE

for (i in 1:length(ide_all)) {
  for (j in 1:length(ide_type)) {    
    ide_mat[i, j] = sum(unlist(str_split(ide_all[i], ";")) %in% ide_type[j])
    }  
}

ide_mat2 <- ide_mat %>% as_tibble()
colnames(ide_mat2) <- ide_type

```

The situation for IDE is similar to languages, it is highly possible
that a developer used multiple IDEs at the same time. 

```{r ide-bar, fig.height=8}

ide_mat3 <- cbind(df_ide$Country, df_ide$DevType2, 
                  df_ide$ConvertedSalary, ide_mat2) %>% 
  as_tibble() %>% 
  rename(Country = `df_ide$Country`,
         DevType = `df_ide$DevType2`,
         ConvertedSalary = `df_ide$ConvertedSalary`) 

ide2 <- ide_mat3 %>% gather(Komodo:TextMate, key = "IDE", value = "Value") %>% 
  select(IDE, Value)

ide3 <- ide2 %>% 
  filter(Value == 1) %>% 
  group_by(IDE) %>% 
  mutate(n = n()) %>% 
  ungroup() %>% 
  select(-2) %>% 
  distinct()

ide3 %>% 
  mutate(count = n / sum(n)) %>% 
  ggplot(aes(x = reorder(IDE, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5]) +
  geom_label(aes(x = IDE, y = 8e-3, 
               label = str_c("(", round(count*100, 2), "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() + 
  labs(x = "", y = "", 
       title = "IDE Usage") + 
  theme(axis.text = element_text(size = rel(1.15)),
        plot.title = element_text(size = rel(1.15), face = "bold"))
  
```

**Visual Studio Code* is the king of IDE, with **Visual Studio** being the runner up. My favorite is Emacs, which is just as popular as Rstudio, with which I prepared this RMarkdown document.

# Artificial Intelligence: Is the Winter Coming? {.tabset .tabset-fade .tabset-pills}

## Is AI really dangerous?

Let first take a look at the possible answers:

```{r ai-danger}

df_ai <- survey %>% 
  select(Country, Employment, DevType2, 
         AIDangerous, AIInteresting, AIResponsible, AIFuture) %>% 
  as_tibble()

df_ai %>% 
  filter(!is.na(AIDangerous)) %>% 
  select(AIDangerous) %>%  
  distinct() %>% 
  kable() %>% 
  kable_styling()

```

According to the answers to this item, no negative feedback is given.

```{r ai-danger2}

ai_danger <- df_ai %>% 
  filter(!is.na(AIDangerous)) %>% 
  group_by(AIDangerous) %>% 
  mutate(danger_n = n()) %>% 
  ungroup()

ai_danger %>% 
  select(AIDangerous, danger_n) %>% 
  distinct() %>% 
  mutate(count = danger_n / sum(danger_n)) %>% 
  # print()

  ggplot(aes(x = "", y = count, fill = factor(AIDangerous))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar(theta = "y", start = 0) +
   scale_fill_manual(values = colors[c(2,1,5,6)])  +
    geom_text(aes(y = c(12, 30, 60, 85)/100,
                  label = percent(count)), size = 4, color = "white") +
  blank_theme +
  theme(axis.text.x = element_blank()) +
  labs(fill = "AI Dangerous?")

```

## Is AI interesting?

```{r ai-interesting}

df_ai %>% filter(!is.na(AIInteresting)) %>% 
  group_by(AIInteresting) %>% 
  mutate(count = n()) %>% 
  ungroup() %>% 
  select(count, AIInteresting) %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
   # print()

  ggplot(aes(x = "", y = count, fill = factor(AIInteresting))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar(theta = "y", start = 0) +
   scale_fill_manual(values = colors[c(2,1,5,6)])  +
    geom_text(aes(y = c(86, 20, 65, 46)/100,
                  label = percent(count)), size = 4, color = "white") +
  blank_theme +
  theme(axis.text.x = element_blank()) +
  labs(fill = "AI Interesting?")


```

## Is AI reposible?

```{r ai-responsile}

df_ai %>% 
  filter(!is.na(AIResponsible)) %>% 
  group_by(AIResponsible) %>% 
  mutate(count = n() ) %>% 
  ungroup() %>% 
  select(count, AIResponsible) %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
  # print()

  ggplot(aes(x = "", y = count, fill = factor(AIResponsible))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar(theta = "y", start = 0) +
   scale_fill_manual(values = colors[c(2,1,5,6)])  +
    geom_text(aes(y = c(25, 85,  55, 68)/100,
                  label = percent(count)), size = 4, color = "white") +
  blank_theme +
  theme(axis.text.x = element_blank()) +
  labs(fill = "AI Responsible?")

```

## Will AI have a bright future?
```{r ai-future}

  survey %>% 
  select(AIFuture) %>% 
  filter(!is.na(AIFuture)) %>% 
  group_by(AIFuture) %>% 
  mutate(count = n()) %>% 
  ungroup() %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
  
  ggplot(aes(x = "", y = count, fill = factor(AIFuture))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar(theta = "y", start = 0) +
   scale_fill_manual(values = colors[c(2,1,6)])  +
    geom_text(aes(y = c(50, 95, 12)/100,
                  label = percent(count)), size = 4, color = "white") +
  blank_theme +
  theme(axis.text.x = element_blank()) +
  labs(fill = "AI Future")

```

Most people are optimistic about, may be the winter is not so close.


# Database  {.tabset .tabset-fade .tabset-pills}
## Database Worked with
```{r database}

database <- survey %>% 
  select(Country, Employment, DevType2, ConvertedSalary, 
         DatabaseDesireNextYear, DatabaseWorkedWith) %>% 
  filter(!is.na(DatabaseWorkedWith)) %>% 
  as_tibble() 

db_type <- str_split(database$DatabaseWorkedWith, ";") %>% 
  unlist() %>% 
  unique()

kable(matrix(db_type, ncol=2)) %>% kable_styling()

num_db <- length(db_type)
db_mat <- matrix(ncol = num_db, nrow = dim(database)[1])
db_all <- database$DatabaseWorkedWith

for (i in seq_along(db_all)) {
  for (j in seq_along(db_type)) {    
    db_mat[i, j] = if_else(grepl(db_type[j], db_all[i], fixed = T), 1, 0)
    }  
}

db_mat2 <- db_mat %>% as.data.frame()
colnames(db_mat2) <- db_type

```

```{r db-prefer, fig.height=10}
db_mat3 <- cbind(database$Country, database$DevType2, database$ConvertedSalary, db_mat2) %>% 
  as_tibble() %>% 
  rename(Country = `database$Country`,
         DevType = `database$DevType2`,
         ConvertedSalary = `database$ConvertedSalary`) 

database2 <- db_mat3 %>% gather(Redis:Neo4j, key = "database", value = "Value") %>% 
  select(database, Value) %>% 
  filter(Value == 1) %>% 
  group_by(database) %>% 
  mutate(n = n()) %>% 
  ungroup() %>% 
  select(-2) %>% 
  distinct() 

database2 %>% 
  mutate(count = n / sum(n)) %>% 
  ggplot(aes(x = reorder(database, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5]) +
  geom_label(aes(x = database, y = 2e-2, 
                label = str_c("(", round(count*100, 2), "%)")),
            fontface = "bold", hjust = 0.5, vjust = 0.5, 
            size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", 
       title = "Database Preference") +
  theme(plot.title = element_text(size = rel(1.25), face = "bold"))

```
Obviously, MySQL is still the most popular database.

## Database Desire Next Year

```{r include=FALSE}

database2 <- survey %>% 
  select(Country, Employment, DevType2, ConvertedSalary, 
         DatabaseDesireNextYear) %>% 
  filter(!is.na(DatabaseDesireNextYear)) %>% 
  as_tibble() 

db_type2 <- str_split(database2$DatabaseDesireNextYear, ";") %>% 
  unlist() %>% 
  unique()

kable(matrix(db_type2, ncol = 2)) %>% kable_styling()

num_db2 <- length(db_type2)
db_mat2 <- matrix(ncol = num_db2, nrow = dim(database2)[1])
db_all2 <- database2$DatabaseDesireNextYear

for (i in seq_along(db_all2)) {
  for (j in seq_along(db_type2)) {    
    db_mat2[i, j] = if_else(grepl(db_type2[j], db_all2[i], fixed = T), 1, 0)
    }  
}

db_mat2 <- db_mat2 %>% as_tibble()
colnames(db_mat2) <- db_type2
```

```{r}

db_mat2 <- cbind(database2$Country, database2$DevType2, 
                 database2$ConvertedSalary, db_mat2) %>% 
  as_tibble() %>% 
  rename(Country = `database2$Country`,
         DevType = `database2$DevType2`,
         ConvertedSalary = `database2$ConvertedSalary`) 

database2 <- db_mat2 %>% gather(Redis:Neo4j, key = "database", value = "Value") %>% 
  select(database, Value) %>% 
  filter(Value == 1) %>% 
  group_by(database) %>% 
  mutate(n = n()) %>% 
  ungroup() %>% 
  select(-2) %>% 
  distinct() 

database2 %>% 
  mutate(count = n / sum(n)) %>% 
  ggplot(aes(x = reorder(database, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5]) +
  geom_label(aes(x = database, y = 0.03, 
                label = str_c("(", round(count*100, 2), "%)")),
            fontface = "bold", hjust = 0.5, vjust = 0.5, 
            size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", 
       title = "Database Desire Next Year") +
  theme(plot.title = element_text(size = rel(1.25), face = "bold"))

```

Like in the database worked with, MYSQL still is the most popular database. The Redis is the most popular NoSQL database in the list.

# Frameworks  {.tabset .tabset-fade .tabset-pills}
## Frameworks Worked With

```{r frames-worked}

survey %>% select(FrameworkWorkedWith) %>% 
  filter(!is.na(FrameworkWorkedWith)) %>% 
  mutate(FrameworkWorkedWith = str_split(FrameworkWorkedWith, ";")) %>% 
  unnest(FrameworkWorkedWith) %>% 
  group_by(FrameworkWorkedWith) %>%
  mutate(count = n()) %>%
  ungroup() %>% 
  distinct() %>%
  mutate(count = count / sum(count)) %>% 
  # print()

  ggplot(aes(x = reorder(FrameworkWorkedWith, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = FrameworkWorkedWith, y = 2e-2, 
               label = str_c("(", round(count*100, 2), "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Framework Worked with") +
  theme(plot.title = element_text(face = "bold"))


```

## Frameworks Desire Next Year

```{r frame-worked-designed}

survey %>% select(FrameworkDesireNextYear) %>% 
  filter(!is.na(FrameworkDesireNextYear)) %>% 
  mutate(FrameworkDesireNextYear = str_split(FrameworkDesireNextYear, ";")) %>% 
  unnest(FrameworkDesireNextYear) %>% 
  group_by(FrameworkDesireNextYear) %>%
  mutate(count = n()) %>%
  ungroup() %>% 
  distinct() %>%
  mutate(count = count / sum(count)) %>% 
  # print()

  ggplot(aes(x = reorder(FrameworkDesireNextYear, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = FrameworkDesireNextYear, y = 3e-2, 
               label = str_c("(", round(count*100, 2), "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Framework Desire Next Year") +
  theme(plot.title = element_text(face = "bold"))

```

Tensorflow turns the 5th framework to get next year, and spark also gained some  more popularity.

# Job Satisfaction  {.tabset .tabset-fade .tabset-pills}

## Overall distribution
```{r job-satisfaction}

survey %>% 
  select(ConvertedSalary, JobSatisfaction, DevType2, Age, Gender) %>% 
  as_tibble() %>% 
  filter(!is.na(JobSatisfaction)) %>% 
  group_by(JobSatisfaction) %>%
  mutate(count = n()) %>%
  ungroup() %>% 
  select(count, JobSatisfaction) %>%
  distinct() %>%
  mutate(count = count / sum(count),
         JobSatisfaction = factor(JobSatisfaction)) %>%

  ggplot(aes(x = reorder(JobSatisfaction, count), 
             y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = JobSatisfaction, y = 5e-2, 
               label = str_c("(", round(count*100, 2), "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Job Satisfaction") +
  theme(plot.title = element_text(face = "bold"))

```

## Relationship with Salary?

```{r}
survey %>% 
  select(Country, ConvertedSalary, JobSatisfaction, DevType2, Age) %>% 
  filter(!is.na(JobSatisfaction), !is.na(ConvertedSalary)) %>% 
  group_by(JobSatisfaction) %>% 
  mutate(median_salary = median(ConvertedSalary),
         min_salary = min(ConvertedSalary),
         max_salary = max(ConvertedSalary)) %>%
  ungroup() %>% 
  select(JobSatisfaction, median_salary, min_salary, max_salary) %>%
  distinct() %>% 
  
  ggplot(aes(x = reorder(JobSatisfaction, median_salary), y = median_salary)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = JobSatisfaction, y = 1e4, 
               label = str_c("(", median_salary, " USD)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
  # scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Job Satisfaction versus Salary") +
  theme(plot.title = element_text(face = "bold"))

```

So, are people with salary located in the middle of the salary range struggling?

# Career Satisfaction  {.tabset .tabset-fade .tabset-pills}
## Overall Distribution
```{r}

career <- survey %>% select(CareerSatisfaction, 
                  ConvertedSalary, 
                  Gender) %>% 
  filter(!is.na(CareerSatisfaction)) %>% 
  mutate(gender2 = split_fn(Gender)) %>% 
  group_by(CareerSatisfaction) %>% 
  mutate(count = n()) %>% 
  ungroup() 

career %>% 
  select(CareerSatisfaction, count) %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
  # print()

  ggplot(aes(x = reorder(CareerSatisfaction, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  # scale_fill_manual(values = colors[c(1, 2, 5)]) +
  geom_label(aes(x = CareerSatisfaction, y = 3e-2, 
               label = str_c("(", round(count, 3)*100, "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Career Satisfaction", fill = "") +
  theme(plot.title = element_text(face = "bold"))

```


## Correlation with Salary?
```{r}

career %>% 
  filter(!is.na(ConvertedSalary)) %>% 
  group_by(CareerSatisfaction) %>% 
  mutate(med_sal = median(ConvertedSalary)) %>% 
  ungroup() %>% 
  select(med_sal, CareerSatisfaction) %>% 
  distinct() %>% 
  # print()
  
   ggplot(aes(x = reorder(CareerSatisfaction, med_sal), y = med_sal)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  # scale_fill_manual(values = colors[c(1, 2, 5)]) +
  geom_label(aes(x = CareerSatisfaction, y = 1e4, 
               label = str_c("(", round(med_sal/1000, 0), "K USD)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   # scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Career Satisfaction with Median Salary", 
       fill = "") +
  theme(plot.title = element_text(face = "bold"))
```

Generally, better paid careers have significantly higher satisfaction.

# Time Arrangement {.tabset .tabset-fade .tabset-pills}

## How long can developers be productive? 

```{r productive}
survey %>% select(TimeFullyProductive) %>% 
  drop_na() %>% 
  group_by(TimeFullyProductive) %>% 
  mutate(count = n()) %>% 
  ungroup() %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
  
  ggplot(aes(x = reorder(TimeFullyProductive, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = TimeFullyProductive, y = 3e-2, 
               label = str_c("(", round(count, 2)*100, "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Productive Time") +
  theme(plot.title = element_text(face = "bold"))

```


## Wake Time 
```{r wake-time}
survey %>% select(WakeTime) %>% 
  drop_na() %>% 
  group_by(WakeTime) %>% 
  mutate(count = n()) %>% 
  ungroup() %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
  
  ggplot(aes(x = reorder(WakeTime, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = WakeTime, y = 3e-2, 
               label = str_c("(", round(count, 3)*100, "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Wake Time") +
  theme(plot.title = element_text(face = "bold"))

```

## Hours Staring Computer
```{r}
survey %>% select(HoursComputer) %>% 
  drop_na() %>% 
  group_by(HoursComputer) %>% 
  mutate(count = n()) %>% 
  ungroup() %>% 
  distinct() %>% 
  mutate(count = count / sum(count)) %>% 
  
  ggplot(aes(x = reorder(HoursComputer, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = HoursComputer, y = 5e-2, 
               label = str_c("(", round(count, 3)*100, "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Hours Using Computer") +
  theme(plot.title = element_text(face = "bold"))
```

### Computer Hours and Salary?

Consider only full-time employed participants.

```{r}

survey %>% select(HoursComputer, ConvertedSalary, Employment) %>% 
  filter(Employment == employed) %>% 
  drop_na() %>% 
  group_by(HoursComputer) %>% 
  mutate(median_salary = median(ConvertedSalary)) %>% 
  ungroup() %>% 
  select(-c(2)) %>% 
  distinct() %>% 
  # print()

   ggplot(aes(x = reorder(HoursComputer, median_salary), y = median_salary)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = HoursComputer, y = 2e4, 
               label = str_c("(", median_salary, " USD)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Longer Hours Facing Computer Better Paid?") +
  theme(plot.title = element_text(face = "bold"))

```

Longer computer on computer = higher salary? No, maybe it is productivitity that matters.


## Hours Outside

```{r}
survey %>% select(HoursOutside, ConvertedSalary, Employment) %>% 
  filter(Employment == employed) %>% 
  drop_na() %>% 
  group_by(HoursOutside) %>% 
  mutate(median_salary = median(ConvertedSalary),
         count = n()) %>% 
  ungroup() %>% 
  select(-c(2)) %>% 
  distinct() %>%
  mutate(count = count / sum(count)) %>% 
  # print()

  ggplot(aes(x = reorder(HoursOutside, count), y = count)) +
  geom_bar(stat = "identity", fill = colors[5])  +
  geom_label(aes(x = HoursOutside, y = 3e-2, 
               label = str_c("(", round(count, 3)*100, "%)")),
           fontface = "bold", hjust = 0.5, vjust = 0.5, 
           size = 4, color = "black", fill = colors[3]) +
   scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = "", y = "", title = "Hours Outside") +
  theme(plot.title = element_text(face = "bold"))
  
```


# More are on the way, please stay tuned!
