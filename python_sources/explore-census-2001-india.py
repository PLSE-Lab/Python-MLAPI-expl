

<h1><center> Govt. of India Census, 2001 </center></h1>

<h4>Census of India - History</h4>

<p>The decennial Census of India has been conducted 15 times, As of 2011. It has been conducted every 10 years, beginning in 1872, however the first complete census was taken in the year 1881.[1] Post 1949, it has been conducted by the Registrar General and Census Commissioner of India under the Ministry of Home Affairs, Government of India. All the census since 1951 are conducted under 1948 Census of India Act.</p>

<br>

<h4> What is Census? </h4>
<p>Population Census is the total process of collecting, compiling, analyzing or otherwise disseminating demographic, economic and social data pertaining, at a specific time, of all persons in a country or a well-defined part of a country. As such, the census provides snapshot of the country's population and housing at a given point of time.</p>

<br>

<h4> Why Census? </h4>
<p> The census provides information on size, distribution and socio-economic, demographic and other characteristics of the country's population. The data collected through the census are used for administration, planning and policy making as well as management and evaluation of various programmes by the government, NGOs, researchers, commercial and private enterprises, etc. Census data is also used for demarcation of constituencies and allocation of representation to parliament, State legislative Assemblies and the local bodies. Researchers and demographers use census data to analyze growth and trends of population and make projections. The census data is also important for business houses and industries for strengthening and planning their business for penetration into areas, which had hitherto remained, uncovered.<p>

<br>

<h4> Key Points </h4>
* Census is one of the biggest workout conducted by the Govt of India to understand the demographies of the country.
* It is conducted every 10 years. 
* It was initiated by the Britishers during the colonial times. 
* It helps in big way to come up with social and economic programs for the country.
* It also helps to identify the problems in the country

<br>
<br>


```{r setup, include=FALSE}

knitr::opts_chunk$set(echo = FALSE, warning = FALSE, fig.width = 9, fig.height = 8, fig.align = "center")

library("viridisLite")
library(highcharter)
library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)
library(scales)
library(gridExtra)
library(grid)

dshmstops <- data.frame(q = c(0, exp(1:5)/exp(5)),c = substring(viridis(5 + 1, option = "D"), 0, 7)) %>%  list_parse2()

india <- read.csv("../input/all.csv")

year <- c(1901, 1911, 1921, 1931, 1941, 1951, 1961, 1971, 1981, 1991, 2001)
population <- c(238.4, 252.09, 251.32, 278.98, 318.16, 439.23, 548.16, 683.33, 846.42, 1028.74, 1210.19)

population_growth <- as.data.frame(cbind(year, population))

```


##### Before Exploring the 2001 Census data, let us take a look at Indian Population Growth from 1901 - 2001 

```{r}


hchart(population_growth, hcaes(x = year, y = population, color = year), type = "column") %>% 
  hc_credits(enabled = TRUE, text = "Source : Wikipedia") %>%
  hc_add_theme(hc_theme_darkunica())  %>%
  hc_title(text = "Growth of Indian Population from 1901 to 2001") %>%
  hc_subtitle(text = "Population is in Millions")

```



<br>
<br>

### Explore the data from States perspective 

```{r}

india$Rural <- as.numeric(as.character(india$Rural))
india$Urban <- as.numeric(as.character(india$Urban))
india$Growth..1991...2001. <- as.numeric(as.character(india$Growth..1991...2001.))
india$Primary.Health.Centre <- as.numeric(as.character(india$Primary.Health.Centre))
india$Medical.facility <- as.numeric(as.character(india$Medical.facility))
india$Primary.Health.Sub.Centre <- as.numeric(as.character(india$Primary.Health.Sub.Centre))

india[is.na(india)] <- 0

states <- india %>% group_by(State) %>% summarise(Total = n(), Population = sum(Persons), Males = sum(Males), Females = sum(Females), Growth_In_10_Years = sum(Growth..1991...2001.)/Total, TotalRural = sum(Rural), TotalUrban = sum(Urban), TotalLiterates = sum(Persons..literate), TotalMaleLiterates = sum(Males..Literate), TotalFemaleLiterates = sum(Females..Literate), LiteracyRate = sum(Persons..literacy.rate)/Total, MaleLiteracyRate = sum(Males..Literatacy.Rate)/Total, FemaleLiteracyRate = sum(Females..Literacy.Rate)/Total, Age0_4 = sum(X0...4.years), Age5_14 = sum(X5...14.years), Age15_59 = sum(X15...59.years), Age60_above = sum(X60.years.and.above..Incl..A.N.S..), ImpTown1Population = sum(Imp.Town.1.Population), ImpTown2Population = sum(Imp.Town.2.Population), ImpTown3Population = sum(Imp.Town.3.Population), TotalMedicalFacilities = sum(Medical.facility), TotalPrimaryCentres = sum(Primary.Health.Centre), TotalPrimarySubCentres = sum(Primary.Health.Sub.Centre),  TotalMedicalFacilitiesRate = Population/TotalMedicalFacilities, TotalPrimaryCentresRate = Population/TotalPrimaryCentres, TotalPrimarySubCentresRate = Population/TotalPrimarySubCentres)



```

<br>
<br>

### Population{.tabset}
```{r}

tree <- ( hchart(states, hcaes(x = State, value = Population, color = Population), type = "treemap")  %>% hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_db())  %>%
  hc_title(text = "Indian States and Union Territories by Population(Treepmap)") %>%
  hc_colorAxis(stops = color_stops(n = 10, colors = c("#440154", "#21908C", "#FDE725"))) )


bar <- ( hchart(states, hcaes(x = State, y = Population, color = State), type = "column") %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "Indian States and Union Territories by Population(Barplot)") )
   
```

<br>
<br>

#### Treemap
```{r}
tree
```

#### Barplot
```{r}
bar
```

<br>
<br>

### Percentage Contribution of Every State{.tabset}
```{r}

states$PopulationPercent <- states$Population/sum(states$Population) * 100

tree <- (hchart(states, hcaes(x = State, value = PopulationPercent, color = PopulationPercent), type = "treemap")  %>% hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "States by Population Percentage") %>%
  hc_colorAxis(stops = color_stops(n = 10, colors = c("#440154", "#21908C", "#FDE725"))) )
  
  
bar <- (hchart(states, hcaes(x = State, y = PopulationPercent, color = State), type = "column")  %>% hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "States by Population Percentage"))

```

<br>
<br>

#### Treemap
```{r}
tree
```

#### Barplot
```{r}
bar
```

<br>
<br>

### Population by Gender{.tabset}
```{r}

male <- ( hchart(states, hcaes(x = State, y = Males, color = State), type = "column") %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "Indian States and Union Territories by Male Population") )

female <-  ( hchart(states, hcaes(x = State, y = Females, color = State), type = "column") %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "Indian States and Union Territories by Female Population") )
  
```

<br>
<br>

####Male
```{r}
male
```

####Female
```{r}
female
```


<br>
<br>


```{r}

hchart(states, "scatter", hcaes(x = Males, y = TotalMaleLiterates, group = State)) %>%
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flatdark())  %>%
  hc_title(text = "Male Population vs Female Population")

```

<br>
<br>
<br>


```{r}

population <- states[,c(1,4,5)]
population$Diff <- population$Females - population$Males

hchart(population, hcaes(x = State, y = Diff, color = State), type = "column") %>%
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_title(text = "Which States have more Female Population than Male Population?") %>%
  hc_add_theme(hc_theme_flat())

```

<br>
<br>

### Key takeaways from State-wise Population Count Check
* UttarPradesh is the biggest state by population
* UP contributes 16% of India's Population.
* Top 5 states by population size contribute nearly 50% of India's Population
* Followed by Maharashtra, Bihar and West Bengal
* Lakshdweep and Andaman Nicobar Islands are the smallest ones.
* Kerala and Pondichery have more female population than male
* Biggest states have huge difference in male and female population.
* It is very evident than there is Male skewed population

<br>
<br>

### Population Growth Rate

<br>
<br>

```{r}

hchart(states, hcaes(x = State, y = Growth_In_10_Years, color = State), type = "column")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_db())  %>%
  hc_title(text = "Which States or Union Territories had the highest Population Growth Rate?")

```

<br>
<br>

```{r}

hchart(states, hcaes(x = State, value = Growth_In_10_Years, color = Growth_In_10_Years), type = "treemap")  %>% hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_db())  %>%
  hc_title(text = "Which States or Union Territories had the highest Population Growth Rate?") %>%
  hc_colorAxis(stops = color_stops(n = 10, colors = c("#440154", "#21908C", "#FDE725")))

```

<br>
<br>

* Nagaland had the highest growth rate
* Diu and Daman, Dadar and Nagarhaveli had next highest growth.
* All South Indian States are at the bottom

<br>
<br>


### How many are Literates?{.tabset}


<br>
<br>

```{r}

total <- ( hchart(states, hcaes(x = State, y = TotalLiterates, color = State), type = "bubble")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_darkunica())  %>%
  hc_title(text = "Total Literates") )

male <- ( hchart(states, hcaes(x = State, y = TotalMaleLiterates, color = State), type = "bubble")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_darkunica())  %>%
  hc_title(text = "Total Male Literates") )

female <- ( hchart(states, hcaes(x = State, y = TotalFemaleLiterates, color = State), type = "bubble")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_db())  %>%
  hc_title(text = "Total Female Literates") )

```


#### Total
```{r}
total
```


#### Male
```{r}
male
```


#### Female
```{r}
female
```


<br>
<br>


### What is the literacy rate?{.tabset}

```{r}

total <- ( hchart(states, hcaes(x = State, y = LiteracyRate, color = State), type = "bubble")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "Total Literacy Rate") )

male <- ( hchart(states, hcaes(x = State, y = MaleLiteracyRate, color = State), type = "bubble")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "Male Literacy Rate") )

female <- ( hchart(states, hcaes(x = State, y = FemaleLiteracyRate, color = State), type = "bubble")  %>% 
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_flat())  %>%
  hc_title(text = "Female Literacy Rate") )

```


#### Total
```{r}
total
```


#### Male
```{r}
male
```


#### Female
```{r}
female
```


<br>
<br>


```{r, fig.width=6}


literacy <- states[,c(1,13,14)]
literacy <- gather(literacy, 'Male_Female', 'LiteracyRate', 2:3)

ggplot(literacy, aes(Male_Female, State, fill = round(LiteracyRate))) + 
  geom_tile(color = "white", size = 1) +
  ggtitle("Heat Map of Male and Female Population") +
  scale_fill_viridis() +
  geom_text(aes(label=round(LiteracyRate)), color='white')

```


<br>
<br>

### Key takeaways from Literacy data

* Kerala has the highest literacy rate.
* Number of literates does not give the right picture as bigger the state bigger is the number.
* Bihar one of the biggest state has the least literacy rate.
* Lakshdweep , Pondicherry and Mizoram have very good literacy rates.
* All the states have higger Male literacy rate than female literacy rate.

<br>

### Are there any states with more Female Literacy rate than Male literacy rate?

```{r}

literacy <- states[,c(1,13,14)]
literacy$Diff <- literacy$FemaleLiteracyRate - literacy$MaleLiteracyRate

hchart(literacy, hcaes(x = State, y = Diff, color = State), type = "column") %>%
  hc_credits(enabled = TRUE, text = "Source : Census of India 2001") %>%
  hc_add_theme(hc_theme_darkunica())

```


#### Answer is None and there is huge difference in literacy rate of male and female.


### How many people should be treated by every Medical and Primary healthcare facility based on Population of the State?


```{r, fig.height = 7}

medical <- ggplot(states, aes(reorder(State, TotalMedicalFacilitiesRate), TotalMedicalFacilitiesRate)) +
  geom_bar(stat = "identity", fill = "darkred") +
  coord_flip() +
  labs(x = "States and Union Territories", y = "Number of People", title = "Medical Facility")

primary <- ggplot(states, aes(reorder(State, TotalPrimaryCentresRate), TotalPrimaryCentresRate)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  coord_flip() +
  labs(x = "States and Union Territories", y = "Number of People", title = "Primary Healthcare Facility")


grid.arrange(medical, primary, ncol = 2)

```





