---
title: "Zomato Initial EDA"
author: "Pradeep Adhokshaja"
date: "6/11/2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


## Zomato

Zomato is a restaurant search appilication which was founded in 2008. It currently operates in 23 countries.


## Libraries

```{r}

library(tidyverse)
library(ggplot2)
library(ggridges)
library(plotly)
library(tidytext)

zomato <- read.csv('../input/zomato.csv')

zomato %>%
  str()

```


## Where are these restaurants located?


To find out the locations of the restaurant, I have used the world map. Each point on the world map is colored acording to the price range they fall in.


```{r}

WorldData <- map_data('world')
WorldData %>% filter(region != "Antarctica") -> WorldData
WorldData <- fortify(WorldData)

zomato_world <- zomato %>% select(Latitude,Longitude,Price.range) %>% mutate(Price.range=as.factor(Price.range))


p <- ggplot()
p <- p + geom_map(data=WorldData, map=WorldData,
                  aes(x=long, y=lat, group=group, map_id=region),
                  fill="white", colour="#7f7f7f", size=0.5)
p <- p + geom_point(data=zomato_world,aes(x=Longitude,y=Latitude,color=Price.range))

p+theme(panel.background = element_blank(),axis.title = element_blank(),axis.text=element_blank(),legend.position = 'bottom',legend.background = element_blank(),plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),legend.key = element_rect(fill='white',color = 'transparent'))+labs(title='Positions of Profiled Restaurants',color='Price Range')


```


* A majority of restaurants are located in the US

In this script, I try to create a high level analysis of Indian Restaurants.

## How many restaurants have been profiled in India?

The variable `Restaurant.ID` has been used here as the key.


```{r}
zomato %>%
  filter(Country.Code==1) %>%
  select(Restaurant.ID) %>%
  unique() %>%
  nrow()


```

In total, there are 8652 restaurants that are from India in the dataset.

## In which cities are these restaurants present in ?

```{r}
zomato %>% filter(Country.Code==1) %>%
  select(Restaurant.ID,City) %>%
  unique() %>%
  group_by(City) %>%
  summarise(n=n()) %>%
  ggplot(aes(x=reorder(City,n),y=n))+geom_bar(stat = 'identity',fill='#cb202d')+
  coord_flip()+
  theme(panel.background = element_blank(),
        strip.background = element_blank(),
        axis.title = element_text(color = '#2d2d2d'),
        strip.text.x = element_text(color='#2d2d2d',face='bold',size=10),
        plot.title = element_text(hjust=0.5,face='bold',size=15))+
  labs(x='City',y='Number of Restaurants',title="Number of Restaurants by City")
  



```


* Most of the profiled Indian restaurants are present in New Delhi. A major portion of restaurants in India belong to North India, as per the dataset.

## How do the prices vary for each type of price range?


```{r}

zomato %>%
  filter(Country.Code==1) %>%
ggplot(aes(y = as.factor(Price.range))) +
  geom_density_ridges(aes(x = Average.Cost.for.two),
                      alpha = .5, color = "white", from = 0, to = 8000,fill='#cb202d') +
  labs(x = "Average Cost for Two",
       y = "Price Raneg",
       title = "Price Ranges",
       subtitle = "",
       caption = "")+
  theme(plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),panel.background = element_blank(),
        strip.text = element_text(face='bold',color='#2d2d2d'),axis.text.x = element_text(face='bold',color='#2d2d2d'),
        axis.text = element_text(face='bold',color='#2d2d2d'))


```


*  The price range labelled as `1` has the lowest price ranges.


## For each price range what were the top rated places?


```{r}

Sys.setlocale('LC_ALL','C')
for(i in 1:4){
  
  assign(paste("test"),zomato %>% filter(Country.Code==1,Price.range==i) %>%
           group_by(Restaurant.Name,Price.range) %>% 
           summarise(avg_rating=mean(Aggregate.rating)) %>% ungroup() %>%
           unique() %>%
           arrange(desc(avg_rating)) %>%
           head(10))
  #print(test)
  if(i==1){
    zomato_ranking_for_price_range <- test
  }else{
    zomato_ranking_for_price_range <- bind_rows(zomato_ranking_for_price_range,test)
  }
  #print(dim(test))
  
  
}



d <- zomato_ranking_for_price_range %>% 
  ungroup() %>%
  arrange(Price.range,avg_rating) %>%
  mutate(.r=row_number()) %>%
  mutate(Restaurant.Name=gsub('\"'," ",(Restaurant.Name),fixed=TRUE))

d$Restaurant.Name <- as.character(d$Restaurant.Name)
#d$Restaurant.Name <- utf8::utf8_format(d$Restaurant.Name)
p <- 
ggplot(d,aes(.r,avg_rating))+geom_bar(stat = 'identity',fill='#cb202d')+
  facet_wrap(~Price.range,scales = 'free')+
  scale_x_continuous(
    breaks=d$.r,
    labels= d$Restaurant.Name
  )+
  coord_flip()+
  theme(plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),panel.background = element_blank(),
        strip.text = element_text(face='bold',color='#2d2d2d'),axis.text.x = element_text(face='bold',color='#2d2d2d'),
        axis.text = element_text(face='bold',color='#2d2d2d'))+
  labs(y='Average Rating',x='Restaurant',title='Top Rated Restaurants by Price Ranges')

p



```


## How many restaurants have Online Delivery in India?


```{r}

zomato %>%
  filter(Country.Code==1) %>%
  select(Restaurant.ID,Has.Online.delivery,City) %>%
  unique() %>%
  group_by(City,Has.Online.delivery) %>%
  summarise(n=n()) %>%
  ungroup() %>%
  rename(`Online Delivery Service`=Has.Online.delivery) %>%
  ggplot(aes(x=reorder(City,n),y=n,fill=`Online Delivery Service`))+
  geom_bar(stat='identity',position = 'dodge',width = 0.5)+
  labs(x='City',y='Number of Restaurants')+coord_flip()+
  theme(plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),panel.background = element_blank(),
        strip.text = element_text(face='bold',color='#2d2d2d'),axis.text.x = element_text(face='bold',color='#2d2d2d'),
        axis.text = element_text(face='bold',color='#2d2d2d'))

```

* A majority of restaurants do not have an online delivery capability, according to the dataset.



## What are the most popular cuisines in India?


The `Cuisines` variable is comprised of long texts that explain the cuisines of the profiled restaurant. Here, I have tried to break down the cusines into bi-grams using the unnest_tokens function contained in the `tidytext` package.


```{r}

zomato %>%
  filter(Country.Code==1) %>%
  select(Restaurant.ID,Cuisines,Average.Cost.for.two) %>%
  unique() %>%
  mutate(Cuisines=as.character(Cuisines)) %>%
  unnest_tokens(ngram,Cuisines,token='ngrams',n=2) %>%
  group_by(ngram) %>%
  summarise(n=n()) %>%
  arrange(desc(n)) %>%
  top_n(10) %>%
  ggplot(aes(x=reorder(ngram,n),y=n))+geom_bar(stat='identity',fill='#cb202d')+
 theme(plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),panel.background = element_blank(),
        strip.text = element_text(face='bold',color='#2d2d2d'),axis.text.x = element_text(face='bold',color='#2d2d2d'),
        axis.text = element_text(face='bold',color='#2d2d2d'))+
  coord_flip()+labs(x='Cuisine',y='Number of Mentions',title='Popular Cuisines by Mentions')




```




## Is there a relationship between Price range,Votes and Average Cost for two?



```{r}
zomato %>%
  filter(Country.Code==1) %>%
  mutate(Price.range=as.factor(Price.range)) %>%
  ggplot(aes(x=Votes,y=Average.Cost.for.two,color=Price.range))+
  geom_point(alpha=0.4)+
  labs(x='Votes',y='Average Cost for Two')+
  theme(panel.background = element_blank(),plot.title = element_text(hjust = 0.5))+
  facet_wrap(~Price.range)+
  geom_smooth(method='lm')+
  labs(title='Relationship between Price and Votes')+
  theme(plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),panel.background = element_blank(),
        strip.text = element_text(face='bold',color='#2d2d2d'),axis.text.x = element_text(face='bold',color='#2d2d2d'),
        axis.text = element_text(face='bold',color='#2d2d2d'))
  
  


```


## Average Votes for restaurants that do/ do not have table booking, that do/do not have online delivery option and that do/ do not have switch to order menu


```{r}


zomato %>%
  filter(Country.Code==1) %>%
  select(Votes,Switch.to.order.menu,Has.Table.booking,Has.Online.delivery) %>%
  tidyr::gather(type,value,2:4) %>%
  mutate(type=gsub("[.]"," ",type))%>%
  group_by(type,value) %>%
  summarise(mean_votes = mean(Votes,na.rm=TRUE)) %>%
  ggplot(aes(x=value,y=mean_votes))+geom_bar(stat='identity',fill='#cb202d')+facet_wrap(~type)+
  labs(x='Answer',y='Average Number of Votes',title='Average Votes for different kinds of restaurants')+
  theme(plot.title = element_text(hjust=0.5,face='bold',color='#2d2d2d'),panel.background = element_blank(),
        strip.text = element_text(face='bold',color='#2d2d2d'),axis.text.x = element_text(face='bold',color='#2d2d2d'),
        axis.text = element_text(face='bold',color='#2d2d2d'))
  


```


* Restaurants with an online deliver tend to have a higher rating on average.
* Table booking facility tends to be an important factor in a higher average number of votes.
* Nothing much can be said about the importance of order menu as there are no restaurants that provide that facility

## Final Results

* A majority of restaurants are located in India
* A majority of Indian restaurants are located in New Delhi
* A majority of Indian restaurants do not have online delivery
* There is no discernable relationship between average cost and number of votes.