---
title: "Exploratory Data Analysis for Earthquakes in Greece "
author: "Antonis Angelakis"
date: "1st of June 2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

###Import Libraries

```{r Import libraries,warning=FALSE,message=FALSE}
library(rworldmap)
library(zoo)
library(data.table)
library(ggplot2)
library(leaflet)
library(ggmap)
library(DT)
library(knitr)
library(dplyr)
library(plotly)
```

###Import Dataset

```{r}
EarthQuakes_in_Greece <- read.csv("../input/EarthQuakes in Greece.csv")
```

##Rename columns

```{r Rename columns}
earthquakes=EarthQuakes_in_Greece[,c(1,2,3,4,5,6,7,8)]
colnames(earthquakes)[6] <- "Lat"
colnames(earthquakes)[7] <- "Lon"
colnames(earthquakes)[8] <- "Richter"
```

##Explore Variables

```{r explore, echo = TRUE}
str(earthquakes)
```
**For better results and visualisation purposes we convert 3 int columns into 1 of date type**

##Change format of Date

```{r Change format of Date}
earthquakes$DateFormatted <- as.Date(paste(earthquakes$Month, earthquakes$Date , earthquakes$Year), "%m %d %Y")
```

##Point each earthquake in Greece map

```{r Point each earthquake in Greece map} 
newmap <- getMap(resolution = "low")
plot(newmap, xlim = c(18, 30), ylim = c(33, 42), asp = 1)
points(earthquakes$Lon, earthquakes$Lat, col = "red", cex = .6)
```  



**Points of earthquakes are more sparsed in the 4 corners of Greece**



##Count number of earthquakes for each year

```{r Count number of earthquakes for each year}
sumYear <- earthquakes %>% 
  select(Year)  %>% 
  group_by(Year)  %>% 
  summarize(count = n())
datatable(sumYear)  
```



**It seems normal to have more recorded earthquakes as time passes by, because of technology's evolution and better tools.**



##Plot Number of earthquakes for each year

```{r Plot Number of earthquakes for each year}
ggplot(sumYear, aes(x =  Year, y = count, colour = "yellow" , alpha = 1))  + 
  geom_point(colour = "blue")  + geom_line(colour = "black")  
```


**Most earthquakes happened between 2012-2015**



##Plot Density

```{r Plot density}
ggplot(earthquakes, aes(x  = Richter, 
                       fill = "red", 
                       alpha = 0.5))  + 
  geom_density(colour="blue")  
```



**Earthquakes' magnitude probabilities are more accurate from 1 to 4 Richter**



##Create dataframe for average magnitude per year

```{r Create dataframe for average magnitude per year}
years <- unique(earthquakes$Year)
averageRichter <- 1:length(years)
for (i in years){
  temporary <- earthquakes %>% filter(Year==i)
  averageRichter[years==i] <- mean(temporary$Richter)
}
averageRichterPerYear<- data.frame('year' = years, 'averageRichter' = averageRichter)  
```
##Plot Earthquake Average Magnitude by year

```{r Plot Earthquake Average Magnitude by year}
plt <- ggplot(averageRichterPerYear, aes(x = year, y = averageRichter)) +
  geom_line(aes(group=1), colour="#000045")+ geom_point(size=1, colour="#CC0000") + 
  xlab("Year") + ylab("Average Richter") +  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Earthquake Average Magnitude by year ")  
plt
```


**We have a decreasing average magnitude for earthquakes as time passes**




#Heat Map by frequency

```{r Heat Map by frequency}
heatMap <- ggplot() + borders(colour="gray", fill="gray")+ geom_point() +xlim(18, 30) +ylim(33, 42)
heatMapCompleted <- heatMap + geom_density2d(data = earthquakes, aes(x=earthquakes$Lon, y=earthquakes$Lat, color=Richter), size = 0.1) +
  stat_density2d(data = earthquakes, 
                 aes(x=earthquakes$Lon, y=earthquakes$Lat, fill = ..level.., alpha = ..level..),
                 size = 0.1, bins = 15, geom = "polygon") + scale_fill_gradient(low = "yellow", high = "red") + 
  scale_alpha(range = c(0, 1), guide = FALSE) + xlab("Longtitude") + ylab("Latitude") + 
  ggtitle("Earthquake Positions in Greece (1901-2018) (Heat Map by frequency)") + coord_fixed(ratio = 1)
heatMapCompleted  
```



**Most earthquakes occur between Central Greece, Western Greece and Peloponnese**




#Map by position according to Magnitude

```{r Map by position according to Magnitude}
heatMapPos <- ggplot(data = earthquakes) + borders(colour="gray", fill="gray") +  xlim(18, 30) + ylim(33, 42)
heatmapPosCompleted <- heatMapPos + geom_point(aes(x=Lon, y=Lat, color=Richter), size=0.00001) + xlab("Longtitude") +
  ylab("Latitude") + ggtitle("Earthquake Positions from 1965 to 2016") + 
  scale_colour_gradient(low = "darkkhaki", high = "darkmagenta") + coord_fixed(ratio = 1)
heatmapPosCompleted  
```


##Leaflet Plot for map clustering

```{r Leaflet Plot for map clustering}
earthquakes %>%
  leaflet() %>%
  addTiles() %>%
  addMarkers(lat=earthquakes$Lat, lng=earthquakes$Lon, clusterOptions = markerClusterOptions(),
             popup= paste("<br><strong>Richter: </strong>", earthquakes$Richter,
                          "<br><strong>Date: </strong>", earthquakes$DateFormatted
             ))  
```



**Leafplot helps us with a significant way to confirm our insights. Zoom in for a better view of earthquakes seperated by cluster field**










