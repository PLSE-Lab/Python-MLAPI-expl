---
title: "Exploratory Analysis using Mobile Ad Click Data"
author: "Yuqi Nie"
date: "5/28/2018"
output: html_document
---

```{r, echo=F, message=F, warning=F, results=F}
library(knitr)
library(tidyverse)
library(stringr)
library(geosphere)
library(leaflet)
library(highcharter)
library(reshape2)
library(sp)
library(RColorBrewer)



dat_all <- readRDS(file = "../input/dat_all_wf.rds")

```



### 0. Introduction

For this report we are using the ad click data from a mobile ad company. In the original dataset there are 500000 observations and 9 variables, including vendor_id, campaign_id, click_id, conversion_id, device_platform_id, track_type, track_party, event_time and remote_id. Each observation indicates a single click.

- vendor_id - Unique identifier for vendor
- campaign_id - Unique identifier for ad campaign
- click_id - Unique identifier for tracking link. Each tracking link only corresponds to a single vendor and a single campaign
- conversion_id - Unique identifier generated when a user clicks on an ad, irrelevant to time
- device_platform_id - Unique device identifier for advertising purposes
- track_type - Record type. In this case, these are all click records. In this dataset all track_types are 1, therefore we would not use this variable for analysis.
- track_party - MMP; Mobile attribution partner - the 3rd party SDK responsible for attributing installs
- event_time - Time the event (click) occurred
- remote_ip - IP Address of the device



variable     | number of unique values
-------------|-------------------------
vendor_id    | 40
campaign_id  | 59
click_id     |  137
conversion_id|408913
device_platform_id|47023
track_type   | 1
track_party  | 5
remote_ip    |166066


To make better use of IP address, we performed IP geo-location and looked up the below information of IP addresses:


- country
- region name (state)
- city (county)
- zip code
- latitude
- longitude
- time zone
- network service provider



### 1. How are clicks and users distributed across the world?

Since one user (one IP address) may click on more than one ad, or click on one ad more than one time, we describe the number of clicks and unique users per country/region as followed. The majority of clicks come from the United States. Singapore generates 6244 clicks with only two users, which is a sign of abnormal behavior. Out of the 15 regions that generate most clicks, only one region is outside US - Tokyo. The number of users from Tokyo is disproportionately large compared to other regions. (Detailed stats will pop up when cursor is placed on the bars.)

#### {.tabset}
##### By Country (clicks)
```{r clicks by country, echo=F}

df_temp <- dat_all %>% group_by(country) %>% summarise(Total = n(), UniqueUser = n_distinct(remote_ip)) %>% arrange(desc(Total)) %>% head(5)

highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Clicks - By Country (Top 5)") %>% 
  hc_subtitle(text = "Based on all clicks",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = df_temp$country) %>% 
  hc_add_series(data = df_temp$Total,
                name = "Number of clicks", color = "#99CCFF") %>%
  hc_legend(FALSE)

```


##### By Country (IP)
```{r user by country, echo=F}
highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Users - By Country (Top 5 in Clicks)") %>% 
  hc_subtitle(text = "Based on unique IP addresses",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = df_temp$country) %>% 
  hc_add_series(data = df_temp$UniqueUser,
                name = "Number of users", color = "pink") %>%
  hc_legend(FALSE)
```


##### By Region (clicks)
```{r clicks by rg, echo=F}

df_temp_rg <- dat_all %>% group_by(region_name) %>% summarise(Total = n(), UniqueUser = n_distinct(remote_ip)) %>% arrange(desc(Total)) %>% head(15)

highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Clicks - By Region (Top 15)") %>% 
  hc_subtitle(text = "Based on all clicks",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = df_temp_rg$region_name) %>% 
  hc_add_series(data = df_temp_rg$Total,
                name = "Number of clicks", color = "#99CCFF") %>%
  hc_legend(FALSE)
```


##### By Region (IP)

```{r user by rg, echo=F}

highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Users - By Region (Top 15 in Clicks)") %>% 
  hc_subtitle(text = "Based on unique IP addresses",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = df_temp_rg$region_name) %>% 
  hc_add_series(data = df_temp_rg$UniqueUser,
                name = "Number of users", color = "pink") %>%
  hc_legend(FALSE)

rm(df_temp, df_temp_rg)
```


### 2. How are clicks and users distributed across California?

According to the last section, California is the region that generates the most clicks. Here we present the number of clicks and users of the 15 Californian counties that generate most clicks.

#### {.tabset}
##### By County (clicks)
```{r clicks by ct, echo=F}

df_temp_ca <- dat_all[dat_all$region_code == "CA", ] %>% group_by(city) %>% summarise(Total = n(), UniqueUser = n_distinct(remote_ip)) %>% arrange(desc(Total)) %>% head(10)

highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Clicks in California - By County (Top 10)") %>% 
  hc_subtitle(text = "Based on all clicks in CA",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = df_temp_ca$city) %>% 
  hc_add_series(data = df_temp_ca$Total,
                name = "Number of clicks", color = "#99CCFF") %>%
  hc_legend(FALSE)


```


##### By County (IP)

```{r user by ct, echo=F}

highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Users - By County (Top 10 in Clicks)") %>% 
  hc_subtitle(text = "Based on unique IP addresses in CA",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = df_temp_ca$city) %>% 
  hc_add_series(data = df_temp_ca$UniqueUser,
                name = "Number of users", color = "pink") %>%
  hc_legend(FALSE)

rm(df_temp_ca)

```


### 3. How conversions are contributed by track parties?

In the following graph we describe the number of clicks received by each vendor. The y axis indicated the name of each vendor. Vendor 132 and vendor 3 outperform the rest of the vendors. Each bar consists of different color blocks - the five color blocks stand for the 5 track parties. Track party 3 mainly contributes to clicks received by vendor 132 and vendor 3. Track party 6 and track party 8 appear to be the most widely used services. (Color blocks can be hidden or shown by clicking on the corresponding legends.)

```{r conv & vendor, echo=F}
dat_temp <- dat_all %>% group_by(vendor_id) %>% summarise(Total = n()) %>% arrange(desc(Total)) %>% head(10)

dat_temp2 <- dat_all %>% filter(track_party == "2") %>% group_by(vendor_id) %>% 
  summarise(TrackParty2 = n()) %>% filter(vendor_id %in% dat_temp$vendor_id)

dat_temp3 <- dat_all %>% filter(track_party == "3") %>% group_by(vendor_id) %>% 
  summarise(TrackParty3 = n()) %>% filter(vendor_id %in% dat_temp$vendor_id)

dat_temp4 <- dat_all %>% filter(track_party == "4") %>% group_by(vendor_id) %>% 
  summarise(TrackParty4 = n()) %>% filter(vendor_id %in% dat_temp$vendor_id)

dat_temp6 <- dat_all %>% filter(track_party == "6") %>% group_by(vendor_id) %>% 
  summarise(TrackParty6 = n()) %>% filter(vendor_id %in% dat_temp$vendor_id)

dat_temp8 <- dat_all %>% filter(track_party == "8") %>% group_by(vendor_id) %>% 
  summarise(TrackParty8 = n()) %>% filter(vendor_id %in% dat_temp$vendor_id)

dat_temp$TrackParty2 <- dat_temp2$TrackParty2[match(dat_temp$vendor_id, dat_temp2$vendor_id)]

dat_temp$TrackParty3 <- dat_temp3$TrackParty3[match(dat_temp$vendor_id, dat_temp3$vendor_id)]

dat_temp$TrackParty4 <- dat_temp4$TrackParty4[match(dat_temp$vendor_id, dat_temp4$vendor_id)]

dat_temp$TrackParty6 <- dat_temp6$TrackParty6[match(dat_temp$vendor_id, dat_temp6$vendor_id)]

dat_temp$TrackParty8 <- dat_temp8$TrackParty8[match(dat_temp$vendor_id, dat_temp8$vendor_id)]


dat_temp[is.na(dat_temp)] <- 0

rm(dat_temp2, dat_temp3, dat_temp4, dat_temp6, dat_temp8)

highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Clicks Received by each Vendor (Top 15)") %>% 
  hc_subtitle(text = "Divided into 5 track parties",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = dat_temp$vendor_id) %>% 
  hc_plotOptions(bar = list(
    dataLabels = list(enabled = FALSE), stacking = "normal")) %>% 
  hc_series(list(name="track party 2", data = dat_temp$TrackParty2, color = "dimgray"),
            list(name="track party 3", data = dat_temp$TrackParty3, color = "dodgerblue"),
            list(name="track party 4", data = dat_temp$TrackParty4, color = "MediumSeaGreen"),
            list(name="track party 6", data = dat_temp$TrackParty6, color = "pink"),
            list(name="track party 8", data = dat_temp$TrackParty8, color = "orange"))
```


### 4. How vendors contribute to ad campaigns?

Each ad campaign may work with multiple vendors to gain exposure. Here we present the number of clicks received by each ad campaign divided into vendors. Campaign 466 and campaign 481 significantly outperform the rest. Vendor 132 and vendor 3 are the top click contributors.

```{r vend camp, echo=F}
dat_temp <- dat_all %>% group_by(campaign_id) %>% summarise(Total = n(), NumVendor = n_distinct(vendor_id)) %>% arrange(desc(Total)) %>% head(10)


VendorLs <- as.list(unique(dat_all$vendor_id))

for (i in VendorLs) {
  dat_tempTemp <- dat_all %>% filter(vendor_id == i) %>% group_by(campaign_id) %>% 
    summarise(VendorTemp = n()) %>% filter(campaign_id %in% dat_temp$campaign_id)
  
  dat_temp$VendorTemp <- dat_tempTemp$VendorTemp[match(dat_temp$campaign_id, dat_tempTemp$campaign_id)]
  
  colnames(dat_temp)[colnames(dat_temp) == "VendorTemp"] <- paste0("Vendor", i)
}

dat_temp[is.na(dat_temp)] <- 0
rm(dat_tempTemp)

# delete column if all values are zero
dat_temp <- dat_temp[, colSums(dat_temp != 0) > 0]


highchart() %>% 
  hc_chart(type = "bar") %>% 
  hc_title(text = "Number of Clicks by Campaign (Top 10)") %>% 
  hc_subtitle(text = "Divided into vendors",
              style = list(color = "gray", fontWeight = "bold")) %>%
  hc_xAxis(categories = dat_temp$campaign_id) %>% 
  hc_plotOptions(bar = list(
    dataLabels = list(enabled = FALSE), stacking = "normal")) %>% 
  hc_series(list(name="Vendor132", data = dat_temp$Vendor132, color = "#87CEFA"),
            list(name="Vendor156", data = dat_temp$Vendor156, color = "dodgerblue"),
            list(name="Vendor3", data = dat_temp$Vendor3, color = "orange"),
            list(name="Vendor87", data = dat_temp$Vendor87, color = "pink"),
            list(name="Vendor158", data = dat_temp$Vendor158, color = "yellow"),
            list(name="Vendor160", data = dat_temp$Vendor160, color = "green"),
            list(name="Vendor139", data = dat_temp$Vendor139, color = "violet"),
            list(name="Vendor42", data = dat_temp$Vendor42, color = "red"),
            list(name="Vendor100", data = dat_temp$Vendor100, color = "#9ACD32"),
            list(name="Vendor45", data = dat_temp$Vendor45, color = "#008080"),
            list(name="Vendor109", data = dat_temp$Vendor109, color = "#00BFFF"),
            list(name="Vendor146", data = dat_temp$Vendor146, color = "#FF1493"),
            list(name="Vendor153", data = dat_temp$Vendor153, color = "#F5DEB3"),
            list(name="Vendor30", data = dat_temp$Vendor30, color = "#8B4513"),
            list(name="Vendor4", data = dat_temp$Vendor4, color = "#D2691E"),
            list(name="Vendor140", data = dat_temp$Vendor140, color = "#006400"),
            list(name="Vendor76", data = dat_temp$Vendor76, color = "#00FFFF"),
            list(name="Vendor142", data = dat_temp$Vendor142, color = "#6495ED"),
            list(name="Vendor102", data = dat_temp$Vendor102, color = "#800080"),
            list(name="Vendor161", data = dat_temp$Vendor161, color = "#FF00FF"),
            list(name="Vendor101", data = dat_temp$Vendor101, color = "#FFFACD"),
            list(name="Vendor159", data = dat_temp$Vendor159, color = "#BC8F8F"),
            list(name="Vendor86", data = dat_temp$Vendor86, color = "#B0C4DE"),
            list(name="Vendor110", data = dat_temp$Vendor110, color = "#CD5C5C"),
            list(name="Vendor69", data = dat_temp$Vendor69, color = "#808000"),
            list(name="Vendor5", data = dat_temp$Vendor5, color = "#0000FF")) 


```


### 5. User location and behavior in New York City

Below is an interactive visualization of user location in New York City. All clicks in New York come from Manhattan, Brooklyn, Staten Island and the Bronx, but not Queens. The color of the circles represent each track party. The size of the circles indicate the number of clicks generated by that users. More information (IP, location, network service, number of clicks, track party) will pop up when clicking on the circles. Concentric circles mean there are multiple users located very close to each other. Users can also zoom in, zoom out or drag the map to see more areas of New York.

```{r map, echo=F, warning=F}
dat_ny <- dat_all[dat_all$region_name == "New York", ]

dat_nyc <- dat_ny[dat_ny$city == "New York" | dat_ny$city == "Staten Island" |dat_ny$city == "Brooklyn" |dat_ny$city == "The Bronx" |dat_ny$city == "Queens", ]

m <- leaflet(dat_nyc) %>% 
  addTiles("http://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png") %>%
  setView(-73.9949344, 40.7179112, zoom = 11)

pal = colorFactor("Set1", domain = dat_nyc$track_party)
color_offsel1 <- pal(dat_nyc$track_party)
# Set color of points

content <- paste("IP:", dat_nyc$remote_ip,"<br/>",
                 "Number of clicks:", dat_nyc$freq_ip,"<br/>",
                 "Where:", dat_nyc$city,"<br/>",
                 "Network service:", dat_nyc$ISP_name,"<br/>",
                 "Track party:", dat_nyc$track_party)               

m %>% 
  addCircles(lng = ~as.numeric(lon), lat = ~as.numeric(lat),
                 color=color_offsel1, radius = 7*(dat_nyc$freq_ip)^1.3,
                 opacity = 0.1, fillOpacity = 0.01, weight = 1,
                 popup = content) 

```

