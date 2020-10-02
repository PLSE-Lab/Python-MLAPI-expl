---
title: "Calories we eat...in Mcdonald's"
author: "Devi Sangeetha"
output: html_document
---

```{r setup, include=FALSE}
#knitr::opts_chunk$set(echo = TRUE)
```

##**Introduction :**  
McDonald's is an American hamburger and fast food restaurant chain.Today, McDonald's is the world's largest restaurant chain,  
serving approximately 68 million customers daily in 120 countries across approximately 36,900 outlets  
They provide a nutrition analysis of  menu items to help customers balance their McDonald's meal with other foods they eat.  
Key highlight of McDonald's is that customers can select any item to view the complete nutritional information including calories, fat, sodium and Weight Watchers points. 
Customers can use calorie filter to find the McDonald's menu item that best fits your diet.  
So here we will analyse the varaious categories Mc Donald's offers,and its Menu items and their nutritional values.  
  
##Analysis  
### Loading required libraries  
```{r echo=FALSE,warning=FALSE}
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))
suppressMessages(library(plotly))
suppressMessages(library(RColorBrewer))
suppressMessages(library(devtools))
suppressMessages(library(yarrr))
suppressMessages(library(tidyr))
suppressMessages(library(gridExtra))
```  
  
###Load the data  
```{r}
menu<-read.csv("../input/menu.csv")
```

Viewing the structure of data  
```{r}
str(menu)
```
  
###Food Categories  
Distribution of Category  

```{r echo=FALSE,warning=FALSE}
m <- list(
  l = 50,
  r = 50,
  b = 100,
  t = 100,
  pad = 4
)
marker = list(color = brewer.pal(9, "Set1"))
p <- plot_ly(
  x = menu$Category,
  marker = list(color = '#45171D'),
    type = "histogram"
)%>%
  layout(         xaxis = list(title = ""),
         yaxis = list(title = ""),
         autosize = T)
p
```
  
  
##Catogories and its Nutritional Value {.tabset .tabset-fade}  
  
### Calories  
```{r echo=FALSE,warning=FALSE}


new_col<-c("grey50", "blue","hotpink","Magenta","seagreen","violet","brown","maroon","navyblue")
plot_ly(x = menu$Category, y=menu$Calories,color = menu$Category,colors =new_col , type = "box")%>%   layout(
         xaxis = list(title = ""),
         yaxis = list(title = "Calories"),
         showlegend=FALSE,
         autosize = T)

```  
  
### Protien  
```{r echo=FALSE,warning=FALSE}
plot_ly(x = menu$Category, y=menu$Protein,color = menu$Category,colors =new_col , type = "box")%>% layout(
         xaxis = list(title = ""),
         yaxis = list(title = ""),
         showlegend=FALSE,
         autosize = T)
```
  
  
### Carbohydrates  
```{r echo=FALSE,warning=FALSE}
plot_ly(x = menu$Category, y=menu$Carbohydrates,color = menu$Category,colors =new_col , type = "box") %>% layout(
         xaxis = list(title = ""),
         yaxis = list(title = ""),
         showlegend=FALSE,
         autosize = T)
```


```{r echo=FALSE,warning=FALSE}

 
p<-plot_ly(x=menu$Calories, y=menu$Carbohydrates, type="scatter", mode = "markers" , 
marker=list( color=ifelse(menu$Calories>500,"red","blue") , opacity=0.5 , size=20) ) 
p

```
    
### Fat  
```{r echo=FALSE,warning=FALSE}
plot_ly(x = menu$Category, y=menu$Total.Fat,color = menu$Category,colors =new_col , type = "box") %>% layout(
         xaxis = list(title = ""),
         yaxis = list(title = ""),
         showlegend=FALSE,
         autosize = T)

```


Comapring different Fats in items(category= Beef & Park, Chicken & Fish) 
```{r echo=FALSE,warning=FALSE}
m1<-menu%>%filter(Category %in% c("Beef & Pork","Chicken & Fish"))%>%arrange(desc(Total.Fat....Daily.Value.,Saturated.Fat....Daily.Value.,Trans.Fat))
p <- plot_ly(m1, x = ~factor(Item,levels=Item), y = ~Total.Fat....Daily.Value., name = 'Total Fat DV', type = 'scatter', mode = 'lines+markers', width = 2,color = I('red')) %>%
    add_trace(y = ~Saturated.Fat....Daily.Value., name = 'Saturated Fat DV',color=I('blue')) %>%
  add_trace(y = ~Trans.Fat, name = 'Trans Fat',color=I("hotpink")) %>%
      layout(title = 'Camparing Fat in Items',
         xaxis = list(title = "",
                      showgrid = FALSE),
         yaxis = list(title = "value",
                      showgrid = FALSE),
         legend=list(orientation="r",xanchor="center"))
p
```
  
Chicken Nuggets takes the first place of having high fat, next comes Double Quarter Pounder with Cheese, Bacon Clubhouse Burger takes the 3rd position. 
Saturated Fat is high seems very high for these items, since saturated fat increases the blood cholestrol level.
    
### Sugars  
According to the American Heart Association (AHA), the maximum amount of added sugars you should eat in a day are (7):  
  
Men: 150 calories per day (37.5 grams or 9 teaspoons).  
Women: 100 calories per day (25 grams or 6 teaspoons).  
```{r echo=FALSE,warning=FALSE}
plot_ly(x = menu$Category, y=menu$Sugars,color = menu$Category,colors =new_col , type = "bar") %>% layout(title = "Sugars",
         xaxis = list(title = ""),
         yaxis = list(title = ""),
         showlegend=FALSE,
         autosize = T)
```  
 
 
```{r echo=FALSE,warning=FALSE}
library(stringr)
ss<-menu%>%select(Category,Item,Sugars,Serving.Size,Sugars)%>%filter(Category=="Smoothies & Shakes")
ss$size<-NULL
ss$size[str_detect(ss$Item,"Small")]<-"Small"
ss$size[str_detect(ss$Item,"Medium")]<-"Medium"
ss$size[str_detect(ss$Item,"Large")]<-"Large"
ss$size[str_detect(ss$Item,"Snack")]<-"Snack"
ss%>%filter(!size == "Snack")%>%arrange(desc(Sugars))%>%ggplot(aes(x=factor(Item,level=Item),y=Sugars,group=size,fill=size))+
  geom_bar(stat="identity",position="dodge",alpha=0.7)+theme(axis.text.x = element_text(angle=90))+coord_flip()+
  labs(x="Item",title="Sugar content in Smoothies & Shakes")
```


### Cholestrol  

```{r}
menu %>% arrange(desc(Cholesterol,Cholesterol....Daily.Value.))%>%plot_ly( x = ~factor(Item,level=Item), y = ~Cholesterol, type="scatter",color=~Item, size=~Cholesterol,colors='Paired',mode = "markers" , marker=list( opacity=0.7) ) %>% 
      layout(title = "Cholesterol Content",
         xaxis = list(title = ""),
         yaxis = list(title = "Total Cholestrol"),
         showlegend=FALSE,autosize = T)
```
 
 

```{r echo=FALSE,warning=FALSE}
ch<-menu %>% select(Category,Item,Cholesterol)%>% arrange(desc(Cholesterol))%>%head(25)
p1<-plot_ly(ch, x=factor(ch$Item,level=ch$Item),y=ch$Cholesterol,color=ch$Category,type="bar")%>%layout(title="Cholestrol Rich Items",height=400)
ch1<-menu %>% select(Category,Item,Cholesterol)%>% arrange(desc(Cholesterol))%>%filter(Cholesterol >5 & Cholesterol<25)
p2<-plot_ly(ch1, x=factor(ch1$Item,level=ch1$Item),y=ch1$Cholesterol,color=ch1$Category,type="bar")%>%layout(title="Cholestrol Low Items",height=400)
ggplotly(p1)
ggplotly(p2)
```
  
  
### Dietary Fibre
  
Vegetables, Legumes, fruits are the high source of fibre content. So lets take items in all category  and find its dietary fibre content.  

```{r echo=FALSE,warning=FALSE}
menu %>%  
plot_ly( x = menu$Item, y = menu$Dietary.Fiber....Daily.Value., type="scatter", mode = "markers" , marker=list( color=colorRampPalette(brewer.pal(30,"Spectral"))(100) , opacity=0.7 , size=~Dietary.Fiber....Daily.Value.) ) %>% layout(title = "Dietry Fibre Daily Content ",
         xaxis = list(title = ""),
         yaxis = list(title = "Daily Dietary fibre"),
         showlegend=FALSE,autosize = F, width = 1000, height = 400,margin=m)

```   
 
#
All over the world, lets take most popular items in Mcdonald and see its nutritive values.  
```{r echo=FALSE}


menu %>% filter(Item %in% c("Egg McMuffin","Big Mac","Chicken McNuggets (10 piece)","Large French Fries","Baked Apple Pie","Double Cheeseburger"))%>% 
select(Item,Cholesterol....Daily.Value.,Sodium....Daily.Value.,Carbohydrates....Daily.Value.,Dietary.Fiber....Daily.Value.,Vitamin.A....Daily.Value.,
Calcium....Daily.Value.,Iron....Daily.Value.,Total.Fat....Daily.Value.,Saturated.Fat....Daily.Value.)%>%
gather(nut,value,2:10)%>%ggplot(aes(x="",y=value,fill=nut))+geom_bar(stat="identity",width=1)+
coord_polar(theta = "y", start=0)+facet_wrap(~Item)+theme(legend.position = "bottom",legend.text=element_text(size=5))+labs(title="Nutritive values in most popular items",fill="Nutrients")

```   
  
##Nutrients contributing to Calories  
```{r echo=FALSE}
g1<-menu%>%ggplot(aes(x=Cholesterol,y=Calories))+geom_point(col="hotpink")+geom_smooth(method="lm",col="hotpink")
g2<-menu%>%ggplot(aes(x=Carbohydrates,y=Calories))+geom_point(col="navyblue")+geom_smooth(method="lm",col="navyblue")
g3<-menu%>%ggplot(aes(x=Total.Fat,y=Calories))+geom_point(col="magenta")+geom_smooth(method="lm",col="magenta")
g3<-menu%>%ggplot(aes(x=Sugars,y=Calories))+geom_point(col="darkorchid4")+geom_smooth(method="lm",col="darkorchid4")
g4<-menu%>%ggplot(aes(x=Protein,y=Calories))+geom_point(col="firebrick4")+geom_smooth(method="lm",col="firebrick4")
g5<-menu%>%ggplot(aes(x=Sodium,y=Calories))+geom_point(col="olivedrab4")+geom_smooth(method="lm",col="olivedrab4")
g6<-menu%>%ggplot(aes(x=Saturated.Fat,y=Calories))+geom_point(col="orange4")+geom_smooth(method="lm",col="orange4")
g7<-menu%>%ggplot(aes(x=Dietary.Fiber,y=Calories))+geom_point(col="tomato4")+geom_smooth(method="lm",col="tomato4")
g8<-menu%>%ggplot(aes(x=Trans.Fat,y=Calories))+geom_point(col="slateblue4")+geom_smooth(method="lm",col="slateblue4")
grid.arrange(g1,g2,g3,g4,g5,g6,g7,g8,nrow=3,ncol=3)
``` 
  
Sugar,Carbohydrates,Protein,Saturated Fat & cholestrol are the nutrients contributing more to the calorie.

Protein and Fat content in category "Chicken & Fish"  

```{r echo=FALSE}
library(ggrepel)
menu %>% select(Category,Item,Protein,Total.Fat)%>%arrange(desc(Protein))%>%filter(Category =="Chicken & Fish")%>%ggplot(aes(x=Item,y=Protein,col=Item))+geom_point(size=3)+theme(legend.position = "none",axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+geom_label_repel(aes(label=substr(Item,1,20)),size=2)+labs(title="High Protein Item in Chicken & Fish Category")+geom_bar(aes(y=Total.Fat),alpha=0.5,stat="identity")
```  
  
  
Milk products is one of high source of calcium, let us find out what other items have calcium.  
  
```{r echo=FALSE}
options(repr.plot.width=10, repr.plot.height=50)
avg_cal<-mean(menu$Calcium....Daily.Value.)

cal<-menu %>% select(Category,Item,Calcium....Daily.Value.)%>%arrange(desc(Calcium....Daily.Value.))%>%filter(! Category %in% c("Coffee & Tea","Smoothies & Shakes","Beverages","Desserts") )
cal$level1<-ifelse(cal$Calcium....Daily.Value.>avg_cal,"Above Avg","Below Avg")
cal<-cal[order(cal$Calcium....Daily.Value.),]
#cal$Item<-substr(cal$Item,1,20)
ggplot(cal, aes(x=factor(cal$Item,levels=cal$Item), y=Calcium....Daily.Value.)) + 
  geom_bar(stat='identity',aes(fill=level1),width=.5)  +
    coord_flip()+
  scale_fill_manual(name="Calcium", 
                    labels = c("Above Average", "Below Average"), 
                    values = c("Above Avg"="red4", "Below Avg"="green4")) +labs(title="Daily calcium in other than milk Items",x="item")
```  





The average daily iron intake from foods and supplements is 13.7–15.1 mg/day in children aged 2–11 years, 16.3 mg/day in children and teens aged 12–19 years,  
and 19.3–20.5 mg/day in men and 17.0–18.9 mg/day in women older than 19.  
The median dietary iron intake in pregnant women is 14.7 mg/day.  
The upper limit -- the highest dose that can be taken safely -- is 45 mg a day.  
```{r echo=FALSE}
c<-16.3
m<-20.5
w<-18.9
menu %>% select(Category,Item,Iron....Daily.Value.)%>%arrange(desc(Iron....Daily.Value.))%>%filter(Iron....Daily.Value.>=15)%>%ggplot(aes(x=substr(Item,1,15),y=Iron....Daily.Value.,col=Category,size=Iron....Daily.Value.))+geom_point(fill="red")+theme(axis.text.x = element_text(angle=90),legend.position="bottom")+geom_hline(yintercept =c,col="red",linetype="dashed")+geom_text(aes( 0, c, label = "Children",vjust=-1,hjust=0,col="red"), size = 3)+geom_hline(yintercept =m,col="blue",linetype="dashed")+geom_text(aes( 0, m, label = "Men",vjust=-1,hjust=0), size = 3,col="blue")+geom_hline(yintercept =w,col="green",linetype="dashed")+geom_text(aes( 0, w, label = "Women",vjust=-0.5,hjust=0), size = 3,col="green")+labs(title="Mcdonald's Item -Daily requirement of Iron",x="Item")
```  
  
