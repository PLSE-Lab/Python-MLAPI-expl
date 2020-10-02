---
title: "Interactive Sales Visualizations with `dygraph`"
output: html_document
---

As a MATLAB user who transitioned to R for data science, one of the things I miss the most is fantastic interactive plotting. MATLAB lets you zoom and pan with ease, and multiple graphs can be stacked vertically with linked axes, so you can navigate multiple time series simultaneously. Recently I was pleasantly surprised to discover R's `dygraphs` package, which lets you do almost everything you can do in MATLAB. 

Let's have a look at Rossmann sales data using `dygraph`:

```{r, echo=F, warning=F, message=F}
# Data
## Load & clean
require(dplyr)
train = data.frame(read.csv('../input/train.csv',as.is=T),split='train')
test  = data.frame(read.csv('../input/test.csv' ,as.is=T),split='test')
dat = bind_rows(train,test) %>% arrange(Store,Date)
rm(train,test)
dat$Date = as.Date(dat$Date)
dat$Open = dat$Open==1
dat$Promo = dat$Promo==1
dat$SchoolHoliday = dat$SchoolHoliday==1
dat = dat %>% rename(
  sales=Sales,
  dayOfWeek=DayOfWeek,date=Date,open=Open,promo=Promo,
  schoolHoliday=SchoolHoliday,store=Store,stateHoliday=StateHoliday)

# Model
## Fit
train = dat[dat$split=='train',]
train = train[train$sales>0,]
preds=c('store','dayOfWeek','promo')
mdl = train %>% group_by_(.dots=preds) %>% summarise(predSales=median(sales)) %>% ungroup()
predict.dplyr = function(mdl,newdata) {
  x=(newdata %>% left_join(mdl,by=preds) %>% select(Id,predSales) %>% rename(sales=predSales))$sales
  x[is.na(x)]=0
  x
}
## Predict
dat$predSales=predict.dplyr(mdl,newdata=dat) 

# Graph
## Select a store 
s=1081
gd=dat$store==s 
x=dat[gd,]
## Find promo periods and holidays
gd=x$stateHoliday!='0'
sth = data.frame(date=x$date[gd],stateHoliday=x$stateHoliday[gd])
sth$color = plyr::mapvalues(sth$stateHoliday,from=c('a','b','c'),to=c('black','blue','red'))
promos = data.frame(
  start = x$date[x$promo-lag(x$promo,default=0)>0], 
  end = c(x$date[x$promo-lag(x$promo,default=0)<0],x$date[nrow(x)]))
## Make the graph
require(xts)
require(dygraphs)
dyEvents = function(x,date,label=NULL,labelLoc='bottom',color='black',strokePattern='dashed') {
  for (i in 1:length(date)) x = x %>% dyEvent(date[i],label[i],labelLoc,color[i],strokePattern)
  x
}
dyShadings = function(x,from,to,color="#EFEFEF") {
  for (i in 1:length(from)) x = x %>% dyShading(from[i],to[i],color)
  x
}
y=cbind(sales=xts(x$sales,x$date),
        predSales=xts(x$predSales,x$date))
dygraph(y, main = "Real & Pred Sales", group = "q", width=800) %>% 
  dySeries('sales', drawPoints = TRUE, color = 'blue') %>%
  dySeries('predSales', drawPoints = TRUE, color = 'green') %>%
  dyRoller(rollPeriod=1) %>%
  dyShadings(promos$start,promos$end) %>%
  dyEvents(sth$date,color=sth$color) %>% 
  dyRangeSelector(dateWindow=as.Date(c('2014-01-01','2014-06-01')))
```

This graph is initialized to show the first 6 months of 2014, but it contains the entire sales history of store 1081 along with the predictions of a simple model (median by store, dayOfWeek, and promo period). Promo periods are slightly shaded. State holidays are marked with dashed lines, color-coded to the value of StateHoliday. 

Here's how you interact with the graph:

* **Pan:** Drag the range selector at the bottom
* **Zoom horizontally:** click on the graph and drag left or right
* **Zoom vertically:** click on the graph and drag up or down
* **Zoom out:** double-click on graph window
* **Moving average:** enter a number of days in the lower left box

Try setting the moving average to 14 days: this removes the alternating promo/non-promo periods and shows slower trends, as well as the holiday behavior. 

Another great way to use `dygraph` is within the RStudio and Shiny ecosystem:

* Run the script that generates the graph above, and you'll get an interactive graph in the RStudio plot pane. 
* Want a GUI control to specify the store? Just stick dygraph into a Shiny app and run the app in RStudio. 
* Want more context and control? Use Shiny to make a full-featured dashboard summarizing your model's performance in a variety of ways, and allowing for as many break-outs, drill-downs, and zoom-ins as you have the patience to code.

(I'm currently working on that last one! ;) )
