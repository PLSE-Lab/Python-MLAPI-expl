#!/usr/bin/env python
# coding: utf-8

# In[ ]:


library(tidyverse)
library(jsonlite)
library(scales)
library(lubridate)
library(repr)
library(ggrepel)
library(gridExtra)
library(lightgbm)


# In[ ]:


train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")

