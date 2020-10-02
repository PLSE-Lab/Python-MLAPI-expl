# I'm trying to identify things I do habitually when using R
# and then finding equivalent python syntax
# Hopefully these will be cute baby steps when I look back a couple months later

# import similar to require() or library()

import numpy as np      # vectors
import pandas as pd     # dataframes

import os               # These are just base R functions too.
                        # Feels like we cut all non-essential stuff for effeciency

print(os.listdir("../input"))   ## print(dir("../input"))

##deaths = read_csv("../input/character-deaths.csv")    # not allowed even though there's only 1 possible read_csv
                                                        # to me this is an engineer's mentality
                                                        # they're worried that 2 packages will have the same function
                                                        # and I'll get confused

deaths = pd.read_csv('../input/character-deaths.csv')   ## read.csv(deaths, stringsAsFactors=F)
                                                        # we have to type the package name every time
                                                        # thankfully you can set an alias with 'as pd'

deaths.sample(6)    ## deaths[sample(1:nrow(deaths), 6),]
                    # wow I like this

deaths.info()   ## Summary(deaths) 
                # except I don't get the summary stats

deaths.columns  ## colnames(deaths)
                # Why does it say 'index'?

deaths['Gender'].value_counts()     ## table(deaths$Gender)

pd.crosstab(deaths['Nobility'], deaths['Gender'])   ##table(deaths$Nobility, deaths$Gender)
                                                    # I'd expect deaths[['Nobility', 'GoT']].crosstab()
                                                    # I don't understand when to expect a function and when a property
                                                    # Is there a pattern to it?

deaths[['Gender', 'Death Year']].groupby(by='Gender').mean()    ## require(sqldf)
                                                                ## sqldf("SELECT Gender, avg(Death Year) avgdeathyear FROM deaths GROUP BY Gender", drv="SQLite")
                                                                # This seems extremely useful to me

deaths['Death Year'] = deaths['Death Year'].fillna(value=deaths['Death Year'].mean())  ## deaths[is.na(deaths$DeathYear),]$DeathYear <- mean(deaths$DeathYear, na.rm=T)

np.log(deaths['Death Year'] + 1)    ## log(deaths$DeathYear + 1, base=exp(1))
                                    # The numpy version has a different function for every logarithm base
                                    # I only really use base e anyway