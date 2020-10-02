import numpy as np
import pandas as pd
from ggplot import *

dataUsers = pd.read_csv("../input/Users.csv")
dataUsers.RegisterDate = dataUsers.RegisterDate.apply(lambda x: pd.to_datetime(x[0:10]))

def catTiers(x):
    if x == 0:
        return "Not Approved"
    elif x == 1:
        return "Not Approved"
    elif x == 2:
        return "Not Approved"
    elif x == 3:
        return "Novice"
    elif x == 4:
        return "Kaggler"
    elif x == 10:
        return "Master"
    else:
        return
    
dataUsers['Tier2'] = dataUsers['Tier']
dataUsers['Tier'] = dataUsers.Tier2.apply(lambda x : catTiers(x))

dataUsers = dataUsers[dataUsers.Tier != "Not Approved"]

A = ggplot(aes(x='RegisterDate',y = 'Points', color = 'Tier'), data=dataUsers)
A = A + geom_point() + scale_color_brewer(type = 'qual', palette = 3)

B = ggplot(aes(x='Id',y = 'Points', color = 'Tier'), data=dataUsers)
B = B + geom_point() + scale_color_brewer(type = 'qual', palette = 3)

C = ggplot(aes(x='RegisterDate',y = 'Id', color = 'Tier'), data=dataUsers)
C = C + geom_point()

ggsave(A, 'date_points.png')
ggsave(B, 'id_points.png')
ggsave(C, 'date_id.png')