import numpy as np
import pandas as pd


veriler = pd.read_csv("../input/A_Z Handwritten Data/A_Z Handwritten Data.csv")
import pandas as pd


A = veriler.loc[veriler['0'] == 0]
#D = veriler.loc[veriler['0'] == 3]
#S = veriler.loc[veriler['0'] == 18]
#W = veriler.loc[veriler['0'] == 22]

del(veriler)

#frames = [A, D, S, W]
#final_data = pd.concat(frames)

#final_data.to_csv('WASD_dataset.csv', index=False)
A.to_csv('WASD_dataset.csv', index=False)


#print(os.listdir("../input/A_Z Handwritten Data/A_Z Handwritten Data.csv"))