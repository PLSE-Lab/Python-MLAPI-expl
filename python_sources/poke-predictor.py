import pandas as pd;
import numpy as np;
import pickle as pickle;
import seaborn as sns;
import matplotlib.pyplot as plt;
import matplotlib as mtl;

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


data_table=pd.read_csv("../input/pokemon.csv")
data_table_copy=data_table;

type_vals=pd.read_csv(".../input/Type_table.csv");

#get fighter IDs
Fighter_Ids=pd.read_csv("../input/combats.csv")
Fighter_Idscopy=Fighter_Ids;
combatTrial=[];
count=0;



#combatTrial=[151,231]

#Get combatants Data
def Model(j):
 combatents_Data=pd.DataFrame();
 for x in j:
  for c in data_table_copy["#"]:
   if x==c:
    combatents_Data=combatents_Data.append(data_table_copy[data_table_copy["#"]==x]);
   else:
    continue
 return combatents_Data


#print(c)

#calculat and decide victor
def Score():
 values=[50,10,25,8,15,6];
 divisonlst=[0,1,2,3,4,5]
 c1=[];
 c2=[];
 division1=[];
 division2=[];
 score1=0;
 score2=0;
 score={};
 count=0;
 type1=[];
 type2=[];
 Spatkpts1=0;
 spdefpts1=0;
 Spatkpts2=0;
 spdefpts2=0;
 for T in c["Type 1"]:
  if count==0:
   type1.append(H);
   count=1;
  elif count==1:
   type2.append(H);
   count=0;
 for x in zip(type1,type2):
  typeValue1=type_vals.get_value(index=x[0],col=x[1]);
  Spatkpts1=((typeValue1/184)*100);
  spdefpts1=((typeValue1/210)*100);
 for x in zip(type2,type1):
  typeValue2=type_vals.get_value(index=x[0],col=x[1]);
  Spatkpts2=((typeValue2/184)*100);
  spdefpts2=((typeValue2/210)*100); 
 for H in c["HP"]:
  if count==0:
   c1.append(H);
   count=1;
  elif count==1:
   c2.append(H);
   count=0;
 for A in c["Attack"]:
  if count==0:
   c1.append(A);
   count=1;
  elif count==1:
   c2.append(A);
   count=0;
 for D in c["Defense"]:
  if count==0:
   c1.append(D);
   count=1;
  elif count==1:
   c2.append(D);
   count=0;
 for SpA in c["Sp. Atk"]:
  if count==0:
   c1.append((SpA+Spatkpts1));
   count=1;
  elif count==1:
   c2.append((SpA+Spatkpts2));
   count=0;
 for SpD in c["Sp. Def"]:
  if count==0:
   c1.append((SpD+spdefpts1));
   count=1;
  elif count==1:
   c2.append((SpD+spdefpts2));
   count=0;
 for SPD in c["Speed"]:
  if count==0:
   c1.append(SPD);
   count=1;
  elif count==1:
   c2.append(SPD);
   count=0;
 #print (c1," ",c2)
 for num in divisonlst:
    r=((c1[num]*2)/values[num]);
    #print(r,"R")
    division1.append(r);
 for num in divisonlst:
    p=((c2[num]*2)/values[num]);
    #print(p,"P")
    division2.append(p);
    #print(x)
 for d1 in division1:
   score1=score1+d1;
   score["score1"]=score1;
 for d2 in division2:
   score2=score2+d2;
   score["score2"]=score2;
 if score["score1"]>score["score2"]:
    return 0
 elif score["score2"]>score["score1"]:
    return 1    
 elif score["score2"]==score["score1"]:
    return "Draw"
 #print(c1,"c1"," ",c2,"c2") 
  

drawdata=pd.DataFrame();
mismatch=pd.DataFrame();
percentage=0;
for row in Fighter_Idscopy.itertuples():
 del combatTrial[:];
 count=count+1;
 fighter1=row[1];
 fighter2=row[2];
 combatTrial.append(fighter1)
 combatTrial.append(fighter2)
 c=Model(combatTrial);
 d=Score();
 if d=="Draw":
    drawdata=drawdata.append(c)
 if d!="Draw":
  winner=combatTrial[d];
  if winner==row[3]:
     #print("Its a match winner is ", winner);
     percentage=percentage+1;
  else:
     mismatch=mismatch.append(c);
 if count==5000:
   break
   
accuracy=((100*percentage)/5000);
print(accuracy," % acurate");
#print(drawdata.head(),"draw data")
#print(mismatch.head(),"Mismatch")
csv_out1=pd.DataFrame.to_csv(drawdata,"Drawdata.csv");
csv_out2=pd.DataFrame.to_csv(mismatch,"Mismatch.csv"); 
#print(combatTrial)