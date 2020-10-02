# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import csv

# %% [code]
# Create auctions.csv

with open('/kaggle/input/classic-world-of-warcraft-auction-data-raw/AuctionDB.lua') as file:
    elapsed = ''
    final = "scan-time,itemid,modifier,player,duration,quantity,minbid,buyout"
    for line in file:
        
        if line.startswith('			["elapsed"]'):
            elapsed = line.strip()
            elapsed = elapsed[14:]
            elapsed = elapsed[:-1]
        
        if line.startswith('			["data"]'):
            line = line.strip()
            line = line[13:]
            
            line = re.sub("0 i","0, i",line)
            line = re.sub("1 i","1, i",line)
            line = re.sub("2 i","2, i",line)
            line = re.sub("3 i","3, i",line)
            line = re.sub("4 i","4, i",line)
            line = re.sub("5 i","5, i",line)
            line = re.sub("6 i","6, i",line)
            line = re.sub("7 i","7, i",line)
            line = re.sub("8 i","8, i",line)
            line = re.sub("9 i","9, i",line)
            
            
            data = line.split(", i")
            
            for i in data:
                output = ''
                
                if i[-1:] == ",":
                     i = i + ","
                
                item = i.split("!")[0]
                
                players = i.split("!")
                for p in players[1:]:
                    
                    player = p.split("/")[0]
                    
                    auctions = p.split("/")[1].split("&")
                    for a in auctions:
                                            
                        if a[-1:] == ",":
                            a = a[:-1]
                        
                        if a[-2:] == '",':
                            a = a[:-3]
                    
                        if a.count(",") > 3:
                            a = a.split(",")[0] + (",") + a.split(",")[1] + (",") + a.split(",")[2] + (",") + a.split(",")[4] 
            
                        
                        if len(item.split("?")) > 1 :
                            output = output + "\n" + elapsed + "," + item.split("?")[0] + "," + item.split("?")[1] + "," + player + "," + a
                        else :
                            output = output + "\n" + elapsed + "," + item.split("?")[0] + "," + "" + "," + player + "," + a
            
                final = final + output

    File_object = open("auctions.csv","w")
    File_object.write(final)


# %% [code]
# Create items.csv
            
with open('/kaggle/input/classic-world-of-warcraft-auction-data-raw/AuctionDB.lua') as file:
    read = 1
    itemtext = ''
    

    for line in file:
        if line.startswith('	["itemDB_2"]'):
            read = 0
        if read == 0:
            
            itemmod = str(re.findall(r'\?.+?\"\] \=',line))
            itemmod = itemmod[3:]
            itemmod = itemmod[:-6]
            
            itemnum = str(re.findall(r'Hitem:.+?:',line))
            itemnum = itemnum[8:]
            itemnum = itemnum[:-3]

            itemdesc = str(re.findall('\|h\[.+?\]\|h\|r',line))
            itemdesc = itemdesc[5:]
            itemdesc = itemdesc[:-7]                

            if itemnum:
                itemline = (itemnum + ',' + itemmod + ',' + itemdesc + ',' + "https://classic.wowhead.com/item=" + itemnum)
                itemtext = (itemtext + itemline + "\n")

File_object = open("items.csv","w")
File_object.write("itemid,modifier,name,url\n" + itemtext)
    