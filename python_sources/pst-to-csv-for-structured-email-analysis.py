# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:49:54 2018

@author: vivekkalyanarangan
"""

# Load the PST file in your Microsoft Outlook and then run this code...
import win32com.client
import win32com
import os
import pickle
import pandas as pd
outlook = win32com.client.Dispatch("outlook.Application").GetNameSpace("MAPI")

inbox = outlook.Folders("<name_of_account_folder>").Folders("Inbox").Folders("completed")

message = inbox.items
infolist = []
i=0
for message2 in message:
    try:
        Body = message2.Body
        CC = message2.CC
        ConversationID = message2.ConversationID
        ConversationIndex = message2.ConversationIndex
        ConversationTopic = message2.ConversationTopic
        CreationTime = str(message2.CreationTime)
        HTMLBody = message2.HTMLBody
        Importance = message2.Importance
        ReceivedByName = message2.ReceivedByName
        ReceivedOnBehalfOfName = message2.ReceivedOnBehalfOfName
        ReceivedTime = str(message2.ReceivedTime)
        Recipients = ','.join([str(i) for i in message2.Recipients])
        RTFBody = str(message2.RTFBody)
        Sender = str(message2.Sender)
        SenderEmailAddress = message2.SenderEmailAddress
        SenderName = message2.SenderName
        Subject = message2.Subject
        To = message2.To
        
        row = [Body, 
               CC, 
               ConversationID, 
               ConversationIndex, 
               ConversationTopic,
               CreationTime, 
               #HTMLBody, 
               Importance, 
               ReceivedByName, 
               ReceivedOnBehalfOfName,
               ReceivedTime, 
               Recipients, 
               #RTFBody, 
               Sender, 
               SenderEmailAddress, 
               SenderName, 
               Subject, 
               To
               ]
        infolist.append(row)
    except Exception as e:
        print(str(e))
        i+=1
        continue
    message2.Save
    message2.Close(0)
    #break
print(i)
df_email = pd.DataFrame(infolist, columns=['Body',
                                           'CC',
                                           'ConversationID',
                                           'ConversationIndex',
                                           'ConversationTopic',
                                           'CreationTime',
                                           #'HTMLBody',
                                           'Importance',
                                           'ReceivedByName',
                                           'ReceivedOnBehalfOfName',
                                           'ReceivedTime',
                                           'Recipients',
                                           #'RTFBody',
                                           'Sender',
                                           'SenderEmailAddress',
                                           'SenderName',
                                           'Subject',
                                           'To'
                                           ])
df_email.to_csv(r'/path/to/output.csv', index=False)