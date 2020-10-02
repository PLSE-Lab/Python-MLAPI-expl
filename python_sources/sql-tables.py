#!/usr/bin/env python
# coding: utf-8

# Following is the code I used to create a database in SQL Server that contains all the table for the NFL Punt competition.
# Listed here in case anyone else wants to use it. Must use SQL Server version 2017 or above.
# 
# -- creates database and puts in all nfl punt data
# -- using decimal type for GSISID as ngs-2016-post.csv has 
# --   a decimal point in that field
# -- version 1.0 - no indexes on any fields
# -- version 3.0 - decimal field correction, now decimal(12,8), was just decimal, which did not have any decimal places. oops.
# -- version 4.0 - added input for 2017 data, chanded event length for ngs to 30
# 
# USE master;
# GO
# 
# CREATE DATABASE KaggleNFLPunt3
# GO
# 
# USE KaggleNFLPunt3
# GO
# 
# CREATE TABLE game_data( 
#   GameKey        smallint, 
#   Season_Year    smallint, 
#   Season_Type    char(4), 
#   Week           tinyint, 
#   Game_Date      char(23), 
#   Game_Day       char(9), 
#   Game_Site      char(15), 
#   Start_Time     char(5), 
#   Home_Team      char(20), 
#   HomeTeamCode   char(3), 
#   Visit_Team     char(20), 
#   VisitTeamCode  char(3), 
#   Stadium        char(35), 
#   StadiumType    char(28), 
#   Turf           char(26), 
#   GameWeather    varchar(80), 
#   Temperature    char(5), 
#   OutdoorWeather varchar(80))
# 
# bulk insert game_data
# from 'c:\nfl\all\game_data.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE play_information( 
#   Season_Year          smallint,   
#   Season_Type          char(4),  
#   GameKey              smallint, 
#   Game_Date            char(23), 
#   Week                 tinyint, 
#   PlayID               smallint, 
#   Game_Clock           char(5), 
#   YardLine             char(6), 
#   Quarter              tinyint, 
#   Play_Type            char(4), 
#   Poss_Team            char(3), 
#   Home_Team_Visit_Team char(7), 
#   Score_Home_Visiting  char(7), 
#   PlayDescription      varchar(max))
# 
# bulk insert play_information
# from 'c:\nfl\all\play_information.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE player_punt_data( 
#   GSISID   int, 
#   Number   char(3), 
#   Position char(3))
# 
# bulk insert player_punt_data
# from 'c:\nfl\all\player_punt_data.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE video_review( 
#   Season_Year                      smallint,   
#   GameKey                          smallint,   
#   PlayID                           smallint, 
#   GSISID                           decimal,   
#   Player_Activity_Derived          char(8), 
#   Turnover_Related                 char(3), 
#   Primary_Impact_Type              char(16), 
#   Primary_Partner_GSISID           char(7), 
#   Primary_Partner_Activity_Derived char(8), 
#   Friendly_Fire                    char(7))
# 
# bulk insert video_review
# from 'c:\nfl\all\video_review.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE play_player_role_data( 
#   Season_Year smallint,   
#   GameKey     smallint,   
#   PlayID      smallint, 
#   GSISID      decimal,   
#   Role        char(4))
# 
# bulk insert play_player_role_data
# from 'c:\nfl\all\play_player_role_data.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE video_footage_control( 
#   season          smallint, 
#   Season_Type     char(4), 
#   Week            tinyint, 
#   Home_team       char(20), 
#   Visit_Team      char(20), 
#   Qtr             tinyint, 
#   PlayDescription varchar(max), 
#   gamekey         smallint, 
#   playid          smallint, 
#   Preview_Link    varchar(max))
# 
# bulk insert video_footage_control
# from 'c:\nfl\all\video_footage-control.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE video_footage_injury( 
#   season          smallint, 
#   Type            char(4), 
#   Week            tinyint, 
#   Home_team       char(20), 
#   Visit_Team      char(20), 
#   Qtr             tinyint, 
#   PlayDescription varchar(max), 
#   gamekey         smallint, 
#   playid          smallint, 
#   Preview_Link    varchar(max))
# 
# bulk insert video_footage_injury
# from 'c:\nfl\all\video_footage-injury.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# CREATE TABLE ngs( 
#   Season_Year smallint, 
#   GameKey     smallint, 
#   PlayID      smallint,  
#   GSISID      decimal, 
#   Time        char(23),
#   x           decimal, 
#   y           decimal, 
#   dis         decimal, 
#   o           decimal, 
#   dir         decimal, 
#   Event       char(30))
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2016-pre.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2016-reg-wk1-6.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2016-reg-wk7-12.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2016-reg-wk13-17.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2016-post.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2017-pre.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2017-reg-wk1-6.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2017-reg-wk7-12.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2017-reg-wk13-17.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# bulk insert ngs
# from 'c:\nfl\all\ngs-2017-post.csv'
# with (
#   FORMAT = 'CSV', 
#   FIELDQUOTE = '"',
#   FIRSTROW = 2,
#   FIELDTERMINATOR = ',',  --CSV field delimiter
#   ROWTERMINATOR = '0x0a',   --Use to shift the control to next row
#   TABLOCK
#   )
# 
# go
# 
# 
# 
