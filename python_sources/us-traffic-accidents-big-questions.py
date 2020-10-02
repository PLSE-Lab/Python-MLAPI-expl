#!/usr/bin/env python
# coding: utf-8

# # Designing Exploratory Analysis
# This week Rachael Tatman have started an awesome [5-day SQL Scavenge course](https://www.kaggle.com/rtatman/sql-scavenger-hunt-handbook) covering the very essentials of the Big Query and SQL techniques. It actually happens to be an introduction to the real world BigQuery datasets.
# Today we've quered US Traffic Fatality Record dataset. So, I was cooking my pasta with mushrooms and cheese and kept thinking: hey, it is a very comprehensive, detailed real data, right there, and it is very available. What can we do with it?
# 
# Ultimately we want to make impactful findigs, reveal correlations and reasons in order _"to help suggest solutions, and to help provide an objective basis to evaluate the effectiveness of motor vehicle safety standards and highway safety programs."_ That is why the system was created in the first place.

# **Where to start then?** - The best approach is to land a good basis - observe a wider picture. Only then we would be able to choose where to turn our attention to and narrow down the exploration.

# **The Big Questions**, or rather, first steps would be:
# 
# 1) See traffic fatality trend over time since 1975. We can presume, the number of accidents would increase, due to carpool and population growth. Hense it is important to consider:
#  - number of cars
#  - overall across US and per each State
#     Basically a plot of fatal accidents/car number per each state over time should give a good picture.
#     
# Here is a sketch of how it might look like:
#     
# <img src="https://image.prntscr.com/image/sqPpGKnfRhiuU3tDQGNfpQ.png" alt="Traffic Fatalities Trend per state" style="width: 500px;"/>    
# 
# 2) Then we can choose a slice (a year) from the plot 1. and project it on a map. That might reveal any geographical patterns. For example, what if find out that northen states have less accidents, which could be a strong clue for further exploration.
#     
#      The map projection could look like this:
# <img src="https://image.prntscr.com/image/0BFuvzKYRKmMvMGTFcjy3Q.png" alt="Vehicle Fatal accidents per state" style="width: 600px;"/>
# 
# 3)* As a bonus a plot can be interactive, projecting data on a map as we move a slice across plot 1. That'd be cool.

# Having those visuals at hand we can quickly distinguish any otliers (for example, some states having very high/low accident trend), start deepening the search from there, and hopefully find the explanations.

# # Designing Exploratory Analysis

# Learning more from the available dataset, turns out we have data only for 2016, 2017 years, so we can't see trends over time. Also, overall number of cars not presented.
# 
# Which already narrows our exploration. So, we can dive straight into more specific questions. Most of the questions would involve slicing data by different groups and see which of those have strong correlation to the fatalities trend.
# 
# 1. Understand overall National Trend since 1975 would be great. We can use [report from NHTSA page](https://cdan.nhtsa.gov/SASStoredProcess/guest)
# 2. It is yet makes sense to project overall data as chloropleth map.
# 
# using accident_{year} table
# - Slice the data by the road type: route_signing and route_signing_name of accident_{year} table
# - Slice the data by the intersection: type_of_intersection of accident_{year} table
# - Was it a workzone (construction site)? work_zone of accident_{year} table
# - By light condition: light_condition, light_condition_name of accident_{year} table
# - By Atomospheric condition: atmospheric_conditions_1_name of accident_{year} table
# - By hour of the day, by week day, by month of the year - all from timestamp_of_crash of accident_{year} table
# - Drunk driver involvement: number_of_drunk_drivers of accident_{year} table
# - See by state number of persons not in a vehicle involved: number_of_persons_not_in_motor_vehicles_in_transport_mvitof accident_{year} table by state
# 
# nmcrash_{year} table
# - For events involving non-motorists, find out contribution of non-vehicle person actions to the event: non_motorist_contributing_circumstances, non_motorist_contributing_circumstances_name
#     
#     
# using distract_{year} table
# - See the distraction inputs recorded prior to the event: driver_distracted_by, driver_distracted_by_name of distract_{year} table
# 
# using drimpair_{year} table
# - By driver impairment: condition_impairment_at_time_of_crash_driver, condition_impairment_at_time_of_crash_driver_name of drimpair_{year} table
# 
# using factor_{year} table
# - See contribution of possible defects and maintenance conditions of the vehicles: contributing_circumstances_motor_vehicle, contributing_circumstances_motor_vehicle_name
#     
# maneuver_{year} table
# - By what was on the road, that driver tried to avoid prior the event: driver_maneuvered_to_avoid, driver_maneuvered_to_avoid_name
# 
# nmcrash_{year} table
# - For events involving non-motorists, find out contribution of non-vehicle person actions to the event: non_motorist_contributing_circumstances, non_motorist_contributing_circumstances_name
#     
# nmimpair_{year} table
# - Any physical impairments of non motorists? condition_impairment_at_time_of_crash_non_motorist, condition_impairment_at_time_of_crash_non_motorist_name
# 
# parkwork_{year} table
# - parked and working vehicles involved. See fraction of such vehicles to the whole. unit_type column
# - Possibly explore models of the vehicles, if the previous fraction large enough.
# 
# pbtype_{year} table
# - By a pedestrian/byciclist type
# 
# person_{year} table
# - Slice by age, sex, race involvement into the event
# - See who was a driver by age, sex, race
# - See slice by alcohol
# - See slice by drugs
# - See fraction of misuse of the restraint system: indication_of_misuse_of_restraint_system_helmet
#     
# safetyeq_{year} table
# - For non-motorists involved into the event, slice by the safetyequipment used: non_motorist_safety_equipment_use
# 
# vehicle_{year} table
# - Slice by vehicle manufacturer: vehicle_make, vehicle_make_name
# - Slice by body type: body_type, body_type_name
# - Slice by the year manufactured: vehicle_model_year
# - Fraction of speed related: speeding_related
#     - Slice by speed: travel_speed
# - a fraction of any prev moving violations or convictions: previous_other_moving_violation_convictions
# - Slice by type of traffic: trafficway_description
# - Slice by road way surface: roadway_surface_type
# - Slice if the driver was drinking: driver_drinking (coincides with 21)
#     
# vindecode_{year} table
# - by veicle type: vehicle_type
# - Slices by the type of systems used in the vehicles.
# 
# vision_{year} table
# - Slice by vision impediments: drivers_vision_obscured_by

# Some of the slices above would be informative only if we compare them to the larger picture, like overall traffic data, or events not necessary fatal. For example: when we slice the traffic fatalities data by sex, age or race, we would need to compare it to the slice of drivers overall nationally, not necessarily involved in the accident. Or, if we compare certain systems, like brakes in a car, we would want to compare it with overall accident data (both fatal and non fatal). Only then we would be able to define causation and say: hey this types of brakes are dangerous, or this group of people cause more accidents.
