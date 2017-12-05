# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:37:51 2017

@author: JG
"""
#################################
#Loading Olympic edition DataFrame
#################################

#Import pandas
import pandas as pd

# Create file path: file_path
file_path = 'editions.csv'

# Load DataFrame from file_path: editions
editions = pd.read_csv(file_path)

# Extract the relevant columns: editions
editions = editions[['Edition', 'Grand Total', 'City', 'Country']]

# Print editions DataFrame
print(editions)

###########################
#Loading IOC codes DataFrame
###########################

# Import pandas
import pandas as pd

# Create the file path: file_path
file_path = 'ioc_codes.csv'

# Load DataFrame from file_path: ioc_codes
ioc_codes = pd.read_csv(file_path)

# Extract the relevant columns: ioc_codes
ioc_codes = ioc_codes[['Country', 'NOC']]

# Print first and last 5 rows of ioc_codes
print(ioc_codes.head())
print(ioc_codes.tail())

'''<script.py> output:
                  Athlete  NOC   Medal  Edition
    0       HAJOS, Alfred  HUN    Gold     1896
    1    HERSCHMANN, Otto  AUT  Silver     1896
    2   DRIVAS, Dimitrios  GRE  Bronze     1896
    3  MALOKINIS, Ioannis  GRE    Gold     1896
    4  CHASAPIS, Spiridon  GRE  Silver     1896
                        Athlete  NOC   Medal  Edition
    29211        ENGLICH, Mirko  GER  Silver     2008
    29212  MIZGAITIS, Mindaugas  LTU  Bronze     2008
    29213       PATRIKEEV, Yuri  ARM  Bronze     2008
    29214         LOPEZ, Mijain  CUB    Gold     2008
    29215        BAROEV, Khasan  RUS  Silver     2008'''
    
##################################################
#Counting medals by country/edition in a pivot table
##################################################

#import Excel file
data=pd.ExcelFile('medals.xlsx')
medals = data.parse(0)

# Construct the pivot_table: medal_counts
medal_counts = medals.pivot_table(index='Edition', values='Athlete', columns='NOC', aggfunc='count')

# Print the first & last 5 rows of medal_counts
print(medal_counts.head())
print(medal_counts.tail())

