# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:35:22 2017

@author: JG
"""
#####################################
#Reading DataFrames from multiple files
######################################

# Import pandas
import pandas as pd

# Read 'Bronze.csv' into a DataFrame: bronze
bronze = pd.read_csv('Bronze.csv')

# Read 'Silver.csv' into a DataFrame: silver
silver = pd.read_csv('Silver.csv')

# Read 'Gold.csv' into a DataFrame: gold
gold = pd.read_csv('Gold.csv')

# Print the first five rows of gold
print(gold.head())

#########################################
#Sorting DataFrame with the Index & columns
#########################################

# Import pandas
import pandas as pd

# Read 'monthly_max_temp.csv' into a DataFrame: weather1
weather1 = pd.read_csv('monthly_max_temp.csv', index_col='Month')

# Print the head of weather1
print(weather1.head())

# Sort the index of weather1 in alphabetical order: weather2
weather2 = weather1.sort_index()

# Print the head of weather2
print(weather2.head())

# Sort the index of weather1 in reverse alphabetical order: weather3
weather3 = weather1.sort_index(ascending=False)

# Print the head of weather3
print(weather3.head())

# Sort weather1 numerically using the values of 'Max TemperatureF': weather4
weather4 = weather1.sort_values('Max TemperatureF')

# Print the head of weather4
print(weather4.head())

'''<script.py> output:
           Max TemperatureF
    Month                  
    Jan                  68
    Feb                  60
    Mar                  68
    Apr                  84
    May                  88
           Max TemperatureF
    Month                  
    Apr                  84
    Aug                  86
    Dec                  68
    Feb                  60
    Jan                  68
           Max TemperatureF
    Month                  
    Sep                  90
    Oct                  84
    Nov                  72
    May                  88
    Mar                  68
           Max TemperatureF
    Month                  
    Feb                  60
    Jan                  68
    Mar                  68
    Dec                  68
    Nov                  72'''

###############################
#Reindexing DataFrame from a list
###############################

weather1b = pd.read_csv('weather1.csv', header=None, names=['Month','Mean TemperatureF'], index_col=(0))

data=pd.ExcelFile('weather1.xlsx', index_col='Month')
weather1 = data.parse(0)

year = ['Jan',
 'Feb',
 'Mar',
 'Apr',
 'May',
 'Jun',
 'Jul',
 'Aug',
 'Sep',
 'Oct',
 'Nov',
 'Dec']

# Import pandas
import pandas as pd

# Reindex weather1 using the list year: weather2
weather2 = weather1b.reindex(year)

# Print weather2
print(weather2)

# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1b.reindex(year).ffill()

# Print weather3
print(weather3)

########################################
#Reindexing using another DataFrame Index
########################################

names_1981 = pd.read_csv('names_1981.csv', header=None, names=['name','gender','count'], index_col=(0,1))
names_1881 = pd.read_csv('names_1881.csv', header=None, names=['name','gender','count'], index_col=(0,1))

# Import pandas
import pandas as pd

# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)

# Print shape of common_names
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()

# Print shape of new common_names
print(common_names.shape)

###########################
#Adding unaligned DataFrames
###########################

january = pd.read_csv('january1.csv', header=0, names=['Company','Units'], index_col=0)
february = pd.read_csv('february1.csv', header=0, names=['Company','Units'], index_col=0)

total = january + february
total

###################################
#Broadcasting in arithmetic formulas
###################################

weather = pd.read_csv('weather.csv', index_col='Date', parse_dates=True)

# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5/9

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')

# Print first 5 rows of temps_c
print(temps_c.head())

###################################
#Computing percentage growth of GDP
###################################

import pandas as pd

# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP.csv', parse_dates=True, index_col='DATE')

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp['2008':]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change()*100

# Print yearly again
print(yearly)

#############################
#Converting currency of stocks
##############################

# Import pandas
import pandas as pd

# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', parse_dates=True, index_col='Date')
sp500.head()

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', parse_dates=True, index_col='Date')

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis='rows')

# Print the head of pounds
print(pounds.head())

#######################################
#Appending Series with nonunique Indices
#######################################

bronze = pd.read_csv('bronze.csv', index_col='Country')
silver = pd.read_csv('silver.csv', index_col='Country')

combined = bronze.append(silver)
combined

combined.loc['United States']

#######################
#Appending pandas Series
#######################

# Import pandas
import pandas as pd

# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('jan.csv', parse_dates=True, index_col='Date')

# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('feb.csv', parse_dates=True, index_col='Date')

# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('mar.csv', parse_dates=True, index_col='Date')

# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']

# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']

# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])

# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Compute & print total sales in quarter1
print(quarter1.sum())

##########################################
#Concatenating pandas Series along row axis
##########################################

# Initialize empty list: units
units = []

# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month['Units'])

# Concatenate the list: quarter1
quarter1 = pd.concat(units, axis='rows')

# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

#######################################
#Appending DataFrames with ignore_index
#######################################

names_1981 = pd.read_csv('names_1981.csv', header=1, names=['name','gender','count'])
names_1981.head()
names_1881 = pd.read_csv('names_1881.csv', header=1, names=['name','gender','count'])
names_1881.head()

# Add 'year' column to names_1881 and names_1981
names_1881['year'] = 1881
names_1981['year'] = 1981
names_1981.head()
names_1881.head()

# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981, ignore_index=True)
combined_names

# Print shapes of names_1981, names_1881, and combined_names
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)

# Print all rows that contain the name 'Morgan'
print(combined_names.loc[combined_names['name']=='Morgan'])

#################################################
#Concatenating pandas DataFrames along column axis
#################################################

weather_max = pd.read_csv('weather_max.csv', header=1, names=['Month','Max TemperatureF'])
weather_max
weather_mean = pd.read_csv('weather_mean.csv', header=1, names=['Month','Mean TemperatureF'])
weather_mean

# Concatenate weather_max and weather_mean horizontally: weather
weather = pd.concat([weather_max, weather_mean], axis=1)

# Print weather
print(weather)

###########################################
#Reading multiple files to build a DataFrame
###########################################

medals = []
medal_types = ['bronze', 'silver', 'gold']


for medal in medal_types:

    # Create the file name: file_name
    file_name = "%s_top5.csv" % medal
    
    # Create list of column names: columns
    columns = ['Country', medal]
    
    # Read file_name into a DataFrame: df
    medal_df = pd.read_csv(file_name, header=0, index_col='Country', names=columns)

    # Append medal_df to medals
    medals.append(medal_df)

# Concatenate medals horizontally: medals
medals = pd.concat(medals, axis='columns')

# Print medals
print(medals)

###########################################
#Reading multiple files to build a DataFrame
###########################################

gold_top5_1 = pd.read_csv('gold_top5_1.csv')
gold_top5_1
gold_top5_1.pivot(index='Country', columns='gold', values='Count')

# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)

# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])

# Print data about silver medals
print(medals_sorted.loc['silver'])

# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice

# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:,'United Kingdom'],:])

########################################
#Concatenating DataFrames with inner join
########################################

bronze = pd.read_csv('bronze.csv', header=0, index_col='Country')
bronze
silver = pd.read_csv('silver.csv', header=0, index_col='Country')
silver
gold = pd.read_csv('gold.csv', header=0, index_col='Country')
gold

# Create the list of DataFrames: medal_list
medal_list = [bronze, silver, gold]

# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list, keys=['bronze', 'silver', 'gold'], axis=1, join='inner')

# Print medals
print(medals)

#####################################################
#Resampling & concatenating DataFrames with inner join
#####################################################

china = pd.read_csv('china.csv', parse_dates=True, index_col='Year')
china
us = pd.read_csv('us.csv', header=0, parse_dates=True, index_col='Year')
us

# Resample and tidy china: china_annual
china_annual = china.resample('A').pct_change(10).dropna()
china_annual

# Resample and tidy us: us_annual
us_annual = us.resample('A').pct_change(10).dropna()

# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual, us_annual], join='inner', axis=1)

# Resample gdp and print
print(gdp.resample('10A').last())

'''<script.py> output:
                   China        US
    Year                          
    1971-12-31  0.988860  1.073188
    1981-12-31  0.972048  1.749631
    1991-12-31  0.962528  0.922811
    2001-12-31  2.492511  0.720398
    2011-12-31  4.623958  0.460947
    2021-12-31  3.789936  0.377506'''

############################    
#Merging on a specific column
#############################

revenue = pd.read_csv('revenue.csv')
revenue
managers = pd.read_csv('managers.csv')
managers

managers_1 = pd.read_csv('managers_1.csv')
managers_1

# Merge revenue with managers on 'city': merge_by_city
merge_by_city = pd.merge(revenue, managers, on='city')

# Print merge_by_city
print(merge_by_city)

# Merge revenue with managers on 'branch_id': merge_by_id
merge_by_id = pd.merge(revenue, managers, on='branch_id')

# Print merge_by_id
print(merge_by_id)

# Merge revenue with managers_1 on 'branch_id': merge_by_id
merge_by_id = pd.merge(revenue, managers_1, on='branch_id')

# Print merge_by_id
print(merge_by_id)

###########################################
#Merging on columns with non-matching labels
###########################################

managers_2 = pd.read_csv('managers_2.csv')
managers_2

# Merge revenue & managers on 'city' & 'branch': combined
combined = pd.merge(revenue, managers_2, left_on='city', right_on='branch')

# Print combined
print(combined)

###########################
#Merging on multiple columns
###########################

# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX','CO','IL','CA']

# Add 'state' column to managers: managers['state']
managers['state'] = ['TX','CO','CA','MO']

# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers, on=['branch_id', 'city', 'state'])

# Print combined
print(combined)

########################################
#Left & right merging on multiple columns
########################################

sales = pd.read_csv('sales.csv')
sales
managers
revenue

# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(revenue, sales, how='right', on=['city', 'state'])

# Print revenue_and_sales
print(revenue_and_sales)

# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales, managers_2, how='left', left_on=['city', 'state'], right_on=['branch', 'state'])

# Print sales_and_managers
print(sales_and_managers)

##################################
#Merging DataFrames with outer join
##################################

sales_and_managers = pd.read_csv('sales_and_managers.csv')
revenue_and_sales = pd.read_csv('revenue_and_sales.csv')

# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers, revenue_and_sales)

# Print merge_default
print(merge_default)

# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers, revenue_and_sales, how='outer')

# Print merge_outer
print(merge_outer)

# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers, revenue_and_sales, how='outer', on=['city','state'])

# Print merge_outer_on
print(merge_outer_on)

####################
#Using merge_ordered()
####################

austin = pd.read_csv('austin.csv')
houston = pd.read_csv('houston.csv')

# Perform the first ordered merge: tx_weather
tx_weather = pd.merge_ordered(austin, houston)

# Print tx_weather
print(tx_weather)

# Perform the second ordered merge: tx_weather_suff
tx_weather_suff = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus','_hus'])

# Print tx_weather_suff
print(tx_weather_suff)

# Perform the third ordered merge: tx_weather_ffill
tx_weather_ffill = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus','_hus'],fill_method='ffill')

# Print tx_weather_ffill
print(tx_weather_ffill)

##################
#Using merge_asof()
##################

auto = pd.read_csv('auto.csv', parse_dates=['yr']).sort_values('yr')
auto.head()
oil = pd.read_csv('oil.csv', parse_dates=['Date']).sort_values('Date')
oil.head()

# Merge auto and oil: merged
merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')

# Print the tail of merged
print(merged.tail())

# Resample merged: yearly
yearly = merged.resample('A', on='Date')[['mpg','Price']].mean()

# Print yearly
print(yearly)

# print yearly.corr()
print(yearly.corr())

