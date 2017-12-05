# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:27:40 2017

@author: JG
"""
############
#Cheat sheet

#Inside pd.melt(), pass in the columns you do not wish to melt as a list to the id_vars parameter

#import panda

import pandas as pd
import numpy as np

#import Excel file
data=pd.ExcelFile('airquality_head2.xlsx')
airquality = data.parse(0)
airquality
airquality.head()


############################
#Reshaping your data using melt

# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])

# Print the head of airquality_melt
print(airquality_melt.head())

'''<script.py> output:
       Ozone  Solar.R  Wind  Temp  Month  Day
    0   41.0    190.0   7.4    67      5    1
    1   36.0    118.0   8.0    72      5    2
    2   12.0    149.0  12.6    74      5    3
    3   18.0    313.0  11.5    62      5    4
    4    NaN      NaN  14.3    56      5    5
       Month  Day variable  value
    0      5    1    Ozone   41.0
    1      5    2    Ozone   36.0
    2      5    3    Ozone   12.0
    3      5    4    Ozone   18.0
    4      5    5    Ozone    NaN'''

########################    
#Customizing melted data

# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())

###########
#Pivot data

# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())

##################################
#Resetting the index of a DataFrame

# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the new index of airquality_pivot
print(airquality_pivot.index)

# Print the head of airquality_pivot
print(airquality_pivot.head())

#########################
#Pivoting duplicate values

#import csv
data=pd.ExcelFile('airquality_dup.xlsx')
airquality_dup = data.parse(0)
airquality_dup.head()

# Melt airquality: airquality_melt
airquality_dup = pd.melt(airquality_dup, id_vars=['Month', 'Day'], var_name='measurement', 
                              value_name='reading')

# Pivot airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', 
                                              aggfunc=np.mean)

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())

############################
#Splitting a column with .str

#import csv
data=pd.ExcelFile('tb.xlsx')
tb = data.parse(0)
tb.head()

# Melt tb: tb_melt
tb_melt = pd.melt(tb, id_vars=['country', 'year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())

#import csv
data=pd.ExcelFile('ebola.xlsx')
eb = data.parse(0)
eb.head()

data=pd.ExcelFile('ebola_2.xlsx')
eb2 = data.parse(0)
eb2.head()

###########################################
#Splitting a column with .split() and .get()

import pandas as pd
import numpy as np

'''# Melt ebola: ebola_melt
ebola_melt = pd.melt(eb2, id_vars=['Date','value], var_name='type_country', value_name='counts')'''

# Create the 'str_split' column
eb2['str_split'] = eb2.type_country.str.split('_')
print(eb2.head())
print(ebola_melt.head())

# Create the 'type' column
eb2['type'] = eb2.str_split.str.get(0)

# Create the 'country' column
eb2['country'] = eb2.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())
print(eb2.head())

#######################
#Combining rows of data

data=pd.ExcelFile('uber-raw-data-apr14_100.xlsx')
ub_april = data.parse(0)
ub_april.head()

data=pd.ExcelFile('uber-raw-data-may14_100.xlsx')
ub_may = data.parse(0)
ub_may.head()

data=pd.ExcelFile('uber-raw-data-jun14_100.xlsx')
ub_jun = data.parse(0)
ub_jun.head()

uber1=ub_april
uber2=ub_may
uber3=ub_jun

# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1, uber2, uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())

##########################
#Combining columns of data

# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis=1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())

#<script.py> output:
#    (1952, 6)
#            Date  Day status_country  counts status country
#    0    1/5/2015  289   Cases_Guinea  2776.0  Cases  Guinea
#    1    1/4/2015  288   Cases_Guinea  2775.0  Cases  Guinea
#    2    1/3/2015  287   Cases_Guinea  2769.0  Cases  Guinea
#    3    1/2/2015  286   Cases_Guinea     NaN  Cases  Guinea
#    4  12/31/2014  284   Cases_Guinea  2730.0  Cases  Guinea

##################################
#Finding files that match a pattern

# Import necessary modules
import glob
import pandas as pd

# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())

#######################################
#Iterating and concatenating all matches

# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

###################
#1-to-1 data merge

data=pd.ExcelFile('site.xlsx')
site = data.parse(0)
site.head()

data=pd.ExcelFile('visited.xlsx')
visited = data.parse(0)
visited.head()

# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Print o2o
print(o2o)

####################
#Many-to-1 data merge

data=pd.ExcelFile('visited2.xlsx')
visited2 = data.parse(0)
visited2.head()

# Merge the DataFrames: m2o
m2o = pd.merge(left=site, right=visited2, left_on='name', right_on='site')

# Print m2o
print(m2o)

#######################
#Many-to-many data merge

data=pd.ExcelFile('survey.xlsx')
survey = data.parse(0)
survey.head()

# Merge site and visited: m2m
m2m = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Merge m2m and survey: m2m
m2m = pd.merge(left=m2m, right=survey, left_on='ident', right_on='taken')

# Print the first 20 lines of m2m
print(m2m.head(20))

#####################
#Converting data types

data=pd.ExcelFile('tips2.xlsx')
tip = data.parse(0)
tip.head()

# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())

#########################
#Working with numeric data

data=pd.ExcelFile('tips3.xlsx')
tips = data.parse(0)
tips.head()
tips.info()

# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')

# Print the info of tips
print(tips.info())

#######################################
#String parsing with regular expressions

# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result = prog.match('1123-456-7890')
print(bool(result))

########################################
#Extracting numerical values from strings

# Import the regular expression module
import re

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)

################
#Pattern matching

# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)

##############################
#Custom functions to clean data

data=pd.ExcelFile('tips2.xlsx')
tips = data.parse(0)
tips.head()
tips.info()

# Define recode_sex()
def recode_sex(sex_value):

    # Return 1 if sex_value is 'Male'
    if sex_value == 'Male':
        return 1
    
    # Return 0 if sex_value is 'Female'    
    elif sex_value == 'Female':
        return 0
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['sex_recode'] = tips.sex.apply(recode_sex)

# Print the first five rows of tips
print(tips.head())

################
#Lambda functions

import re

data=pd.ExcelFile('tips5.xlsx')
tips = data.parse(0)
tips.head()
tips.info()

# Convert the total_dollar column to type 'object'
tips.total_dollar = tips.total_dollar.astype('object')
tips.info()

# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('*', ''))
tips.head()

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())

#######################
#Dropping duplicate data

data=pd.ExcelFile('billboard.xlsx')
billboard = data.parse(0)
billboard.head()
billboard.info()

# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())

####################
#Filling missing data

# Print the info of airquality
print(airquality.info())
print(airquality.head())

# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)

# Print the info of airquality
print(airquality.info())
print(airquality.head())

# Calculate the mean of the Solar.R column: oz_mean
solar_mean = airquality['Solar.R'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Solar.R'] = airquality['Solar.R'].fillna(solar_mean)

# Print the info of airquality
print(airquality.info())
print(airquality.head())

##############################
#Testing your data with asserts

# Assert that there are no missing values
assert pd.notnull(eb).all().all()

# Assert that all values are >= 0
assert (eb >= 0).all().all()

####################################
#assumptions about the data are true?

'''Before continuing, however, it's important to make sure that the following assumptions about the data are true:

'Life expectancy' is the first column (index 0) of the DataFrame.
The other columns contain either null or numeric values.
The numeric values are all greater than or equal to 0.
There is only one instance of each country.'''

def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()[1:-1]
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert g1800s['Life expectancy'].value_counts()[0] == 1

####################
#Assembling your data

# Concatenate the DataFrames row-wise
gapminder = pd.concat([g1800s, g1900s, g2000s])

# Print the shape of gapminder
print(gapminder.shape)

# Print the head of gapminder
print(gapminder.head())

#<script.py> output:
#    (780, 218)
#        1800   1801   1802   1803   1804   1805   1806   1807   1808   1809  \
#    0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN   
#    1  28.21  28.20  28.19  28.18  28.17  28.16  28.15  28.14  28.13  28.12   
#    2    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN   
#    3  35.40  35.40  35.40  35.40  35.40  35.40  35.40  35.40  35.40  35.40   
#    4  28.82  28.82  28.82  28.82  28.82  28.82  28.82  28.82  28.82  28.82   
#    
#               ...            2008  2009  2010  2011  2012  2013  2014  2015  \
#    0          ...             NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
#    1          ...             NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
#    2          ...             NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
#    3          ...             NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
#    4          ...             NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   
#    
#       2016        Life expectancy  
#    0   NaN               Abkhazia  
#    1   NaN            Afghanistan  
#    2   NaN  Akrotiri and Dhekelia  
#    3   NaN                Albania  
#    4   NaN                Algeria  
#    
#    [5 rows x 218 columns]

###################
#Reshaping your data

# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder, id_vars='Life expectancy')

# Rename the columns
gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())

#######################
#Checking the data types

# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder.year)

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtypes == np.float64

############################
#Looking at country spellings

#Anchor the pattern to match exactly what you want by placing a ^ in the beginning and $ in the end.
#Use A-Za-z to match the set of lower and upper case letters, \. to match periods, and \s to match whitespace between words.
#Invert the mask by placing a ~ before it.

# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

###########################
#Assert & Drop missing data

# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.year).all()

# Drop the missing values
gapminder = gapminder.dropna(axis=0, how='any')

# Print the shape of gapminder
print(gapminder.shape)

#######################
#Visualization, Groupby

# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot(kind='hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')

