# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:40:05 2017

@author: JG
"""
#######################
#Stacking & unstacking I
######################

# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level='weekday')

# Print the byweekday DataFrame
print(byweekday)

# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level='weekday'))

'''<script.py> output:
            visitors      signups    
    weekday      Mon  Sun     Mon Sun
    city                             
    Austin       326  139       3   7
    Dallas       456  237       5  12
                    visitors  signups
    city   weekday                   
    Austin Mon           326        3
           Sun           139        7
    Dallas Mon           456        5
           Sun           237       12'''

########################
#Stacking & unstacking II
########################

# Unstack users by 'city': bycity
bycity = users.unstack(level='city')

# Print the bycity DataFrame
print(bycity)

# Stack bycity by 'city' and print it
print(bycity.stack(level='city'))

'''<script.py> output:
            visitors        signups       
    city      Austin Dallas  Austin Dallas
    weekday                               
    Mon          326    456       3      5
    Sun          139    237       7     12
                    visitors  signups
    weekday city                     
    Mon     Austin       326        3
            Dallas       456        5
    Sun     Austin       139        7
            Dallas       237       12'''

#########################
#Restoring the index order
#########################

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0, 1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

'''<script.py> output:
                    visitors  signups
    city   weekday                   
    Austin Mon           326        3
    Dallas Mon           456        5
    Austin Sun           139        7
    Dallas Sun           237       12
                    visitors  signups
    city   weekday                   
    Austin Mon           326        3
           Sun           139        7
    Dallas Mon           456        5
           Sun           237       12'''

############################
#Adding names for readability
############################

# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = visitors_by_city_weekday.reset_index() 

# Print visitors_by_city_weekday
print(visitors_by_city_weekday)

# Melt visitors_by_city_weekday: visitors
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')

# Print visitors
print(visitors)

'''<script.py> output:
    city weekday  Austin  Dallas
    0        Mon     326     456
    1        Sun     139     237
      weekday    city  visitors
    0     Mon  Austin       326
    1     Sun  Austin       139
    2     Mon  Dallas       456
    3     Sun  Dallas       237''''

#######################
#Going from wide to long
#######################

# Melt users: skinny
skinny = pd.melt(users, id_vars=['weekday','city'])

# Print skinny
print(skinny)

Going from wide to long

# Melt users: skinny
skinny = pd.melt(users, id_vars=['weekday','city'])

# Print skinny
print(skinny)

'''<script.py> output:
      weekday    city  variable  value
    0     Sun  Austin  visitors    139
    1     Sun  Dallas  visitors    237
    2     Mon  Austin  visitors    326
    3     Mon  Dallas  visitors    456
    4     Sun  Austin   signups      7
    5     Sun  Dallas   signups     12
    6     Mon  Austin   signups      3
    7     Mon  Dallas   signups      5'''

#####################################
#Obtaining key-value pairs with melt()
#####################################

# Set the new index: users_idx
users_idx = users.set_index(['city','weekday'])

# Print the users_idx DataFrame
print(users_idx)

# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx, col_level=0)

# Print the key-value pairs
print(kv_pairs)

'''<script.py> output:
                    visitors  signups
    city   weekday                   
    Austin Sun           139        7
    Dallas Sun           237       12
    Austin Mon           326        3
    Dallas Mon           456        5
       variable  value
    0  visitors    139
    1  visitors    237
    2  visitors    326
    3  visitors    456
    4   signups      7
    5   signups     12
    6   signups      3
    7   signups      5'''

########################
#Setting up a pivot table
########################

# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(index='weekday', columns='city')

# Print by_city_day
print(by_city_day)

'''<script.py> output:
            visitors        signups       
    city      Austin Dallas  Austin Dallas
    weekday                               
    Mon          326    456       3      5
    Sun          139    237       7     12'''

########################################
#Using other aggregations in pivot tables
########################################

# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday', aggfunc='count')

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index='weekday', aggfunc=len)

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))

'''<script.py> output:
             city  signups  visitors
    weekday                         
    Mon         2        2         2
    Sun         2        2         2
    ==========================================
    True'''

############################################
#Using margins in pivot tables, givese totals
############################################

# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index='weekday', aggfunc=sum)

# Print signups_and_visitors
print(signups_and_visitors)

# Add in the margins: signups_and_visitors_total 
signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc=sum,margins=True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)

'''<script.py> output:
             signups  visitors
    weekday                   
    Mon            8       782
    Sun           19       376
             signups  visitors
    weekday                   
    Mon          8.0     782.0
    Sun         19.0     376.0
    All         27.0    1158.0'''

############################
#Grouping by multiple columns
############################

# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
count_by_class = by_class['survived'].count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked','pclass'])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()

'''script.py> output:
    pclass
    1    323
    2    277
    3    709
    Name: survived, dtype: int64
    embarked  pclass
    C         1         141
              2          28
              3         101
    Q         1           3
              2           7
              3         113
    S         1         177
              2         242
              3         495
    Name: survived, dtype: int64'''

##########################
#Grouping by another series
##########################

# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])

# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())

'''<script.py> output:
    region
    America                       74.037350
    East Asia & Pacific           73.405750
    Europe & Central Asia         75.656387
    Middle East & North Africa    72.805333
    South Asia                    68.189750
    Sub-Saharan Africa            57.575080
    Name: 2010, dtype: float64'''

################################################
#Computing multiple aggregates of multiple columns
################################################

# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max','median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:,('fare','median')])

'''<script.py> output:
    pclass
    1    80.0
    2    70.0
    3    74.0
    Name: (age, max), dtype: float64
    pclass
    1    60.0000
    2    15.0458
    3     8.0500
    Name: (fare, median), dtype: float64'''

##################################
#Aggregating on index levels/fields
##################################

# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv', index_col=['Year','region','Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level=['Year','region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated 
print(aggregated.tail(6))

'''script.py> output:
                                       population  child_mortality       gdp
    Year region                                                             
    2013 America                     9.629087e+08        17.745833   49634.0
         East Asia & Pacific         2.244209e+09        22.285714  134744.0
         Europe & Central Asia       8.968788e+08         9.831875   86418.0
         Middle East & North Africa  4.030504e+08        20.221500  128676.0
         South Asia                  1.701241e+09        46.287500   11469.0
         Sub-Saharan Africa          9.205996e+08        76.944490   32035.0'''

###################################
#Grouping on a function of the index
###################################

# Read file: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Create a groupby object: by_day
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum
units_sum = by_day['Units'].sum()

# Print units_sum
print(units_sum)

'''<script.py> output:
    Mon    48
    Sat     7
    Thu    59
    Tue    13
    Wed    48
    Name: Units, dtype: int64'''

################################
#Detecting outliers with Z-Scores
################################

# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)

"""<script.py> output:
                 fertility    life  population  child_mortality     gdp  \
    Country                                                               
    Guatemala        3.974  71.100  14388929.0             34.5  6849.0   
    Haiti            3.350  45.000   9993247.0            208.8  1518.0   
    Tajikistan       3.780  66.830   6878637.0             52.6  2110.0   
    Timor-Leste      6.237  65.952   1124355.0             63.8  1777.0   
    
                                region  
    Country                             
    Guatemala                  America  
    Haiti                      America  
    Tajikistan   Europe & Central Asia  
    Timor-Leste    East Asia & Pacific"""

###########################################
#Filling missing data (imputation) by group
###########################################

# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))

'''script.py> output:
          pclass  survived                                     name     sex   age  \
    1299       3         0                      Yasbeck, Mr. Antoni    male  27.0   
    1300       3         1  Yasbeck, Mrs. Antoni (Selini Alexander)  female  15.0   
    1301       3         0                     Youseff, Mr. Gerious    male  45.5   
    1302       3         0                        Yousif, Mr. Wazli    male  25.0   
    1303       3         0                    Yousseff, Mr. Gerious    male  25.0   
    1304       3         0                     Zabour, Miss. Hileni  female  14.5   
    1305       3         0                    Zabour, Miss. Thamine  female  22.0   
    1306       3         0                Zakarian, Mr. Mapriededer    male  26.5   
    1307       3         0                      Zakarian, Mr. Ortin    male  27.0   
    1308       3         0                       Zimmerman, Mr. Leo    male  29.0   
    
          sibsp  parch  ticket     fare cabin embarked boat   body home.dest  
    1299      1      0    2659  14.4542   NaN        C    C    NaN       NaN  
    1300      1      0    2659  14.4542   NaN        C  NaN    NaN       NaN  
    1301      0      0    2628   7.2250   NaN        C  NaN  312.0       NaN  
    1302      0      0    2647   7.2250   NaN        C  NaN    NaN       NaN  
    1303      0      0    2627  14.4583   NaN        C  NaN    NaN       NaN  
    1304      1      0    2665  14.4542   NaN        C  NaN  328.0       NaN  
    1305      1      0    2665  14.4542   NaN        C  NaN    NaN       NaN  
    1306      0      0    2656   7.2250   NaN        C  NaN  304.0       NaN  
    1307      0      0    2670   7.2250   NaN        C  NaN    NaN       NaN  
    1308      0      0  315082   7.8750   NaN        S  NaN    NaN       NaN'''

##################################
#Other transformations with .apply
##################################

def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})

# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])

#<script.py> output:
#                    regional spread(gdp)    z(gdp)
#    Country                                       
#    United States                47855.0  3.013374
#    United Kingdom               89037.0  0.572873
#    China                        96993.0 -0.432756
    
####################################
#Grouping and filtering with .apply()
####################################

def c_deck_survival(gr):

    c_passengers = gr['cabin'].str.startswith('C').fillna(False)

    return gr.loc[c_passengers, 'survived'].mean()

# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival and print the result
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)

'''<script.py> output:
    sex
    female    0.913043
    male      0.312500
    dtype: float64'''

#####################################
#Grouping and filtering with .filter()
#####################################

# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)

'''<script.py> output:
    Company
    Acme Coporation    34
    Hooli              30
    Initech            30
    Mediacore          45
    Streeplex          36
    Name: Units, dtype: int64
                           Company   Product  Units
    Date                                           
    2015-02-02 21:00:00  Mediacore  Hardware      9
    2015-02-04 15:30:00  Streeplex  Software     13
    2015-02-09 09:00:00  Streeplex   Service     19
    2015-02-09 13:00:00  Mediacore  Software      7
    2015-02-19 11:00:00  Mediacore  Hardware     16
    2015-02-19 16:00:00  Mediacore   Service     10
    2015-02-21 05:00:00  Mediacore  Software      3
    2015-02-26 09:00:00  Streeplex   Service      4'''

##################################    
#Filtering and grouping with .map()
##################################

# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10,'pclass'])['survived'].mean()
print(survived_mean_2)

'''<script.py> output:
    age
    over 10     0.366748
    under 10    0.609756
    Name: survived, dtype: float64
    age       pclass
    over 10   1         0.617555
              2         0.380392
              3         0.238897
    under 10  1         0.750000
              2         1.000000
              3         0.446429
    Name: survived, dtype: float64'''

#############################
#Case Study:  Summer Olympics
#############################

filename='C:/Users/Y/Documents/Python Scripts/Summer Olympic medallists 1896 to 2008 - ALL MEDALISTS.tsv'
medals=pd.read_csv(filename, sep='\t',skiprows=4)

#########################
#Grouping and aggregating

USA_edition_grouped = medals.loc[medals.NOC == 'USA'].groupby('Edition')
USA_edition_grouped['Medal'].count()

#############################################
#Using .pivot_table() to count medals by type

# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC', values='Athlete', columns='Medal', aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))

'''<script.py> output:
    Medal  Bronze    Gold  Silver  totals
    NOC                                  
    USA    1052.0  2088.0  1195.0  4335.0
    URS     584.0   838.0   627.0  2049.0
    GBR     505.0   498.0   591.0  1594.0
    FRA     475.0   378.0   461.0  1314.0
    ITA     374.0   460.0   394.0  1228.0
    GER     454.0   407.0   350.0  1211.0
    AUS     413.0   293.0   369.0  1075.0
    HUN     345.0   400.0   308.0  1053.0
    SWE     325.0   347.0   349.0  1021.0
    GDR     225.0   329.0   271.0   825.0
    NED     320.0   212.0   250.0   782.0
    JPN     270.0   206.0   228.0   704.0
    CHN     193.0   234.0   252.0   679.0
    RUS     240.0   192.0   206.0   638.0
    ROU     282.0   155.0   187.0   624.0'''

###########################    
#Applying .drop_duplicates()

# Select columns: ev_gen
ev_gen = medals[['Event_gender', 'Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)

'''<script.py> output:
          Event_gender Gender
    0                M    Men
    348              X    Men
    416              W  Women
    639              X  Women
    23675            W    Men'''
    
Finding possible errors with .groupby()

# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender', 'Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)

'''<script.py> output:
                          City  Edition  Sport  Discipline  Athlete    NOC  Event  \
    Event_gender Gender                                                             
    M            Men     20067    20067  20067       20067    20067  20067  20067   
    W            Men         1        1      1           1        1      1      1   
                 Women    7277     7277   7277        7277     7277   7277   7277   
    X            Men      1653     1653   1653        1653     1653   1653   1653   
                 Women     218      218    218         218      218    218    218   
    
                         Medal  
    Event_gender Gender         
    M            Men     20067  
    W            Men         1  
                 Women    7277  
    X            Men      1653  
                 Women     218'''
                 
########################
#Locating suspicious data

# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals[sus]

# Print suspect
print(suspect)

'''<script.py> output:
             City  Edition      Sport Discipline            Athlete  NOC Gender  \
    23675  Sydney     2000  Athletics  Athletics  CHEPCHUMBA, Joyce  KEN    Men   
    
              Event Event_gender   Medal  
    23675  marathon            W  Bronze'''
 
#############################################    
#Using .nunique() to rank by distinct sports

# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))

'''<script.py> output:
    NOC
    USA    34
    GBR    31
    FRA    28
    GER    26
    CHN    24
    AUS    22
    ESP    22
    CAN    22
    SWE    21
    URS    21
    ITA    21
    NED    20
    RUS    20
    JPN    20
    DEN    19
    Name: Sport, dtype: int64'''

#############################################
#Counting USA vs. USSR Cold War Olympic Sports

# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = (medals['Edition'] >= 1952) & (medals['Edition'] <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA', 'URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)

'''<script.py> output:
    NOC
    URS    21
    USA    20
    Name: Sport, dtype: int64'''

#############################################
#Counting USA vs. USSR Cold War Olympic Medals

# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition', columns='NOC', values='Athlete', aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_usr_medals
cold_war_usa_usr_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

# Create most_medals 
most_medals = cold_war_usa_usr_medals.idxmax(axis='columns')

# Print most_medals.value_counts()
print(most_medals.value_counts())

'''<script.py> output:
    URS    8
    USA    2
    dtype: int64'''
    
#################################################
#Visualizing USA Medal Counts by Edition: Line Plot

# Create the DataFrame: usa
usa = medals[medals.NOC=='USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()

#################################################
#Visualizing USA Medal Counts by Edition: Area Plot

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()

######################################################################
#Visualizing USA Medal Counts by Edition: Area Plot with Ordered Medals

# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values = medals.Medal, categories=['Bronze', 'Silver', 'Gold'],
                ordered=True)
medals.info()

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()




