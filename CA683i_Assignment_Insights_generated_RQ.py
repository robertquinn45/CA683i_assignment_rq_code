#######################
# Author: Robert Quinn
# CA683i Data Analytics Data Mining Assignment
#######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
import os #, sys
#nb_path = '/content/notebooks'
#os.symlink('/content/drive/My Drive/Colab Notebooks', nb_path)
#sys.path.insert(0,nb_path)
os.chdir("/content/drive/My Drive/CA683i_data_analytics_mining_assignment")
#appended_matched_data = []  # list to append dataframes at end of for loop
#appended_evaluation_data=[] # list to append dataframes at end of for loop
train_full_df = pd.read_csv('bank-additional-full.csv', sep=';')
test_full_df=pd.read_csv('bank-additional.csv', sep=';')
print('The shape of train_full_df: %d x %d' % train_full_df.shape)
#print(train_full_df.head)
print('The shape of test_full_df: %d x %d' % test_full_df.shape)
#print(test_full_df.head)
# merge both dataframe togheter for analysis of full customer base
df_full_cust_base=pd.concat([train_full_df, test_full_df])
print('The shape of df_full_cust_base: %d x %d' % df_full_cust_base.shape)
#print(df_full_cust_base.head)
print("Total customer targets is ",df_full_cust_base['y'].value_counts()['yes']) 
print("Total customer non-targets is ",df_full_cust_base['y'].value_counts()['no']) 
df_target = df_full_cust_base[(df_full_cust_base["y"] == 'yes')]
print('The shape of df_target: %d x %d' % df_target.shape)
df_non_target = df_full_cust_base[(df_full_cust_base["y"] == 'no')]
print('The shape of df_non_target: %d x %d' % df_non_target.shape)
print('The targets average age is:',df_target['age'].mean())
print('The targets median age is:',df_target['age'].median())
print('The targets mode age is: ',df_target['age'].mode())
print('The non-targets average age is:',df_non_target['age'].mean())
print('The non-targets median age is:',df_non_target['age'].median())
print('The non-targets mode age is: ',df_non_target['age'].mode())
print('The full customer base average age is:',df_full_cust_base['age'].mean())
print('The full customer base median age is:',df_full_cust_base['age'].median())
print('The full customer base mode age is: ',df_full_cust_base['age'].mode())
pd.set_option("display.max.columns", None) # make sure pandas doesn’t hide any columns
#pd.options.display.float_format = '{:.1f}'.format
print('Describe the full customer base')
print(df_full_cust_base.describe())
print('Describe the non-target base')
print(df_non_target.describe())
df_non_target.boxplot(column='age', sym='o', return_type='axes')
print('Describe the target base')
print(df_target.describe())
df_target.boxplot(column='age', sym='o', return_type='axes')
job_column = df_full_cust_base["job"]
type(job_column)
#job_column.plot(kind="hist")
# age boxplot and statistics
print('age statistics for targets')
df_target['age'].describe()
print('box plot for targets age')
df_target.boxplot(column='age', sym='o', return_type='axes')

print('age statistics for non-targets')
df_non_target['age'].describe()
print('box plot for non-targets age')
df_non_target.boxplot(column='age', sym='o', return_type='axes')

print('age statistics for full customer base')
df_full_cust_base['age'].describe()
print('box plot for full customer base age')
df_full_cust_base.boxplot(column='age', sym='o', return_type='axes')

# Education plots:
print('bar chart for targets education')
df_target['education'].value_counts().plot(kind='bar')

print('bar chart for non-targets education')
df_non_target['education'].value_counts().plot(kind='bar')

print('bar chart for full customer base education')
df_full_cust_base['education'].value_counts().plot(kind='bar')

# default plots:
print('bar chart for targets default')
df_target['default'].value_counts().plot(kind='bar')

print('bar chart for non-targets default')
df_non_target['default'].value_counts().plot(kind='bar')

print('bar chart for full customer base default')
df_full_cust_base['default'].value_counts().plot(kind='bar')

# housing loan plots
print('bar chart for targets housing')
df_target['housing'].value_counts().plot(kind='bar')

print('bar chart for non-targets housing')
df_non_target['housing'].value_counts().plot(kind='bar')

print('bar chart for full customer base housing')
df_full_cust_base['housing'].value_counts().plot(kind='bar')

# Personal loan plots
print('bar chart for targets loan')
df_target['loan'].value_counts().plot(kind='bar')

print('bar chart for non-targets loan')
df_non_target['loan'].value_counts().plot(kind='bar')

print('bar chart for full customer base loan')
df_full_cust_base['loan'].value_counts().plot(kind='bar')

# contact communication type (phone/cellular) plots
print('bar chart for targets contact')
df_target['contact'].value_counts().plot(kind='bar')

print('bar chart for non-targets contact')
df_non_target['contact'].value_counts().plot(kind='bar')

print('bar chart for full customer base contact')
df_full_cust_base['contact'].value_counts().plot(kind='bar')

# last contact month of year breakdown
print('bar chart for targets month')
df_target['month'].value_counts().plot(kind='bar')

print('bar chart for non-targets month')
df_non_target['month'].value_counts().plot(kind='bar')

print('bar chart for full customer base month')
df_full_cust_base['month'].value_counts().plot(kind='bar')

# last contact day of the week  breakdown
print('bar chart for targets day_of_week')
df_target['day_of_week'].value_counts().plot(kind='bar')

print('bar chart for non-targets day_of_week')
df_non_target['day_of_week'].value_counts().plot(kind='bar')

print('bar chart for full customer base day_of_week')
df_full_cust_base['day_of_week'].value_counts().plot(kind='bar')

# duration of last contact in seconds - boxplot and statistics - Not to be used for prediction purposes
print('duration of last contact in seconds statistics for targets')
df_target['duration'].describe()
print('box plot for targets duration of last contact in seconds')
df_target.boxplot(column='duration', sym='o', return_type='axes')

print('duration of last contact in seconds statistics for non-targets')
df_non_target['duration'].describe()
print('box plot for non-targets duration of last contact in seconds')
df_non_target.boxplot(column='duration', sym='o', return_type='axes')

print('duration of last contact in seconds statistics for full customer base')
df_full_cust_base['duration'].describe()
print('box plot for full customer base duration of last contact in seconds')
df_full_cust_base.boxplot(column='duration', sym='o', return_type='axes')

# campaign: number of contacts performed during this campaign and for this client
print('bar chart for number of contacts made to targets during this camapaign')
df_target['campaign'].value_counts().plot(kind='bar')

print('bar chart for number of contacts made to non-targets during this camapaign')
df_non_target['campaign'].value_counts().plot(kind='bar',figsize=(12, 4))  # widening the distcnae between points on the x axis for readability

# pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
print('bar chart for number of days that passed by after the targets were last contacted from a previous campaign')
df_target['pdays'].value_counts().plot(kind='bar',yticks = range(0,4001,200))

print('bar chart for number of days that passed by after the non-targets were last contacted from a previous campaign')
df_non_target['pdays'].value_counts().plot(kind='bar',yticks = range(0,40001,2000),figsize=(8, 8))

# previous: number of contacts performed before this campaign and for this non-target/target
print('bar chart for number of contacts performed before this campaign and for this target')
df_target['previous'].value_counts().plot(kind='bar',yticks = range(0,3501,250),figsize=(8, 6))
# grouping per unique "previous" vale
df_target.groupby(['previous']).size()

print('bar chart for number of contacts performed before this campaign and for this non-target')
df_non_target['previous'].value_counts().plot(kind='bar',yticks = range(0,36501,2000),figsize=(8, 6))

# poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
print('bar chart for outcome of the previous marketing campaign for targets')
df_target['poutcome'].value_counts().plot(kind='bar') #,yticks = range(0,3501,250),figsize=(8, 6))
# grouping per unique "poutcome" vale
df_target.groupby(['poutcome']).size()

print('bar chart for outcome of the previous marketing campaign for non-targets')
df_non_target['poutcome'].value_counts().plot(kind='bar') #,yticks = range(0,36501,2000),figsize=(8, 6))

# emp.var.rate: number of targets/non-targets per employment variation rate - quarterly indicator 

print('bar chart for number of targets per employment variation rate - quarterly indicator ')
df_target['emp.var.rate'].value_counts().plot(kind='bar')

plt.hist(df_target["emp.var.rate"])
plt.show()

# grouping per unique emp.var.rate value
df_target.groupby(['emp.var.rate']).size()

print('bar chart for number of non-targets per employment variation rate - quarterly indicator ')
df_non_target['emp.var.rate'].value_counts().plot(kind='bar') #,figsize=(12, 4))  # widening the distance between points on the x axis for readability

# cons.price.idx: number of targets/non-targets per consumer price index - monthly indicator

print('bar chart for number of targets per consumer price index - monthly indicator ')
# plotting a histogram
plt.hist(df_target["cons.price.idx"])
plt.show()

df_target['cons.price.idx'].value_counts().plot(kind='bar')
# grouping per unique cons.price.idx value
df_target.groupby(['cons.price.idx']).size()

print('bar chart for number of non-targets per consumer price index - monthly indicator')
df_non_target['cons.price.idx'].value_counts().plot(kind='bar') #,figsize=(12, 4))  # widening the distance between points on the x axis for readability

plt.hist(df_non_target["cons.price.idx"])
plt.show()

# cons.conf.idx consumer confidence index - monthly indicator
plt.hist(df_target["cons.conf.idx"])
plt.show()

plt.hist(df_non_target["cons.conf.idx"])
plt.show()

# euribor3m euribor 3 month rate - daily indicator
plt.hist(df_target["euribor3m"])
plt.show()

plt.hist(df_non_target["euribor3m"])
plt.show()

# nr.employed number of employees - quarterly indicator 
plt.hist(df_target["nr.employed"])
plt.show()

plt.hist(df_non_target["nr.employed"])
plt.show()