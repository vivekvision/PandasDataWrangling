# Introduction
Pandas is short/ acronym for Panel Data Structure 

import pandas as pd
Import numpy as np


Following are used heavily in Pandas

np.array 
np.ndarray    

nd means n dimensional - it can have arbitrary number of dimension


Two fundamental pandas type

pd.Series()    ->   represents dataset with a single column 

pd.DataFrame()  -> table like format - multicolumn data 


Different data sources for Pandas 

Test files - CSV, JSON, HTML table

Binary files - Optimize I/O Performance 

Relational database - SQL Query 


# Reading CSV file into Pandas Dataframe 

import pandas as pd 

#read just 5 rows 

df = pd.read_csv(PATH, nrows=5)


#make specific column as index

df = pd.read_csv(PATH, nrows=5, index_col=’id’)

#in absence of index_col a default serial number is generated for index 



#read only specific columns 

df = pd.read_csv(PATH, nrows=5, index_col='id', usecols = ['id', 'art'])



Reading pipe separated file 

df = pd.read_csv('./data.txt',  sep='|' , engine = 'python')


# Read Excel File into Pandas dataframe

df = pd.read_excel("./fileName.xlsx")

#read a specific sheet


#read mulit-header & multi-index excel
df = pd.read_excel("./fileName.xlsx", header = [0,1,2,3,4,5,6], index_col=[0,1])


# Numpy array to Pandas dataframe

import numpy as np
import pandas as pd

my_array = np.array([[11,22,33],[44,55,66]])

df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

print(df)
print(type(df))


# Selecting Rows & Columns in dataframe

.iloc   -   selects rows & columns by numbers only  (0-based numbers)
.loc    -   selects rows & columns by label


#select all rows and only last column
df.iloc[ : , -1] 

#select all rows and all columns excluding last column
df.iloc[ : , : -1] 

#select all rows and 2nd column

df.iloc[ : , 1] 


# Filter on Pandas Data-frame 

df = df.loc[<condition>]


#Choose only data rows with column ‘Col1’ not null & remove all rows with column ‘Col1’ has null

df = df.loc[~df[‘Col1’].isnull()]


#Choose data rows with value of column ‘col1’ is equal to “constant1”

df = df.loc[df[‘Col1’]==”constant1”]


#Filter with multiple conditions 

df = df.loc[~( (df[‘Col1’] ==”Const1”) & (df[‘Col2’]==”Const2”) )]



# Select only Specific Columns

Select only columns ‘Col1’, ‘Col2’ and ‘Col3’ (Discard all other columns)

df = df[[‘Col1’,’Col2’,’Col3’]]


# New Column as Sum of two other column values 

df[‘NewColumn’] = df[‘Col1’] + df[‘Col2’]




# Select unique values of a column 

df[‘Col1’].unique()


# Join two data-frame

Import pandas as pd

Join two pandas data frame df1 and df2 on column ‘Key’ 


# Inner Join 

df = pd.merge(df1, df2, on=[‘Key’], how=’inner’)

Default type of join is ‘inner’

# Left Join 

df = pd.merge(df1, df2, on=[‘Key’], how =’left’)

column Key should be present in both data-frame and should be common data-type 
 
# Join on two Columns 

df = pd.merge(A_df, B_df, how=’left’,  left_on = [‘A_c1’, ’A_c2’], right_on = [‘B_c1’, ’B_c2’])


# Outer Join on Multiple columns

df = pd.merge(df1, df2, on=[‘col1’, ‘col2’, ‘col3’], how=’outer’)


# Apply function on multiple columns

df [[‘col1’,’col2’]] = df [[‘col1’,’col2’]] .apply()


# Divide numeric column values by a number

df [[‘col1’,’col2’]]  = df [[‘col1’,’col2’]] /10


# Apply lambda function 


df['identifier3'] = df.apply(lambda x: '_'.join([x['Make'], x['Model'], x['Origin']]), axis=1)


# First sort & then concatenate
df['identifier1'] = df.apply(lambda x: '_'.join(sorted({x.Make, x.Model, x.Origin})), axis=1)

df['identifier1'] = df.apply(lambda x: '_'.join(sorted([x.Make, x.Model, x.Origin])), axis=1)


{ } results in set
 
[ ]  results in list 



df['key_list'] = df.apply(lambda x: [x['Make'], x['Model'], x['Origin']], axis=1)

df['MSRP'] = df.apply(lambda x: str(x['MSRP']).replace('$', ''), axis=1)


# apply & eval function


eval() allows you to evaluate arbitrary Python expressions from a string-based or compiled-code-based input.

fruits["favorite_fruits"] = fruits["favorite_fruits"].apply(eval)

if 'favorite_fruits' column look like this: “[‘strawberry’, ‘apple’, ‘orange’]”, apply(eval) can get into separate columns of dataframe 


# Apply,  Apply Map & Map 

import pandas as pd
data = [(3,5,7), (2,4,6),(5,8,9)]
df = pd.DataFrame(data, columns = ['A','B','C'])
print(df)

# Use Pandas DataFrame apply() Function to Single Column
df["A"]=df["A"].apply(lambda x: x/10)
print(df)

# Use DataFrame.apply() method
df2 = df.apply(lambda x: x/10)
print(df2)

# Use Pandas DataFrame.applymap() method
df2 = df.applymap(lambda a: a*10)
print(df2)

# Use DataFrame.applymap() method
df2 = df.applymap(lambda a: str(a)+".00")
print(df2)


# Use  map() Method on series/ a column of dataframe 
df["Col 1"]=df["Col 1"].map(lambda x: x/100)
print(df)


# Column value isin list 

Select rows of data from the data-frame based on the condition:
Values of ‘column1’ is present in the list ‘ListOfKeys’


ListOfKeys = [‘437’, ’534’]

df[df[‘Col1’].isin(ListOfKeys)]


# Convert column value to Upper/Lower case

Convert all values of a column to upper case /lower case

df[‘Col1’] = df[‘Col1’].str.upper()

df[‘Col1’] = df[‘Col1’].str.lower()


# Convert all column names/headers to Upper/Lower case 

df.columns = df.columns.str.upper()
df.columns = df.columns.str.lower()


# Rename Column

df.rename(columns = {'OldColName' : 'NewColName'}, inplace=True)


# Renaming multiple columns:

df.rename(columns= {'OldCol1Name' : 'NewCol1Name', 'OldCol2Name': 'NewCol2Name'}, inplace=True)


# Remove space from a Column value

df1['employee_id'] = df1['employee_id'].str.replace(" ","" ,regex=True)

df['Invoice'] = df['Invoice'].str.replace(r'$', r'', regex=True)

 
# Remove leading and trailing space from a Column value

df1['State'] = df1['State'].str.strip()


# strip leading space
df1['State'] = df1['State'].str.lstrip()


# strip trailing space
df1['State'] = df1['State'].str.rstrip()



#Replace NaN with zero

df[‘col1’] = df[‘col1’].replace(np.nan, 0)


#Set a column as index 

df  = df.set_index(‘Col1’)

#revert back
df = df.reset_index()




#Convert data-frame to Dictionary


df has two columns ‘Col1’ and ‘Col2’ 

Col1 - will form keys (must be unique values)
Col2 - will form values for corresponding keys 

dict = df.set_index(‘Col1’)[‘Col2’].to_dict()
dict_keys = list(dict.keys())



# Drop Duplicate Rows

Syntax to drop duplicate rows from a data-frame 

df = df.drop_duplicate(subset=[‘Col1’,’Col2’], keep=’first’)




# Drop rows with NA

df = df.dropna(axis=0, thresh=1, how='any')

axis=0 -> rows
axis =1  -> columns 


df  = df.dropna(subset=['col1'], axis = 0)

Following is equivalent to above:

df.dropna(subset=['col1'], axis = 0, inplace = True)


# Drop columns with NA

df = df.dropna(axis=1) 

axis = 1 represents columns 


# Remove/Drop a column 

df.drop(x)  # x is index number, zero for first row - drop defaults to row and first row 

df = df.drop(‘column1’, axis=1)    # axis=1 represents column 

df = df.drop([‘col1’,’col2’], axis=1)

df.drop([‘col1’,’col2’], axis=1, inplace=True)



# Group by & Aggregate 

Group by column1, column2 & column3 and take aggregate of ColumnAmount by applying function sum


df = df.groupby([‘column1’, ’column2’, ’column3’], as_index=False).agg({‘ColumnAmount’:’sum’})



# Multiple column aggregates 

df = df.groupby([‘column1’,’column2’,’column3’], as_index=False).agg({‘ColumnAmount1’:’sum’, ‘ColumnAmount2’:’sum’, ’ColumnAmount3’:’mean’})




# Apply abstract on column value 

df[‘col1’] .abs()

abs(df[‘col1’])




# Stack two data-frame


#Stack data-frame df1 and data-frame df2
df = pd.concat([df1, df2], axis=0, ignore_index=True, sort=False)

#Befor concat fucntion call, ensure all column names are matching, each column must have matching data-type
#matching column names and column data-type is essential to avoid mess up in concat() function


# If else conditional on Column value 

Numpy function where and select can be used with pandas data-frame to perform data wrangling involving one or more conditions on column values


Create or Modify a column based on certain condition involving other columns 

import numpy as np
condition = ((df[‘column1’]==”Test1”) &(df[‘column2’]==”Test2”))
trueResult = df[‘column3’]
falseResult = df[‘column4’]

df[‘ColumnNew’] = np.where(condition1, trueResult, falseResult)

where function can be nested 

# Multiple Condition if else 

#Create or modify a column based on multiple conditions involving other columns 

# Put all conditions into list 
Conditions = [ (df[‘col1’]==”T1”), (df[‘col2’] > df[‘col3’]), (df[‘col4’]==”T3”) ]

# Put all corresponding results into list 
Results = [ (“R1”), (df[‘col5’]), (df[‘col6’]) ]

# Call the select function to evaluate condition and populate corresponding results 
df[‘ColumnNew’] = np.select(Conditions, Results, default=””)


# Lookup from data-frame

#Data-frame df1 has two columns ‘ColumnKey’ & ‘ColumnValue’
#Data-frame df2 has the primary data with a column ‘ColKey’

#Use case: Lookup & populate a new column named ‘ColVal’ in data-frame df2 from data-frame df1

#First step: create a dictionary from data-frame df1 

df1_dict = df1.set_index(‘ColumnKey’)[‘ColumnValue’].to_dict()


#Second step: lookup from dictionary 

df2[‘ColVal’] = df1[‘ColKey’].map(df1_dict)


# Ignore Pandas warnings 

import warnings
warnings.filterwarnings(“ignore”)



# Logging the messages 


# Basic logging 

import logging
logging.basicConfig(level=logging.INFO)
logging.info('Hello World')


# File logging 

# importing module
import logging
 
# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
#Creating an object
logger = logging.getLogger()
 
#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
 
#Test messages
logger.debug("Harmless debug Message")
logger.info("Just an information")
logger.warning("Its a Warning")
logger.error("Did you try to divide by zero")
logger.critical("Internet is down")


# Remove space/special character from column names

df.columns = df.columns.str.replace(“ “,””)
df.columns = df.columns.str.replace(“)”,””)

df.columns = df.columns.str.replace(“_”,””)
df.columns = df.columns.str.replace(“-”,””)


# Line break within a python code line 


Special char ‘\’ can be used to introduce line break within a python code line 

For example, following python code line 

df = pd.merge(df1, df2, how=’left’,  left_on=[‘x1’],  right_on=[‘x2’])


Can be written as by making use of special char \ in the code:

df = pd.merge(df1, df2, how=’left’, left_on=[‘x1’], \
                                                       right_on = [‘x2’])


# Null check on column value 
#Null check on column
df[‘Col1’].isnull()


# Filter out all null values of ‘Col1’

df = df.loc[~df[‘Col1’].isnull()]




# Case statement/ Case when structure 

#Create or Modify a column based on multiple conditions involving other columns 

#put all conditions in a list 
Conditions = [(df[‘col1’]==”T1”), (df[‘col2’] > df[‘col3’]), (df[‘col4’]==”T3”)]

#put all the corresponding results into list 
Results = [(“R1”), (df[‘col5’]), (df[‘col6’])]

# call the select function to evaluate condition and populate corresponding results 
df[‘ColumnNew’]  = np.select(Conditions, Results, default=””)

 
# Replace NaN with zero or empty 

import numpy as np

#Change NaN to zero only for numeric columns 

#For one column
df[‘col1’] = df[‘col1’].replace(np.nan, 0)

#For whole data-frame
df = df.replace(np.nan, 0)
#Replace NaN with empty string 

#For one column
df[‘col1’] = df[‘col1’].replace(np.nan, "")

#For whole data-frame 
df = df.replace(np.nan,"")


# Pivot & Melt/Unpivot in Pandas data-frame
 


Grouping & aggregation 
df = df.groupby([‘Col1’, ’Col2’, ’Col3’, ’Account’]).agg({‘Amount’:’sum’})

Pivot Account & Amount 
df = df.pivot_table(index=[‘Col1’, ’Col2’, ’Col3’], columns=”Account”, values=”Amount” ). reset_index()

df = df.rename_axis(None, axis=1, inplace=True)


Unpivot /melt

.melt() can be called to unpivot data-frame from wide format to long format 

import pandas as pd 

df

Name, Course, Age
“John”, “Masters”, 27
“Bob”, “Graduate”, 23

#Name is id_vars and course is value_vars
df = pd.melt(df, id_vars=[‘Name’], value_vars=[‘Course’])

#Multiple columns can be included in id_vars
df = pd.melt(df, id_vars=[‘Name’,’Age’], value_vars=[‘Course’])

 
# Compare float with Zero

np.isclose(df[‘col1’], 0.0)

#Check for zero value while performing division 
df[‘div’] = np.where( (np.isclose(df[‘col1’], 0.0)), np.nan, (df[‘col1’]/df[‘col2’]) )


# Converting to float dealing with empty column value

df[‘col1’]  = df[‘col1’].replace(“”, np.nan).astype(float).fillna(0.0)



# Converting to float dealing with comma in column value

df[‘col1’]  = df[‘col1’].str.replace(r’,’ , r’’ , regex=True)

df[‘col1’]  = df[‘col1’].astype(float).fillna(0.0)

df['Invoice'] = df['Invoice'].str.replace(r'$', r'', regex=True)
Fill NA

NA can be filled with:
Zero
Front Filling 
Back Filling
Average value 

df[‘col1’]  = df[‘col1’].isna().sum()   will return none zero if any NA is present

Python considers True to 1 and False to 0
df[‘col1’].isna() returns False if value is present  True if NA



# Front Filling 

df[‘col1’]  = df[‘col1’].fillna(method = “ffill”)


# Back Filling

df[‘col1’]  = df[‘col1’].fillna(method = “bfill”)

# Average value filling

df[‘col1’]  = df[‘col1’].fillna(value = df[‘col1’].mean())




# String Concatenation 


X = “hello” + “world” 


# Substring on Column 

df['col'] = df['col'].str[:9]


df['col'] = df['col'].str.slice(0, 9)


# Column names to lowercase 


df.columns  = df.columns.str.lower()

# Creating Dataframe in code 

import pandas as pd

df = pd.DataFrame({‘Name’:{0:”John”, 1:”Bob”},’Course’:{0:”Master”,0:”Graduate”},’Age’:{0:27,1:23}})


#Another style

data= {“Name:[“Tom”,”Nick”], “Age”:[20,21]}

df = pd.DataFrame(data)



# Create empty Data-frame with column names 

  
data = {'Name': [], 'Age': []}  
  
#Create DataFrame  
df = pd.DataFrame(data)  
  

#add rows to empty data-frame
df = df.append({‘Name’: “xyz”, ‘Age’: 23})
#Delete Columns

df.drop([‘col1’,’col2’,’col3’], axis=1, inplace=True)
# original df object changed when inplace=True 



# Delete Dataframe

del df



# If else expression 


Value_1 if condition else value_2

#Do not use with pandas column, instead use np.where()


# Datatype

#Imported data - often wrong data type is assigned to columns - must check & correct 

int64
float64
object  - string or text 

float and object type in dataframe can contain null values 


#display datatype of all columns 
df.dtypes 


#display datatype of one column ‘col1’
dataframe.col1.dtype


# Convert entire data-frame to string type 

df = df.astype(‘str’)


# Convert Column data-type 

df[‘col1’]  = df[‘col1’].astype(‘float’)

df[‘col2’]  = df[‘col2’].astype(‘int’)

df[‘col3’]  = df[‘col3’].astype(‘str’)



# Convert data-type of multiple columns

df [[‘col1’, ‘col2’, ‘col3’]] = df [[‘col1’, ‘col2’, ‘col3’]].astype(‘str’)


# Convert to Date

df[‘col2’] = pd.to_datetime(df[‘col2’])


df['just_date'] = df['dates'].dt.date

df['dateTypeValue'] = pd.to_datetime(df['yyyymmddString'], format='%Y%m%d')

# Size/ Row count 


len(df.index)
df.shape[0]


# Differencing with previous values


df['diff_1'] = df['Passengers'].diff(periods=1)
df['diff_2'] = df['Passengers'].diff(periods=2)



# Ignore warnings

import warnings
warnings.filterwarnings(‘ignore’)


# Sampling 


df_sample = df.sample(n=60, replace=True)



# Cumulative sum : Cumsum 


s = pd.Series([3, np.nan, 4, -5, 0])
s




0    3.0
1    NaN
2    4.0
3   -5.0
4    0.0


s.cumsum()




0    3.0
1    NaN
2    7.0
3    2.0
4    2.0
dtype: float64


By default, NA values are ignored





# Sort by a column
df = df.sort_values(‘date’)

df = df.sort_values(by=’date’, ascending=True)




 

# Regular expressions 

Check valid regex online 
https://regex101.com/r/2EaKua/1










# Debugging Tips 

#Display column schema 
df.info()

#Display all columns of a data-frame 

df.columns

df.columns.values

#Find out all the float columns 
float_columns = df.select_dtypes(include=[‘float’]).columns


x = df.columns.values
x.sort()
x

# Difference of two list 

list(set(df1.columns.values) - set(df2.schema.names))


# Check datatype of column

Display datatype of one column

dataframe.col1.dtype


For all columns:

dataframe.dtypes




# Get all Columns of Dataframe in  List

columnNames = df.columns.tolist()


# Display type of object

type(obj)


# Sort list elements

list.sort() - sorts the original list / modifies the order of elements in the list 
list.sort(reverse=True) - to sort elements from higher to lower 


sorted() function sorts given iterable object in specific order (either ascending or descending)

sorted(data)

sorted(iterable, key=None, reverse=False)

key (Optional) - A function that serves as a key for the sort comparison. Defaults to None


sorted(iterable, key=len)
len() is Python's in-built function to count the length of an object
The list is sorted based on the length of the element, from the lowest count to highest




# random list
random = [(2, 2), (3, 4), (4, 1), (1, 3)]

# sort list with key
sorted_list = sorted(random, key=lambda x: x[1])

# axis in pandas data-frame
axis=0 represents row 
axis =1 represents columns 





