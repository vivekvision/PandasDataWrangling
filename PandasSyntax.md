Pandas is short/ acronym for Panel Data Structure 

import pandas as pd
Import numpy as np


Following are used heavily in Pandas

np.array 
np.ndarray    -> nd means n dimensional -> which mean it can have arbitrary number of                                                                                  dimension 


Two fundamental pandas type

pd.Series()    ->   represents dataset with a single column 

pd.DataFrame()  -> table like format - multicolumn data 


Different data sources for Pandas 

Test files - CSV, JSON, HTML table

Binary files - Optimize I/O Performance 

Relational database - SQL Query 


Reading CSV in Dataframe 

import pandas as pd 

#read just 5 rows 

df = pd.read_csv(PATH, nrows=5)


# make specific column as index

df = pd.read_csv(PATH, nrows=5, index_col=’id’)

# in absence of index_col a default serial number is generated for index 



# read only specific columns 

df = pd.read_csv(PATH, nrows=5, index_col='id', usecols = ['id', 'art'])



Reading pipe separated file 

df = pd.read_csv('./data.txt',  sep='|' , engine = 'python')






