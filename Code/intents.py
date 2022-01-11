
import pandas as pd

#read dataset into dataframe from pickle file
df=pd.read_pickle('Code/required_dataset.pkl')
# printing the unique categories in the dataset
print(df.main_cat.unique())
print(len(df))
rslt=df.loc[df['main_cat']=="Computers"]
print(rslt.loc[:,['question','answer']])
print(len(rslt))