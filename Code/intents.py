
import pandas as pd

#read dataset into dataframe from pickle file
df=pd.read_pickle('Code/required_dataset.pkl')
# printing the unique categories in the dataset
print(df.main_cat.unique())

rslt=df[df['main_cat']=='All Electronics']
print(rslt.loc[0:5,['question','answer']])
print(rslt.main_cat.unique())