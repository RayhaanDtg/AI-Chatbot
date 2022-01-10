import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
    if i>=300000:
        break
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('qa_Electronics.json.gz')
df2=getDF('meta_Electronics.json.gz')
print(df.columns)
print(df2.columns)
df2_new=df2.loc[:,['category','title','asin','main_cat']]
print(df2_new.head())


df3=df.merge(df2_new,on=['asin'])
df3.to_pickle('dataset_final.pkl')
print(df3.head())

print(df3.loc[4500,['title','category','question','main_cat']])
