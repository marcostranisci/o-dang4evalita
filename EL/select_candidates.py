import pandas as pd
import numpy as np

def filter_names(a,b):
   if type(b) is not str:
       b= None
   elif len(b.split(' '))==2:
       b
   elif b==a.split(' ')[0]:
       b
   else: b = None
   return b

names = pd.read_csv('names.csv')
df = pd.read_csv('../evalita/PolitcIT/candidates.csv')
cos = pd.read_csv('../evalita/PolitcIT/cosine.csv')

df = df.merge(names[['name','gender']],left_on='mention',right_on='name',how ='left')

df['to_filter'] = df.apply(lambda x:filter_names(x.wp_title,x['name']),axis=1)
df = df[df.to_filter.isna()]

df = df.merge(cos)
df['ranking'] = df.ranking.apply(lambda x:1-(x/9))
df['avg'] = (df.ratio+df.ranking+df.cosine)/3
filtered = df.groupby('mention')['avg'].apply(list).reset_index()

filtered.avg = filtered.avg.apply(lambda x:max(x))

df = df.merge(filtered)

df = df[df.avg>0.8]

df.to_csv('../evalita/PolitcIT/filtered_candidates.csv',index=False)
