import gensim,csv,time
import pandas as pd
import regex as re
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from transformers import AutoModel,AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('../evalita/evalita_hodi/candidates.csv')

df_ = df.groupby('mention').wp_title.apply(list).reset_index()
model = AutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
tok = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

l = list()
for row in tqdm(df_.to_numpy()):
    ent = tok.encode_plus(row[0],return_tensors='pt')
    ent = model(ent['input_ids'],ent['attention_mask'])
    ent = ent[0].squeeze()[1:-1].mean(axis=0).detach().numpy()
    for wp in row[1]:
        wp_emb = tok.encode_plus(wp,return_tensors='pt')
        wp_emb = model(wp_emb['input_ids'],wp_emb['attention_mask'])
        wp_emb = wp_emb[0].squeeze()[1:-1].mean(axis=0).detach().numpy()

        sim = cosine_similarity([ent,wp_emb]).mean()
        l.append((row[0],wp,sim))
    time.sleep(0.5)

pd.DataFrame(l,columns=['mention','wp_title','cosine']).to_csv('../evalita/evalita_hodi/cosine.csv',index=False)

'''print(df)

df['keep'] = df.mention.apply(lambda x:1 if len(x.strip().split(' '))>1 else 0)

candidates = df.wp_title.to_list()

df = df[df.keep==1]

mentions = {x:re.sub(' ','_',x) for x in df.mention.to_numpy()}


train = pd.read_csv('../evalita/ACTI/subtaskA_train.csv')
ents = pd.read_csv('../evalita/ACTI/entities.csv')

train = train.merge(ents)
train = train.groupby(['Id','comment_text']).entity.apply(list).reset_index()

emb_l = list()
for row in train.to_numpy():
    mod = row[1]
    for ent in row[-1]:
        if ent in mentions:
            try:
                mod = re.sub(ent,mentions[ent],row[1])
            except:
                continue
    mod = ' '.join(mod.split("'"))
    mod = re.sub('(\?|!|\.|,|;|:|-|\(|\)|#|@)','',mod)
    emb_l.append(word_tokenize(mod.lower()))



file = open('../evalita/ACTI/wp_pages_lt.csv')
reader = csv.DictReader(file,fieldnames=['sentence'])

for row in tqdm(reader):
    row = ' '.join(row['sentence'].split("'"))
    row = re.sub('(\?|!|\.|,|;|:|-|\(|\))','',row)
    emb_l.append(word_tokenize(row.lower()))

model = Word2Vec(sentences=emb_l, vector_size=100, window=5, min_count=1, workers=4)

model.save('../evalita/ACTI/embedding.model')
'''
