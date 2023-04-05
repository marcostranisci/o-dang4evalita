import pandas as pd
from bs4 import BeautifulSoup
import regex as re
import requests,spacy,csv
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

nlp = spacy.load('it_core_news_lg')

url = 'https://it.wikipedia.org/wiki/'
file = open('../evalita/ACTI/wp_pages_lt.csv', mode='w')
writer = csv.DictWriter(file,fieldnames=['sentence'])
writer.writeheader()
df = pd.read_csv('../evalita/ACTI/candidates.csv')

ents = list(set(df[df.ranking<5].wp_title.to_list()))


for ent in tqdm(ents):
    try:
        name = re.sub(' ','_',ent)
        ments = ent.split(' ')
        ments.append(name)
        req = requests.get('{}{}'.format(url,ent))
        if req.status_code==200:
            text = ''
            soup = BeautifulSoup(req.text, 'html.parser')
            for el in soup.find_all('p'):
                text += el.text
                text = re.sub(r'\[.*?\]', '', text)
                text = re.sub(ent,name,text)
                tok = sent_tokenize(text)
                tok_ments = list()
                for ment in ments:
                    tok_ments.append([x for x in tok if ment in x])
                tok_ments = list(set(list([x for y in tok_ments for x in y])))
                for sent in tok_ments:
                    try:
                        writer.writerow({'sentence':sent})
                    except Exception as e:print(e)
    except Exception as e:print(e)
