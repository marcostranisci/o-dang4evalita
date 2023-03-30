import requests,yaml,sys,time
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

url = 'https://it.wikipedia.org/w/api.php'



f = open(sys.argv[1])
vars = yaml.load(f, Loader=yaml.FullLoader)


df = pd.read_csv(vars['data']['input_file'])

entities = pd.read_csv(vars['data']['entities_file'])

mentions = list(set([x[1] for x in df.to_numpy()]))
l = list()
for mention in tqdm(mentions):
    time.sleep(0.5)

    params = {'action':'query','srsearch':mention,'list':'search','srqiprofile':'engine_autoselect','format':'json'}
    try:
        req = requests.get(url,params=params,timeout=3)
        if req.status_code==200:
            for idx,res in enumerate(req.json()['query']['search']):
                ent = entities[entities.wp_title==res['title']]
                if len(ent)>0:
                    match = SequenceMatcher(None,mention,res['title']).ratio()
                    l.append((ent['id'].values[0],mention,res['title'],match,idx))
    except Exception as e:
        print(e)
        continue

pd.DataFrame(l,columns=['id','mention','wp_title','ratio','ranking']).to_csv(vars['data']['output_file'],index=False)
