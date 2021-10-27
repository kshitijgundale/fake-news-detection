import pandas as pd
import os
from tqdm import tqdm

data_path = './preprocessed_data/politifact'

features = None
data = []
for label in ['fake', 'real']:
    for news in tqdm(os.listdir(os.path.join(data_path, label))):
        df = pd.read_csv(os.path.join(data_path, label, news, f'{news}_stats.csv'))
        if features is None:
            features = list(df['feature'])
            features.append('id')
        data.append(list(df['value']) + [news])

dr = pd.DataFrame(data=data, columns=features)
dr.to_csv('graph_features.csv', index=False)
        