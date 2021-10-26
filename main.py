import os
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from propagation_graph import PropagationGraphBuilder
from tqdm import tqdm

v = SentimentIntensityAnalyzer()

DATASETS = [
    # {'source': 'politifact', 'label': 'fake'},
    {'source': 'politifact', 'label': 'real'},
    # {'source': 'gossipcop', 'label': 'fake'},
    # {'source': 'gossipcop', 'label': 'real'}
]


for i in DATASETS:
    data_path = f'./data/{i["source"]}/{i["label"]}'
    save_path = f'./preprocessed_data/{i["source"]}/{i["label"]}'

    for folder in tqdm(os.listdir(data_path)):
        news_path = os.path.join(data_path, folder)

        p = PropagationGraphBuilder(label='fake', news_id=folder, news_path=news_path, retweet_gap=3, vader=v)

        l = p.build_edges(True, os.path.join(save_path, folder))
        if l:
            p.build_node_features(True, os.path.join(save_path, folder))
            p.graph_statistics(True, os.path.join(save_path, folder))
