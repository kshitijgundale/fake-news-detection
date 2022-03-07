import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from dataset_builder.propagation_graph import PropagationGraphBuilder

v = SentimentIntensityAnalyzer()

DATASETS = [
    # {'source': 'politifact', 'label': 'fake'},
    # {'source': 'politifact', 'label': 'real'},
    # {'source': 'gossipcop', 'label': 'fake'},
    {'source': 'gossipcop', 'label': 'real'}
]


for i in DATASETS:
    data_path = f'./raw_data/{i["source"]}/{i["label"]}'
    save_path = f'./preprocessed_data_25h/{i["source"]}/{i["label"]}'

    for folder in tqdm(os.listdir(data_path)):
        news_path = os.path.join(data_path, folder)

        p = PropagationGraphBuilder(label=i['label'], news_id=folder, news_path=news_path, retweet_gap=3, vader=v, deadline=25*3600)

        l = p.build_edges(True, os.path.join(save_path, folder))
        if l:
            p.build_node_features(True, os.path.join(save_path, folder))
            p.graph_statistics(True, os.path.join(save_path, folder))