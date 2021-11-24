import pandas as pd
import numpy as np
import os
import json
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from tqdm import tqdm

def gdl_dataset_builder(data_path, test_size, dataset_name, val_size=None, cv=None):    
    labels = ['fake', 'real']

    data_list = []
    for label in labels:
        for news_article in tqdm(os.listdir(os.path.join(data_path, label))):
            path_n = os.path.join(data_path, label, news_article)

            with open(os.path.join(path_n, f"{news_article}_text.txt"), encoding="utf-8") as f:
                text = f.read()
                if not text:
                    continue
            
            with open(os.path.join(path_n, f"{news_article}_graph.txt")) as f:
                edges = json.load(f)
            
            node_features = pd.read_csv(os.path.join(path_n, f"{news_article}_nf.csv"))
            node_features['type'] = node_features['type'].map({'tweet': 1, 'retweet':2})
            ids = node_features['id']
            node_features = node_features.drop(['id'], axis=1)
            ss = StandardScaler()
            node_features = pd.DataFrame(data=ss.fit_transform(node_features), columns=node_features.columns)
            node_features = node_features.to_dict(orient="records")



            g = nx.DiGraph(text=text)
            for id, i in zip(ids, node_features):
                g.add_node(str(id), **i)
            g.add_edges_from(edges)
            nx.set_node_attributes(g, {news_article: {k: 0 for k in i.keys()}})
            g = nx.convert_node_labels_to_integers(g)
            data = from_networkx(g, group_node_attrs=list(i.keys()))

            data.y = 1 if label == "real" else 0
            data.text = text
            data.news_id = news_article

            data_list.append(data)
    
    train_dataset, test_dataset = train_test_split(data_list, test_size=test_size, stratify=[i.y for i in data_list])
    print("Length of test_dataset: " + str(len(test_dataset)))

    dataset = {}
    news_ids = {}

    if cv and not val_size:
        print("Length of train_dataset: " + str(len(train_dataset)))

        kfolds = []
        kf = StratifiedKFold(n_splits=cv)
        for i,j in kf.split(train_dataset, [i.y for i in train_dataset]):
            kfolds.append((i.tolist(), j.tolist()))
        dataset['kfolds'] = kfolds
        dataset['test_dataset'] = test_dataset
        dataset['train_dataset'] = train_dataset

        news_ids['kfolds'] = kfolds
        news_ids['test_dataset'] = [i.news_id for i in test_dataset]
        news_ids['train_dataset'] = [i.news_id for i in train_dataset]

    else:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_size/(1-test_size), stratify=[i.y for i in train_dataset])
        print("Length of train_dataset: " + str(len(train_dataset)))
        print("Length of val_dataset: " + str(len(val_dataset)))
        dataset['train_dataset'] = train_dataset
        dataset['val_dataset'] = val_dataset
        dataset['test_dataset'] = test_dataset

        news_ids['test_dataset'] = [i.news_id for i in test_dataset]
        news_ids['train_dataset'] = [i.news_id for i in train_dataset]
        news_ids['val_dataset'] = [i.news_id for i in val_dataset]

    torch.save(dataset, f'{dataset_name}_gdl_dataset.pt')
    json.dump(news_ids, open(f"{dataset_name}_news_ids_dataset.json", 'w'))

gdl_dataset_builder(
    data_path="../preprocessed_data/politifact",
    test_size=0.20,
    cv = 10,
    dataset_name="politifact"
)

        

