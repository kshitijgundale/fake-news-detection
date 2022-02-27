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
from torch_geometric.data import Data

class PairData(Data):
    def __init__(self, edge_index_h, x_h, edge_index_f, x_f, y, text, news_id):
        super().__init__()
        self.edge_index_h = edge_index_h
        self.x_h = x_h
        self.edge_index_f = edge_index_f
        self.x_f = x_f
        self.y = y
        self.text = text
        self.news_id = news_id
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

def pair_gdl_dataset_builder(f_data_path, h_data_path, test_size, early_size, dataset_name, val_size=None, cv=None):    
    labels = ['fake', 'real']

    data_list = []
    for label in labels:
        for news_article in tqdm(os.listdir(os.path.join(h_data_path, label))):
            path_f = os.path.join(f_data_path, label, news_article)
            path_h = os.path.join(h_data_path, label, news_article)

            with open(os.path.join(path_f, f"{news_article}_text.txt"), encoding="utf-8") as f:
                text = f.read()
                if not text:
                    continue
            
            with open(os.path.join(path_f, f"{news_article}_graph.txt")) as f:
                edges_f = json.load(f)
            
            with open(os.path.join(path_h, f"{news_article}_graph.txt")) as f:
                edges_h = json.load(f)
            
            # Complete Propagation graph
            node_features = pd.read_csv(os.path.join(path_f, f"{news_article}_nf.csv"))
            node_features['type'] = node_features['type'].map({'tweet': 1, 'retweet':2})
            ids = node_features['id']
            node_features = node_features.drop(['id'], axis=1)
            ss = StandardScaler()
            node_features = pd.DataFrame(data=ss.fit_transform(node_features), columns=node_features.columns)
            node_features = node_features.to_dict(orient="records")

            g = nx.DiGraph()
            for id, i in zip(ids, node_features):
                g.add_node(str(id), **i)
            g.add_edges_from(edges_f)
            nx.set_node_attributes(g, {news_article: {k: 0 for k in i.keys()}})
            g = nx.convert_node_labels_to_integers(g)
            data_f = from_networkx(g, group_node_attrs=list(i.keys()))

            # Propagation graph at deadline t
            node_features = pd.read_csv(os.path.join(path_h, f"{news_article}_nf.csv"))
            node_features['type'] = node_features['type'].map({'tweet': 1, 'retweet':2})
            ids = node_features['id']
            node_features = node_features.drop(['id'], axis=1)
            ss = StandardScaler()
            node_features = pd.DataFrame(data=ss.fit_transform(node_features), columns=node_features.columns)
            node_features = node_features.to_dict(orient="records")

            g = nx.DiGraph()
            for id, i in zip(ids, node_features):
                g.add_node(str(id), **i)
            g.add_edges_from(edges_h)
            nx.set_node_attributes(g, {news_article: {k: 0 for k in i.keys()}})
            g = nx.convert_node_labels_to_integers(g)
            data_h = from_networkx(g, group_node_attrs=list(i.keys()))

            # PairData object

            data = PairData(
                edge_index_f = data_f.edge_index,
                edge_index_h = data_h.edge_index,
                x_h = data_h.x,
                x_f = data_f.x,
                y = 1 if label == "real" else 0,
                text = text,
                news_id = news_article
            )

            data_list.append(data)
    
    train_dataset, test_dataset = train_test_split(data_list, test_size=test_size, stratify=[i.y for i in data_list])
    train_dataset, early_dataset = train_test_split(train_dataset, test_size=early_size/(1-test_size), stratify=[i.y for i in train_dataset])
    print("Length of test_dataset: " + str(len(test_dataset)))
    print("Length of early_dataset: " + str(len(early_dataset)))

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
        dataset['early_dataset'] = early_dataset

        news_ids['kfolds'] = kfolds
        news_ids['test_dataset'] = [i.news_id for i in test_dataset]
        news_ids['train_dataset'] = [i.news_id for i in train_dataset]
        news_ids['early_dataset'] = [i.news_id for i in early_dataset]

    else:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_size/(1-(test_size+early_size)), stratify=[i.y for i in train_dataset])
        print("Length of train_dataset: " + str(len(train_dataset)))
        print("Length of val_dataset: " + str(len(val_dataset)))
        dataset['train_dataset'] = train_dataset
        dataset['val_dataset'] = val_dataset
        dataset['test_dataset'] = test_dataset
        dataset['early_dataset'] = early_dataset

        news_ids['test_dataset'] = [i.news_id for i in test_dataset]
        news_ids['train_dataset'] = [i.news_id for i in train_dataset]
        news_ids['val_dataset'] = [i.news_id for i in val_dataset]
        news_ids['early_dataset'] = [i.news_id for i in early_dataset]

    torch.save(dataset, f'{dataset_name}_pair_gdl_dataset.pt')
    json.dump(news_ids, open(f"{dataset_name}_pair_news_ids_dataset.json", 'w'))

pair_gdl_dataset_builder(
    f_data_path="../preprocessed_data/politifact",
    h_data_path="../preprocessed_data_5h/politifact",
    early_size = 0.15,
    test_size=0.1,
    cv = 10,
    dataset_name="politifact"
)