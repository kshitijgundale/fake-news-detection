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
            return self.x_h.size(0)
        if key == 'edge_index_t':
            return self.x_f.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

def pair_gdl_dataset_builder(base, f_data_path, h_data_path, dataset_name):    

    dataset = {}
    with open(f'./{base}_news_ids_dataset.json', 'r') as f:
        news_ids = json.load(f)
    for d in news_ids:
        if d == "kfolds":
            dataset[d] = news_ids[d]
        else:
            dataset[d] = []
            for news_article, l in tqdm(news_ids[d]):
                label = "real" if l == 1 else "fake"
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

                dataset[d].append(data)

    torch.save(dataset, f'{dataset_name}_pair_gdl_dataset.pt')

pair_gdl_dataset_builder(
    base="gossipcop",
    f_data_path="../preprocessed_data/gossipcop",
    h_data_path="../preprocessed_data_5h/gossipcop",
    dataset_name="gossipcop_5h"
)