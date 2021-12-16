import os
from os import path
import networkx as nx
from pyvis.network import Network
import pandas as pd
import json

data_path = "./preprocessed_data"
dataset = 'gossipcop'
labels = ['real', 'fake']

for label in labels:
    cnt = 0

    for news in os.listdir(path.join(data_path, dataset, label)):
        if cnt < 4:
            with open(path.join(data_path, dataset, label, news, f"{news}_graph.txt")) as f:
                edges = json.load(f)

            node_features = pd.read_csv(path.join(data_path, dataset, label, news, f"{news}_nf.csv"))
            max_followers = node_features['num_followers'].max()
            min_followers = node_features['num_followers'].min()

            g = nx.DiGraph()

            node_features = node_features.to_dict(orient="records")
            if len(node_features) > 50 and len(node_features) < 75:
                for i in node_features:
                    g.add_node(str(i['id']), **i)

                g.add_node(news, **{k:0 for k in i.keys()})

                g.add_edges_from(edges)

                net = Network(directed=True)
                net.from_nx(g)

                for node in net.nodes:
                    node['label'] = ""

                    if node['type'] == "tweet":
                        node['shape'] = "square"
                        node['color'] = 'darkgray'
                        node['size'] = (((node['num_followers'] - min_followers)/(max_followers - min_followers)) * 20) + 10
                    elif node['type'] == "retweet":
                        node['shape'] = "triangle"
                        node['color'] = 'darkgray'
                        node['size'] = (((node['num_followers'] - min_followers)/(max_followers - min_followers)) * 20) + 10
                    else:
                        if label == "fake":
                            node['color'] = 'black'
                        else:
                            node['color'] = "orange"

                    if node['num_followers'] > 10000:
                        node['color'] = 'red'

                    if node['is_verified'] == 1:
                        node['color'] = 'blue'

                for edge in net.edges:
                    edge['color'] = "black"

                net.show(f'{label}_{cnt}.html')
                cnt += 1
