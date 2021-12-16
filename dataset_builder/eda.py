import json
import os
from os import path
import pandas as pd

data_path = "./preprocessed_data"
dataset = 'gossipcop'
labels = ['real', 'fake']

meta = {}

for label in labels:
    news_count = 0
    tweet_count = 0
    retweet_count = 0

    for news in os.listdir(path.join(data_path, dataset, label)):
        pass