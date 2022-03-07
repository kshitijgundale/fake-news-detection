import torch
from imblearn.over_sampling import RandomOverSampler
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from models.gnn import GCN
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv

import re
from sentence_transformers import SentenceTransformer
from models.text import TextModule

dataset = torch.load("politifact_gdl_dataset.pt")
kfolds = dataset['kfolds']
train_dataset = dataset['train_dataset']
test_dataset = dataset['test_dataset']

###################################
########## GNN Training ###########
###################################

conv_classes = [GraphConv, GCNConv, SAGEConv, GATConv]

for conv_class in conv_classes:

  gnn_scores = []

  for fold, (train_index, val_index) in enumerate(kfolds):
    print(f"Conv Layer: {conv_class}")
    print("------------------------")
    print(f"Fold: {fold + 1}")
    print("------------------------")

    model = GCN(hidden_channels=64, conv_class=conv_class)

    train_index = np.array(train_index)
    indexes, _ = RandomOverSampler().fit_resample(train_index.reshape(-1,1), [train_dataset[i].y for i in train_index])
    X_train = [train_dataset[i] for i in indexes.ravel()]
    X_val = [train_dataset[i] for i in val_index]

    train_loader = DataLoader(X_train, batch_size=64, shuffle=False)
    val_loader = DataLoader(X_val, batch_size=64, shuffle=False)

    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
      max_epochs=50,
      deterministic=True,
      log_every_n_steps = 1,
      callbacks=[checkpoint]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = GCN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, hidden_channels=64)
    f1 = trainer.validate(model, val_loader)[0]['f1']
    gnn_scores.append(f1)

  print(f"GNN Val accuracy using : {conv_class} : " + str(sum(gnn_scores)/len(gnn_scores)))


###################################
######### Text Training ###########
###################################

def text_to_embedding(transformer, emb_type, data):
  X_train = []
  y_train = []
  for i in data:
    s = i.text
    s = s.lower()
    s = s.replace("\n", "")
    s = re.sub('[^A-Za-z0-9 ]+', '', s)

    tokens = s.split()
    if emb_type.lower() == "first512":
      tokens = tokens[:500]
    elif emb_type.lower() == "last512":
      tokens = tokens[-500:]
    elif emb_type.lower() == "fl256":
      tokens = tokens[:256] + tokens[-256:]

    s = " ".join(tokens)

    X_train.append(s)
    y_train.append(i.y)
  
  transformer.max_seq_length = 500
  X_train = transformer.encode(X_train)
  
  return X_train, y_train

models = ['all-mpnet-base-v2', 'all-distilroberta-v1', 'all-roberta-large-v1', 'all-MiniLM-L12-v1']
emb_types = ['first512', 'last512', 'fl256']

for model in models:
  for emb_type in emb_types:

    transformer = SentenceTransformer(model_name_or_path=model)

    X_train, y_train = text_to_embedding(transformer, emb_type, train_dataset)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    text_train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    scores = []

    for fold, (train_index, val_index) in enumerate(kfolds):
      print(f"Model: {model}, emb_type: {emb_type}")
      print("------------------------")
      print(f"Fold: {fold + 1}")
      print("------------------------")

      model = TextModule(d_in=768)

      train_index = np.array(train_index)
      indexes, _ = RandomOverSampler().fit_resample(train_index.reshape(-1,1), [text_train_dataset[i][1] for i in train_index])
      X_train = [text_train_dataset[i] for i in indexes.ravel()]
      X_val = [text_train_dataset[i] for i in val_index]

      train_loader = DataLoader(X_train, batch_size=64, shuffle=False)
      val_loader = DataLoader(X_val, batch_size=64, shuffle=False)

      checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

      trainer = pl.Trainer(
        max_epochs=25,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
      )
      trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
      model = TextModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, d_in=768)
      f1 = trainer.validate(model, val_loader)[0]['f1']
      scores.append(f1)

    print(f"Text Val accuracy using : {model} and {emb_type} : " + str(sum(scores)/len(scores)))