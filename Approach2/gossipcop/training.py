import torch
from imblearn.over_sampling import RandomOverSampler
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from models.gnn import GCN
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv
from sklearn.utils.class_weight import compute_class_weight

import re
from sentence_transformers import SentenceTransformer
from models.text import TextModule

dataset = torch.load("politifact_gdl_dataset.pt")
val_dataset = dataset['val_dataset']
train_dataset = dataset['train_dataset']
test_dataset = dataset['test_dataset']

###################################
########## GNN Training ###########
###################################

conv_classes = [GraphConv, GCNConv, SAGEConv, GATConv]

for conv_class in conv_classes:

    class_weights = compute_class_weight(class_weight="balanced", classes=[0,1], y=[i.y for i in train_dataset])
    class_weights = torch.FloatTensor(class_weights)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
        max_epochs=100,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    model = GCN(hidden_channels=64, class_weights=class_weights)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = GCN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, hidden_channels=64)
    f1 = trainer.validate(model, val_loader)[0]['f1']

    print(f"GNN Val accuracy using {conv_class} : {f1}")


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
    X_val, y_val = text_to_embedding(transformer, emb_type, val_dataset)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.long)
    text_train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    text_val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    class_weights = compute_class_weight(class_weight="balanced", classes=[0,1], y=[i.y for i in train_dataset])
    class_weights = torch.FloatTensor(class_weights)

    train_loader = DataLoader(text_train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(text_val_dataset, batch_size=64, shuffle=False)

    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
        max_epochs=100,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    model = GCN(hidden_channels=64, class_weights=class_weights)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = GCN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, hidden_channels=64)
    f1 = trainer.validate(model, val_loader)[0]['f1']

    print(f"Text Val accuracy using {model} and {emb_type}: {f1}")
    


