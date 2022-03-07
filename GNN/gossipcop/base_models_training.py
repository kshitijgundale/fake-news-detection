import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from models.gnn import GNN
from models.text import TextModule
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from sklearn.utils.class_weight import compute_class_weight

import re
from sentence_transformers import SentenceTransformer
from models.text import TextModule

dataset = torch.load("gossipcop_gdl_dataset.pt")
val_dataset = dataset['val_dataset']
train_dataset = dataset['train_dataset']
test_dataset = dataset['test_dataset']

###################################
########## GNN Training ###########
###################################

conv_classes = [GraphConv, GCNConv, GATConv]

def train_gnn(conv_class):

    class_weights = compute_class_weight(class_weight="balanced", classes=[0,1], y=[i.y for i in train_dataset])
    class_weights = torch.FloatTensor(class_weights)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
        max_epochs=50,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    model = GNN(conv_class=conv_class, hidden_channels=64, class_weights=class_weights)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = GCN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, conv_class=conv_class, hidden_channels=64, class_weights=class_weights)
    f1 = trainer.validate(model, val_loader)[0]['f1']
    precision = trainer.validate(model, val_loader)[0]['precision']
    recall = trainer.validate(model, val_loader)[0]['recall']
    acc = trainer.validate(model, val_loader)[0]['acc']

    print(f"GNN metrics using {model} and {emb_type}: f1:{f1} precision:{precision} recall:{recall} acc:{acc}")


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
    if emb_type == "first512":
      s = " ".join(s.split()[:500])
    elif emb_type == "last512":
      s = " ".join(s.split()[-500:])
    elif emb_type == "fl256":
      if len(s.split()) > 500:
        s = " ".join(s.split()[:250] + s.split()[-250:])
      else:
        s = " ".join(s.split()[:250])

    s = " ".join(tokens)

    X_train.append(s)
    y_train.append(i.y)
  
  transformer.max_seq_length = 500
  X_train = transformer.encode(X_train)
  
  return X_train, y_train

models = ['all-mpnet-base-v2', 'all-distilroberta-v1', 'all-MiniLM-L12-v1']
emb_types = ['first512', 'last512', 'fl256']

def train_text(emb_model, emb_type):
    transformer = SentenceTransformer(model_name_or_path=emb_model)

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
        max_epochs=50,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    model = TextModule(d_in=len(X_train[0]), class_weights=class_weights)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = TextModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, d_in=len(X_train[0]))
    f1 = trainer.validate(model, val_loader)[0]['f1']
    precision = trainer.validate(model, val_loader)[0]['precision']
    recall = trainer.validate(model, val_loader)[0]['recall']
    acc = trainer.validate(model, val_loader)[0]['acc']

    print(f"Text Val metrics using {emb_model} and {emb_type}: f1:{f1} precision:{precision} recall:{recall} acc:{acc}")
    


