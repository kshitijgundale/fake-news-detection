from models.late_fusion import GNN, TextModule
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from sklearn.utils.class_weight import compute_class_weight

def prepare_dataset(transformer, emb_type, data):
  transformer.max_seq_length = 512

  for i in data:
    s = i.text
    s = s.lower()
    s = s.replace("\n", "")
    s = re.sub('[^A-Za-z0-9 ]+', '', s)

    if emb_type == "first512":
      s = " ".join(s.split()[:500])
    elif emb_type == "last512":
      s = " ".join(s.split()[-500:])
    elif emb_type == "fl256":
      if len(s.split()) > 500:
        s = " ".join(s.split()[:250] + s.split()[-250:])
      else:
        s = " ".join(s.split()[:500])
    else:
      raise Exception()
  
    i.text_emb = torch.tensor(transformer.encode(i.text), dtype=torch.float)
    del i.text

def train_gnn(conv_class):
    class_weights = compute_class_weight(class_weight="balanced", classes=[0,1], y=[i.y for i in train_dataset])
    class_weights = torch.FloatTensor(class_weights)
    loss_f = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
        max_epochs=50,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    model = GNN(conv_class=conv_class, hidden_channels=64, loss_f=loss_f)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = GCN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, conv_class=conv_class, hidden_channels=64, loss_f=loss_f)
    f1 = trainer.validate(model, val_loader)[0]['f1']
    precision = trainer.validate(model, val_loader)[0]['precision']
    recall = trainer.validate(model, val_loader)[0]['recall']
    acc = trainer.validate(model, val_loader)[0]['acc']

    print(f"Base GNN metrics using {model} and {emb_type}: f1:{f1} precision:{precision} recall:{recall} acc:{acc}")

    return model

def train_text(emb_model, emb_type):
    class_weights = compute_class_weight(class_weight="balanced", classes=[0,1], y=[i.y for i in train_dataset])
    class_weights = torch.FloatTensor(class_weights)
    loss_f = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
        max_epochs=50,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    model = TextModule(d_in=len(X_train[0]), loss_f=loss_f)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = TextModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, d_in=len(X_train[0]), loss_f=loss_f)
    f1 = trainer.validate(model, val_loader)[0]['f1']
    precision = trainer.validate(model, val_loader)[0]['precision']
    recall = trainer.validate(model, val_loader)[0]['recall']
    acc = trainer.validate(model, val_loader)[0]['acc']

    print(f"Base Text Val metrics using {emb_model} and {emb_type}: f1:{f1} precision:{precision} recall:{recall} acc:{acc}")

    return model

def train_late_fusion_mean(conv_class, emb_model, emb_type):
    text_to_embedding(transformer, "first512", train_dataset)
    text_to_embedding(transformer, "first512", val_dataset)

    gnn_model = train_gnn(conv_class)
    text_model = train_text(emb_model, emb_type)

    gcn_model.eval()
    text_model.eval()

    preds = []

    predict_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for i in predict_dataloader:
        text_logits = text_model(i.text_emb)
        gcn_logits = gcn_model(i.x, i.edge_index, i.batch)

        probs = (0.5*torch.exp(text_logits)) + (0.5*torch.exp(gcn_logits)[0])
        c = probs.argmax(dim=-1)

        preds.append(c)
    true = [i.y for i in val_dataset]

    f1_score = f1_score(true, preds, average="macro")
    precision = precision_score(true, preds)
    recall = recall_score(true, preds)
    acc = accuracy_score(true, preds)

    print(f"Text Val metrics using {emb_model} and {emb_type}: f1:{f1} precision:{precision} recall:{recall} acc:{acc}")

