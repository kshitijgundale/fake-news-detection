import torch
from imblearn.over_sampling import RandomOverSampler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from models.gnn import GCN

dataset = torch.load("politifact_gdl_dataset.pt")
kfolds = dataset['kfolds']
train_dataset = dataset['train_dataset']
test_dataset = dataset['test_dataset']

###################################
########## GNN Training ###########
###################################

gnn_scores = []

for fold, (train_index, val_index) in enumerate(kfolds):
  print("------------------------")
  print(f"Fold: {fold + 1}")
  print("------------------------")

  model = GCN(hidden_channels=64)

  train_index = np.array(train_index)
  indexes, _ = RandomOverSampler().fit_resample(train_index.reshape(-1,1), [train_dataset[i].y for i in train_index])
  X_train = [train_dataset[i] for i in indexes.ravel()]
  X_val = [train_dataset[i] for i in val_index]

  train_loader = DataLoader(X_train, batch_size=64, shuffle=False)
  val_loader = DataLoader(X_val, batch_size=64, shuffle=False)

  early_stopping = EarlyStopping('val_loss', patience=10)
  checkpoint = ModelCheckpoint(monitor='f1', mode='max', save_top_k = 1)

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

print("GNN Val accuracy: " + str(sum(gnn_scores)/len(gnn_scores)))