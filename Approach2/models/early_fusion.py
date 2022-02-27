import torch
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

class GNN(pl.LightningModule):
    def __init__(self, hidden_channels, conv_class):
        super(GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_class
        self.conv1 = self.conv_class(12, self.hidden_channels)
        self.conv2 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, 32)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin1(x).relu()

        return x

class TextModule(pl.LightningModule):

  def __init__(self, d_in):
    super(TextModule, self).__init__()
    self.lin1 = Linear(d_in, 128)
    self.lin2 = Linear(128, 64)
    self.lin3 = Linear(64, 32)

  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x).relu()
    x = self.lin3(x).relu()
    return x

class EarlyFusion(pl.LightningModule):

  def __init__(self, d_in, loss_f, hidden_channels):
    super(Concat, self).__init__()
    self.d_in = d_in
    self.hidden_channels = hidden_channels
    self.text_module = TextModule(self.d_in)
    self.gnn = GNN(self.hidden_channels)
    self.lin1 = Linear(64, 32)
    self.lin2 = Linear(32, 16)
    self.lin3 = Linear(16, 2)

    self.loss_f = loss_f

  def forward(self, text_emb, x, edge_index, batch):
    text_out = self.text_module(text_emb)
    gnn_out = self.gnn(x ,edge_index, batch)
    concat = torch.hstack([text_out, gnn_out])
    x = self.lin1(concat).relu()
    x = self.lin2(x).relu()
    x = self.lin3(x).relu()
    
    return F.log_softmax(x, dim=-1)

  def training_step(self, batch, batch_idx):
      data = batch
      d = int(data.text_emb.shape[0]/self.d_in)
      out = self(data.text_emb.resize_((d,self.d_in)), data.x, data.edge_index, data.batch)
      loss = self.loss_f(out, data.y)
      return loss

  def validation_step(self, batch, batch_idx):
    data = batch
    d = int(data.text_emb.shape[0]/self.d_in)
    out = self(data.text_emb.resize_((d, self.d_in)), data.x, data.edge_index, data.batch)  
    val_loss = self.loss_f(out, data.y)
    self.log("val_loss", val_loss, on_epoch=True)

    preds = out.argmax(dim=1).cpu().detach().numpy()
    truth = data.y.cpu().detach().numpy()

    return val_loss, preds, truth

  def validation_epoch_end(self, outputs):
    preds = np.hstack([i[1] for i in outputs])
    truth = np.hstack([i[2] for i in outputs])

    f1 = f1_score(truth, preds, average='macro')
    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    accuracy = accuracy_score(truth, preds)

    self.log("f1", f1)
    self.log("precision", precision)
    self.log("accuracy", accuracy)
    self.log("recall", recall)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)