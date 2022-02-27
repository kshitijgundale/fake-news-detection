import torch
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

class GNN(pl.LightningModule):
    def __init__(self, loss_f, conv_class, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv_class = conv_class
        self.hidden_channels = hidden_channels
        self.conv1 = self.conv_class(12, self.hidden_channels)
        self.conv2 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, int(self.hidden_channels/2))
        self.lin1 = Linear(int(self.hidden_channels/4), int(self.hidden_channels/4))
        self.lin3 = Linear(int(self.hidden_channels/2), 2)

        self.loss_f = loss_f

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
        
    def training_step(self, batch, batch_idx):
        data = batch
        out = self(data.x, data.edge_index, data.batch)
        loss = self.loss_f(out, data.y)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        out = self(data.x, data.edge_index, data.batch)  
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

class TextModule(pl.LightningModule):

  def __init__(self, d_in, loss_f):
    super(TextModule, self).__init__()
    torch.manual_seed(12345)
    self.d_in = d_in
    self.lin1 = Linear(self.d_in, 128)
    self.lin2 = Linear(128, 64)
    self.lin3 = Linear(64, 32)
    self.lin4 = Linear(32, 2)

    self.loss_f = loss_f

  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x).relu()
    x = self.lin3(x).relu()
    x = self.lin4(x).relu()

    return F.log_softmax(x, dim=-1)

  def training_step(self, batch, batch_idx):
    data = batch
    d = int(data.text_emb.shape[0]/self.d_in)
    out = self(data.text_emb.resize_((d, self.d_in)))
    loss = self.loss_f(out, data.y)
    return loss

  def validation_step(self, batch, batch_idx):
    data = batch
    d = int(data.text_emb.shape[0]/self.d_in)
    out = self(data.text_emb.resize_((d, self.d_in)))  
    val_loss = self.loss_f(out, data.y)
    self.log("val_loss", val_loss, on_epoch=True)

    preds = list(out.argmax(dim=1)) 
    truth = list(data.y)

    return val_loss, preds, truth

  def validation_epoch_end(self, outputs):
    preds = []
    truth = []
    for _, batch_preds,batch_truth in outputs:
      preds += batch_preds
      truth += batch_truth
      val_l.append(val_loss)
    
    f1 = f1_score(truth, preds, average='macro')
    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    accuracy = accuracy_score(truth, preds)

    self.log("f1", f1)
    self.log("precision", precision)
    self.log("recall", recall)
    self.log("accuracy", accuracy)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)