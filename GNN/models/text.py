import torch
from torch.nn import Linear
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.nn.functional as F

class TextModule(pl.LightningModule):

  def __init__(self, d_in, class_weights=None):
    super(TextModule, self).__init__()
    torch.manual_seed(12345)
    self.class_weights = class_weights
    self.lin1 = Linear(d_in, 128)
    self.lin2 = Linear(128, 64)
    self.lin3 = Linear(64, 32)
    self.lin4 = Linear(32, 2)

    self.loss_f = torch.nn.CrossEntropyLoss(weight=self.class_weights)

  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x).relu()
    x = self.lin3(x).relu()
    x = self.lin4(x).relu()

    return F.log_softmax(x, dim=-1)

  def training_step(self, batch, batch_idx):
    data = batch
    out = self(data[0])
    loss = self.loss_f(out, data[1])
    return loss

  def validation_step(self, batch, batch_idx):
    data = batch
    out = self(data[0])  
    val_loss = self.loss_f(out, data[1])
    self.log("val_loss", val_loss, on_epoch=True)

    preds = list(out.argmax(dim=1)) 
    truth = list(data[1])

    return val_loss, preds, truth

  def validation_epoch_end(self, outputs):
    preds = []
    truth = []
    for _,batch_preds,batch_truth in outputs:
      preds += batch_preds
      truth += batch_truth
    f1 = f1_score(truth, preds, average='macro')
    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    acc = accuracy_score(truth, preds)
    self.log("f1", f1)
    self.log("precision", precision)
    self.log("recall", recall)
    self.log("acc", acc)
    

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)