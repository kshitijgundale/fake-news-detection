import torch
from torch.nn import Linear
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class MultiModal(pl.LightningModule):
  
  def __init__(self, d_in, class_weights=None):
    super(MultiModal, self).__init__()
    torch.manual_seed(12345)
    self.class_weights = class_weights
    self.lin1 = Linear(d_in, 32)
    self.lin2 = Linear(32, 2)

    self.loss_f = torch.nn.CrossEntropyLoss(weight=self.class_weights)

  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x).relu()

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
    print(f1)
    self.log("f1", f1)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)