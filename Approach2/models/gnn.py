import torch
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch.nn.functional as F

class GCN(pl.LightningModule):
    def __init__(self, conv_class, hidden_channels, class_weights=None):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv_class = conv_class
        self.hidden_channels = hidden_channels
        self.class_weights = class_weights
        self.conv1 = self.conv_class(12, self.hidden_channels)
        self.conv2 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.conv5 = self.conv_class(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, int(hidden_channels/2))
        self.lin2 = Linear(int(hidden_channels/2), 2)

        self.loss_f = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
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

        preds = list(out.argmax(dim=1)) 
        truth = list(data.y)

        return val_loss, preds, truth

    def validation_epoch_end(self, outputs):
        preds = []
        truth = []
        for _,batch_preds,batch_truth in outputs:
          preds += batch_preds
          truth += batch_truth
        f1 = f1_score(truth, preds, average='weighted')
        print(f1)
        self.log("f1", f1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)