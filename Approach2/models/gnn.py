import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from sklearn.metrics import f1_score


class GCN(pl.LightningModule):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(12, 64)
        self.conv2 = GraphConv(64, 64)
        self.conv3 = GraphConv(64, 64)
        self.conv4 = GraphConv(64, 64)
        self.conv5 = GraphConv(64, 64)
        self.lin1 = Linear(64, 32)
        self.lin2 = Linear(32, 2)

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
        loss = torch.nn.CrossEntropyLoss()(out, data.y)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        out = self(data.x, data.edge_index, data.batch)  
        val_loss = F.cross_entropy(out, data.y)
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