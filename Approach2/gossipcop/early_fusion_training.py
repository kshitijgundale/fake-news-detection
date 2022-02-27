from models.early_fusion import EarlyFusion
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sentence_transformers import SentenceTransformer

dataset = torch.load("gdrive/MyDrive/FakeNewsDetection/gossipcop_gdl_dataset.pt")
val_dataset = dataset['val_dataset']
train_dataset = dataset['train_dataset']
test_dataset = dataset['test_dataset']

def text_to_embedding(transformer, emb_type, data):
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

def train_early_fusion(emb_model, emb_type):
    transformer = SentenceTransformer(emb_model)

    text_to_embedding(transformer, emb_type, train_dataset) 
    text_to_embedding(transformer, emb_type, val_dataset)
    
    class_weights = compute_class_weight(class_weight="balanced", classes=[0,1], y=[i.y for i in train_dataset])
    class_weights = torch.FloatTensor(class_weights)

    loss_f = torch.nn.CrossEntropyLoss(weight=class_weights)
    model = Concat(d_in=384, loss_f=loss_f)

    X_train = train_dataset
    X_val = val_dataset

    train_loader = DataLoader(X_train, batch_size=64, shuffle=False)
    val_loader = DataLoader(X_val, batch_size=64, shuffle=False)

    early_stopping = EarlyStopping('val_loss', patience=10)
    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k = 1)

    trainer = pl.Trainer(
        max_epochs=50,
        deterministic=True,
        log_every_n_steps = 1,
        callbacks=[checkpoint]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = Concat.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, d_in=384, loss_f=loss_f)
    f1 = trainer.validate(model, val_loader)[0]['f1']
    acc = trainer.validate(model, val_loader)[0]['accuracy']
    recall = trainer.validate(model, val_loader)[0]['recall']
    precision = trainer.validate(model, val_loader)[0]['precision']

    print(f"Early Fusion metrics using {model} and {emb_type}: f1:{f1} precision:{precision} recall:{recall} acc:{acc}")