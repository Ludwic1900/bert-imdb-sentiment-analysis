import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path

from dataset import MyDataset
from collate import collate_fn
from model import TextClassificationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Data
dataset_path = Path("3000.xlsx")
df = pd.read_excel(dataset_path)
df['text'] = df['text'].fillna("").astype(str)

# train / val split
validation_ratio = 0.1
df_val = df.sample(frac=validation_ratio, random_state=42)
df_train = df.drop(df_val.index)

train_dataset = MyDataset(df_train)
val_dataset = MyDataset(df_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

#Model
model = TextClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

def to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}

#Validation
def validate():
    model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch, targets in val_loader:
            batch = to_device(batch)
            targets = targets.to(device)

            outputs = model(batch)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, pred = torch.max(outputs, dim=1)
            total_correct += (pred == targets).sum().item()

    accuracy = total_correct / len(val_dataset)
    return accuracy, total_loss / len(val_loader)

#Training Loop
best_acc = 0

for epoch in range(1):
    model.train()
    for i, (batch, targets) in enumerate(train_loader):
        batch = to_device(batch)
        targets = targets.to(device)

        outputs = model(batch)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            print(f"Epoch 1 Step {i}: loss={loss.item():.4f}")

    acc, val_loss = validate()
    print(f"Validation accuracy={acc:.4f}, val_loss={val_loss:.4f}")

    if acc > best_acc:
        torch.save(model.state_dict(), "model_best.pt")
        best_acc = acc
