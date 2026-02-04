import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X = np.load("data/X.npy")
y = np.load("data/y.npy") - 1   
meta = np.load("data/meta.npy")

#Adding This Line Because GPT Told that else it wont use my GPU
X = X.astype(np.float32)
y = y.astype(np.int64)


X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta,
    test_size=0.2,
    stratify=y,
    random_state=42
)


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]              
        x = x.permute(2, 0, 1)       
        return x, self.y[idx]


train_loader = DataLoader(
    Data(X_train, y_train),
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    Data(X_test, y_test),
    batch_size=16,
    shuffle=False
)

class RCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(
            input_size=64 * 32,
            hidden_size=128,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, (hn, _) = self.lstm(x)

        x = hn[-1]
        x = self.classifier(x)
        return x
    
model = RCNN(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 67
best_val_loss = float("inf")
patience = 4
counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    y_true, y_pred , y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
            probs = torch.softmax(preds, dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.argmax(dim=1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    val_loss /= len(test_loader)
    train_loss /= len(train_loader)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, "
          f"Val Loss={val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "rcnn_torch.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = np.mean(y_pred == y_true)
macro_f1 = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)

entropy = -np.sum(y_prob * np.log(y_prob + 1e-9), axis=1)
mean_entropy = np.mean(entropy)

male_idx = meta_test[:, 1] == 0
female_idx = meta_test[:, 1] == 1

male_acc = np.mean(y_pred[male_idx] == y_true[male_idx])
female_acc = np.mean(y_pred[female_idx] == y_true[female_idx])

male_f1 = f1_score(y_true[male_idx], y_pred[male_idx], average="macro")
female_f1 = f1_score(y_true[female_idx], y_pred[female_idx], average="macro")


f = open('Log_RCNN.txt','w')
f.write(f"Test Accuracy: {accuracy:.4f}\n")
f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
f.write(f"Mean Prediction Entropy: {mean_entropy:.4f}\n\n")
f.write("Confusion Matrix:\n")
f.write(np.array2string(cm))
f.write("\n\n")
f.write(f"Entropy : {mean_entropy:0.4f}\n")
f.write(f"Male Accuracy: {male_acc:.4f}\n")
f.write(f"Female Accuracy: {female_acc:.4f}\n")
f.write(f"Accuracy Gap Between Male and Female : {abs(male_acc - female_acc):.4f}\n\n")
f.write(f"Male Macro F1: {male_f1:.4f}\n")
f.write(f"Female Macro F1: {female_f1:.4f}\n")
f.close()