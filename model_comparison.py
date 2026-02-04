import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,log_loss,roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rcnn_model import RCNN

X = np.load("data/X.npy")
y = np.load("data/y.npy") - 1
_, X_test, _, y_test = train_test_split(X, y,test_size=0.2,stratify=y,random_state=42)
num_classes = len(np.unique(y_test))

cnn_model = tf.keras.models.load_model("cnn_model.keras")
cnn_probs = cnn_model.predict(X_test)
cnn_preds = np.argmax(cnn_probs, axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rcnn_model = RCNN(num_classes).to(device)
rcnn_model.load_state_dict(torch.load("rcnn_model.pth", map_location=device))
rcnn_model.eval()
X_test_torch = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

with torch.no_grad():
    rcnn_logits = rcnn_model(X_test_torch)
    rcnn_probs = torch.softmax(rcnn_logits, dim=1).cpu().numpy()
    rcnn_preds = np.argmax(rcnn_probs, axis=1)


cnn_acc = accuracy_score(y_test, cnn_preds)
cnn_f1 = f1_score(y_test, cnn_preds, average="macro")
cnn_auc = roc_auc_score(y_test, cnn_probs, multi_class="ovr", average="macro")
cnn_logloss = log_loss(y_test, cnn_probs)

rcnn_acc = accuracy_score(y_test, rcnn_preds)
rcnn_f1 = f1_score(y_test, rcnn_preds, average="macro")
rcnn_auc = roc_auc_score(y_test, rcnn_probs, multi_class="ovr", average="macro")
rcnn_logloss = log_loss(y_test, rcnn_probs)

print("CNN")
print(f" Accuracy    : {cnn_acc:.4f}")
print(f" Macro F1    : {cnn_f1:.4f}")
print(f" ROC AUC     : {cnn_auc:.4f}")
print(f" Log Loss    : {cnn_logloss:.4f}\n")

print("RCNN")
print(f" Accuracy    : {rcnn_acc:.4f}")
print(f" Macro F1    : {rcnn_f1:.4f}")
print(f" ROC AUC     : {rcnn_auc:.4f}")
print(f" Log Loss    : {rcnn_logloss:.4f}")

# AI ed This Because I didnt Know How To Plot The curve rather know that its a comparison metric between models

plt.figure(figsize=(7, 6))

for name, probs in [("CNN", cnn_probs), ("RCNN", rcnn_probs)]:
    fpr, tpr, _ = roc_curve(
        (y_test == 0).astype(int), probs[:, 0]
    )
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Example Class)")
plt.legend()
plt.show()

if rcnn_f1 > cnn_f1 and rcnn_auc > cnn_auc:
    print("RCNN outperforms CNN across Macro F1 and ROC-AUC.")
else:
    print("CNN performs comparably")
