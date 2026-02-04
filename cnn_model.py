import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models


gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print('GPU in usage ')


X = np.load("data/X.npy")
y = np.load("data/y.npy")
meta = np.load("data/meta.npy") 
y = y - 1
num_classes = len(np.unique(y))

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta,
    test_size=0.2,
    stratify=y,
    random_state=42
)

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu",
                      input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

cnn_model = build_cnn()

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()

history = cnn_model.fit(
    X_train, y_train,
    epochs=23,
    batch_size=32,
    validation_split=0.2
)

cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)

y_pred_probs = cnn_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = np.mean(y_pred == y_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

entropy = -np.sum(y_pred_probs * np.log(y_pred_probs + 1e-9), axis=1)
mean_entropy = np.mean(entropy)

male_idx = meta_test[:, 1] == 0
female_idx = meta_test[:, 1] == 1

male_acc = np.mean(y_pred[male_idx] == y_test[male_idx])
female_acc = np.mean(y_pred[female_idx] == y_test[female_idx])

male_f1 = f1_score(y_test[male_idx], y_pred[male_idx], average="macro")
female_f1 = f1_score(y_test[female_idx], y_pred[female_idx], average="macro")

cnn_model.save("cnn_model.keras")

f = open('Log_CNN.txt','w')
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

