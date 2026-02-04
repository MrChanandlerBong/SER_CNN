import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from preprocessing import load_and_clean_audio, audio_to_logmel, fix_length
from rcnn_model import RCNN

# Copying all constraints from preprocessing.py

SR = 22050
Number_Of_Mel_Frames = 128
Length_Of_FFT_Window = 2048
Hop_Length = 512
Max_length = 128

Emotions = ["Neutral", "Calm", "Happy", "Sad","Angry", "Fearful", "Disgust", "Surprised"]

def preprocess(path):
    audio = load_and_clean_audio(path)
    mel = audio_to_logmel(audio)
    mel = fix_length(mel)
    return mel.astype(np.float32)

def predict_cnn(log_mel):
    model = tf.keras.models.load_model("cnn_model.keras")

    x = log_mel[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]

    d = {}
    for i in range(8):
        d[Emotions[i]] = float(probs[i])
    return d

def predict_rcnn(log_mel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RCNN().to(device)
    model.load_state_dict(torch.load("rcnn_torch.pth", map_location=device))
    model.eval()

    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

    d = {}
    for i in range(8):
        d[Emotions[i]] = float(probs[i])
    return d

wav_path = "predict.wav"
log_mel = preprocess(wav_path)

dict1 = predict_cnn(log_mel)
labels = list(dict1.keys())
values = list(dict1.values())

max_prob_id = np.argmax(values)
print(f"CNN predicts that the emotion is {labels[max_prob_id]} with probability {values[max_prob_id]}")

for k, v in dict1.items():
    print(f"Probability for the Emotion {k} is {v}")

dict1 = predict_rcnn(log_mel)
keys = list(dict1.keys())
values = list(dict1.values())

max_prob_id = np.argmax(values)
print(f"RCNN predicts that the emotion is {labels[max_prob_id]} with probability {values[max_prob_id]}")

for k, v in dict1.items():
    print(f"Probability for the Emotion {k} is {v}")
