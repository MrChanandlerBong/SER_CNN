import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SR = 22050
Number_Of_Mel_Frames = 128
Length_Of_FFT_Window = 2048
Hop_Length = 512
Max_length = 128

Emotion_Mappings = {}
for k in range(1, 9):
    Emotion_Mappings[str(0) + str(k)] = k

def metadata(filename):
    parts = filename.split("-")
    emotion = Emotion_Mappings[parts[2]]
    actor_id = int(parts[-1].split(".")[0])
    
    if actor_id % 2 == 1 :
        gender = 0
    else:
        gender = 1
    return emotion, actor_id, gender

def load_and_clean_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    return y

def audio_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=Length_Of_FFT_Window,
        hop_length=Hop_Length,
        n_mels=Number_Of_Mel_Frames
    )
    return librosa.power_to_db(mel, ref=np.max)

def fix_length(mel):
    if mel.shape[1] < Max_length:
        mel = np.pad(
            mel,
            ((0, 0), (0, Max_length - mel.shape[1])),
            mode="constant"
        )
    else:
        mel = mel[:, :Max_length]
    return mel

def add_AWGN(y):
    AWGN = np.random.randn(len(y))
    return y + 0.005 * AWGN


def pitch_shift(y, sr, steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def build_dataset(root_dir, augment=True):
    X, y, meta = [], [], []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue

            path = os.path.join(root, file)
            emotion, actor, gender = metadata(file)

            audio = load_and_clean_audio(path)
            audio_variants = [audio]

            if augment:
                audio_variants.append(add_AWGN(audio))
                audio_variants.append(pitch_shift(audio, SR, steps=2))
                audio_variants.append(time_stretch(audio, rate=1.1))

            for a in audio_variants:
                mel = audio_to_logmel(a)
                mel = fix_length(mel)
                mel = mel[..., np.newaxis]

                X.append(mel)
                y.append(emotion)
                meta.append([actor, gender])

    return np.array(X), np.array(y), np.array(meta)

X, y, meta = build_dataset("Audio_Speech_Actors_01-24", augment=True)

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)
np.save("data/processed/meta.npy", meta)


# ===============================================================
# EXTRAS Made By AI - For My Visualisation and Maybe Show To You
# ===============================================================
def spectral_stats(y):
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR).mean()
    energy = np.mean(y ** 2)
    return centroid, bandwidth, energy


angry_centroids, sad_centroids = [], []

for root, _, files in os.walk("Audio_Speech_Actors_01-24"):
    for file in files:
        if not file.endswith(".wav"):
            continue

        emotion, _, _ = metadata(file)
        if emotion not in [5, 4]:
            continue

        audio = load_and_clean_audio(os.path.join(root, file))
        centroid, _, _ = spectral_stats(audio)

        if emotion == 5:
            angry_centroids.append(centroid)
        else:
            sad_centroids.append(centroid)

plt.hist(angry_centroids, bins=30, alpha=0.7, label="Angry")
plt.hist(sad_centroids, bins=30, alpha=0.7, label="Sad")
plt.legend()
plt.title("Spectral Centroid Distribution")
plt.show()

idx_angry = np.where(y == 5)[0][0]
idx_sad = np.where(y == 4)[0][0]

energy_angry = X[idx_angry].squeeze().mean(axis=0)
energy_sad = X[idx_sad].squeeze().mean(axis=0)

plt.plot(energy_angry, label="Angry")
plt.plot(energy_sad, label="Sad")
plt.legend()
plt.title("Energy Over Time")
plt.show()

sample_path = None
for root, _, files in os.walk("Audio_Speech_Actors_01-24"):
    for file in files:
        if file.endswith(".wav"):
            sample_path = os.path.join(root, file)
            break
    if sample_path:
        break

sample_audio = load_and_clean_audio(sample_path)

variants = [
    sample_audio,
    add_AWGN(sample_audio),
    pitch_shift(sample_audio, SR, 2),
    time_stretch(sample_audio, 1.1)
]

titles = ["Original", "AWGN", "Pitch Shift", "Time Stretch"]

plt.figure(figsize=(10, 8))
for i, v in enumerate(variants):
    mel = audio_to_logmel(v)
    plt.subplot(2, 2, i + 1)
    plt.imshow(mel, aspect="auto", origin="lower")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Augmentation Sanity Check")
plt.show()

X_flat = X.reshape(len(X), -1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.6)
plt.colorbar()
plt.title("Emotion Separability (PCA)")
plt.show()

male_avg = X[meta[:, 1] == 0].mean(axis=0).squeeze()
female_avg = X[meta[:, 1] == 1].mean(axis=0).squeeze()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(male_avg, aspect="auto", origin="lower")
plt.title("Male Average Mel")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(female_avg, aspect="auto", origin="lower")
plt.title("Female Average Mel")
plt.axis("off")

plt.show()
