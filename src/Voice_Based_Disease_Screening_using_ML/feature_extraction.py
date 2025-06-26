import librosa
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from dataclasses import dataclass

#class features_class:
local_file_path="/Users/vaibhavkavdia/Desktop/Projects_for_Resume/Medical/Voice-Based-Disease-Screening-using-ML//Coswara-Data"

def extract_features(file_path, sr=22050):
        y, sr = librosa.load(file_path, sr=sr)
        
        # Basic features
        #1. MFCC — Mel-Frequency Cepstral Coefficients:A compact representation of the timbre (tone quality) of the audio.
        # ZCR - Zero Crossing Rate:The rate at which the signal changes sign (positive ↔ negative):Smooth speech = lower ZCR
        #RMS — Root Mean Square Energy:A measure of the loudness or energy in the audio signal.
        
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        #librosa.load: Loads the .wav file and resamples to 22.05 kHz.
        #y: waveform as a NumPy array
        #sr: sampling rate
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        #ZCR is calculated for each short frame (~20-50 ms),then averaged over the whole file.

        #RMS is also computed frame-wise and then averaged.
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Combine into single feature vector
        features = np.concatenate([mfcc, [zcr], [rms]])
        
        if y.size == 0:
            raise ValueError("Empty audio file")

        
        return features

def load_features_from_metadata(metadata_path, data_dir):
        metadata = pd.read_csv(metadata_path)
        X, y = [], []

        for _, row in metadata.iterrows():
            uid = row["uid"]
            label = row["covid_status"]
            audio_path = os.path.join(data_dir, f"{uid}_cough.wav")

            if os.path.exists(audio_path) and label in ("positive", "negative"):
                features = features.extract_features(audio_path)
                X.append(features)
                y.append(1 if label == "positive" else 0)

        return np.array(X), np.array(y)

def file_path_fun(base_dir):
    file_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    print(f"✅ Found {len(file_paths)} .wav files.")
    return file_paths


def generate_feature_dataframe(base_dir):
    paths = file_path_fun(base_dir)
    data = []

    for path in tqdm(paths):
        try:
            features = extract_features(path)
            label = os.path.basename(os.path.dirname(path))  # folder name
            data.append([path] + list(features) + [label])
        except Exception as e:
            print(f"❌ Error extracting {path}: {e}")

    columns = ["path"] + [f"mfcc_{i+1}" for i in range(13)] + ["zcr", "rms", "label"]
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    BASE_DIR = "/Users/vaibhavkavdia/Desktop/Projects_for_Resume/Medical/Voice-Based-Disease-Screening-using-ML/Coswara-Data"
    df = generate_feature_dataframe(BASE_DIR)
    df.to_csv("features.csv", index=False)
    print("✅ Saved features.csv with shape:", df.shape)

#audio_files=file_path_fun()
#print(f"Total audio files found: {len(audio_files)}")   
#print("Sample files:")
#for path in audio_files[:]:
#    print(path)


