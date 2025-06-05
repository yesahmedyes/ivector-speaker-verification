import os

import librosa

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def load_dataset():
    audios, labels = [], []

    for i in range(1, 41):
        speaker_dir = os.path.join("./voices", str(i))

        for file in os.listdir(speaker_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_dir, file)

                audio, _ = librosa.load(file_path, sr=16000)

                audios.append(audio)
                labels.append(i)

    return audios, labels


def extract_features(audios):
    features = []

    for audio in audios:
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)

        delta_mfcc = librosa.feature.delta(mfcc)

        combined_features = np.concatenate([mfcc, delta_mfcc], axis=0)

        features.append(combined_features)

    return features


def train_ubm(features, n_components=64):
    # One Gaussian might be modelling vowels, another consonants, another silence, etc.

    all_features = np.vstack([f.T for f in features])

    ubm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=5,
        verbose=1,
        random_state=42,
    )

    ubm.fit(all_features)

    return ubm


def create_supervector(features, ubm, relevance_factor=16.0):
    ubm_means = ubm.means_  # 64, 26

    posteriors = ubm.predict_proba(features)  # n_frames, 64

    n_i = posteriors.sum(axis=0)  # summed over all frames

    f_i = np.zeros_like(ubm_means)  # 64, 26

    for c in range(ubm.n_components):
        post = posteriors[:, c].reshape(-1, 1)  # posterior for cth gaussian

        # probability of each feature vector belonging to cth gaussian
        f_i[c] = np.sum(post * features, axis=0)

    # if n_i = 10, then alpha = 10 / (10 + 16) = 0.38
    # we'll use 38% of speaker data and 62% of UBM data
    alphas = n_i / (n_i + relevance_factor)

    adapted_means = np.zeros_like(ubm_means)

    for c in range(ubm.n_components):
        new_mean = f_i[c] / (n_i[c] + 1e-10)

        adapted_means[c] = (alphas[c] * new_mean) + ((1 - alphas[c]) * ubm_means[c])

    supervector = adapted_means.flatten()  # 64 * 26 = 1664

    return supervector


def extract_speaker_supervectors(features, ubm):
    supervectors = []

    for feat in features:
        feat = feat.T

        sv = create_supervector(feat, ubm)

        supervectors.append(sv)

    return np.array(supervectors)


def factor_analysis(x, n_components=200):
    # get rid of unwanted variability (e.g., differences in microphones, environments, and sessions)

    fa = FactorAnalysis(n_components=n_components, random_state=42)

    fa.fit(x)

    return fa


def extract_speaker_factors(x, fa):
    x = np.array(x)

    speaker_factors = fa.transform(x)

    return speaker_factors


def compute_wccn(speaker_factors, labels):
    # Distances or similarities between embeddings may still be influenced by residual variability unrelated to speaker identity
    # Thus, we do within class covariance normalization
    # Transformed embeddings have an identity matrix as their within-class covariance making cosine similarity more reliable

    labels = np.array(labels)
    unique_speakers = np.unique(labels)

    n_speakers = len(unique_speakers)

    dim = speaker_factors.shape[1]

    W = np.zeros((dim, dim))

    for speaker in unique_speakers:
        speaker_data = speaker_factors[labels == speaker]
        speaker_mean = np.mean(speaker_data, axis=0)
        centered_data = speaker_data - speaker_mean

        W += np.dot(centered_data.T, centered_data)

    W /= n_speakers

    eigenvalues, eigenvectors = np.linalg.eigh(W)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    B = np.dot(eigenvectors * (1.0 / np.sqrt(eigenvalues)), eigenvectors.T)

    return B


def train(speaker_factors, labels):
    model = LinearDiscriminantAnalysis()
    model.fit(speaker_factors, labels)

    lda_factors = model.transform(speaker_factors)

    wccn_matrix = compute_wccn(lda_factors, labels)

    return model, wccn_matrix


def test(test_factors, model, wccn_matrix):
    lda_factors = model.transform(test_factors)

    wccn_factors = np.dot(lda_factors, wccn_matrix)

    return wccn_factors


def cosine_score(speaker_embeddings, test_embeddings):
    speaker_norm = (
        speaker_embeddings / np.linalg.norm(speaker_embeddings, axis=1)[:, np.newaxis]
    )

    test_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1)[:, np.newaxis]

    scores = np.dot(test_norm, speaker_norm.T)

    return scores