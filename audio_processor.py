"""
audio_processor.py
------------------
Processes UrbanSound8K audio files for the UWSN routing simulation.

Workflow:
  1. Auto-detect UrbanSound8K folder by scanning for .wav files in fold1..fold10.
  2. Resample each audio file to 2400 Hz (bandwidth-limited underwater acoustic channel).
  3. Extract 13 Mel-Frequency Cepstral Coefficients (MFCCs) per frame.
  4. Return the mean MFCC vector and the payload byte length.
  5. If no .wav files are found, generate synthetic 2400 Hz acoustic waveforms with noise.
"""

import os
import glob
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# Suppress noisy audio backend warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------
# Constants
# -------------------------------------------------
TARGET_SR: int = 2400          # Underwater acoustic channel sample rate (Hz)
N_MFCC: int = 13               # Number of MFCC coefficients
N_FFT: int = 256               # FFT window size (suitable for 2400 Hz)
HOP_LENGTH: int = 128          # Hop length for STFT
N_MELS: int = 40               # Number of Mel filter banks
PAYLOAD_BITS_PER_FLOAT: int = 32   # 32-bit float representation


# -------------------------------------------------
# Dataset auto-discovery
# -------------------------------------------------
def find_urbansound_folder(base_dir: Optional[str] = None) -> Optional[str]:
    """
    Scan the project directory tree to find the UrbanSound8K dataset folder.
    Returns the path to the folder that contains fold subdirectories with .wav files.
    """
    if base_dir is None:
        base_dir = str(Path(__file__).parent)

    # Check explicit 'Dataset' subfolder first
    explicit = os.path.join(base_dir, "Dataset")
    if os.path.isdir(explicit):
        for fold in range(1, 11):
            fold_path = os.path.join(explicit, f"fold{fold}")
            if os.path.isdir(fold_path):
                wav_files = glob.glob(os.path.join(fold_path, "*.wav"))
                if wav_files:
                    logger.info("UrbanSound8K dataset detected at: %s", explicit)
                    return explicit

    # Generic recursive search for any folder containing fold1..fold10
    for root, dirs, _ in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        fold1 = os.path.join(root, "fold1")
        if os.path.isdir(fold1):
            wav_files = glob.glob(os.path.join(fold1, "*.wav"))
            if wav_files:
                logger.info("UrbanSound8K dataset detected at: %s", root)
                return root

    logger.warning("UrbanSound8K dataset not found in '%s'. Will use synthetic audio.", base_dir)
    return None


def collect_wav_files(dataset_folder: str, max_per_fold: int = 5) -> List[str]:
    """
    Collect a representative subset of .wav files from fold1-fold10.
    """
    files: List[str] = []
    for fold in range(1, 11):
        fold_path = os.path.join(dataset_folder, f"fold{fold}")
        if not os.path.isdir(fold_path):
            continue
        wavs = sorted(glob.glob(os.path.join(fold_path, "*.wav")))
        files.extend(wavs[:max_per_fold])
    return files


# -------------------------------------------------
# Low-level DSP helpers (no librosa dependency)
# -------------------------------------------------
def _read_wav_scipy(file_path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file using scipy; return (mono float32 array, sample_rate)."""
    rate, data = wavfile.read(file_path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)

    return data, rate


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample audio from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def _preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to boost high frequencies."""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def _framing(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice audio into overlapping frames. Returns shape (n_frames, frame_length)."""
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        audio = np.pad(audio, (0, frame_length - len(audio)))
        n_frames = 1
    shape = (n_frames, frame_length)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()


def _power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Compute power spectrum for each frame."""
    windowed = frames * np.hamming(frames.shape[1])
    spectrum = np.fft.rfft(windowed, n=n_fft)
    return (np.abs(spectrum) ** 2) / n_fft


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbanks(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1)."""
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        for k in range(left, center):
            if center != left:
                fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right != center:
                fbank[m - 1, k] = (right - k) / (right - center)
    return fbank


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """Return DCT-II matrix of shape (n_mfcc, n_mels)."""
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    dct[0] *= 1.0 / np.sqrt(2)
    return dct * np.sqrt(2.0 / n_mels)


def compute_mfcc(audio: np.ndarray, sr: int,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS) -> np.ndarray:
    """
    Compute MFCC coefficients from raw audio using pure NumPy/SciPy.
    Returns mfcc_matrix of shape (n_mfcc, n_frames).
    """
    audio = _preemphasis(audio)
    frames = _framing(audio, frame_length=n_fft, hop_length=hop_length)
    if frames.shape[0] == 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)

    power_spec = _power_spectrum(frames, n_fft)
    fbank = _mel_filterbanks(sr, n_fft, n_mels)
    mel_energy = np.dot(power_spec, fbank.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel = np.log(mel_energy)
    dct = _dct_matrix(n_mfcc, n_mels)
    mfcc = np.dot(dct, log_mel.T)
    return mfcc.astype(np.float32)


# -------------------------------------------------
# Try librosa first, fall back to custom DSP
# -------------------------------------------------
def _try_librosa_mfcc(file_path: str, target_sr: int, n_mfcc: int) -> Optional[np.ndarray]:
    """Attempt MFCC extraction via librosa; return None on failure."""
    try:
        import librosa  # type: ignore
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH,
                                     n_mels=N_MELS)
        return mfcc
    except Exception as exc:
        logger.debug("librosa failed (%s); falling back to scipy pipeline.", exc)
        return None


# -------------------------------------------------
# Synthetic audio fallback
# -------------------------------------------------
def generate_synthetic_audio(duration: float = 2.0,
                              sr: int = TARGET_SR,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a synthetic underwater acoustic waveform at sr Hz.
    Composed of multiple low-frequency tones + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    freqs = [80.0, 120.0, 200.0, 350.0, 600.0, 800.0, 1000.0]
    amps = rng.uniform(0.05, 0.25, len(freqs)).astype(np.float32)
    phases = rng.uniform(0, 2 * np.pi, len(freqs)).astype(np.float32)

    audio = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phases):
        audio += a * np.sin(2 * np.pi * f * t + p)

    noise_std = float(np.max(np.abs(audio))) / 10.0
    audio += rng.normal(0, noise_std, len(t)).astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


# -------------------------------------------------
# Public API
# -------------------------------------------------
def process_audio(file_path: Optional[str],
                  target_sr: int = TARGET_SR,
                  n_mfcc: int = N_MFCC) -> Tuple[np.ndarray, int]:
    """
    Process a single audio file (or generate synthetic audio) and extract MFCCs.

    Args:
        file_path:  Absolute path to a .wav file, or None for synthetic mode.
        target_sr:  Target sample rate in Hz (default 2400).
        n_mfcc:     Number of MFCC coefficients (default 13).

    Returns:
        feature_vector: np.ndarray of shape (n_mfcc,) --- mean MFCCs.
        payload_bytes:  Integer byte size of the feature payload.
    """
    mfcc_matrix: Optional[np.ndarray] = None

    if file_path and os.path.isfile(file_path):
        mfcc_matrix = _try_librosa_mfcc(file_path, target_sr, n_mfcc)

        if mfcc_matrix is None:
            try:
                audio, orig_sr = _read_wav_scipy(file_path)
                audio = _resample(audio, orig_sr, target_sr)
                if len(audio) < N_FFT:
                    audio = np.pad(audio, (0, N_FFT - len(audio)))
                mfcc_matrix = compute_mfcc(audio, target_sr, n_mfcc)
            except Exception as exc:
                logger.warning("scipy pipeline failed for '%s': %s. Using synthetic.", file_path, exc)
                mfcc_matrix = None
    else:
        if file_path:
            logger.warning("File not found: '%s'. Using synthetic audio.", file_path)

    if mfcc_matrix is None:
        audio = generate_synthetic_audio(duration=2.0, sr=target_sr)
        mfcc_matrix = compute_mfcc(audio, target_sr, n_mfcc)

    if mfcc_matrix.ndim == 2 and mfcc_matrix.shape[1] > 0:
        feature_vector = np.mean(mfcc_matrix, axis=1)
    else:
        feature_vector = np.zeros(n_mfcc, dtype=np.float32)

    if feature_vector.shape[0] != n_mfcc:
        feature_vector = np.zeros(n_mfcc, dtype=np.float32)

    payload_bytes = n_mfcc * (PAYLOAD_BITS_PER_FLOAT // 8)
    return feature_vector.astype(np.float32), payload_bytes


class AudioPayloadManager:
    """
    Manages a pool of audio feature vectors drawn from the UrbanSound8K dataset.
    Sensor nodes call get_payload() to obtain a pre-computed MFCC feature vector
    embedded into UWSN data packets.
    """

    def __init__(self, dataset_folder: Optional[str] = None,
                 max_files: int = 50,
                 target_sr: int = TARGET_SR,
                 n_mfcc: int = N_MFCC):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self._features: List[np.ndarray] = []
        self._payload_bytes: int = n_mfcc * (PAYLOAD_BITS_PER_FLOAT // 8)
        self._rng = np.random.default_rng(42)

        if dataset_folder is None:
            dataset_folder = find_urbansound_folder()

        if dataset_folder:
            wav_files = collect_wav_files(dataset_folder, max_per_fold=max(1, max_files // 10))
            wav_files = wav_files[:max_files]
            logger.info("Pre-processing %d audio files from UrbanSound8K...", len(wav_files))
            for fp in wav_files:
                try:
                    feat, _ = process_audio(fp, target_sr, n_mfcc)
                    self._features.append(feat)
                except Exception as exc:
                    logger.warning("Skipped '%s': %s", fp, exc)
            logger.info("Audio pre-processing complete. %d feature vectors cached.", len(self._features))

        if not self._features:
            logger.warning("No real audio features cached. Building synthetic pool of 50 vectors.")
            for seed in range(50):
                feat, _ = process_audio(None, target_sr, n_mfcc)
                self._features.append(feat)

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    def get_payload(self) -> Tuple[np.ndarray, int]:
        idx = int(self._rng.integers(0, len(self._features)))
        return self._features[idx].copy(), self._payload_bytes

    def __len__(self) -> int:
        return len(self._features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Audio Processor Self-Test")
    print("=" * 60)

    ds_folder = find_urbansound_folder()
    test_file: Optional[str] = None
    if ds_folder:
        wavs = collect_wav_files(ds_folder, max_per_fold=1)
        if wavs:
            test_file = wavs[0]

    print(f"\nDataset folder : {ds_folder}")
    print(f"Test file      : {test_file}")

    features, payload_bytes = process_audio(test_file, TARGET_SR, N_MFCC)
    print(f"\nTarget SR      : {TARGET_SR} Hz")
    print(f"MFCC vector    : {np.round(features, 4)}")
    print(f"Feature shape  : {features.shape}")
    print(f"Payload size   : {payload_bytes} bytes")

    mgr = AudioPayloadManager(ds_folder, max_files=20)
    feat, pb = mgr.get_payload()
    print(f"\nManager pool size : {len(mgr)}")
    print(f"Sample payload    : {np.round(feat[:5], 4)} ... ({pb} bytes)")
    print("\nSelf-test PASSED.")
