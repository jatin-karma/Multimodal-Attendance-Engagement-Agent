"""
Audio Utility Functions.

Provides helper functions for audio file handling, format validation,
and basic preprocessing operations used by the Audio Module.
"""

import os
import numpy as np
from config import ALLOWED_AUDIO_EXTENSIONS, AUDIO_CONFIG


def is_valid_audio_file(filename):
    """
    Check if a filename has a valid audio extension.
    
    Args:
        filename (str): The filename to validate.
    
    Returns:
        bool: True if the file has a valid audio extension.
    """
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_AUDIO_EXTENSIONS


def get_audio_duration(audio_signal, sample_rate=None):
    """
    Calculate the duration of an audio signal in seconds.
    
    Args:
        audio_signal (np.ndarray): Audio signal array.
        sample_rate (int, optional): Sample rate. Defaults to config value.
    
    Returns:
        float: Duration in seconds.
    """
    if sample_rate is None:
        sample_rate = AUDIO_CONFIG["sample_rate"]
    return len(audio_signal) / sample_rate


def normalize_audio(audio_signal):
    """
    Normalize an audio signal to the range [-1, 1].
    
    Args:
        audio_signal (np.ndarray): Raw audio signal.
    
    Returns:
        np.ndarray: Normalized audio signal.
    """
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        return audio_signal / max_val
    return audio_signal


def segment_audio(audio_signal, segment_duration, sample_rate=None):
    """
    Split an audio signal into fixed-duration segments.
    
    Args:
        audio_signal (np.ndarray): Audio signal array.
        segment_duration (float): Duration of each segment in seconds.
        sample_rate (int, optional): Sample rate. Defaults to config value.
    
    Returns:
        list[np.ndarray]: List of audio segments.
    """
    if sample_rate is None:
        sample_rate = AUDIO_CONFIG["sample_rate"]
    
    segment_length = int(segment_duration * sample_rate)
    segments = []
    
    for start in range(0, len(audio_signal), segment_length):
        segment = audio_signal[start:start + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments


def calculate_rms_energy(audio_signal):
    """
    Calculate the Root Mean Square (RMS) energy of an audio signal.
    
    Args:
        audio_signal (np.ndarray): Audio signal array.
    
    Returns:
        float: RMS energy value.
    """
    return float(np.sqrt(np.mean(audio_signal ** 2)))
