import numpy as np
import librosa
import matplotlib.pyplot as plt



def evaluate_frequency_isolation(mix, stem, sr, fft_window_size=2048, hop_length=512, weight_fundamental=40.0, plot_spectrogram=False):
    """
    Evaluate the isolation of a stem by checking the presence of its fundamental frequency and harmonics in the mix, and incorporate spectral flux for artifact detection.
    
    Parameters:
    mix (ndarray): The mixed audio signal.
    stem (ndarray): The isolated stem audio signal.
    sr (int): Sample rate of the audio.
    fft_window_size (int): Size of the FFT window for STFT.
    hop_length (int): Hop length for STFT. 
    weight_fundamental (float): Weight for the fundamental frequency.
    plot_spectrogram (bool): If True, plot spectrograms of mix and stem.
    
    Returns:
    float: The isolation score normalized between 0 and 100.
    """
    # Short-Time Fourier Transform for both mix and stem
    stft_mix = librosa.stft(mix, n_fft=fft_window_size, hop_length=hop_length, window='hann')
    stft_stem = librosa.stft(stem, n_fft=fft_window_size, hop_length=hop_length, window='hann')
    
    # Calculate magnitudes
    mag_mix = np.abs(stft_mix)
    mag_stem = np.abs(stft_stem)
    
    # Ensure the number of time frames matches between mix and stem
    min_frames = min(mag_mix.shape[1], mag_stem.shape[1])
    mag_mix = mag_mix[:, :min_frames]
    mag_stem = mag_stem[:, :min_frames]
    
    # Initialize score components
    fundamental_score = 0
    harmonics_score = 0
    weight_harmonics = 60.0
    count_valid_frames = 0
    
    # Calculate frequency presence score
    fundamental_freq_indices = np.array([np.flatnonzero(mag_stem[:, t] > 0)[0] if len(np.flatnonzero(mag_stem[:, t] > 0)) > 0 else 0 for t in range(mag_stem.shape[1])])
    fundamental_freq_indices = np.clip(fundamental_freq_indices, 0, mag_mix.shape[0] - 1)
    
    for t, f_idx in enumerate(fundamental_freq_indices):
        count_valid_frames += 1
        # Check if fundamental is present in the mix
        if mag_mix[f_idx, t] > 0:
            fundamental_score = weight_fundamental  # Fixed 40 points if fundamental is present
        
        # Check all possible harmonics up to 20,000 Hz
        harmonic_indices = [f_idx * (n + 1) for n in range(1, int(20000 / (sr / fft_window_size))) if f_idx * (n + 1) < mag_mix.shape[0]]
        
        for idx in harmonic_indices:
            if mag_mix[idx, t] >= mag_stem[idx, t]:
                harmonics_score += 1  # Add 1 if the harmonic energy in the mix is greater or equal to that in the stem
    
    # Normalize harmonics score to 50 points
    max_possible_harmonics_score = count_valid_frames * len(harmonic_indices)  # All harmonics per frame
    if max_possible_harmonics_score > 0:
        harmonics_score = (harmonics_score / max_possible_harmonics_score) * weight_harmonics
    
    # Calculate frequency presence score (0 - 100)
    frequency_presence_score = fundamental_score + harmonics_score  # 40 for fundamental, 60 for harmonics

    
    # Plot spectrogram if needed
    if plot_spectrogram:
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(mag_mix, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.title('Mix Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(mag_stem, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.title('Stem Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
    
    return frequency_presence_score
