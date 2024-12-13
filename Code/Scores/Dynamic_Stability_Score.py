import numpy as np
import librosa
import matplotlib.pyplot as plt

def evaluate_dynamic_stability(mix, stem, sr, fft_window_size=2048, hop_length=512, plot_spectrogram=False, instrument_type=None):

    """
    Evaluate the dynamic stability of a stem based on RMS and spectral flux.

    Parameters:
    mix (ndarray): The mixed audio signal.
    stem (ndarray): The isolated stem audio signal.
    sr (int): Sample rate of the audio.
    fft_window_size (int): Size of the FFT window for STFT.
    hop_length (int): Hop length for STFT.
    plot_spectrogram (bool): If True, plot spectrogram of the stem.
    instrument_type (str): Type of instrument ('drums', 'bass', 'other').

    Returns:
    float: The dynamic stability score normalized between 0 and 100.
    """
    
    # Calculate RMS
    rms_stem = librosa.feature.rms(y=stem, frame_length=fft_window_size, hop_length=hop_length)[0]
    presence_threshold = 0.1 * np.max(rms_stem)
    active_frames = rms_stem > presence_threshold  # Boolean mask for active frames
    
    # Filter RMS using active frames
    active_rms_stem = rms_stem[active_frames]
    dss = np.mean(active_rms_stem) / (np.std(active_rms_stem) + 1e-6) if len(active_rms_stem) > 0 else 0
    dss = (dss / 3) * 100
    print(f"Dynamic Stability Score: {dss}")
    # Spectral Flux Calculation
    stft_stem = librosa.stft(stem, n_fft=fft_window_size, hop_length=hop_length)
    mag_stem = np.abs(stft_stem)
    flux_differences = np.diff(mag_stem, axis=1) ** 2
    flux_per_frame = np.sum(flux_differences, axis=0)
    
# Filter spectral flux using active frames
    active_flux = flux_per_frame[active_frames[1:]]  # Skip the first frame due to np.diff
    flux_max_reference = 15000
    flux_score = (np.mean(active_flux) / flux_max_reference) * 100 if len(active_flux) > 0 else 0
    flux_score = min(max(flux_score, 0), 30)
     

    print(f"Flux Score: {flux_score}")
    # Final Scoring
    if instrument_type == 'drums':
        final_dynamic_score = dss + flux_score
    elif instrument_type == 'bass':
        final_dynamic_score = dss
    else:
        final_dynamic_score = dss - flux_score
    final_dynamic_score = max(final_dynamic_score, 0)
    
    # Optional Plot
    if plot_spectrogram:
        plt.figure(figsize=(14, 6))
        librosa.display.specshow(librosa.amplitude_to_db(mag_stem, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.title('Stem Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
    
    return final_dynamic_score