import os
import mir_eval
import librosa
import numpy as np
import csv

# Define paths
ground_truth_path = 'F:\\coding\\workspace\\stem sap\\musdb18hq\\train'
output_base_path = 'F:\\coding\\workspace\\stem sap\\sap_output'

# Output CSV file path
evaluation_output_file = 'F:\\coding\\workspace\\stem sap\\evaluation_results.csv'


# Function to evaluate frequency isolation
def evaluate_frequency_isolation(mix, stem, sr, fft_window_size=2048, hop_length=512, weight_fundamental=40.0, plot_spectrogram=False):
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
    
    return frequency_presence_score

# Function to evaluate dynamic stability
def evaluate_dynamic_stability(mix, stem, sr, fft_window_size=2048, hop_length=512, plot_spectrogram=False, instrument_type=None):
    # Calculate RMS
    rms_stem = librosa.feature.rms(y=stem, frame_length=fft_window_size, hop_length=hop_length)[0]
    presence_threshold = 0.1 * np.max(rms_stem)
    active_frames = rms_stem > presence_threshold  # Boolean mask for active frames
    
    # Filter RMS using active frames
    active_rms_stem = rms_stem[active_frames]
    dss = np.mean(active_rms_stem) / (np.std(active_rms_stem) + 1e-6) if len(active_rms_stem) > 0 else 0
    dss = (dss / 3.5) * 100
    
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
    
    # Final Scoring
    if instrument_type == 'drums':
        final_dynamic_score = dss + flux_score
    elif instrument_type == 'bass':
        final_dynamic_score = dss
    else:
        final_dynamic_score = dss - flux_score
    final_dynamic_score = max(final_dynamic_score, 0)
    
    return final_dynamic_score

# Prepare CSV file
with open(evaluation_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Song Name", "Stem Name", "Algorithm Type", "SDR (dB)", "SIR (dB)", "SAR (dB)", "Frequency Isolation Score", "Dynamic Stability Score"])

    # Iterate through each song folder in the ground truth path
    for folder_name in os.listdir(ground_truth_path):
        song_folder = os.path.join(ground_truth_path, folder_name)
        if not os.path.isdir(song_folder):
            continue

        # Iterate over each target (vocals, drums, bass, other)
        for target in ["vocals", "drums", "bass", "other"]:
            # Construct the path to the reference
            reference_path = os.path.join(song_folder, f"{target}.wav")

            # Iterate over algorithms (Demucs and UMX)
            for algorithm in ["Demucs", "UMX"]:
                # Construct the estimate path based on the algorithm
                base_estimate_folder = os.path.join(output_base_path, folder_name, algorithm)

                if algorithm == 'Demucs':
                    # List the base estimate folder to get the model name (e.g., "htdemucs")
                    if os.path.exists(base_estimate_folder):
                        model_folders = os.listdir(base_estimate_folder)
                        if model_folders:
                            model_folder = model_folders[0]  # Assume only one model folder exists
                            final_estimate_folder = os.path.join(base_estimate_folder, model_folder, "mixture")
                        else:
                            continue
                    else:
                        continue
                elif algorithm == 'UMX':
                    # UMX always has a "mixture" folder
                    final_estimate_folder = os.path.join(base_estimate_folder, "mixture")
                else:
                    continue

                # Construct the estimate path
                estimate_path = os.path.join(final_estimate_folder, f"{target}.wav")

                # Check if both reference and estimate files exist
                if not os.path.isfile(reference_path) or not os.path.isfile(estimate_path):
                    continue

                # Load the reference and estimated audio
                ref_audio, sr = librosa.load(reference_path, sr=None)
                est_audio, _ = librosa.load(estimate_path, sr=None)

                # Truncate or pad to match length
                min_len = min(len(ref_audio), len(est_audio))
                ref_audio = ref_audio[:min_len]
                est_audio = est_audio[:min_len]

                # Prepare references and estimates for mir_eval (shape: (n_sources, n_samples))
                reference_sources = np.array([ref_audio])
                estimated_sources = np.array([est_audio])

                # Perform evaluation using mir_eval
                try:
                    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)

                    # Calculate frequency isolation and dynamic stability scores
                    freq_isolation_score = evaluate_frequency_isolation(ref_audio, est_audio, sr)
                    dynamic_stability_score = evaluate_dynamic_stability(ref_audio, est_audio, sr, instrument_type=target)

                    # Write the evaluation results to the CSV file
                    writer.writerow([folder_name, target, algorithm, sdr[0], sir[0], sar[0], freq_isolation_score, dynamic_stability_score])

                    print(f"Evaluation completed for {folder_name} - {target} using {algorithm}.")
                except Exception as e:
                    print(f"Error during evaluation for {folder_name} - {target} using {algorithm}: {e}")
