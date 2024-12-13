import os
import musdb
import museval
import soundfile as sf
import numpy as np

# Define paths
base_estimate_path = r'F:\coding\workspace\stem sap\sap_output\A Classic Education - NightOwl'
evaluation_output_path = 'F:\coding\workspace\stem sap\evaluation_results'

# Ensure the output directory exists
if not os.path.exists(evaluation_output_path):
    os.makedirs(evaluation_output_path)

# Load MUSDB18 dataset
mus = musdb.DB(root='F:\coding\workspace\stem sap\musdb18hq', is_wav=True)

# Find the track to evaluate
track_name = 'A Classic Education - NightOwl'
track = None
for t in mus:
    if t.name == track_name:
        track = t
        break

if track is None:
    print(f"Error: Track '{track_name}' not found in the dataset.")
else:
    # Load the estimated audio for all components
    estimates = {}
    components = ['vocals', 'drums']

    for component in components:
        estimate_path = os.path.join(base_estimate_path, f"{component}.wav")
        if os.path.isfile(estimate_path):
            est_audio, _ = sf.read(estimate_path, dtype='float32')
            if len(est_audio.shape) == 1:
                est_audio = np.expand_dims(est_audio, axis=0)  # Convert mono to (1, samples)
            else:
                est_audio = est_audio.T  # Convert stereo from (samples, channels) to (channels, samples)
            estimates[component] = est_audio
            print(f"Loaded {component}: shape={est_audio.shape}, dtype={est_audio.dtype}")
        else:
            print(f"Warning: Estimate file does not exist for {component} at {estimate_path}")

    # Print estimate details
    for key, value in estimates.items():
        print(f"Estimate '{key}': shape={value.shape}, dtype={value.dtype}")

    # Run evaluation using museval
    try:
        # Run eval_mus_track without output_dir, directly evaluate with loaded estimates
        scores = museval.eval_mus_track(track, estimates)

        # Print evaluation results
        for target in scores.targets:
            print(f"Target: {target.name}")
            print(f"  SDR: {np.nanmean(target.sdr):.2f} dB")
            print(f"  SIR: {np.nanmean(target.sir):.2f} dB")
            print(f"  SAR: {np.nanmean(target.sar):.2f} dB")

        print(f"Evaluation completed for {track_name}.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
