import os
import glob
from openunmix import predict
import subprocess
from datetime import datetime

# Define paths
dataset_path = 'F:\\coding\\workspace\\stem sap\\musdb18hq\\train'
output_base_path = 'F:\\coding\\workspace\\stem sap\\sap_output'
log_file_path = 'F:\\coding\\workspace\\stem sap\\log.txt'



# Open the log file in append mode
with open(log_file_path, 'a') as log_file:
    log_file.write(f"\n--- Separation processing started at {datetime.now()} ---\n")

    # Find all folders in the dataset path
    for folder in glob.glob(os.path.join(dataset_path, '*')):
        if os.path.isdir(folder):
            # Look for mixture.wav in the current folder
            mixture_file = os.path.join(folder, 'mixture.wav')
            
            if os.path.isfile(mixture_file):
                folder_name = os.path.basename(folder)
                output_umx_dir = os.path.join(output_base_path, folder_name, 'UMX')
                output_demucs_dir = os.path.join(output_base_path, folder_name, 'Demucs')
                os.makedirs(output_umx_dir, exist_ok=True)
                os.makedirs(output_demucs_dir, exist_ok=True)

                log_file.write(f"Processing {mixture_file}...\n")

                # Process with Open-Unmix (UMX)
                try:
                    log_file.write("  - Applying Open-Unmix...\n")
                    predict.separate(mixture_file, output_dir=output_umx_dir)
                    log_file.write(f"  - Open-Unmix results saved in {output_umx_dir}\n")
                except Exception as e:
                    log_file.write(f"  - Error with Open-Unmix: {e}\n")

                # Process with Demucs
                try:
                    log_file.write("  - Applying Demucs...\n")
                    subprocess.run(['demucs', '--out', output_demucs_dir, mixture_file], check=True)
                    log_file.write(f"  - Demucs results saved in {output_demucs_dir}\n")
                except Exception as e:
                    log_file.write(f"  - Error with Demucs: {e}\n")

            else:
                log_file.write(f"No mixture.wav found in {folder}\n")

    log_file.write(f"--- Separation processing ended at {datetime.now()} ---\n")
    print("Separation processing completed. Check the log file for details.")
