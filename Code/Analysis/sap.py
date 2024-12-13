import os
import glob
import subprocess
from datetime import datetime

# Define paths
dataset_path = 'F:\\coding\\workspace\\stem sap\\musdb18hq\\100'
output_base_path = 'F:\\coding\\workspace\\stem sap\\sap_output'
log_file_path = 'F:\\coding\\workspace\\stem sap\\log.txt'

# Open the log file in append mode
with open(log_file_path, 'a') as log_file:
    log_file.write(f"\n--- Separation processing started at {datetime.now()} ---\n")
    log_file.flush()

    # Find all folders in the dataset path
    for folder in glob.glob(os.path.join(dataset_path, '*')):
        log_file.write(f"Checking folder: {folder}\n")
        log_file.flush()

        if os.path.isdir(folder):
            # Look for mixture.wav in the current folder
            mixture_file = os.path.join(folder, 'mixture.wav')
            log_file.write(f"Checking for mixture.wav in {folder}\n")
            log_file.flush()

            if os.path.isfile(mixture_file):
                log_file.write(f"Found mixture.wav in {folder}\n")
                log_file.flush()

                folder_name = os.path.basename(folder)
                output_umx_dir = os.path.join(output_base_path, folder_name, 'UMX')
                output_demucs_dir = os.path.join(output_base_path, folder_name, 'Demucs')
                os.makedirs(output_umx_dir, exist_ok=True)
                os.makedirs(output_demucs_dir, exist_ok=True)

                # Process with Open-Unmix (UMX)
                try:
                    log_file.write("  - Applying Open-Unmix...\n")
                    log_file.flush()
                    print(f"Applying Open-Unmix on {mixture_file}...", flush=True)

                    # Run Open-Unmix CLI using subprocess and capture errors
                    result = subprocess.run(
                        ['umx', '--outdir', output_umx_dir, mixture_file],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    log_file.write(f"  - Open-Unmix results saved in {output_umx_dir}\n")
                    log_file.flush()
                    print(f"Open-Unmix results saved in {output_umx_dir}", flush=True)
                except subprocess.CalledProcessError as e:
                    log_file.write(f"  - Error with Open-Unmix: {e.stderr}\n")
                    log_file.flush()
                    print(f"Error with Open-Unmix: {e.stderr}", flush=True)

                # Process with Demucs - Always run regardless of previous outputs
                try:
                    log_file.write("  - Applying Demucs...\n")
                    log_file.flush()
                    print(f"Applying Demucs on {mixture_file}...", flush=True)

                    # Run Demucs using subprocess.run
                    subprocess.run(['demucs', '--out', output_demucs_dir, mixture_file], check=True)

                    log_file.write(f"  - Demucs results saved in {output_demucs_dir}\n")
                    log_file.flush()
                    print(f"Demucs results saved in {output_demucs_dir}", flush=True)
                except subprocess.CalledProcessError as e:
                    log_file.write(f"  - Error with Demucs: {e}\n")
                    log_file.flush()
                    print(f"Error with Demucs: {e}", flush=True)

            else:
                log_file.write(f"No mixture.wav found in {folder}\n")
                log_file.flush()

    log_file.write(f"--- Separation processing ended at {datetime.now()} ---\n")
    log_file.flush()
    print("Separation processing completed. Check the log file for details.", flush=True)
