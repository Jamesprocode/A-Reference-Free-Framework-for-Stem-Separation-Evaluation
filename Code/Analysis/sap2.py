import os
import glob
import subprocess
from datetime import datetime

# Define paths
dataset_path = 'F:\\coding\\workspace\\stem sap\\musdb18hq\\train'
output_base_path = 'F:\\coding\\workspace\\stem sap\\sap_output'
log_file_path = 'F:\\coding\\workspace\\stem sap\\umx_log.txt'  # Separate log file for Open-Unmix only

# Open the log file in append mode
with open(log_file_path, 'a') as log_file:
    log_file.write(f"\n--- Open-Unmix processing started at {datetime.now()} ---\n")
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
                os.makedirs(output_umx_dir, exist_ok=True)

                # Process with Open-Unmix (UMX) if the UMX folder is empty
                if not os.listdir(output_umx_dir):  # Only process if the UMX directory is empty
                    try:
                        log_file.write(f"  - Applying Open-Unmix for {mixture_file}...\n")
                        log_file.flush()
                        print(f"Applying Open-Unmix on {mixture_file}...", flush=True)

                        # Run Open-Unmix CLI using subprocess and capture detailed output
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
                        # Log both the stdout and stderr for debugging
                        log_file.write(f"  - Error with Open-Unmix:\n")
                        log_file.write(f"    stdout: {e.stdout}\n")
                        log_file.write(f"    stderr: {e.stderr}\n")
                        log_file.flush()
                        print(f"Error with Open-Unmix:\nstdout: {e.stdout}\nstderr: {e.stderr}", flush=True)

                else:
                    log_file.write(f"  - UMX folder already has content, skipping...\n")
                    log_file.flush()
                    print(f"UMX folder already has content for {folder_name}, skipping...", flush=True)

            else:
                log_file.write(f"No mixture.wav found in {folder}\n")
                log_file.flush()

    log_file.write(f"--- Open-Unmix processing ended at {datetime.now()} ---\n")
    log_file.flush()
    print("Open-Unmix processing completed. Check the log file for details.", flush=True)
