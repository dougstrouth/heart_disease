import os
import csv

def log_run_results(run_details):
    """
    Logs the details of a successful analysis run to a CSV file.
    """
    log_file = "run_logs.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=run_details.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_details)
    print(f"Run details logged to {log_file}")
