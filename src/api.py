import datetime
import os

updated_log_files = {}


def log_response(function_name, response):
    global updated_log_files

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/{function_name}.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        if function_name not in updated_log_files:
            log_file.write("\n--- Application run start ---\n")
            updated_log_files[function_name] = True
        log_file.write(f"{timestamp}:\n{response}\n")
