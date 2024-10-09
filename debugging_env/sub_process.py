import subprocess
import time
import sys

max_threads = 8
active_processes = []
max_runtime = 60 * 60  # 30 minutes in seconds
max_runs = 10000  # Set your desired maximum number of runs here
total_runs = 0


def start_new_process():
    # get randomint for pliability
    return subprocess.Popen(
        [
            "/home/work/miniconda3/envs/py311/bin/python",
            "/home/work/climate-cooperation-competition/debugging_env/trajectory_mitigation_saving_simulations.py",
        ]
    )


start_time = time.time()

# Start initial batch of processes
for _ in range(max_threads):
    process = start_new_process()
    active_processes.append(process)
    total_runs += 1


def terminate_all_processes():
    print("Terminating all processes.")
    for process in active_processes:
        process.terminate()
    time.sleep(5)  # Give processes 5 seconds to terminate gracefully
    for process in active_processes:
        if process.poll() is None:  # If process is still running
            process.kill()  # Force kill


while active_processes:
    current_time = time.time()
    if current_time - start_time > max_runtime:
        print(f"Maximum runtime of {max_runtime/60} minutes reached.")
        terminate_all_processes()
        break

    if total_runs >= max_runs:
        print(f"Maximum number of runs ({max_runs}) reached.")
        terminate_all_processes()
        break

    for process in active_processes[:]:  # Iterate over a copy of the list
        # Check if the process has finished
        if process.poll() is not None:
            # Remove the process from the active list
            active_processes.remove(process)
            # Start a new process to maintain the number of active threads
            if total_runs < max_runs:
                new_process = start_new_process()
                active_processes.append(new_process)
                total_runs += 1
    # Wait a bit before checking again
    time.sleep(1)

print(f"Program completed. Total runs: {total_runs}")
print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")
