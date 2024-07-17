# main.py
import subprocess
import time

max_threads = 8
active_processes = []


def start_new_process():
    # get randomint for pliability
    return subprocess.Popen(
        [
            "/home/work/miniconda3/envs/py311/bin/python",
            "/home/work/climate-cooperation-competition/debugging_env/trajectory_mitigation_saving_simulations.py",
        ]
    )


# Start initial batch of processes
for _ in range(max_threads):
    process = start_new_process()
    active_processes.append(process)

while active_processes:
    for process in active_processes:
        # Check if the process has finished
        if process.poll() is not None:
            # Remove the process from the active list
            active_processes.remove(process)
            # Start a new process to maintain the number of active threads
            new_process = start_new_process()
            active_processes.append(new_process)
    # Wait a bit before checking again
    time.sleep(1)

print("All processes have been started.")
