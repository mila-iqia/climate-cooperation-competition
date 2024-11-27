import subprocess


# subprocess.call(
#         [
#             "python",
#             "climate-cooperation-competition/train_with_jax.py",
#             "-sc",
#             "opt_in"
#             "-w",
#             "-ng",
#             "-t",
#             "1000"
#         ]
#     )

# Define the command as a list of arguments
command = [
    "python",
    "climate-cooperation-competition/train_with_jax.py",
    "-sc", "opt_in",
    "-w",
    "-ng",
    "-t", "1000"
]

# Run the command
result = subprocess.call(command)
