# Overview

You can train and evaluate your model in an isolated Docker container.
This keeps your RL environment separate from your local environment,
which make reproducibility more robust.

The `Dockerfile_CPU` builds a standalone local Docker image that both
hosts the Jupyter notebook server and also can train and evaluate 
your submission from the command line.

# Quick Start

Build the image and start the notebook server:

```
make
```

Train model from command line:

```
make train
```

Evaluate most recent submission

```
make evaluate
```

# Requirements

You need to have Docker installed in your workstation.


