"""Training studio: train an image classifier on your own folder of images,
on the best hardware this machine has, and keep the result in a model library.

- hardware.py: GPUs and CPU available for training
- datasets.py: reading a folder of images arranged by class
- architectures.py: the model architectures offered
- metrics.py: evaluation, including the confidence threshold for abstaining
- runs.py: starting, following and cancelling training runs
- worker.py: the training process itself, run separately from the app
- library.py: saved models, and using them on new images

Nothing here is a medical device. Trained models are for research and
evaluation until validated for their intended use."""
