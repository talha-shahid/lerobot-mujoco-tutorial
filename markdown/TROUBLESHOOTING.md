# Troubleshooting the MuJoCo Imitation-Learning Project

This document covers recurring setup and workflow issues in this repository.
Start by reading the complete traceback: the final exception and the cell named
in the traceback usually identify the failing stage.

## 1. PyTorch says it cannot use CUDA

### Typical message

```text
AssertionError: Torch not compiled with CUDA enabled
```

### Why it happens

The operating system can recognize an NVIDIA GPU while the Python environment
still has a CPU-only PyTorch build. In this project environment, the observed
version was `torch 2.6.0+cpu`, with `torch.version.cuda is None`.

### What to do

First check the active notebook kernel:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

If it reports a CPU-only build, either:

- Keep using CPU with a compatible device selection:

  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

- Or install a CUDA-enabled PyTorch build into the same virtual environment
  used by the notebook, following the official PyTorch installation selector.
  Restart the kernel and re-check CUDA after installation.

The project machine has been reported to have a GTX 1050 with 4 GB VRAM. If CUDA
works, start with a small training batch; the notebook's default batch size of
64 may be too large for that VRAM.

## 2. `PicklingError` mentioning a lambda

### Typical message

```text
PicklingError: Can't pickle <function <lambda> ...>
```

### Why it happens

On Windows, a DataLoader configured with worker processes must serialize
objects such as the dataset and image transforms. A lambda function is not
normally pickleable in this setup.

### What to do

- Remove `transforms.Lambda(...)` from transforms used by the dataset.
- Put the operation inside a named top-level transform class or function.
- Set `num_workers=0` in the DataLoader used by the training loop.
- Re-run the dataset/loader setup cell, then run the training cell.
- If an old loader was already created in the kernel, recreate it; changing the
  notebook source does not mutate an existing Python object automatically.

The actual loader used by the training loop is the one that matters. Check its
`num_workers` value before iteration if the same error continues.

## 3. Hugging Face reports that a dataset repository is missing

### Typical messages

```text
RepositoryNotFoundError
```

or a request to `huggingface.co/api/datasets/...`.

### Why it happens

The requested local dataset path may not exist or may not contain valid LeRobot
metadata. The library can then try to resolve the repository ID online.

### What to do

- Check that the dataset directory exists relative to the notebook's current
  working directory.
- For the included example, the expected path is `./demo_data_example`.
- Confirm it contains `meta/info.json`, `meta/tasks.jsonl`, and the data files.
- Use the same dataset root when loading metadata and the dataset itself.
- If intentionally loading a private or remote dataset, check repository ID,
  repository type, and authentication.

## 4. `NameError` for `dataloader`, `dataset`, or `policy`

### Why it happens

The training cell was run before the cell that defines its dependencies, or the
kernel was restarted and earlier variables were cleared.

### What to do

Run the notebook cells from the top in order, at least through:

1. Imports and device selection.
2. Dataset metadata and policy configuration.
3. Policy initialization.
4. Dataset, optimizer, and DataLoader creation.
5. Training loop.

After a kernel restart, variables from previous runs do not survive. Source code
in a notebook also does not automatically rerun when edited.

## 5. The policy loads but does not pick up the mug

### Why it can happen

Loading a checkpoint successfully only confirms that the saved model can be
used. It does not guarantee that the model learned a robust strategy. The
included example dataset has one episode with 137 frames, which offers little
variation in object placements and execution styles.

Other possible issues include:

- Deployment uses a different camera image order, resolution, or scaling than
  training.
- The action configuration or checkpoint does not match.
- The policy is being evaluated at a starting position absent from its
  demonstrations.
- The action/gripper mapping or environment configuration differs.

### What to do

1. First test the checkpoint at the demonstrated/default starting arrangement.
2. Confirm the policy inputs have the expected shapes and values.
3. Verify the saved checkpoint and deployment configuration correspond to the
   same training setup.
4. If testing varied placements, collect successful demonstrations for those
   variations and retrain.
5. Evaluate several trials and report the success rate, not only one attempt.

## 6. CUDA out-of-memory during training

### What to do

- Reduce batch size, for example from 64 to 4, 2, or 1.
- Confirm the DataLoader definition used by the loop has that batch size.
- Close other GPU-heavy applications.
- Restart the kernel before retrying if the previous run left memory allocated.
- Keep CPU fallback available if the model or batch does not fit in VRAM.

Smaller batches use less memory, but do not necessarily make total training
faster; they may require more updates to process the same amount of data.

## 7. MuJoCo viewer does not open or exits unexpectedly

- Confirm MuJoCo dependencies and the required project assets are installed.
- Run the notebook from the repository root so relative paths like
  `./asset/example_scene_y.xml` resolve.
- Close any stale viewer instance and restart the notebook kernel before trying
  again.
- Read the error traceback for missing XML, mesh, or texture paths.

## 8. A warning is not always a failure

Warnings such as a missing optional accelerated decoder may indicate that the
software selected a slower fallback. Distinguish warnings from exceptions:
check whether the cell completed, whether expected variables were created, and
whether the next stage can run.

## Quick diagnostic checklist

```text
[ ] Is the notebook using the intended .venv kernel?
[ ] Does the dataset path exist relative to the notebook working directory?
[ ] Did I run the setup cells after the latest kernel restart?
[ ] Does the active DataLoader have num_workers=0 on Windows?
[ ] Does torch.cuda.is_available() match the intended compute device?
[ ] Does the checkpoint match the policy config and dataset metadata?
[ ] Am I distinguishing a warning from the final raised exception?
```
