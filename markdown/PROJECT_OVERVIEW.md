# How This Project Works

## 1. Project in a nutshell

This repository is a robotics imitation-learning tutorial built around a simulated
pick-and-place task. A six-joint Robotis OMY-style manipulator sees a mug and a
plate in MuJoCo. The project can record human keyboard demonstrations, store them
as LeRobot datasets, train or fine-tune policies, and run the resulting policies
back in the simulation.

The main end-to-end loop is:

```text
MuJoCo scene
    -> human teleoperation
    -> observation/action demonstrations
    -> LeRobot dataset
    -> policy training
    -> saved checkpoint
    -> policy controls MuJoCo
```

The main task used by the notebooks is to pick up a mug and place it on a plate.
The notebooks are separate tutorial stages: data collection and training are
not automatically run just because a deployment notebook is opened.

## 2. Repository map

| File or folder | Purpose |
| --- | --- |
| `1.collect_data.ipynb` | Teleoperate the simulation and record ordinary pick-and-place demonstrations. |
| `2.visualize_data.ipynb` | Replay collected data to inspect the actions and corresponding simulation. |
| `3.train.ipynb` | Train an ACT policy from a LeRobot dataset and save a checkpoint. |
| `4.deploy.ipynb` | Load an ACT checkpoint and let the policy control the simulation. |
| `5.language_env.ipynb` | Collect demonstrations in the language-conditioned environment. |
| `6.visualize_data.ipynb` | Inspect/replay the language-conditioned demonstrations. |
| `7.pi0.ipynb` | Run deployment/inference for the pi0 workflow. |
| `8.smolvla.ipynb` | Run deployment/inference for the SmolVLA workflow. |
| `mujoco_env/` | Simulation wrapper, robot/environment helpers, inverse kinematics, and rendering. |
| `asset/` | Robot, mug, plate, and other simulation assets. |
| `demo_data_example/` | Example local LeRobot dataset included for trying the ACT pipeline. |
| `demo_data/` | Local dataset location used by some workflows; keep the dataset root consistent with the notebook being run. |
| `pi0_omy.yaml`, `smolvla_omy.yaml` | Training configuration examples for pi0 and SmolVLA. |
| `train_model.py` | LeRobot training entry point used by the pi0 and SmolVLA configurations. |
| `ckpt/` | Local policy checkpoints. The ACT checkpoint is expected under `ckpt/act_y`. |

## 3. The simulation and task

The simulation wrapper is implemented mainly in `mujoco_env/y_env.py`. It loads
the MuJoCo scene, initializes the robot and objects, exposes images and state,
accepts actions, and checks whether the task was completed.

The environment can be reset to a starting pose and object arrangement. The
robot's end effector is controlled through inverse kinematics when an end-effector
pose action is used. The environment also exposes camera images, including the
agent view and egocentric/wrist-side view used by the notebooks.

In the standard environment, success is checked using simulated object positions
and gripper/robot state: the mug must be close enough to the plate, the gripper
must be open, and the end effector must be raised above the task's threshold.
This is a programmatic simulation success check, not a learned reward function.

## 4. Demonstrations: turning human control into training data

`1.collect_data.ipynb` provides keyboard teleoperation. The README documents
these controls:

- **W/A/S/D** move in the horizontal plane.
- **R/F** move vertically.
- **Q/E** adjust tilt; the arrow keys control other rotations.
- **Space** changes the gripper state.
- **Z** resets the scene and discards the current unfinished episode.

During teleoperation, the environment produces synchronized observations and
actions. An observation includes camera images and robot state; the action records
what the operator commanded. The dataset writer groups those frames into
episodes and saves dataset metadata and statistics as well.

The dataset follows the LeRobot layout. In this checkout, an example has files
such as:

```text
demo_data_example/
├── data/chunk-000/episode_000000.parquet
└── meta/
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── info.json
    ├── stats.json
    └── tasks.jsonl
```

The example dataset metadata specifies a frame rate of 20 FPS and image frames
of 256 by 256 pixels. Its main training signals are:

| Field | Meaning |
| --- | --- |
| `observation.image` | Main camera RGB image. |
| `observation.wrist_image` | Egocentric/wrist camera RGB image. |
| `observation.state` | Six-dimensional end-effector pose: position and orientation (roll, pitch, yaw). |
| `action` | Seven values: six robot joint/action values and a gripper command. |
| `obj_init` | Initial mug and plate pose information; useful metadata, but not a model input in the ACT notebook. |

LeRobot's metadata describes feature shapes, frame rate, episode boundaries,
tasks, and normalization statistics. These statistics are used when constructing
the policy so that input and output values can be normalized consistently.

## 5. Inspecting recorded demonstrations

`2.visualize_data.ipynb` and `6.visualize_data.ipynb` help check that the
recorded data makes sense before training. Playback reconstructs or advances the
simulation using recorded actions and displays dataset camera images alongside
the scene.

This step is important: if the images, action trajectory, episode boundaries, or
task labels are wrong, the model will learn from bad examples. Visualization is
for inspecting data; it does not train the policy.

## 6. ACT training workflow

`3.train.ipynb` is the direct training example for the Action Chunking
Transformer (ACT):

1. Select a local dataset and load its metadata.
2. Convert LeRobot features into policy input/output features.
3. Configure ACT with the feature definitions, dataset statistics, and chunk
   settings.
4. Load frames through a `LeRobotDataset` and a PyTorch `DataLoader`.
5. For each batch, run the policy forward pass, calculate its loss, backpropagate
   the gradients, and update the model using Adam.
6. Save the trained policy checkpoint under `ckpt/act_y`.

The checked-in example uses an action chunk size of 10: one observation is used
to predict a sequence of future actions rather than only a single next action.
This is intended to make manipulation control more temporally coherent. The
training notebook's configured training duration is 3,000 update steps; it can
be changed to suit available compute and the experiment.

The training objective compares predicted actions with the demonstration action
chunks, with ACT's model-specific loss terms. The notebook optimizes that loss;
the separate simulation success checker is not what drives these gradient
updates.

### Dataset worker setting on Windows

The ACT notebook uses `num_workers=0` for its data loaders. This is useful in
this Windows setup because multiprocessing workers need to pickle dataset and
transform objects. A lambda inside a transform can cause a `PicklingError`.
Avoid adding non-pickleable lambdas to worker-loaded transforms; keep
`num_workers=0` here unless the transform and runtime setup have been tested
with worker processes.

## 7. ACT deployment workflow

`4.deploy.ipynb` connects the saved ACT checkpoint to the live MuJoCo task:

1. Select the compute device.
2. Load the dataset metadata to reconstruct feature definitions and
   normalization statistics.
3. Build an ACT configuration compatible with the checkpoint.
4. Load `ckpt/act_y` with `ACTPolicy.from_pretrained`.
5. Create and reset the MuJoCo environment.
6. At the control frequency, read the end-effector state and camera images,
   preprocess the images to tensors, and package the observations.
7. Call `policy.select_action(...)`.
8. Send the predicted action to the environment and render the next frame.
9. Reset or stop when the environment's success condition is met or the viewer
   closes.

The deployment configuration in this notebook uses temporal ensembling and
requests one action at a time from the policy. This is a deployment choice; the
model was trained to predict action chunks, and the policy manages how predicted
actions are selected/combined at inference time.

The notebook's task/instruction string and environment setup need to match the
task represented in the checkpoint's demonstrations. The standard ACT example
is not automatically a language-conditioned policy just because a text label
appears in the deployment observation.

## 8. Language-conditioned data and other policy families

The second set of notebooks extends the tutorial:

- `5.language_env.ipynb` records data in an environment with language/task
  variation (for example, different mug/plate arrangements and instructions).
- `6.visualize_data.ipynb` is used to inspect this collection.
- `7.pi0.ipynb` and `8.smolvla.ipynb` are inference/deployment notebooks for
  those policy families.
- `pi0_omy.yaml` and `smolvla_omy.yaml` configure the training script to use a
  language dataset, set model/chunk parameters, batch size and training steps,
  and specify checkpoint output directories.
- `train_model.py` is the common training entry point for those YAML
  configurations. It delegates dataset/policy setup to LeRobot and handles
  training, logging, and checkpoint saving.

These are related but distinct model workflows. The ACT notebook is a
self-contained example; the pi0 and SmolVLA recipes use their own dataset
configuration, policy settings, and checkpoint paths.

## 9. Compute, checkpoints, and paths

The Python environment in this checkout has been observed to use CPU-only
PyTorch (`torch 2.6.0+cpu`). In that environment, use:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This avoids trying to move a model to CUDA when PyTorch was not compiled with
CUDA support. CPU inference and training are possible but can be significantly
slower. For GPU use, install a compatible CUDA-enabled PyTorch build and verify
that `torch.cuda.is_available()` is `True`.

Notebook paths such as `./demo_data_example` and `./ckpt/act_y` are relative to
the notebook's current working directory. Run notebooks from the repository
root, or adjust those paths to match the actual local dataset/checkpoint
locations. A deploy notebook cannot load a checkpoint that has not been
created or downloaded.

## 10. Suggested run order

For the basic ACT project:

1. Install the dependencies using `requirements.txt` and prepare the MuJoCo
   assets as described in `README.md`.
2. Open `1.collect_data.ipynb` to record your own demonstrations, or use the
   provided `demo_data_example/`.
3. Open `2.visualize_data.ipynb` to inspect demonstrations if you collected
   data.
4. Run `3.train.ipynb` from top to bottom to create or update `ckpt/act_y`.
5. Run `4.deploy.ipynb` to evaluate the trained policy in simulation.

If using the provided checkpoint instead of training, confirm its configuration
matches the deployment notebook and skip the training step. For language
experiments, follow the corresponding `5`/`6` data notebooks and the pi0 or
SmolVLA workflow separately.

## 11. What this project demonstrates (and what it does not)

This project demonstrates a simulation-based imitation-learning pipeline:
human demonstrations are recorded as state/action examples, a policy learns to
predict actions from observations, and the learned policy is evaluated in a
simulated manipulation task.

The standard ACT rollout is a simulated demonstration, not proof of safe
real-world robot operation. A real robot would require hardware interfaces,
calibration, safety limits, latency handling, and validation beyond this
repository's MuJoCo deployment notebook.

## 12. A concise presentation explanation

> “I teleoperate a simulated robot to demonstrate mug-to-plate pick-and-place.
> The environment records camera images, robot state, and actions as a LeRobot
> dataset. ACT learns from those demonstration sequences and saves a checkpoint.
> The deployment notebook loads that checkpoint, observes the simulation, and
> predicts actions in a loop until the simulation reports success.”
