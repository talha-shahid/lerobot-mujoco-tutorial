# Robot Learning Glossary for This Project

This glossary explains the terms used in the MuJoCo pick-and-place project in
plain language. The examples refer to the robot learning to move a mug onto a
plate.

## Simulation and robotics

### MuJoCo

A physics simulator. It represents the robot, mug, plate, cameras, and table in
a virtual scene and calculates what happens when the robot moves or touches an
object.

### Environment

The complete simulated task: the scene, robot, objects, observations, action
interface, reset behavior, and success check. The environment is the world the
policy interacts with.

### Robot state

Numeric information about the robot at the current time. This project's ACT
example uses a six-value end-effector pose: three position values and three
orientation values (roll, pitch, yaw).

### End effector

The part at the end of the robot arm that interacts with objects. Here, it is
the robot's gripper assembly.

### Gripper

The robot mechanism that opens and closes to grasp or release the mug. The
dataset's action has a separate gripper command in addition to the six robot
control values.

### Inverse kinematics (IK)

A calculation that finds robot joint positions needed to place the end effector
at a requested position and orientation. The environment uses IK when it
receives end-effector pose commands.

## Data and learning

### Observation

What the policy is given as input at a particular moment. For this project,
observations include camera images and robot state. An observation is not
necessarily the simulator's complete internal state.

### Action

A command that the policy or human operator sends to the robot. In the dataset,
an action contains six robot-control values and one gripper command.

### Demonstration

An example of a person controlling the robot to perform the task. The system
records observations alongside the actions the person took.

### Frame / timestep

One time-aligned record in a trajectory: the observations and action associated
with a point in time. Frames are sampled at the dataset's recording frequency.

### Trajectory

A time-ordered sequence of frames describing one attempt or behavior.

### Episode

A dataset segment containing one run of the task. An episode commonly starts
from an environment reset and ends when recording is stopped or the task is
complete.

### Dataset

A collection of recorded episodes plus metadata such as feature shapes, task
names, frame rate, and normalization statistics. The example dataset in this
checkout contains one episode with 137 frames.

### LeRobot

The robotics library used here for dataset representation and policy utilities.
It helps standardize how observations, actions, metadata, and checkpoints are
handled.

### Behavior cloning

A supervised learning approach that teaches a model to imitate recorded
observation/action pairs. The model predicts actions, compares them with the
demonstrated actions, and adjusts its parameters to reduce the difference.

### Imitation learning

The broader family of methods that learn behavior from demonstrations. The
basic ACT training example in this project is an imitation-learning /
behavior-cloning workflow.

### ACT

**Action Chunking Transformer**, the policy architecture used in the basic
training workflow. It predicts a short sequence of actions conditioned on the
current observation.

### Action chunk

A group of consecutive future actions predicted together. The basic example
configures `chunk_size=10`, meaning the policy is trained to model short
sequences rather than isolated actions only.

### Loss

A numeric training objective measuring how far model predictions are from the
demonstrated targets, together with ACT's model-specific terms. The optimizer
uses this value to update the model.

### Optimizer

The algorithm that changes model parameters during training to reduce the loss.
The ACT notebook uses Adam.

### Batch

A group of dataset examples processed together for one training update. Batch
size affects memory use and training throughput.

### Checkpoint

A saved model state that can be loaded later for inference. In the ACT example,
the checkpoint is expected under `ckpt/act_y`. It is not the dataset or the
simulator.

### Inference

Using a trained model to make predictions. In deployment, inference means
asking the policy for an action from current camera images and robot state.

### Deployment / rollout

Running a trained policy against an environment. The policy observes, predicts
an action, the simulator advances, and the policy receives the next observation.

### Temporal ensembling

A technique that combines overlapping action predictions from multiple recent
observations to smooth control. The deployment configuration enables it for the
ACT example.

### Generalization

The ability to perform well in conditions different from the training
demonstrations, such as a new mug starting position. A model trained on one
episode should not be assumed to generalize reliably.

### Reinforcement learning versus imitation learning

In reinforcement learning, an agent typically learns from rewards while
interacting with an environment. In this project's basic ACT notebook, the
training targets are demonstrations. The MuJoCo success checker evaluates the
rollout; it is not the reward signal driving ACT's training updates.

## Compute and software terms

### CPU

The computer's general-purpose processor. PyTorch in the current project
environment was observed to be CPU-only, so model operations run on CPU unless
the environment is changed.

### GPU / CUDA

A graphics processor can accelerate many neural-network operations. CUDA is
NVIDIA's platform for using supported NVIDIA GPUs. `nvidia-smi` seeing a GPU
does not by itself mean PyTorch can use CUDA; the installed PyTorch build must
include compatible CUDA support.

### VRAM

Memory on the GPU. The GTX 1050 in this setup has 4 GB. Large batches can exceed
that memory, so a smaller batch size may be needed when training on the GPU.

### DataLoader workers

Separate processes that fetch and prepare dataset samples. On Windows they need
to serialize (pickle) objects passed to worker processes. Lambda functions in
transforms can fail to serialize; `num_workers=0` avoids this worker-process
path in the training example.

## A compact mental model

```text
demonstration = example behavior
dataset       = many recorded examples + metadata
training      = adjust policy to imitate the examples
checkpoint    = saved trained policy
inference     = policy predicts an action
rollout       = policy repeatedly acts in the simulator
evaluation    = check whether the rollout meets the task goal
```
