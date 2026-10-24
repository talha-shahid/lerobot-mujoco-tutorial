# Concepts and End-to-End Pipeline

## The central idea

This project explores **robot imitation learning** in simulation. Instead of
writing a hand-crafted sequence such as “move above the mug, close the gripper,
lift, move above the plate, release,” a person demonstrates the task by
teleoperating a simulated robot. The system records what the robot sensed and
what it did. A policy is trained on those examples and later tries to reproduce
the demonstrated behavior from its current observations.

The experiment is a pick-and-place task: move a mug onto a plate. MuJoCo supplies
the simulated world and robot dynamics; LeRobot provides dataset and policy
building blocks; PyTorch runs model training and inference.

At a high level:

```text
Demonstrate the task
        |
        v
Record observations and actions over time
        |
        v
Organize demonstrations into a dataset
        |
        v
Learn a mapping from observations to actions
        |
        v
Use the learned policy in a simulation control loop
        |
        v
Measure whether the task succeeded
```

## 1. The environment is the world the policy acts in

The MuJoCo environment models a robot, objects, cameras, and physical
interactions. At each point in time it has an internal state: joint positions,
the robot end-effector pose, object poses, gripper state, and other simulated
quantities.

The policy does not need direct access to every internal simulator variable.
Instead, it receives a selected **observation**, mainly:

- An RGB image from the main camera.
- An RGB image from the wrist/egocentric camera.
- A numeric robot state, represented in the dataset as a six-dimensional
  end-effector pose.

The policy returns an **action**. In the recorded dataset this has seven values:
six robot control values and one gripper value. The environment interprets the
action, advances the simulation, and provides the next observation.

This creates a feedback loop:

```text
observation_t -> policy -> action_t -> simulator -> observation_(t+1)
```

The next observation depends on the previous action and the simulated physics.
For that reason, deployment is not just a single model prediction: it repeatedly
observes, predicts, acts, and observes again.

## 2. Demonstrations are time-aligned examples

During teleoperation, a human provides the robot's controls. At regular
intervals, the system records an observation and the action associated with
that moment. A sequence of these synchronized records forms a **trajectory**;
a completed attempt is stored as an **episode**.

Conceptually, one recorded frame contains a pair:

```text
(what the robot observed at time t, what the operator commanded at time t)
```

Many frames make one trajectory, and many trajectories make a dataset. The
policy learns from these examples rather than receiving explicit symbolic
instructions for every movement. Demonstrations therefore need to be
representative and correctly recorded: missing camera data, inconsistent
controls, or incorrect task labels can make the examples less useful.

The project stores data in the LeRobot dataset format. This format keeps the
frame data, episode organization, task descriptions, feature definitions, and
dataset statistics together. This matters because training and deployment must
agree about what each observation/action field means and how values are scaled.

## 3. Imitation learning: learning from the examples

The ACT workflow is an instance of **behavior cloning**, a common form of
imitation learning. In simplified terms, it trains a model to predict
demonstrated actions from recorded observations:

```text
training input:  observation (images + robot state)
training target: demonstrated action or future action sequence
```

The training notebook reads examples from the dataset in batches. For each
batch, the model predicts actions, a loss measures the difference from the
demonstrated actions, and gradient-based optimization updates the model
parameters. Repeating this process over many examples is intended to make the
predictions resemble the demonstrations.

This is **offline supervised training**: the recorded dataset is used for
learning. The simulation's success condition is useful for evaluating a rollout,
but it is not the training signal used by the ACT notebook's gradient updates.
The learned policy can only be expected to behave well in situations reasonably
represented by the demonstrations; it does not automatically discover a safe or
successful strategy for every new scene.

### Why actions are chunked

An ordinary one-step predictor estimates just the next action. ACT can instead
predict a **chunk** of future actions from the current observation. In the
example ACT setup, the training chunk size is 10. Predicting a short sequence
can capture the temporal structure of a demonstrated movement, such as reaching
and grasping, rather than treating every tiny control update as unrelated.

At inference time, the deployment configuration requests one action at a time
and enables temporal ensembling. In broad terms, temporal ensembling combines
overlapping predictions made at different moments to produce smoother control.
Training chunk length and the number of actions used per inference call are
related settings, but they serve different parts of the pipeline.

## 4. Scaling and feature definitions keep the interfaces consistent

Images and robot measurements have different numeric ranges and shapes. LeRobot
metadata describes the policy's inputs and outputs, while dataset statistics
provide the values needed for normalization and later denormalization.

Normalization puts numeric values into ranges that are easier for a neural
network to learn from. The policy configuration and checkpoint must match the
dataset's feature definitions and statistics. If training and deployment
disagree about an image field, state shape, action ordering, or normalization,
the code may fail outright or produce meaningless control.

This is why deployment reloads the same dataset metadata when rebuilding the
policy configuration. Metadata is not just descriptive documentation; it helps
preserve the contract between data, model, and simulator.

## 5. The checkpoint is the boundary between learning and acting

Training updates model parameters and saves a **checkpoint**. The checkpoint is
the learned part of the system; it is not the simulation, the dataset, or a
complete robot program on its own.

To run the checkpoint, deployment also needs:

1. A compatible policy configuration.
2. The matching input/output feature definitions and dataset statistics.
3. A running environment with cameras, robot state, and an action interface.
4. Preprocessing that converts environment observations into the model's
   expected tensors.

The inference step produces a model action; the environment then interprets and
applies it. The deployment loop repeats until the viewer closes or the
environment reports success.

The success check in this project is an explicit simulator-side rule based on
the mug and plate positions, gripper state, and end-effector height. The policy
does not inherently know that it succeeded; the environment evaluates the
result and can signal the notebook to stop or reset.

## 6. How the project is organized conceptually

Rather than thinking of the repository as a collection of notebooks, it is
useful to see it as a set of cooperating layers:

### Simulation and task layer

Defines the robot, scene, physics, sensors/cameras, control interface, reset
behavior, and task-completion rule. This layer answers: **What world is the
robot in, what can it observe, and what happens when it acts?**

### Demonstration and dataset layer

Connects human teleoperation to timestamped observations and actions, then
organizes them as episodes with metadata and statistics. This layer answers:
**What behavior was demonstrated, and how is it represented consistently?**

### Policy and learning layer

Defines model inputs and outputs, loads batches, calculates the learning loss,
updates parameters, and saves checkpoints. ACT is the direct example in the
basic pick-and-place workflow. The language-oriented workflows add policy
families such as pi0 and SmolVLA, which are configured and trained through
LeRobot's training pipeline.

### Evaluation and deployment layer

Loads a checkpoint, converts live observations into the model's input format,
requests actions, and sends them to the simulator. This layer answers:
**Does the learned policy work when it must control the environment in a
closed-loop rollout?**

The boundaries between these layers are important. The dataset format connects
demonstration to training. The policy configuration connects training to
inference. The environment's observation/action contract connects inference to
robot control.

## 7. The language-conditioned branch

The repository also explores language-conditioned manipulation. In that setting,
the demonstration dataset includes task or language information, and the goal
is for a vision-language-action (VLA) policy to use both visual context and the
task instruction when choosing actions.

This is conceptually broader than the basic ACT behavior-cloning example. A
plain text string placed in an observation dictionary does not by itself turn a
policy into a language-conditioned model. The selected policy architecture,
dataset features, training recipe, and inference interface all need to support
language conditioning. The pi0 and SmolVLA paths are separate model workflows
with their own configurations and checkpoints.

## 8. What is being evaluated

There are two complementary ways to inspect performance:

1. **Dataset playback or action comparison:** check whether recorded data is
   coherent and whether model predictions resemble held-out demonstrated
   actions.
2. **Closed-loop simulation rollout:** let the policy control the robot and
   check whether it completes the pick-and-place task under the simulator's
   success rule.

Good action prediction on recorded data does not guarantee successful
closed-loop behavior. Small errors can change what the robot sees next, and
those new observations may differ from the training examples. A rollout tests
this accumulated interaction more directly.

## 9. Practical constraints and interpretation

- **Compute:** The current project environment was observed to have CPU-only
  PyTorch. Device selection should use CUDA only when
  `torch.cuda.is_available()` is true. CPU operation is valid but slower.
- **Data-loader workers:** On Windows, use `num_workers=0` for this example.
  Multiprocessing workers must serialize dataset/transform state, and lambda
  transforms can trigger pickling errors.
- **Paths:** Dataset roots and checkpoint paths are relative to the process's
  working directory. Training and deployment must point at the intended local
  data and saved checkpoint.
- **Simulation versus reality:** A successful MuJoCo rollout demonstrates
  simulated control, not readiness for a physical robot. Real deployment also
  requires hardware drivers, calibration, latency management, safety limits,
  and dedicated validation.

## 10. The whole story in one example

1. A person uses keyboard controls to move the simulated arm, grasp the mug,
   and place it on the plate.
2. The system records camera images, end-effector state, and actions at each
   time step, grouping the sequence into an episode.
3. The ACT learner samples observation/action sequences and adjusts its
   parameters so its predicted action chunks match the demonstrations.
4. The resulting checkpoint is loaded with compatible metadata and
   normalization.
5. In the simulator, the policy repeatedly receives current camera/state
   observations and predicts the next control.
6. MuJoCo advances the scene, and the task checker determines whether the mug
   placement, gripper, and robot pose satisfy the success condition.

That cycle—from human example, to dataset, to learned controller, to interactive
evaluation—is the core concept behind the project.
