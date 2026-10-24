# Project Demo: Presentation Script and Q&A

## The project in one sentence

This project uses robot demonstrations to train a policy to pick up a mug and
place it on a plate in a MuJoCo simulation.

## Suggested presentation flow

Aim for about five minutes. Keep the live simulation as the main visual part.
Do not try to train the model during the presentation; training can be slow on
the current laptop setup.

### 1. Introduce the problem

Say:

> “The goal is to teach a simulated robot arm a pick-and-place task: move a mug
> onto a plate. Instead of programming every movement by hand, I use examples
> of a person controlling the robot.”

Point out that this is a simulation project. The robot in the demo is not a
physical robot.

### 2. Explain how demonstrations become data

Say:

> “While a person teleoperates the arm, the project records what the robot
> observes and what action the person commands. The observations include camera
> images and the robot's state. The actions include robot movement and gripper
> control. A sequence of records forms a demonstration episode.”

If asked, show a short sample replay or the data folder. Avoid spending too long
on file formats unless the audience asks.

### 3. Explain training

Say:

> “The ACT policy is trained from those examples. During training, it predicts
> actions from observations and compares its predictions with the demonstrated
> actions. The optimizer adjusts the model so that its predictions become more
> like the examples.”

Explain that ACT predicts short sequences, or **chunks**, of actions. In the
example setup, the training chunk size is 10. This is still imitation learning:
the model learns from demonstrations rather than discovering the task by
trial-and-error reinforcement learning.

### 4. Show deployment

Load the already-saved policy checkpoint in the deployment or demo notebook.
Then explain the loop as the simulation runs:

```text
camera images + robot state
          -> trained policy
          -> predicted action
          -> simulated robot moves
          -> new images + state
          -> repeat
```

Say:

> “The checkpoint is a saved version of the trained model. The deployment
> notebook loads it, gives it the current camera images and robot state, and
> applies the action it predicts in MuJoCo. This loop continues so the policy
> can react to the changing scene.”

### 5. Explain the result honestly

The current example dataset contains **one episode with 137 frames**. That is
enough to demonstrate the workflow, but it gives the policy very little
variation to learn from. It may imitate the demonstrated setup without
generalizing to a mug at a substantially different position.

If showing the shifted-mug experiment, introduce it as a test rather than a
guaranteed success:

> “I changed the mug's starting position to test generalization. The current
> model did not pick it up reliably. The dataset has only one demonstration, so
> this result shows why varied training examples matter.”

This is a useful finding, not a reason to claim that the policy can handle
arbitrary object placements.

## Ready-to-say short script

> “My project demonstrates imitation learning for robotic manipulation in
> simulation. The task is to move a mug onto a plate using a simulated robot
> arm. A person can teleoperate the arm, and the project records camera images,
> robot state, and the actions taken. Those demonstrations are organized into
> a dataset and used to train an ACT policy. The trained model is saved as a
> checkpoint. During deployment, the notebook loads that checkpoint and
> repeatedly sends the current simulated camera images and robot state to the
> policy. The policy predicts an action, MuJoCo applies it, and the process
> repeats. My example dataset has one short episode, so the model is a proof of
> the pipeline rather than a robust controller. When I moved the mug, it failed
> to pick it up reliably, which illustrates the need for more varied
> demonstrations.”

## Likely teacher questions

### What is the project trying to solve?

It demonstrates how a robot policy can learn a manipulation behavior from
human-controlled examples, using mug-to-plate pick-and-place as the task.

### What is imitation learning?

Imitation learning trains a model from examples of an expert's behavior. Here,
the operator is the expert, and the recorded actions are the examples the policy
tries to reproduce.

### What is ACT?

ACT stands for **Action Chunking Transformer**. It is the policy used in the
basic training example. Rather than predicting only one isolated action, it is
configured to predict a short chunk of future actions.

### What does the model receive, and what does it output?

It receives the observations configured for the policy, including an image and
the robot's six-dimensional end-effector state. It predicts a seven-dimensional
action: six robot control values and a gripper command.

### What is a checkpoint?

A checkpoint is a saved copy of the trained policy's parameters and related
configuration. Training creates or updates it; deployment loads it so the
policy can make predictions without training again.

### Does deployment train the model?

No. Training adjusts the model using the dataset. Deployment loads an existing
checkpoint and uses it to predict actions in a running simulation.

### How do you decide whether the task succeeded?

The MuJoCo environment has a programmatic success check. It checks the mug's
position relative to the plate, whether the gripper is open, and whether the
end effector has moved upward. This check evaluates the simulated result; it is
not the ACT training loss.

### Why did the robot fail after the mug was moved?

The provided example dataset contains one episode with 137 frames, which has
very little variation in starting conditions. A model trained on that data may
not have learned how to handle a new mug position. More varied successful
demonstrations and retraining would be needed to test whether it can generalize
better.

### Is this reinforcement learning?

The basic ACT notebook is supervised imitation learning, also called behavior
cloning. It learns from recorded observation/action examples. The simulation's
success checker is used to evaluate a rollout, not as a reward that drives the
ACT notebook's training loop.

### Is this running on the physical robot?

No. The rollout shown in these notebooks is in MuJoCo, a physics simulator.
Moving to a physical robot would require hardware interfaces, calibration,
safety controls, and separate testing.

### Why is training slow?

The notebook's current Python environment uses CPU-only PyTorch, even though
the laptop has a GTX 1050 GPU. That means PyTorch cannot use the GPU until a
compatible CUDA-enabled PyTorch build is installed in the notebook's
environment. The GPU has 4 GB of VRAM, so training would also likely need a
smaller batch size than 64.

### What would you improve next?

Collect more successful demonstrations with the mug and plate at varied
positions, train a new checkpoint, and evaluate it on positions not included in
the training demonstrations. Compare success across multiple attempts instead
of relying on a single rollout.

## Demo-day checklist

- Start from a fresh notebook kernel and run setup cells in order.
- Confirm that the local dataset, simulation assets, and checkpoint are present.
- Confirm the device is selected with
  `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Test the MuJoCo viewer before presenting.
- Keep a short screen recording available in case the live simulation fails.
- Be clear that the moved-mug experiment is a generalization test and may fail.
- Do not describe the project as a real-robot deployment or as a robust,
  production-ready manipulation system.
