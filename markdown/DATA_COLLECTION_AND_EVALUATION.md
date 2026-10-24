# Collecting Better Demonstrations and Evaluating a Policy

This guide focuses on improving the evidence and data used by the mug-to-plate
experiment. More data is useful only when the demonstrations are successful,
consistent, and varied in relevant ways.

## Why the example can fail on a moved mug

The included example dataset contains one episode with 137 frames. Those frames
are a short sequence from a single demonstration, not 137 independent examples
of different mug placements. The model therefore has very limited evidence of
how the scene looks and how the robot should act in other starting conditions.

Moving the mug is a **generalization test**: it asks the policy to succeed under
a condition that may not be represented in the training data. A failure does
not necessarily mean the software pipeline is broken; it may mean the learned
behavior does not cover that new condition.

## What makes a useful dataset?

Aim for demonstrations that are:

- **Successful:** the mug is grasped, transported, released onto the plate, and
  the robot finishes in a sensible pose.
- **Complete:** record the approach, grasp, lift, transport, release, and
  retreat—not just the middle of the motion.
- **Consistent:** use the same control conventions and task meaning across
  episodes.
- **Varied where desired:** if the policy should handle different mug positions,
  include demonstrations starting from different positions.
- **Representative:** include variations the policy is expected to encounter,
  such as modest changes in object placement or approach path.
- **Cleanly segmented:** start episodes from a reset and discard unfinished or
  corrupted attempts instead of treating them as successful demonstrations.

Collecting many poor or contradictory attempts can make the learning problem
harder. Prioritize clear successful demonstrations over a large count.

## A practical collection plan

1. Start the simulation from a known reset.
2. Choose a mug and plate placement within the reachable tabletop region.
3. Teleoperate a complete, successful pick-and-place.
4. Save that attempt as an episode.
5. Reset and change the starting arrangement slightly.
6. Repeat until the dataset covers the variations you want to test.
7. Replay a sample of episodes before training to check camera views, actions,
   and episode endings.

The environment's reset logic samples object coordinates from a bounded area.
When collecting varied data, make sure the saved episodes actually contain
different initial object positions; do not rely only on assuming that resets
varied the scene.

## Keep a held-out test set

To test whether the policy generalizes rather than merely replays familiar
placements:

1. Use some starting arrangements for training.
2. Reserve other starting arrangements for evaluation.
3. Do not train on the held-out evaluation attempts.
4. Run multiple policy rollouts on held-out starts.
5. Record outcomes and failure types.

If there is only one episode, a meaningful train/test separation is not
possible. Treat results from that dataset as a pipeline demonstration rather
than a robust performance estimate.

## What to measure

For each rollout, record at least:

- Initial mug and plate positions.
- Whether the policy grasped the mug.
- Whether the mug remained in the gripper during transport.
- Whether the mug ended up on the plate.
- Whether the simulator's success check passed.
- Any failure point (missed grasp, collision, dropped object, wrong release,
  timeout).

Report both successes and total attempts. For example, “3 successful rollouts
out of 5 held-out trials” is more informative than showing only the best run.
The simulator success condition is a useful metric, but it is specific to the
environment's programmed rule.

## Training changes for a small GPU

The current project environment was observed to have CPU-only PyTorch, while
the machine has an NVIDIA GTX 1050 with 4 GB of VRAM. To use the GPU:

1. Install a CUDA-enabled PyTorch build in the same `.venv` used by the
   notebook, following the official PyTorch installation selector.
2. Restart the notebook kernel after installation.
3. Verify `torch.cuda.is_available()` returns `True` and that the expected GPU
   name appears.
4. Start with a small training batch such as 1, 2, or 4 and increase only if
   memory permits.

The ACT notebook has batch-size settings in the dataset/DataLoader setup. If
changing batch size, use the same intended value in the DataLoader definition
that is active when the training loop starts. If CUDA reports out-of-memory,
reduce the batch size and restart the kernel to clear allocated memory.

GPU acceleration may reduce training time, but it does not increase the amount
of information in a one-episode dataset. Data quality and variation still
matter.

## Comparing two checkpoints fairly

When comparing the original and improved policies:

- Use the same environment scene and success rule.
- Use the same held-out starting arrangements.
- Reset the environment before every trial.
- Use several trials per checkpoint.
- Keep the camera preprocessing and policy configuration consistent.
- Save checkpoints to different directories so one experiment does not
  overwrite another.

This makes it easier to tell whether a change in data or training actually
improved results.
