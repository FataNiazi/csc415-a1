# Reproducing Domain Randomization for Sim-to-Real
Reproducing "[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)" by Tobin et al. (2017)


## Intro

This repository is my implementation of the domain randomization setup, based on the [implementation by matwilso](https://github.com/matwilso/domrand). The core data generation and training logic is similar, with modifications to support newer library versions and GPUs.

## Setup (same as the original repository)

Follow the instructions at the [mujoco\_py](https://github.com/openai/mujoco-py)
repo.  Installation can be a pain, and see the tips [below](#mujoco) for some GitHub
issues to check out if you have any problems with installation.

You may have to install some extra packages (see my [`apt.txt`](./apt.txt) for 
a (partial) list of these).  


To install the python dependencies:
```
pip install -r requirements.txt
```

## Reproducing Results
Once setup is complete, the results for each part of the paper can be reproduced using the commands below. The ablation experiments were originally run individually with different hyperparameters; an AI-generated script is included to generate all required data and run training in a single command.

### Reproducing the Main Experiment
This script generates data with different numbers of unique randomized textures and runs the same experiment on all of them, saving the results in `texture_ablation_results.csv` and the diagram `texture_ablation_graph.png`. The dataset for each experiment contains 10,000 examples.

```bash
python texture_ablation.py
```

### Reproducing the Ablation Study
This script runs the ablation study experiments, investigating the effect of camera randomization, noise, and distractors in the training dataset on sim-to-real transfer.

The ablation script requires pre-generated data. Run the data generation script first:

```bash
python generate_data.py
```

Then run the experiments:

``` bash
python run_ablation_experiments.py
```

The results are saved in the `ablation_results.csv` file.


<a name="mujoco"></a>

## Mujoco Tips (same as the original repository)
- READ [THIS](https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564) if you are getting ERROR: GLEW initialization error: Missing GL version
- You need to call sim.forward() or sim.step() to get the camera and light modders to update
- You can't scale a mesh after it has been loaded (http://www.mujoco.org/forum/index.php?threads/how-to-scale-already-exist-model.3483/)
- Read this: https://github.com/openai/mujoco-py/issues/148 and this: https://github.com/openai/gym/issues/234
- The maximum number of lights that can be active simultaneously is 8, counting the headlight
- More documentation on lighting can be found here: http://www.glprogramming.com/red/chapter05.html#name10
- To make it so cameras don't look through walls, you need to add:

```
  <visual>
    <map znear=0.01 /> 
  </visual>
```


**Running ablation experiments**

To run all 4 ablation experiments from the paper (full method, no noise, no camera randomization, no distractors):

```
python3 run_ablation_experiments.py
```

For GPU training with XLA (recommended for H100):
```
TF_XLA_FLAGS="--tf_xla_auto_jit=2" python3 run_ablation_experiments.py
```
