# TOP-ERL: Transformer-based Off-policy Episodic RL (ICLR25 Spotlight)

<p align="center">
  <img src='./web_assets/Metaworld.gif' width="200" />
  <img src='./web_assets/Box_Pushing.gif' width="200" />
  <img src='./web_assets/rollout.png' width="243" />
</p>

<br>

<p align="center">
  <img src='./web_assets/results_new.png' width="650" />
</p>

<br><br>

## Episodic RL, What and Why?
Episodic Reinforcement Learning (ERL) [1, 4, 5] is a distinct RL family that emphasizes the maximization of returns over entire episodes, typically lasting several seconds, rather than optimizing the intermediate states during environment interactions. Unlike Step-based RL (SRL) [2, 3], ERL shifts the solution search from per-step actions to a parameterized trajectory space, leveraging techniques like Movement Primitives (MPs) [6, 7, 8] for generating action sequences. This approach enables a broader exploration horizon [4], captures trajectory statistics [9], and ensures smooth transitions between re-planning phases [10].

<p align="center">Exploration Strategies Comparison, SRL vs. ERL [9]</p>

<table align="center">
  <tr>
    <td align="center">
      <img src='./web_assets/SRL.png' width="300" /><br>
      <em>Step-based RL explores per action step</em>
    </td>
    <td align="center">
      <img src='./web_assets/ERL.png' width="300" /><br>
      <em>Episodic RL has consistent exploration</em>
    </td>
  </tr>
</table>



## Use Movement Primitives for Trajectory Generation
Episodic RL often uses the movement primitves (MPs) as a paramterized trajectory generator. In TOP-ERL, we use the ProDMP [8] for fast computation and better initial condition enforcement. A simple illustration of using MPs can be seen as follows:

<p align="center">
  <img src='./web_assets/mp_demo.gif' width="600" /><br>
  <em>MP predicts a trajectory (upper curve) by adjusting the weights of the basis functions (lower curves)</em>
</p>

## Use Transformer as an Action Sequence Critic
In the literature, most of the combinations of RL and Transformers focus on offline, model-based and POMDP settings. Directly using tranformer in online RL for acition sequence value prediction remains highly unexplored. In TOP-ERL, we utilize Transformers as an action sequence value predictor, training it via the N-step future returns. To ensure stable critic learning, we adapt the trajectory segmentation strategy in [9] by splitting the long trajectory into sub-sequences of varying lengths.

<p align="center">
  <img src='./web_assets/critic_animation_gif.gif' width="900" /><br>
  <em>TOP-ERL utilizes a transformer critic that predicts the value of executing a sub-sequence of actions from the beginning of the segment state. </em>
</p>


## Installation Tutorial
0. We tested our installation using the following PC setup:
```
	- Ubuntu 22.04
	- RTX 2060 Super GPU
	- git is installed with "sudo apt install git-all"
	- Github account with ssh access
```

We provide a 12 min long [tutorial video](https://www.youtube.com/watch?v=y-d1E0qkZFM) to guide your installation step-by-step. This video contains the following steps:

1. Install [Mamba](https://github.com/conda-forge/miniforge/releases) (a faster conda release)	

2. Activate mamba in your teminal
```
	source .bashrc  #if you use bash
```

3. Clone the repository
```
	mkdir top_erl
	cd top_erl
	git clone git@github.com:BruceGeLi/TOP_ERL_ICLR25_Code.git
```
4. Install dependencies
```
    cd TOP_ERL_ICLR25_Code
    bash conda_env.sh	
```
Wait for 10 min until finish (depend on the internet speed)

5. Activate the mamba(conda) env by:
```
  mamba activate top_erl_iclr25
```

6. Register a [wandb](https://wandb.ai/) account and login in your local PC:
```
  wandb login --relogin
```

7. Replace the wandb username in the config file, such as "shared_dense.yaml".

8. Run experiment locally, e.g. box pushing dense reward setting
```
  python seq_mp_exp_multiprocessing.py config/box_push_random_init/seq/entire/local_dense.yaml -o --nocodecopy
```

9. To run experiments in slurm-based HPC, you need to adapt your hpc info in our slurm configs. An example of running code in slurm is:
```
  python seq_mp_exp_multiprocessing.py config/box_push_random_init/seq/entire/slurm_dense.yaml -o --nocodecopy
```

10. We used [cw2](https://pypi.org/project/cw2/) to parse our experiment configs into sbatch commands in slurm based HPC system. For more technical details, we refer the documents in cw2.

<br><br>
### References
[1] Darrell Whitley, Stephen Dominic, Rajarshi Das, and Charles W Anderson. Genetic reinforcement learning for neurocontrol problems. Machine Learning, 13:259–284, 1993.

[2] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[3] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. PMLR, 2018a.

[4] Jens Kober and Jan Peters. Policy search for motor primitives in robotics. NIPS, 2008.

[5] Jan Peters and Stefan Schaal. Reinforcement learning of motor skills with policy gradients. Neural networks, 21(4):682–697, 2008.

[6] Stefan Schaal. Dynamic movement primitives-a framework for motor control in humans and humanoid robotics. In Adaptive motion of animals and machines, pp. 261–280. Springer, 2006.

[7] Alexandros Paraschos, Christian Daniel, Jan Peters, and Gerhard Neumann. Probabilistic movement primitives. Advances in neural information processing systems, 26, 2013.

[8] Ge Li, Zeqi Jin, Michael Volpp, Fabian Otto, Rudolf Lioutikov, and Gerhard Neumann. Prodmp:A unified perspective on dynamic and probabilistic movement primitives. IEEE RA-L, 2023.

[9] Ge Li, Hongyi Zhou, Dominik Roth, Serge Thilges, Fabian Otto, Rudolf Lioutikov, and Gerhard Neumann. Open the black box: Step-based policy updates for temporally-correlated episodic reinforcement learning. ICLR 2024.

[10] Fabian Otto, Hongyi Zhou, Onur Celik, Ge Li, Rudolf Lioutikov, and Gerhard Neumann. Mp3: Movement primitive-based (re-) planning policy. arXiv preprint arXiv:2306.12729, 2023.

