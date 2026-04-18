**ECE 567 Final Project Ideas**

**Project 1**: Unsupervised Reinforcement Learning  
Benchmark: [URLB](https://github.com/facebookresearch/controllable_agent)

* Baselines  
1. [Diversity is all you need.](https://arxiv.org/pdf/1802.06070)  
2. [Forward Backward Representation](https://arxiv.org/pdf/2103.07945)  
* Envs:  
  * [Dm\_control](https://github.com/google-deepmind/dm_control) (built on mujoco)  
  * Diverse range of implemented tasks, such as velocity tracking (see [custom tasks](https://github.com/ahmed-touati/url_benchmark/tree/main/custom_dmc_tasks))  
    * Hopper  
    * Cheetah  
    * Quadruped  
    * Walker

**Project 2**: Online Goal-Conditioned Reinforcement Learning  
Benchmark: [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL)

* Baselines:   
  * [Contrastive Reinforcement Learning](https://arxiv.org/pdf/2206.07568)  
  * Goal-conditioned [SAC](https://arxiv.org/pdf/1801.01290)  
  * Goal-conditioned [PPO](https://arxiv.org/pdf/1707.06347)  
  * Goal-conditioned [TD3](https://arxiv.org/pdf/1802.09477)  
* Envs:  
  * [Brax](https://github.com/google/brax) (GPU version of Mujoco)  
  * Have both locomotion and manipulation tasks.

**Project 3**: Offline Goal-Conditioned Reinforcement Learning  
Benchmark: [OGBench](https://github.com/seohongpark/ogbench)

* Baselines:  
1. Offline [CRL](https://arxiv.org/pdf/2206.07568)  
2. [HIQL](https://arxiv.org/pdf/2307.11949)  
3. [QRL](https://arxiv.org/pdf/2304.01203)  
4. [Implicit Q/V Learning](https://arxiv.org/pdf/2110.06169)  
* Envs:  
  * Locomotion maze environments  
  * Manipulation environments.  
  * Powderworld.  
* Example Follow-ups:  
  * TMD: [Code](https://github.com/vivekmyers/tmd-release)  
  * MQE: [Code](https://github.com/WJ2003B/mqe-release)

**Project 4:** Continual Reinforcement Learning  
Benchmark: [CORA](https://github.com/AGI-Labs/continual_rl)

* Baselines:  
  * [EWC](https://arxiv.org/pdf/1612.00796)  
  * [PC](https://arxiv.org/pdf/1805.06370)  
  * [CLEAR](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf)  
* Envs:  
  * Atari  
  * Procgen  
  * Minihack

Envs without Benchmarks:

* [AgarCL](https://github.com/machado-research/AgarCL-benchmark)  
* [Jelly Bean World](https://github.com/eaplatanios/jelly-bean-world)

**Project 5:** Open-ended Reinforcement Learning  
Benchmark: [Craftax baselines](https://github.com/MichaelTMatthews/Craftax_Baselines)

* Baselines:  
  * [PPO](https://arxiv.org/pdf/1707.06347)  
  * [ICM](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)  
  * [RND](https://arxiv.org/pdf/1810.12894)  
* Envs:  
  * [Craftax](https://github.com/MichaelTMatthews/Craftax)

**Project 6**: Safe Reinforcement Learning  
Benchmark: [Omnisafe](https://github.com/PKU-Alignment/omnisafe)

* Baselines:  
  * [CPO](https://arxiv.org/pdf/1705.10528)  
  * [FOCOPS](https://arxiv.org/abs/2002.06506)  
  * [PPO-Lagrangian and TRPO-Lagrangian](https://cdn.openai.com/safexp-short.pdf)  
* Envs:  
  * [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/en/latest/introduction/about_safety_gymnasium.html)  
  * [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym)

**Project 7**: Multi-agent Reinforcement Learning  
Benchmark: [BenchMARL](https://github.com/facebookresearch/BenchMARL)

* Baselines:  
  * [MAPPO](https://arxiv.org/abs/2103.01955)  
  * [IPPO](https://arxiv.org/abs/2011.09533)  
* Envs:  
  * [PettingZoo](https://github.com/Farama-Foundation/PettingZoo/tree/master)  
  * [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator)

**Project 8**: Queueing Network Controls via Reinforcement Learning  
Benchmark: [QGym](https://arxiv.org/html/2410.06170v1)  
Envs: [DiffDiscreteEventSystem](https://github.com/namkoong-lab/QGym)  
**Project 9:  The Nethack environment**   
[Paper](https://arxiv.org/abs/2006.13760)   
[Envs and codebase](https://github.com/NetHack-LE/nle?tab=readme-ov-file) 