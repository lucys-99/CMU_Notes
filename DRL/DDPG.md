##### **What it is**
- An **actor–critic algorithm** for **continuous action spaces**.
- Extends **Deterministic Policy Gradient (DPG)** with deep neural networks.
- Uses **off-policy replay buffer + target networks**, similar to DQN.
##### **Key Equations**
- **Actor (policy):** deterministic mapping a=πθ(s)a = \pi_\theta(s)a=πθ​(s).
- **Critic (Q-function):** $L(\varphi) = \mathbb{E}_{(s,a,r,s') \sim D} \Big[ \big(Q_\varphi(s,a) - (r + \gamma Q^{tgt}_\varphi(s',\pi_\theta(s'))) \big)^2 \Big]$
- **Actor update:**
    $\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim D} \big[ \nabla_\theta \pi_\theta(s) \, \nabla_a Q_\varphi(s,a)|_{a=\pi_\theta(s)} \big]$
#### **Why use it**
- Standard policy gradient (like A2C) struggles in **continuous actions** because of high variance.
- DDPG uses a **deterministic actor** + critic gradient → efficient learning.
#### **Problems it solves**
- Makes **continuous control feasible** (robotics, control tasks).
- More **sample efficient** than on-policy methods like A2C.
### **DDPG vs A2C**
- **DDPG is off-policy** → more sample efficient.
- **Handles continuous actions** directly, unlike A2C which needs Gaussian approximation.
- **But** DDPG is less stable, brittle to hyperparameters, prone to Q-value overestimation.
### Advantages & Disadvantages
##### ✅ **DDPG Advantages**
- Efficient in continuous action tasks.
- Off-policy replay → sample efficiency.
- Simple actor-critic structure.
##### ❌ **DDPG Disadvantages**
- Brittle, unstable training.
- Sensitive to hyperparameters.
- Poor exploration (deterministic policy).
### ✅ **A2C Advantages**
- Simpler to implement.
- Stable for small/discrete tasks.
- On-policy → less sensitive to extrapolation bias.
### ❌ **A2C Disadvantages**
- Sample inefficient.
- High variance updates.
- Struggles in continuous action spaces.
### Takeaway
- **A2C:** Good for discrete, small-scale on-policy tasks.
- **DDPG:** Makes continuous control feasible, but unstable.
- **SAC:** Fixes DDPG’s brittleness with max-entropy RL, now one of the **most popular continuous control algorithms**.