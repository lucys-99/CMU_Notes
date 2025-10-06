Scribe 9 Group 1
##### 1. Core Objective of Offline Actor-Critic
Critic loss: $L(\varphi) = \mathbb{E}_{s,a,r,s' \sim D} \Big[ \big(Q_\varphi(s,a) - y\big)^2 \Big]$
Target: $y = r(s,a) + \gamma \; \mathbb{E}_{a' \sim \pi_\theta(\cdot|s')} \Big[ Q^{tgt}_\varphi(s',a') \Big]$
- $Q_\varphi(s,a)$: critic (action-value function).
- $\pi_\theta(a|s)$: actor (policy).
- D: offline dataset (collected from some behavior policy, not updated further).
- $Q^{tgt}_\varphi$: target critic (stabilizes learning).
So: **critic approximates Q-values**, **actor learns from critic’s gradients**.
##### 2. Motivation for Offline Actor-Critic
- **Q-learning (value-based):** Needs action maximization inside the target $(\max_{a'}Q(s',a')$. In **offline RL**, if we pick actions not in dataset, we extrapolate poorly → overestimation.
- **Actor-Critic (online):** Trains actor $\pi_\theta$​ alongside critic, but needs new samples each step → not possible offline.
- **Offline Actor-Critic:**
    - Avoids unsafe exploration by learning only from dataset D.
    - Uses **actor network** to replace the hard $⁡\max$ in Q-learning with a **soft policy improvement** step.
    - Stabilizes training by coupling actor + critic updates.
👉 Key motivation: Make RL **data-efficient and safe** when no environment interaction is allowed (e.g., healthcare, robotics logs).
#### 3. Training Procedure
1. **Critic update (Q-function):**  
    Minimize TD error using $L(\varphi)$ above.
    - This is similar to Q-learning but with expectation over actor’s actions.
2. **Actor update (policy improvement):**  
    Maximize expected Q-value: $\nabla_\theta J(\pi_\theta) \approx \mathbb{E}_{s \sim D, \, a \sim \pi_\theta} \Big[ \nabla_\theta \log \pi_\theta(a|s) \, Q_\varphi(s,a) \Big]$
    - For deterministic policies (DDPG, TD3): $\nabla_\theta J(\pi_\theta) \approx \mathbb{E}_{s \sim D} \big[ \nabla_\theta \pi_\theta(s) \, \nabla_a Q_\varphi(s,a) \big|_{a=\pi_\theta(s)} \big]$
    - For stochastic policies (SAC): includes entropy bonus.
3. **Target networks:**  
    Slowly update $Q^{tgt}$ to stabilize bootstrapping.
#### 4. How is it Different?
- **vs Value-Based Q-Learning:**
    - Q-learning uses $\max_{a'} Q(s',a')$ in targets → unstable in offline setting.
    - Actor-Critic uses $\pi_\theta$​ instead of max, keeping updates **policy-constrained**.
- **vs Online Actor-Critic:**
    - Online: samples from $\pi_\theta$ continuously.
    - Offline: samples only from fixed dataset DDD. Must be careful of **distribution shift** (if $\pi_\theta$​ picks actions not covered in DDD, critic estimates are unreliable).
#### 5. The 0-Gradient Problem
- In offline Q-learning, if $\pi_\theta$​ picks actions outside dataset support, critic has no reliable value → gradients vanish or explode.
- **Offline actor-critic fixes this** by:
    - Training actor to **stay close to dataset distribution** (constrained updates, KL penalties, conservative critics).
    - Using critic gradients from **actions in support of D**, ensuring non-zero learning signal.
    - Entropy terms (SAC) encourage exploration within support, avoiding collapse.
#### 6. Variants
##### **[[DDPG]] (Deep Deterministic Policy Gradient)**
- Actor: deterministic policy $\pi_\theta(s)$.
- Critic: Q-function trained with TD error.
- Problem: High variance, overestimation.
##### **TD3 (Twin Delayed DDPG)**
- Fixes DDPG instability:
    - **Twin critics** → reduce overestimation (take min of two Qs).
    - **Target policy smoothing** → add noise to actions in target, reduce Q-exploitation of sharp peaks.
    - **Delayed actor updates** → stabilize actor training.
##### **[[SAC]] (Soft Actor-Critic)**
- Actor: stochastic policy with entropy regularization.
- Critic: learns Q-values.
- Update target: $y = r(s,a) + \gamma \, \mathbb{E}_{a' \sim \pi_\theta} \big[ Q^{tgt}(s',a') - \alpha \log \pi_\theta(a'|s') \big]$
- Motivation: maximize reward + exploration (entropy).
- Benefits: robust, good for continuous action spaces, sample efficient.
#### 7. Advantages
✅ **Offline Actor-Critic in general:**
- More stable than offline Q-learning.
- Uses actor to guide critic → avoids extrapolation errors.
- Can reuse logged data without dangerous exploration.
✅ **Algorithm-specific:**
- **DDPG:** Simple, works for continuous actions.
- **TD3:** Stable, reduces bias.
- **SAC:** Stochastic & entropy-regularized → better exploration + robustness.
#### 8. Disadvantages
❌ Offline actor-critic:
- Still suffers from **distribution shift** → policy may pick actions unseen in data.
- Needs **regularization (KL, conservative critics, behavior cloning)** to avoid out-of-distribution actions.
- Actor updates depend on critic accuracy — if critic is biased, actor is misled.
❌ Specific:
- **DDPG:** Unstable, sensitive to noise.
- **TD3:** More stable, but slower due to delayed updates.
- **SAC:** Extra hyperparameter α\alphaα (temperature); harder to tune.
#### 9. Takeaway Summary
- **Offline Actor-Critic** = learn critic from dataset, train actor using critic gradients.
- Key difference vs Q-learning: replaces $\max_{a'}$ with expectation under $\pi_\theta$​.
- Prevents 0-gradient by constraining actor to stay close to data distribution.
- **DDPG, TD3, SAC** are practical actor-critic variants:
    - DDPG = simple deterministic baseline.
    - TD3 = fixes overestimation.
    - SAC = entropy-regularized, robust.

Offline actor-critic is different because it learns exclusively from a **fixed, pre-collected dataset** without any active exploration, whereas online actor-critic continuously interacts with the environment to gather new data.
##### How Offline Actor-Critic is Different
The core difference isn't just the data source, but the fundamental challenge that arises from it. This challenge shapes the entire algorithm.
1. **Data Source**:
    - **Online AC**: Follows the standard loop: act, collect new data, update policy, repeat. It generates its own data.
    - **Offline AC**: Is given a static batch of data (e.g., logs from a human expert or a previous algorithm) and can _never_ interact with the environment to collect more.
2. **The Core Challenge: Distribution Shift**: This is the main technical problem that offline algorithms must solve. The actor, in its attempt to improve, might propose an action that is **out-of-distribution**—meaning, an action that is not present in the static dataset for a given state.
    - When this happens, the critic has no real data to evaluate this new action. It can produce a wildly incorrect and often overly optimistic Q-value, a phenomenon called **extrapolation error**.
    - The actor will then exploit this error, learning to choose this "fantasy" action that the critic _thinks_ is good, leading to a policy that performs terribly in the real world.
3. **The Algorithmic Solution: Policy Constraints**: To combat this, offline actor-critic algorithms must include a mechanism to **constrain the actor's policy**. They force the learned policy to stay "close" to the actions and behaviors already present in the dataset, preventing it from straying into out-of-distribution actions where the critic is unreliable.
##### Why Use Offline Actor-Critic? 🤔
You would use offline actor-critic methods when online data collection is impossible, unsafe, or expensive. The goal is to leverage large, existing datasets to train decision-making policies without the risks of live experimentation.
- **Safety-Critical Applications**: In fields like robotics, autonomous driving, or healthcare, letting a partially trained agent experiment in the real world can have catastrophic consequences. Offline RL allows you to learn a policy from a safe, pre-collected set of data.
- **Expensive Data Collection**: When each interaction is costly (e.g., running complex scientific experiments, conducting user studies, or managing industrial processes), you want to extract the maximum value from the data you already have.
- **Data is the Only Resource**: In many cases, you may only have access to historical logs and cannot interact with the system at all. Examples include learning trading strategies from past market data or optimizing patient treatment plans from electronic health records.