# A2C

#### Actor-Critic Methods
- Combine policy gradient (**actor**) with value function approximation (**critic**).
- Actor: parameterized policy $\pi_\theta(a \mid s)$.
- Critic: estimates value function $V^\pi_w(s)$ or action-value $Q^\pi_w(s,a)$.
- **Update rules:**
    - Actor update:  
        $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A_t$
    - Critic update:  
        $w \leftarrow w - \beta \nabla_w \big(V_w(s_t) - (r_t + \gamma V_w(s_{t+1}))\big)^2$
- **Advantage function (reduces variance):**  
    $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$  
    Using $A^\pi$ instead of raw returns improves stability.
- **Q function:**
	- The **Q-function** (or Action-Value function), denoted as $Q(s, a)$, estimates the total expected future reward an agent will receive if it starts in a specific state `$s$`, takes a specific action $a$, and then follows its policy forever after
- Examples: **A2C, A3C, PPO, DDPG, SAC.**
#### Here’s the intuition:
1. **The Problem with Basic Policy Gradients**: Simple policy gradient methods (like REINFORCE) update the policy based on the total reward of an entire episode. This signal is very noisy; a single lucky action can appear good even if the overall policy is bad, and vice-versa.
2. **The Critic's Role**: To fix this, Actor-Critic methods introduce a **Critic**. The Critic's job is to provide a more stable, lower-variance evaluation of the Actor's actions.
3. **Using V-Values to Get the Advantage** (MC or TD): Instead of learning the value of every state-action pair (which can be inefficient), the A2C Critic learns the value of each **state ($V(s)$)**. It then uses this to calculate the **Advantage ($A(s, a)$)**:
		$A(s_t​,a_t​)≈(r_t​+γV(s_t+1​))−V(s_t​)$
    The Advantage doesn't just tell you if an action was good in an absolute sense; it tells you **how much better or worse it was than the average action** from that state. This is a much more powerful and stable learning signal.
    - If $A > 0$, the action was better than expected. The Actor should increase its probability.
    - If $A < 0$, the action was worse than expected. The Actor should decrease its probability.
- **A2C’s Use of Q-function**
	- A2C doesn’t explicitly learn the full **Q-function**.
	- Instead, it learns **state-value function $V^\pi(s)$** (via the critic network).
	- The **Q-function is implicitly used** in defining the advantage:
	    - When estimating $A(s,a)$, A2C approximates:
	        $A(s_t,a_t) \approx \big(r_t + \gamma V(s_{t+1})\big) - V(s_t)$
	    - Here, $r_t + \gamma V(s_{t+1})$ is essentially a **one-step sample-based estimate of $Q(s_t,a_t)$**.
#### How does N-Step help:
While using a critic helps, there's still a trade-off between bias and variance in how you estimate the advantage.
- **1-Step Return (Low Variance, High Bias)**: A basic Actor-Critic uses a 1-step lookahead: $A(s_t, a_t) \approx (r_t + \gamma V(s_{t+1})) - V(s_t)$. This has low variance because it depends on only one real reward, but it has high bias because it relies heavily on the potentially inaccurate estimate from the critic ($V(s_{t+1})$).
- **Monte Carlo Return (High Variance, Low Bias)**: REINFORCE uses the full return. This is an unbiased estimate of the true return, but it has very high variance.
**N-step returns** provide a compromise between these two extremes. The N-step return is calculated by summing the next `$N$` real rewards and then "bootstrapping" (using the estimated value) from the state you land in after $N$ steps: $G_{t:t+N​}=r_t​+γr_{t+1​}+⋯+γ^{N−1}r_{t+N−1}​+γ^N V(s_{t+N​})$
Using this N-step return to calculate the advantage helps in two ways:
1. **It reduces bias** compared to a 1-step return because it incorporates more real, unbiased reward information before relying on the critic's estimate.
2. **It reduces variance** compared to a full Monte Carlo return because the sequence of random variables is much shorter (`$N$` steps vs. the rest of the episode).
![A2C Diagram](CMU_Notes/DRL/img/A2C.png)
Improve diversity of updates:
[[A3C]]

---

# A3C

### A2C (synchronous).
Multiple environments roll out for T steps, compute gradients, _aggregate synchronously_, then update the global network and broadcast parameters.
### A3C (asynchronous).
Workers maintain local copies, interact with different parts of the environment, compute gradients, and _update the global network asynchronously_ (no locks across workers). This decorrelates data and stabilizes on-policy learning without a replay buffer.
The core idea of A3C is to have multiple "worker" agents interacting with their own copies of the environment simultaneously. These workers update a single "global" network.
- **Global Network**: There is one central network, often called the "target" or "global" network, which contains the master parameters for both the policy (Actor) and the value function (Critic).
- **Worker Agents**: Multiple worker agents are created, each with its own set of network parameters and its own copy of the environment. These workers run in parallel on different CPU threads.
- **Asynchronous Learning Loop**: Each worker follows this process independently:
    - It copies the latest parameters from the global network.
    - It interacts with its environment for a small number of steps (e.g., 5 steps), collecting experiences (states, actions, rewards).
    - It calculates the gradients for both its Actor and Critic networks based on these experiences.
    - It sends these gradients up to the global network.
    - The global network updates its parameters using the gradients from that worker. The worker then repeats the process.
#### Problem:
- The problem with that is the experience is very correlated and sequential which can cause instability when training the neural network. We want to diversify the experience and de-correlated the gradient updates. In other words, we want to parallelize the collection of experiences and stabilize training with multiple threads of experiences and different exploration strategies. --> Asynchronous Deep RL
- Distributed RL
	- Distributed Synchronous RL-A2C: All agents have identical policies. Each one collect experiences, compute gradients, aggregate all of them to update the policy, and reshare.![Distributed_A2C](CMU_Notes/DRL/img/Distributed_A2C.png)
	- Distributed Asynchronous RL-A3C: Agents can have slightly different policies where each worker updates the global network, one at a time.![Distributed_A3C.png](CMU_Notes/DRL/img/Distributed_A3C.png)
![Async_pseudocode.png](CMU_Notes/DRL/img/Async_pseudocode.png)
### Why Use A3C? Solving Key Problems in RL

A3C was designed to address two major challenges faced by earlier algorithms like DQN:
1. It Solves the Correlated Data Problem
Standard online RL algorithms learn from consecutive samples, which are highly correlated. A sequence of states in a game are not independent of each other. This correlation can make learning unstable.
- **DQN's Solution**: Use a large **experience replay buffer**. By storing hundreds of thousands of past experiences and sampling them randomly, DQN breaks the temporal correlations in the data. The downside is that this requires a massive amount of memory.
- **A3C's Solution**: Use **asynchronous parallel workers**. Each worker has a different experience in its own environment. Because the global network receives updates from all these different workers at different times, the updates are naturally decorrelated. This achieves the same stabilizing effect as an experience replay buffer without the high memory cost.
2. It's More Sample Efficient
	By using multiple parallel workers, A3C can explore a much larger part of the state space in the same amount of time compared to a single-agent algorithm. This parallelization dramatically speeds up the training process. The algorithm is also designed to run efficiently on standard multi-core CPUs without requiring expensive GPUs, making it more accessible.
	In summary, A3C's key innovation is using **asynchrony** to replace the need for an experience replay buffer, leading to a more memory-efficient and often faster training algorithm.

---

# AWR


Approximation trick (sample state from $\mu$, actions from $\pi$): 
Scribe page 5-8
![AWR illustration](CMU_Notes/DRL/img/AWR-distribution.png)

#### 1. **AWR Objective**
$\max_{\pi} \; \mathbb{E}_{s \sim d^\mu(s)} \Big[ \mathbb{E}_{a \sim \pi(\cdot|s)} \big[ A^\mu(s,a) \big] - \alpha D_{KL}(\pi \,\|\, \mu) \Big]$
where:
- $\pi$ = new policy we want to learn.
- $\mu$ = behavior policy (the replay buffer distribution that generated the data).
- $d^\mu(s)$ = discounted state distribution under μ.
- $A^\mu(s,a)$ = **advantage function** under the behavior policy.
- $D_{KL}(\pi \,\|\, \mu)$ = KL divergence between new policy $\pi$ and behavior policy $\mu$.
- $\alpha$ = regularization weight.
#### 2. **What is Being Estimated (and Why)?**
1. **Value Function $V^\mu(s)$:**
    - Needed to compute the advantage.
    - Estimated by regression to empirical returns (TD or Monte Carlo).
2. **Advantage Function $A^\mu(s,a)$:**
    - Estimated as: $A^\mu(s,a) = Q^\mu(s,a) - V^\mu(s)$
    - In practice approximated by: $A^\mu(s,a) \approx \Big(\sum_{t=0}^k \gamma^t r_t\Big) + \gamma^k V^\mu(s_k) - V^\mu(s)$
    - Intuition: measures how much better an action a is compared to the baseline behavior $\mu$.
3. **Policy Update $\pi$:**
    - Instead of gradients, AWR updates $\pi$ by **regression toward actions in replay buffer**, but weighted by $\exp(\tfrac{1}{\beta} A^\mu(s,a))$.
    - This comes from the KL-regularized objective in the formula.
4. $d^\pi(s)$ is estimated by $d^\mu(s)$ as long as $\pi$ and $\mu$ are close
#### 3. **Why AWR? (Intuitions)**
- The core idea:
    $\pi(a|s) \propto \mu(a|s) \exp\!\Big(\tfrac{1}{\beta} A^\mu(s,a)\Big)$
    - If $A^\mu(s,a)$ is **large positive** → increase probability of that action.
    - If $A^\mu(s,a)$ is **negative** → downweight that action.
    - The KL penalty keeps $\pi$ from drifting too far away from $\mu$ (ensures stability).
- **Why it works:**    
    - Turns policy improvement into a **weighted supervised learning problem**: imitate good actions more strongly than bad ones.
    - No high-variance importance sampling (unlike naive off-policy policy gradient).
    - More stable than directly estimating Q-values for actor updates.
#### 4. **Advantages of AWR**
- **Off-policy friendly:** Can reuse replay buffer data.
- **Stable:** KL regularization avoids catastrophic policy updates.
- **Simple:** Policy update reduces to weighted maximum likelihood regression.
- **Low variance:** Avoids noisy gradient estimates.
#### 5. **Limitations**
- Relies on accurate **advantage estimates** (bias here directly impacts learning).
- Needs careful tuning of temperature parameter $\beta$.
- May be less aggressive in exploration compared to entropy-regularized algorithms like **SAC**.

✅ In short: AWR says — _take your replay buffer ($\mu$), estimate which actions were good ($A^\mu$), and regress your policy $\pi$ toward those actions, but weight them exponentially by their advantage_.

---

# Advanced Policy Gradients

## Performance Difference Lemma
#### 1. **Definition (Performance Difference Lemma)**
The main motivation of this lecture is to find an algorithm which merges the best attributes of the Actor-Critic and Q-value algorithms. The main advantage of the actor critic algorithm is a probability distribution over actions, while the main benefit of Q-Learning style approaches is their off-policy nature (no need to have an online sampling algorithm). Ideally, we would want to sample rollouts of an old policy 𝜇and update 𝜋, our optimal policy. The Performance Difference Lemma gives an, albeit incomplete, method to provide such a framework.

For two policies $\pi$ and $\mu$, the lemma states:
$J(\pi) - J(\mu) = \mathbb{E}_{\tau \sim P_\pi(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^t A^{\mu}(s_t, a_t) \right] \\ = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi(s), a \sim \pi(\cdot|s)}[A^\mu(s, a)]$
where:
- $J(\pi) = \mathbb{E}_{\tau \sim P_\pi} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$ is the expected discounted return under policy $\pi$.    
- $\tau = (s_0, a_0, s_1, a_1, \ldots)$ is a trajectory generated by policy $\pi$.
- $A^\mu(s,a) = Q^\mu(s,a) - V^\mu(s)$ is the **advantage function** under the baseline policy $\mu$.
- $\gamma \in [0,1)$ is the discount factor.
#### 2. **Motivation**
Why do we need this?
- In policy optimization, we want to know: _How much better is a new policy $\pi$ than some baseline policy $\mu$?_
- Directly comparing $J(\pi)$ and $J(\mu)$ by rollouts is expensive.
- PDL gives a way to **express performance difference using only the advantage of the baseline policy $\mu$** and the distribution of states and actions visited by the new policy $\pi$.
- This is the key theoretical result underlying **policy gradient theorems** and **trust-region policy optimization (TRPO, PPO, etc.)**.
#### 3. **Intuition**
- $A^\mu(s,a)$ tells us whether action aa is **better or worse than average** under policy $\mu$.
- If $\pi$ tends to choose actions with **positive advantage under $\mu$**, then $J(\pi) > J(\mu)$.
- If $\pi$ picks actions with **negative advantage**, then $J(\pi) < J(\mu)$.
- The expectation is taken over trajectories from $\pi$, because we care about how often the new policy visits different states.
👉 So: the performance difference equals the **discounted sum of advantages of the baseline policy**, weighted by how the new policy $\pi$ explores the environment.
- We are essentially computing the gap between J(π) and J(μ), but instead of just computing rewards and taking the difference, we are going to states where π goes, and summing over the advantage that the actions proposed by π have over what μ would have done.
- True fact for two policies. 
#### 4. **Proof Sketch**
Start from the definition of returns. $J(\pi) = \mathbb{E}_{\tau \sim P_\pi} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$
Now insert the baseline value function $V^\mu$: $r(s_t, a_t) = A^\mu(s_t, a_t) + V^\mu(s_t) - \gamma V^\mu(s_{t+1})$
(since $Q^\mu(s_t, a_t) = r(s_t,a_t) + \gamma \mathbb{E}[V^\mu(s_{t+1})]$ and $A^\mu = Q^\mu - V^\mu$).
Substitute back: $J(\pi) = \mathbb{E}_{\tau \sim P_\pi} \left[ \sum_{t=0}^\infty \gamma^t \big( A^\mu(s_t, a_t) + V^\mu(s_t) - \gamma V^\mu(s_{t+1}) \big) \right]$
The telescoping terms with $V^\mu$ collapse, leaving: $J(\pi) = J(\mu) + \mathbb{E}_{\tau \sim P_\pi} \left[ \sum_{t=0}^\infty \gamma^t A^\mu(s_t, a_t) \right]$
which is exactly the statement of the lemma. 
![performance difference lemma proof 1](CMU_Notes/DRL/img/Performance-diff-proof1.png)
![performance proof 2](CMU_Notes/DRL/img/performance-diff-proof2.png)
#### 5. **Key Takeaways**
- **Formal statement:** $J(\pi) - J(\mu) = \mathbb{E}_{\tau \sim P_\pi(\tau)} \Big[ \sum_{t=0}^\infty \gamma^t A^\mu(s_t, a_t) \Big]$
- $A^\mu(s_t, a_t)$ estimated via a critic
- **Motivation:** Lets us evaluate the improvement of a new policy using the advantage of an old one.
- **Intuition:** If $\pi$ consistently picks actions that look better under μ\mu, then it will outperform $\mu$.
- **Proof:** Derived via decomposition of rewards and telescoping cancellation.
✅ In short: the **Performance Difference Lemma** is the theoretical foundation that allows safe and principled policy improvement in RL.
## [[Importance sampling]]
Practical offline gradient descent algorithm
[[Algorithm Approach]]
#### **Option 1: Use states from μ, but estimate \($A^\pi$\).**
- $J_{\pi} - J_{\mu} = \mathbb{E}_{s \sim d^{\mu},\, a \sim \pi} \left[ A^{\pi}(s, a) \right]$
- Here, expectations are under \($d^\mu$\), the state distribution of the data collection policy.
- Requires estimating \($A^\pi$\) (the advantage of the current learning policy)
- This is challenging, but connects to *off-policy TD learning methods*, such as Soft Actor-Critic ([[SAC]]) and [[TD3]], [[Offline Actor Critic]].
- These algorithms train a critic to approximate \($Q^\pi$\), from which \($A^\pi$\) can be derived.
#### **Option 2: Use states from π, but estimate \($A^\mu$\).**
- $J_{\pi} - J_{\mu} = \mathbb{E}_{s \sim d^{\pi},\, a \sim \pi} \left[ A^{\mu}(s, a) \right]$
- Here, expectations are under \($d^\pi$\), but we only need \($A^\mu$\), which is easier to estimate from replay buffers or Monte Carlo rollouts of μ.
- This connects to algorithms like Proximal Policy Optimization ([[PPO]]) and Advantage-Weighted Regression ([[AWR]]).
- In practice, algorithms often approximate \($d^\pi$\) with data collected from something “close” to π.

Approximation trick (sample state from $\mu$, actions from $\pi$): 
Scribe 7a7b page 5-8


---

# Algorithm Approach

**On-Policy vs Off-Policy Workflow**
```
 On-Policy:             Off-Policy:
 πθ (collects data) →   μ (behavior policy collects data)
 Update θ using πθ       Store in replay buffer
 Discard old data        Sample minibatches, update πθ
```
On-policy requires **new rollouts every update**; off-policy can **reuse old transitions many times**, making it more sample-efficient but requiring corrections to avoid bias.
### 1. **Setup: Sampling from $\mu$ vs. $\pi$**
- **$\pi$ = new (target) policy** we are training/evaluating.
- $\mu$ = **behavior policy** that actually generated the data.
So when we talk about sampling:
- From $\pi$: Roll out the _current_ policy we want to optimize.
- **From $\mu$:** Reuse data from some _other_ policy (could be older versions of **$\pi$, or even a completely different policy, like a replay buffer).
### 2. Why the Difference Matters
- The **Performance Difference Lemma** says: $J(\pi) - J(\mu) = \mathbb{E}_{\tau \sim P_\pi} \Big[ \sum_t \gamma^t A^\mu(s_t,a_t) \Big]$
    This expectation is over **$\pi$’s distribution**, not $\mu$.
- But in practice, it’s often easier to get samples from $\mu$ (stored rollouts, replay buffer).
- So algorithms differ in **how they bridge the gap** between $\mu$-samples and the desired $\pi$-expectations.
### 3. **Algorithms**
##### **(a) [[PPO]] (Proximal Policy Optimization)**
- **Sampling:** On-policy → rollouts are generated by the _current policy_ (or a very recent version of it).
- **Why:** Ensures that the data distribution matches **$\pi$, so policy gradient estimates are unbiased.
- **Trick:** Uses clipping/KL constraints to avoid **$\pi$ drifting too far from the old policy within one update.
- **Intuition:** “Trust region” update — only change **$\pi$ where you know you have reliable samples.
- **Pros:** Stable, theoretically grounded.
- **Cons:** Very sample-inefficient (must discard data after one or two updates).
##### **(b) [[AWR]] (Advantage Weighted Regression)**
- **Sampling:** Off-policy → can reuse data from replay buffer (**$\mu$).
- **How:** Uses importance weighting to correct for the mismatch: $\pi(a|s) \propto \mu(a|s) \exp\!\Big(\tfrac{1}{\beta} A^\mu(s,a)\Big)$
- **Why:** Turns off-policy samples into supervised regression targets, weighted by advantage.
- **Intuition:** “Imitate good actions more than bad ones” (soft policy improvement).
- **Pros:** Can reuse lots of past data, more sample-efficient than PPO.
- **Cons:** Still biased if the replay buffer is too different from current policy (distribution shift).
##### **(c) [[SAC]] (Soft Actor-Critic)**
- **Sampling:** Off-policy → samples from replay buffer (**$\mu$).
- **How:** Uses importance weighting _implicitly_ by learning Q-functions. The replay buffer ensures good coverage of states.
- **Why:** Maximizes entropy-regularized return: $J(\pi) = \mathbb{E}_{s \sim \mu, a \sim \pi} [Q^\pi(s,a) - \alpha \log \pi(a|s)]$
- **Intuition:** The replay buffer acts as an approximate state distribution. The entropy bonus keeps exploration broad.
- **Pros:** Very sample-efficient, stable for continuous control.
- **Cons:** Still sensitive to replay buffer distribution mismatch, requires careful hyperparameter tuning.
##### **(d) TD3 (Twin Delayed DDPG)**
- **Sampling:** Off-policy → samples from replay buffer (**$\mu$).
- **How:** Learns Q-functions using replay data, trains deterministic policy to maximize Q.
- **Tricks:**
    - “Twin critics” to reduce Q overestimation bias.
    - Target policy smoothing (adds noise to target actions).
- **Intuition:** Reuse experience many times, rely on critic to guide deterministic actor.
- **Pros:** Very sample-efficient. Strong performance in continuous control.
- **Cons:** Deterministic policy → exploration is harder; sensitive to replay buffer quality.
### 4. **Intuition: Sampling Trade-offs**
- **Sampling from $\pi$ (on-policy):**
    - Matches the desired distribution exactly.
    - Low bias, but very high variance.
    - Data inefficient — can’t reuse past trajectories.
    - Used in **PPO, A2C, A3C**.
- **Sampling from $\mu$(off-policy):**
    - Enables replay buffers → much higher sample efficiency.
    - Needs correction (importance weighting, critics).
    - Introduces potential bias if $\mu$ is very different from **$\pi$.
    - Used in **SAC, TD3, AWR, DDPG**.
### 5. **Summary Table**

| Algorithm | Sampling From                     | Correction Method                   | Pros                     | Cons                                       |
| --------- | --------------------------------- | ----------------------------------- | ------------------------ | ------------------------------------------ |
| **PPO**   | $\pi$(on-policy)                  | Clipping/KL trust region            | Stable, low bias         | Sample inefficient                         |
| **AWR**   | $\mu$ (off-policy)                | Advantage-weighted regression       | Reuses data, simple      | Biased if  $\mu\not\approx \pi$            |
| **SAC**   | $\mu$ (off-policy, replay buffer) | Q-learning + entropy regularization | Sample efficient, robust | Hyperparameter sensitive                   |
| **TD3**   | $\mu$ (off-policy, replay buffer) | Twin critics, smoothing             | Strong performance       | Exploration harder, buffer quality matters |
✅ **Key takeaway:**
- On-policy (sample from $\pi$): unbiased, stable, but data-hungry.
- Off-policy (sample from $\mu$): efficient, but needs correction for distribution shift.
- Algorithms choose one depending on whether stability or efficiency is more critical.
![policy value based summary](CMU_Notes/DRL/img/policy-value-based-methods.png)
**Value-based methods (e.g. Q-learning) struggle with:**
- Large/continuous action spaces
- Instability in function approximation  
**Policy Gradients offer:**
- Direct optimization of the policy. 
- Can learn stochastic Policies
- Naturally handle continuous actions (think how?)
**Problems:**
- High-variance (e.g. REINFORCE)
- Sample Efficiency (on-policy constraints)

---

# DDPG

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

---

# DQN

DQNs are a variant of fitted Q-iteration that use gradient-based updates for the parameters of the value function.
![DQN Replay](CMU_Notes/DRL/img/DQN-replay.png)
To solve the problem of highly correlated updates in on-policy training, two mechanisms are introduced:
1. **Replay buffer**: Stores experience tuples (s,a,r,s′). Updates to $Q_{θ_t​​}$ are made using i.i.d. samples from this buffer, which is populated by trajectories of the current policy πt​.
2. **Target network**: A separate network with parameters θˉ is used to compute the target y(s,a). Its parameters are updated only periodically to stabilize training.
There are two ways of updating targets:
3. **Soft targets**: Update rule defined by $min θ_t​​(Q_{θ_t​​}(s,a)−y_{θ_{t−1}}​​(s,a))^2$. Here, the target y is computed using parameters from the previous step, $θ_{t−1}$​. This results in more frequent but smaller updates, reducing instability while still keeping the target network close to the online network. $\bar{\theta}_{i+1} \leftarrow \alpha\theta_{i+1} + (1-\alpha)\bar{\theta}_i$ , $\alpha$ very small, 0.005
4. **Hard targets**: The target network parameters θˉ are held fixed for K gradient steps on the TD-error, then set equal to the current network parameters θt​. Small K (e.g., K=1) can lead to instabilities due to strong correlations between the target and the Q-function.
 5. **Comparison**

| Aspect          | **Hard Target Update**     | **Soft Target Update**                        |
| --------------- | -------------------------- | --------------------------------------------- |
| Update Rule     | Copy every C steps         | Polyak average with rate $\tau$               |
| Target Change   | Abrupt, discontinuous      | Smooth, continuous                            |
| Stability       | Can oscillate near updates | More stable                                   |
| Hyperparameters | Interval C                 | Smoothing factor $\tau$                       |
| Common Usage    | Original DQN, Atari        | DDPG, TD3, SAC, sometimes modern DQN variants |
. **When to Use Which**
- **Hard Target Updates**
    - If the environment is **discrete-action** and you follow the original DQN setup.
    - Simpler, fewer hyperparameters.
- **Soft Target Updates**
    - If environment is **continuous control** (common in robotics).
    - If training is unstable and you want smoother learning.
    - Preferred in most modern deep RL algorithms (except “classic” DQN).
### **Challenges with DQNs:**
1. **Sampling from replay buffers**: Ensuring that sampled batches are representative and uncorrelated is difficult. Poor sampling can introduce bias and slow convergence.
2. **Overestimation from noise and stochasticity**: Q-learning updates can systematically overestimate values, leading to instability. This issue is commonly referred to as the **overestimation problem**. Accidental errors may lead to divergence. The target network will amplify the error.
#### 1. **Q-learning Loss with MSE**
In DQN, the loss for Q-network parameters $\theta$ is usually:$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \Big[ \big( y - Q_\theta(s,a) \big)^2 \Big]$
where the target is: $y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$
This is a **mean squared error (MSE)** between the target y and the prediction $Q_\theta(s,a)$.
#### 2. **Huber Loss**
The **Huber loss** is a robust alternative to MSE.  
For an error term $\delta = y - Q_\theta(s,a)$:
$\ell_\kappa(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{if } |\delta| \leq \kappa \\[6pt] \kappa \big(|\delta| - \tfrac{1}{2}\kappa \big) & \text{if } |\delta| > \kappa \end{cases}$
- Quadratic when the error is **small** ($|\delta| \leq \kappa$).
- Linear when the error is **large** ($|\delta| > \kappa$).
- $\kappa$ is a threshold parameter (in DQN, usually set to 1).
So the **Huber loss for Q-learning** is:
$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \big[ \ell_\kappa\big( y - Q_\theta(s,a)\big) \big]$
#### 3. **Why Huber Loss Improves Stability**
- **MSE Problem:**
    - Squared error grows rapidly with large TD errors.
    - In Q-learning, targets y can be very noisy (due to bootstrapping + stochastic environment).
    - Large errors → huge gradients → unstable updates.
- **Huber Loss Solution:**
    - Behaves like MSE when errors are small → precise fitting around the optimum.
    - Behaves like MAE (mean absolute error) when errors are large → avoids exploding gradients.
    - Provides a balance: sensitive near the target, robust to outliers.
- **Effect in practice:**
    - Reduces sensitivity to rare, catastrophic TD errors (from bad targets).
    - Leads to smoother training curves and less divergence.
    - Was a key modification in the original DQN paper (Nature 2015).
- **Why it helps:**
    - Limits gradient explosion from large TD errors.
    - Makes learning more stable.
    - Balances accuracy (MSE) with robustness (MAE).
👉 In short: The **Huber loss stabilizes DQN** by making updates robust to noisy or outlier TD errors, while still being accurate for small errors.
### Practical details for Q-learning:
- coverage
- high URD can destabilize training
- improved loss functions: 
	- Hubert loss: 
	- $\ell_\delta(e) =\begin{cases}\frac{1}{2} e^2 & \text{if } |e| \leq \delta, \delta \left( |e| - \frac{1}{2} \delta \right) & \text{otherwise.}\end{cases}$
## Double DQN
$y_t = r_{t+1} + \gamma Q_{\text{target}}(s_{t+1}, \arg \max_a Q_\omega(s_{t+1}, a))$
![DDQN](CMU_Notes/DRL/img/DDQN.png)
### How Double DQN Leverages the Delay: Reducing Overestimation Bias

The specific innovation of **Double DQN** is to address a different problem: **maximization bias**.

- **The Problem**: In the standard DQN target calculation, $\text{target} = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$, the `max` operator tends to pick actions whose values are overestimated due to noise. This leads to a consistent upward bias in the value estimates, which can slow down learning and lead to suboptimal policies.
    
- **The Solution**: Double DQN cleverly uses the two available networks (main and target) to decouple **action selection** from **action evaluation**.
    
    Here's how the target is calculated in Double DQN:
    1. **Action Selection**: First, use the **main network** to ask, "What is the best action to take in the next state $s$?" $a∗=arga′max​Qmain​(s′,a′)$
    2. **Action Evaluation**: Then, use the **target network** to ask, "What is the value of that _specific action_, $a∗$?" $target=r+γQtarget​(s′,a∗)$
        
By using the main network to choose the action and the delayed target network to evaluate it, Double DQN reduces the tendency to select and lock onto overestimated values.

In summary, the **delay itself is for stability**, and Double DQN **leverages the existence of that delayed network** to implement its clever decoupling mechanism, which in turn **reduces overestimation bias**.
### Overestimation Bias
-  UTD update to data Ratio: high UTD amplifies the compounding errors
- target overestimate due to max operator selecting positively noisy actions and compounding through bootstrapping
![double q learning](CMU_Notes/DRL/img/double-q-learning.png)
![deep double q learning ru](CMU_Notes/DRL/img/Deep-double-q-learning-ru.png)
## ChatGPT Summary
#### 1. **The Overestimation Problem in DQN**
In Q-learning (and DQN), the update target is: $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$
- The $\max$ operator **both selects and evaluates** the next action.
- If Q-values are noisy (due to limited data, stochasticity, or approximation errors), the $\max$ will tend to **overestimate** the true expected return.
- This bias accumulates → unstable learning and overly optimistic Q-values.
🔑 Intuition:  
It’s like always assuming the best-case action outcome, even if it’s just noise.
#### 2. **Mitigation Techniques**
##### **(a) Double Q-Learning**
- **Idea:** Decouple **action selection** from **action evaluation**.
- Maintain **two Q-functions**, $Q^A$ and $Q^B$.
- Update one using the other’s estimate: $y = r + \gamma Q^B(s', \arg\max_{a'} Q^A(s',a'))$
    - $Q^A$ chooses the action.
    - $Q^B$ evaluates it.
- Reduces upward bias because action selection noise isn’t reinforced by the same network
##### **(b) Deep Double Q-Learning (DDQN)**
- Adaptation of Double Q-Learning for DQNs.
- Uses **online network** for action selection, **target network** for evaluation: $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s',a'))$
- Much more stable than vanilla DQN.
- Common default in modern implementations.
##### **(c) Deep Double Q-Learning with Random Updates (Ensembles)**

- **Idea:** Maintain an ensemble of Q-networks.
- Randomly select which Q-network(s) to update each step.
- This stochasticity helps decorrelate errors across networks.
- Similar spirit to “bootstrapped DQN” or “randomized value functions” — used for both exploration and reducing bias.
### **(d) N-Step Returns**

- Instead of one-step targets, use **multi-step returns**:   $y_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n Q_{\theta^-}(s_{t+n}, a_{t+n})$
- Benefits:
    - Reduces variance in target estimation.
    - Propagates reward signals faster.
    - Mitigates overestimation because the target relies less on a single max operator.
- Widely used in Rainbow DQN (which combines DDQN + prioritized replay + multi-step returns + dueling).
#### 3. **Comparison**

| Method                            | Fixes Overestimation? | How It Helps                                     |
| --------------------------------- | --------------------- | ------------------------------------------------ |
| **Double Q-Learning**             | ✅ Yes                 | Decouples selection and evaluation               |
| **Deep Double Q-Learning (DDQN)** | ✅ Yes                 | Online net selects, target net evaluates         |
| **Random Updates / Ensembles**    | ✅ Partially           | Averaging across multiple learners reduces noise |
| **N-Step Returns**                | ✅ Indirectly          | Smoother targets, less reliance on max operator  |
#### 4. **Summary**
- **Problem:** Vanilla DQN overestimates Q-values because $\max$ picks noisy high estimates.
- **Solutions:**
    - **Double Q-learning**: two networks for decoupled selection/evaluation.        
    - **Deep Double Q-learning (DDQN)**: practical neural version using online vs. target net.
    - **Randomized / ensemble updates**: reduce bias by decorrelating errors.
    - **N-step returns**: smooth out learning targets, reduce single-step max reliance.    
✅ In practice:
- **DDQN** is the default fix.
- **N-step returns + DDQN** (as in Rainbow DQN) → strong performance.
- Ensembles/randomized updates are useful but more computationally costly.



---

# Evolutionary Methods for Policy Search

Evolutionary methods are a class of "black-box" optimization algorithms that use principles inspired by biological evolution—like reproduction, mutation, and selection—to find optimal solutions for complex problems. In reinforcement learning, they are used to directly search for the best policy parameters without needing to calculate gradients.

You'd typically use these techniques when the reward signal is sparse or deceptive, when the environment is not differentiable, or when you need to explore a wide range of policies to avoid getting stuck in a local optimum. They are particularly good for continuous control problems in robotics.
### General algorithm:
1. Initialize a population of parameter vectors (genotypes).
2. Apply random perturbations (mutations) to each parameter vector.
3. Evaluate the perturbed parameter vectors according to a fitness function.
4. Retain the perturbed vectors if they yield improved performance (selection).
5. Repeat from step 2.

Unlike policy gradient methods, **evolutionary methods** don't rely on **gradient information**. Instead, they explore the parameter space using **random mutations** and **selection**. This makes them a good choice for problems where gradients are unavailable, unreliable, or hard to compute.
### Cross entropy method
The first evolutionary strategy we consider is the **Cross-Entropy Method (CEM)**. In this approach, policy parameters are sampled from a **multivariate Gaussian distribution with a diagonal covariance matrix**. The mean and variance of the parameter distribution are updated towards the samples with the **highest fitness scores**.

Work embarrasingly well in low-dimensions
1. **Initialize**: Start with a probability distribution over the policy parameters (e.g., a Gaussian with mean $\mu$ and variance $\Sigma$).
2. **Sample**: Draw a population of `N` policy parameter vectors from this distribution.
3. **Evaluate**: Test each policy and calculate its fitness (total reward).
4. **Select Elites**: Identify the top `k` percent of policies (the "elites").
5. **Refit**: Calculate a new mean and variance based _only_ on the elite samples. This new, refined distribution is then used to generate the next population.
By repeatedly fitting the distribution to the best-performing samples, CEM quickly hones in on high-reward regions of the parameter space.

![CEM](CMU_Notes/DRL/img/CEM.png)

### Covariance Matrix Adaptation
The second evolutionary strategy discussed is the **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**. This method samples policy parameters from a **multivariate Gaussian distribution** with a **full covariance matrix**, which distinguishes it from strategies that use different variances for each dimension. This full covariance matrix enables CMA-ES to capture **correlations between parameters**. The mean and covariance matrix are then iteratively updated based on the distribution of samples that achieved the highest fitness scores.
- sample
- select elites
- update mean
- update covariance
- iterate

CMA-ES is a more advanced evolutionary strategy that is particularly effective for complex, non-convex optimization problems. It's similar to CEM but with a crucial difference: it intelligently adapts the **covariance matrix** ($\Sigma$) of the search distribution.
Adapting the covariance matrix allows the search to:

- **Elongate or shrink** along different directions, enabling it to navigate narrow valleys or broad plains in the fitness landscape.
- **Rotate** its orientation to align with ridges in the landscape.

This makes CMA-ES much more efficient at finding optimal solutions when the parameters are correlated.
![CMA-ES](CMU_Notes/DRL/img/CMA-ES.png)
### Evolution for Action Selection
- Generate diverse candidate action trajectories, assess each by summing cumulative rewards, retain the highest-scoring trajectories, and apply iterative perturbations until a trajectory meeting predefined performance criteria is identified.
- Rather than optimizing policy parameters directly, perform **model-based trajectory optimization** by searching in the action space to maximize total reward or reach a target optimal state.
- A **zero-order method** that does not compute the gradient. Just perturbing and computing fitness scores and weighted average of them based on the scores.
### Natural Evolutionary Strategies (NES)

NES bridges the gap between evolutionary methods and gradient-based methods. Instead of using a simple gradient, NES uses the **natural gradient**.

The key idea is to find the direction in the _distribution space_ that maximally increases the expected fitness. It treats the search distribution itself as the object to be optimized and updates its parameters (e.g., the mean and covariance of a Gaussian) to produce better samples. It effectively computes a "gradient" for the entire population, making it a more principled and often more efficient search strategy than simpler methods.

![policy vs evolutionary](CMU_Notes/DRL/img/policy-vs-evolutionary.png)

How to sample from gaussian distribution: $z \sim \mathcal{N}(\mu, \Sigma)$
$z = \mu + \Sigma^{\frac{1}{2}}\epsilon \text{ where } \epsilon \sim \mathcal{N}(0, I)$
### Distributed Evolution
Communications between GPUs becomes bottleneck. Sending vectors to each worker is very expensive.
![NES](CMU_Notes/DRL/img/NES.png)

Instead, send random seeds to workers and sample locally.
![distributed evolution](CMU_Notes/DRL/img/distributed-evolution.png)
Despite these advantages, ES can still get stuck in local optima. A simple strategy to improve robustness is to evaluate candidate policies across multiple related tasks or environments. This forces exploration to generalize beyond a single setting, reducing the chance of converging prematurely. While there are no formal guarantees, empirical studies show this approach helps ES escape poor local solutions and improves overall performance.

---
# ChatGPT summary
# 1. **Evolutionary Methods for Policy Search**
- **What it is:**  
    Instead of using policy gradients or Q-learning, we treat the policy parameters θ\theta as individuals in a population and evolve them over time.
- **How it works:**
    - Generate multiple policy candidates.
    - Evaluate them on the environment (fitness = expected return).
    - Select the best, mutate or recombine, repeat.
- **Problem solved:**  
    Policy search when gradients are unavailable, noisy, or misleading.
- **Good for:**
    - Non-differentiable environments.
    - Sparse rewards (reward signal is too weak for policy gradients).
    - Parallel training on many CPUs.
- **Limitations:**
    - Low sample efficiency (requires many rollouts).
    - Doesn’t scale well to high-dimensional neural networks unless combined with variance reduction tricks.
# 2. **Cross-Entropy Method (CEM)**
- **What it is:**  
    A stochastic optimization method where you iteratively update a sampling distribution toward better-performing solutions.
- **How it works:**
    - Sample parameters θ(i)\theta^{(i)} from a Gaussian.
    - Evaluate returns.
    - Keep the top $\rho\%$ ("elite set").
    - Update mean and variance of the Gaussian based on elites.
- **Problem solved:**  
    Find good policies without gradients, using elite selection.
- **Good for:*
    - Simple implementation.
    - Global exploration with convergence toward local optimum.
- **Limitations:**
    - Still sample-inefficient.
    - Tends to converge prematurely (loss of diversity).
# 3. **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**
- **What it is:**  
    A more advanced evolution strategy that adapts not just mean/variance but the full covariance matrix of the search distribution.
- **How it works:**
    - Maintains a multivariate Gaussian over parameters.
    - Updates covariance to capture correlations between dimensions.
- **Problem solved:**  
    Handles high-dimensional, non-separable search spaces better than CEM.
- **Good for:**
    - Black-box optimization in continuous domains.
    - Medium-scale problems (hundreds to a few thousand parameters).
- **Limitations:**
    - Computationally expensive (covariance matrix is O(d2)O(d^2) in dimension).
    - Not suited for huge neural networks.
# 4. **Evolution for Action Selection**
- **What it is:**  
    Use evolution not to search over policy parameters, but directly to select or evolve sequences of actions.
- **How it works:**
    - Generate candidate action sequences (like model predictive control).
    - Evaluate rollouts, choose best sequence, execute first action.
- **Problem solved:**  
    Planning in environments where model-based gradients are unavailable.
- **Good for:**
    - Short-horizon planning.
    - Situations where dynamics are known or can be simulated.
- **Limitations:**
    - Not scalable to very long horizons.
    - Doesn’t generalize well beyond the simulated rollout window.
# 5. **Natural Evolutionary Strategies (NES)**
- **What it is:**  
    A family of methods that estimate gradients of expected fitness with respect to the parameters of a search distribution, using the **log-likelihood trick** (similar to REINFORCE).
- **How it works:**
    - Parameterize search distribution pθ(⋅)p_\theta(\cdot).
    - Compute update using: $\nabla_\theta J(\theta) = \mathbb{E}_{x \sim p_\theta} [f(x)\nabla_\theta \log p_\theta(x)]$
    - Equivalent to a black-box policy gradient.
- **Problem solved:**  
    Gradient-free optimization with variance reduction.
- **Good for:**
    - High-dimensional optimization with smoother updates than simple evolution strategies.
- **Limitations:**
    - Variance in gradient estimates still high.
    - Needs many samples per update.
# 6. **Distributed Evolution**
- **What it is:**  
    Run evolutionary methods across many machines/CPUs/GPUs in parallel.
- **How it works:**
    - Each worker samples policies/actions.
    - Evaluate independently.
    - Aggregate fitness, update distribution.
- **Problem solved:**  
    The sample inefficiency of evolution is mitigated by massive parallelism.
- **Good for:**
    - Scalability in cloud/cluster settings.
    - Fast exploration of large policy spaces.
- **Limitations:**
    - Requires significant compute resources.
    - Still less sample-efficient than gradient-based RL.
# 7. **When and Why to Use These Techniques**

✅ **Use evolutionary methods when:**
- The environment or policy is **non-differentiable**.
- Rewards are **sparse/delayed** (policy gradients may fail).
- You can afford **large-scale parallelism**.
- You want simple, robust optimization without delicate gradient tuning.
❌ **Limitations:**
- Very **sample-inefficient** (need millions of rollouts).
- Scale poorly to **very high-dimensional neural policies** (though hybrid methods exist).
- Often **slower convergence** than actor-critic or Q-learning.
# 8. **Summary Table**

| Method                          | Problem Solved                    | Good For                         | Limitation                          |
| ------------------------------- | --------------------------------- | -------------------------------- | ----------------------------------- |
| Evolutionary Policy Search      | Gradient-free policy optimization | Non-differentiable envs          | Sample inefficient                  |
| Cross-Entropy Method (CEM)      | Simple stochastic search          | Easy implementation              | Premature convergence               |
| CMA-ES                          | Correlated parameter search       | Medium-dim continuous problems   | Expensive, doesn’t scale to huge NN |
| Evolution for Action Selection  | Planning without gradients        | Short-horizon control            | Doesn’t scale to long horizons      |
| Natural Evolutionary Strategies | Black-box policy gradients        | Variance reduction in ES         | Still high variance, many samples   |
| Distributed Evolution           | Parallelizing evolution           | Leveraging clusters, fast search | Needs lots of compute               |
👉 In short: **evolutionary methods trade off sample efficiency for robustness and parallelism.** They’re often used as baselines, in black-box tasks, or when gradients fail.


---

# Fitted Q Iteration

Value iteration with neural networks: $Q(s, a) = r(s, a) + \mathbb{E}_{s'}\left( \max_{a'} Q(s', a') \right)$
### Loss function: 
As we've shown $V(s′)=maxa′​Q(s′,a′)$, we want we can rewrite $Q(s,a)=r(s,a)+Es′​(maxa′​Q(s′,a′))$. People have aimed to approximate Qθ​ with a neural network, but the challenge is defining a suitable loss function to run gradient descent.

Let us define $y(s,a)=r(s,a)+γmaxa′​Q(s′,a′)$. Gradient descent is difficult since y changes as Qθ​ changes, so keeping our algorithm on-policy is not optimal.

We can draw inspiration from **TD-learning** (temporal difference) to construct an off-policy algorithm called **fitted Q-iteration**. We want to train $Q_θ​(s,a)$ to mimic $y(s,a)$, such that
$\theta' = \min_{\theta} \mathbb{E}_{s,a \sim \text{somewhere}} [(Q_\theta(s,a) - y(s,a))^2]$ 
$s′∼P(⋅∣s,a)$
This algorithm can use data from **any policy θ** to construct $y(s,a)$. A general pipeline would be: starting with policy $θ_0$​, calculating $y_{θ0}​​(s,a)$, obtaining $θ_1$​ from $y_{θ_0}​​(s,a)$, and so on until we converge.

If we don't already have (s,a) pairs, we need data to train on. These (s,a) samples can come from any distribution, but the distribution should provide **good coverage** over the state-action space. Ideally, it is close to uniform, though we can't assume uniform coverage at the start since many states may not have been visited yet. In practice, we collect tuples (s,a,r,s′) from experience. Training then amounts to minimizing the squared error over these tuples, and using squared error ensures the learning remains unbiased with respect to the sampled data.

**Important clarification.** Although we use gradients to minimize the Bellman regression loss, this is _not_ the same as performing gradient descent directly on the policy objective (i.e., it's not a policy gradient step). Instead:
- We are performing supervised-style regression to make $Q_θ$​ consistent with the Bellman equation,
- and only after (or during) fitting $Q_θ$​ we extract a policy by greedy improvement: m$π(s)=argmax_a​ Q_θ​(s,a)$.
Fitted Q-Iteration (FQI) is an offline, batch-based reinforcement learning algorithm that uses a function approximator to learn the optimal Q-function in environments with large or continuous state spaces. It reframes the problem of Q-learning as a sequence of supervised learning (regression) problems.
### How Fitted Q-Iteration Works 🔄
The core idea is to iteratively improve an estimate of the Q-function by training a supervised learning model on targets generated by the Bellman equation.
1. **Collect a Static Dataset**: First, collect a dataset $\mathcal{D}$ of transitions `$(s, a, r, s')$` by having an agent interact with the environment. This is done only once, and the policy used for collection doesn't need to be optimal (it can even be random).
2. **Initialize the Q-Function**: Start with an initial function approximator, $\hat{Q}_0$, which can be as simple as a function that returns zero for all inputs.
3. **Iterate and Fit**: Repeat the following steps for a fixed number of iterations ($k=0, 1, 2, \dots$):
    - **Create Targets**: For every transition $(s_i, a_i, r_i, s'_i)$ in your dataset, create a target value $y_i$ using the Q-function from the _previous_ iteration, $Q^​k$​: $y_i​=r_i​+γmax_{a′}​Q^​k​(s_i′​,a′)$
    - **Train a New Model**: Create a new training set where the inputs are the state-action pairs $(s_i, a_i)$ and the labels are the targets $y_i$. Train a _new_ supervised learning model ($\hat{Q}_{k+1}$) on this dataset to minimize the prediction error.
4. **Extract Policy**: After the final iteration, the optimal policy is the one that acts greedily with respect to the final Q-function estimate, $\hat{Q}_K$.
##### Why It Works
FQI works because it is a practical, sample-based implementation of the **Value Iteration** algorithm. Each iteration of FQI approximates one full backup of the Bellman optimality operator.
- The target calculation step, $y_i = r_i + \gamma \max_{a'} \hat{Q}_k(s'_i, a')$, is a direct application of the Bellman update rule using the collected data samples.
- The supervised learning step finds a function $\hat{Q}_{k+1}$ that best approximates the result of this Bellman update across all the states and actions in the dataset.
By repeatedly applying this process, the learned function $\hat{Q}_k$ converges towards the optimal Q-function, $Q^*$, just as tabular Value Iteration would. The function approximator allows this value information to be generalized to unseen states.
##### Advantages and Disadvantages of FQI
###### Advantages 👍
- **High Sample Efficiency**: As a **batch** and **off-policy** algorithm, FQI reuses the same dataset in every iteration. It can extract a significant amount of information from a limited number of samples, making it far more sample-efficient than on-policy methods.
- **Handles Continuous Spaces**: By using a function approximator (like a random forest or neural network), it can naturally handle continuous state and action spaces where tabular methods are impossible.
- **Stable and Reproducible**: Since it operates on a fixed dataset, the training process is deterministic and can be more stable than online methods like DQN that learn from a continuous stream of potentially correlated data.
###### Disadvantages 👎
- **Error Accumulation**: This is the primary drawback. Small errors made by the function approximator in one iteration ($\hat{Q}_k$) are "baked into" the training targets for the next iteration. These errors can compound over many iterations, potentially causing the algorithm to become unstable or converge to a poor solution.
- **Computationally Expensive**: Training a new supervised learning model from scratch at every iteration can be very slow and resource-intensive, especially with large datasets.
- **Offline Only**: FQI is not designed for online learning. It cannot easily incorporate new experiences without re-running the entire iterative process on an updated dataset.

|Feature|Fitted Q-Iteration (FQI)|Q-Learning / DQN|
|---|---|---|
|**Data Usage**|**Batch (Offline)**: Learns from a fixed dataset.|**Incremental (Online)**: Learns from a stream of data.|
|**Update Mechanism**|**Retrains a new model** from scratch at each iteration.|**Updates an existing model** with each new experience.|
|**Sample Efficiency**|**Very High**: Reuses the entire dataset at every iteration.|**Lower**: Online methods are typically less sample-efficient.|
|**Computational Cost**|**High per iteration**: Must train a full model.|**Low per step**: A single, quick update.|
|**Typical Use Case**|Offline RL, where you have a pre-existing log of data.|Online RL, where an agent is actively exploring.|


---

# Importance sampling

### 1. **Definition**
**Importance sampling (IS)** is a statistical technique for estimating expectations under one distribution using samples from another distribution.

Suppose we want: $\mathbb{E}_{x \sim p(x)}[f(x)]$
but we only have samples from a different distribution $q(x)$.

We can rewrite: $\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$
Here:
- $\frac{p(x)}{q(x)}$ is the **importance weight**.
- It corrects for the mismatch between the sampling distribution q (what we have) and the target distribution p (what we want).
### 2. **In Reinforcement Learning**
In RL:
- **Target policy**: $\pi$ (the policy we want to evaluate/improve).
- **Behavior policy**: $\mu$ (the policy that generated the data, e.g. from replay buffer).
    
We want: $\mathbb{E}_{a \sim \pi(\cdot|s)}[f(s,a)]$
but we only have actions sampled from $\mu(a|s)$.

Using importance sampling: $\mathbb{E}_{a \sim \pi}[f(s,a)] = \mathbb{E}_{a \sim \mu}\left[ \frac{\pi(a|s)}{\mu(a|s)} f(s,a) \right]$
- The ratio $\frac{\pi(a|s)}{\mu(a|s)}$ is the **importance weight**.
- If $\pi$ and $\mu$ are very different, this ratio can blow up → high variance.

### 3. **Trajectory Importance Sampling**
For entire trajectories $\tau = (s_0,a_0,\ldots)$: $\frac{P_\pi(\tau)}{P_\mu(\tau)} = \prod_{t=0}^T \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}$
- Used in **off-policy policy gradient** methods (e.g. REINFORCE off-policy, AWR, importance-weighted actor-critic).
- But products of many ratios explode → variance grows exponentially with horizon.

### 4. **Motivation & Intuition**
- **Why use it?**  
    Lets us reuse off-policy data (from $\mu$) while still estimating expectations w.r.t. the target policy ($\pi$).
- **Intuition:**  
    You’re “re-weighting” samples from one distribution to look like they came from another.  
    Example: If $\pi$ favors an action more than $\mu$, its weight gets boosted.
### 5. **Advantages**
- Allows **off-policy learning** (reuse old trajectories, sample efficiency).
- The estimator is **unbiased** (in expectation, gives the correct value).
- Theoretically clean correction between policies.
### 6. **Disadvantages**
- **High variance**, especially with long horizons (trajectory weights multiply).
- If $\mu(a|s)$ is very small while $\pi(a|s)$ is large → ratio explodes.
- Requires knowing both $\pi$ and $\mu$ probabilities (not always easy).
### 7. **Fixes / Variants**
- **Truncated importance sampling:** Cap the ratio to avoid huge variance.
- **Per-decision IS:** Apply IS step-by-step instead of whole trajectory.
- **Weighted IS:** Normalize weights to reduce variance (biased but stable).
- **Algorithms using IS:**
    - Off-policy policy gradient
    - AWR (advantage weighted regression)
    - V-trace (used in IMPALA)

✅ **Summary:**
- Importance sampling lets us estimate expectations under one policy using data from another.
- In RL, it enables **off-policy learning** by correcting for distribution mismatch.
- But the variance problem is real, so most modern algorithms use **variants** (clipped, normalized, or trust-region constraints) to stabilize it.
    

---

# Introduction

Notes for CMU DRL Fall 2025 Course. Modified and cleaned with assist of AI tools.
### Introduction to Deep RL (Lecture 2) 
#### **Definitions in RL**
- **State ($s_t​$)**: A complete description of the state of the world at a specific time t. For example, in a chess game, the state is the position of all pieces on the board.
- **Action ($a_t​$)**: A choice made by the agent at time t. For example, moving a specific chess piece.
- **Observation ($o_t​$)**: A potentially incomplete description of the state that the agent perceives. In many simple problems, the observation is the same as the state ($o_t​=s_t$). In a poker game, your observation would be your own cards and public cards, but not the other players' cards (which are part of the true state).
- **Policy ($\pi(a\mid s)$)**: The agent's strategy or "brain." It's a function that maps states to a probability distribution over actions.
    - A **stochastic policy** gives probabilities for each action, e.g., $\pi(left \mid s) = 0.8, \pi(right \mid s) = 0.2$. A common way to represent this for discrete actions is with a **softmax function** on some learned values.
    - A **deterministic policy** maps each state to a single action, e.g., $a = \mu(s)$.
- **Markov Property**: The future is independent of the past, given the present. This means the next state $s_{t+1}$ depends only on the current state $s_t$ and action $a_t$, not on any previous states or actions.
    - **Correct Formulation**: $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0)$
#### **The Goal of RL** 
The primary goal is to find an optimal policy ($\pi^*$) that maximizes the expected cumulative reward.
- **Markov Decision Process (MDP)**: The mathematical framework for modeling RL problems. An MDP is a tuple $M = (S, A, P, R, \gamma)$.
    - $S$: A finite set of states.
    - $A$: A finite set of actions.
    - $P$: The state transition probability function, $P(s' \mid s, a)$, which gives the probability of transitioning to state $s'$ from state $s$ after taking action $a$.
    - $R$: The reward function, $r(s, a)$, which gives the immediate reward for taking action $a$ in state $s$.
    - $\gamma$: The discount factor ($0 \le \gamma \le 1$), which balances immediate vs. future rewards.
- **Partially Observable MDP (POMDP)**: Used when the agent can't observe the full state. It adds:
    - $\Omega$: A set of observations.
    - $O$: An observation function $O(o \mid s', a)$, the probability of observing $o$ after transitioning to state $s'$ by taking action $a$.
- **The Optimization Problem**: Find the policy $\pi^*$ that maximizes the expected return (sum of rewards).
    - $\pi^{*} = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[\sum_{t=0}^T r(s_t, a_t)\right]$
    - $\tau$ represents a **trajectory** (also called a rollout or episode), which is a sequence of states and actions: $\tau = (s_0, a_0, s_1, a_1, \dots)$.
- **Discounting**: In infinite-horizon problems, the total reward $\sum_{t=0}^{\infty} r(s_t, a_t)$ could be infinite. The discount factor $\gamma$ ensures the sum converges and prioritizes sooner rewards over later ones.
    - **Discounted Return**: $\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)$
    - The following identity holds due to the linearity of expectation, allowing us to view the total expected reward as a sum of expected rewards at each step: $\sum_{t=0}^{\inf} r(s_t, a_t)$ -> $\sum_{t=0}^{\inf} \gamma_t r(s_t, a_t)$
	- Claim: $\mathbb{E}_{\tau \sim P_{\pi}(\tau)} \left[\sum_{t=0}^T r(s_t, a_t)\right] = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim p_{\pi}(s_t, a_t)}[r(s_t, a_t)]$
- **State Occupancy Distribution ($\rho_{\pi}(s)$)**: The discounted distribution of states you visit when following policy $\pi$. It represents how often you expect to be in each state
#### **The Generic RL Algorithm Loop** 
Most RL algorithms follow a general iterative process:
1. **Act & Collect**: Execute the current policy $\pi$ in the environment to generate trajectories (collect data).
2. **Evaluate**: Use the collected data to estimate how good the current policy is. This often involves calculating returns or fitting a value function.
3. **Improve**: Update the policy $\pi$ based on the evaluation to increase the expected return.
4. **Repeat** until the policy converges.
#### **Examples of RL Algorithms**
Algorithms can be broadly categorized as **model-based** or **model-free**.
- **Model-Based Methods**: The agent first learns a model of the environment (the transition dynamics $P$ and reward function $R$). It then uses this learned model to plan the best course of action.
    - **How it works**:
        1. Collect data by interacting with the environment.
        2. Use this data to learn approximations $P_{\theta}(s' \mid s, a)$ and $R_{\phi}(s, a)$.
        3. Use the learned model $P_{\theta}, R_{\phi}$ to find a good policy, for example, by using planning algorithms (like trajectory optimization) or by generating simulated experiences to train a model-free method.
    - **Examples**: Dyna-Q, World Models.
- **Model-Free Methods**: The agent learns a policy or value function directly from experience, without explicitly learning a model of the environment
    - **Value-Based Methods**: Learn a value function that predicts the expected return. The policy is often implicit (e.g., act greedily with respect to the values)        
        - **Q-Learning**: Learns the state-action value function, $Q(s, a)$.
        - **Deep Q-Networks (DQN)**: Uses a deep neural network to approximate the Q-function, enabling it to handle large state spaces like images from Atari games.
    - **Policy-Based Methods**: Directly learn the parameters of a policy $\pi_{\theta}(a \mid s)$ that maximizes the expected return.
        - **REINFORCE**: A foundational policy gradient algorithm that increases the probability of actions that lead to high returns.
        - **Actor-Critic Methods**: A hybrid approach. The "Actor" is a policy that decides which action to take, and the "Critic" is a value function that evaluates how good that action was. Examples include **A2C** and **A3C**.
#### **The Value Function** 
A value function is a prediction of the expected future reward. It helps the agent evaluate the "goodness" of a given state or state-action pair.
- State-Value Function (Vπ(s)): The expected discounted return when starting in state $s$ and following policy $\pi$ thereafter. Expected cumulative reward you get when use  to act starting from $s_t$ :
	 $V(s^{*}) = \mathbb{E}_{\tau}[\sum_{t=0}^{\inf} \gamma^{t} r(s_t,a_t)\mid s_0=s^{*}]$
- Action-Value Function (Qπ(s,a)): The expected discounted return when starting in state $s$, taking action $a$, and then following policy $\pi$ thereafter.
    $Q^{\pi}(s^{*},a^{*}) = r(s^{*},a^{*}) +\gamma \mathbb{E}_{\tau}[\sum_{t=0}^{\inf} \gamma^{t} r(s_t,a_t)\mid s_{-1}=s^{*}]$
- **The Bellman Equations**: These are fundamental to RL. They express the value of a state recursively, relating it to the values of subsequent states. This recursive structure is what allows for dynamic programming and temporal-difference learning.
    - For $V^\pi$: $V^\pi(s)=\sum_a \pi(a\mid s)[r(s,a)+\gamma \sum_{s'} P(s'\mid s,a)V^\pi(s')]$
    - For $Q^\pi$:  $Q^\pi(s,a) = r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \sum_{a'} \pi(a' \mid s') Q^\pi(s',a')$
- **Bellman optimality equation (for $Q^*$):**  $Q^*(s,a) = r(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \max_{a'} Q^*(s',a')$




---

# MaxEnt RL

### 1. Standard RL Objective
In classical RL, the goal is:
$J(\pi) = \mathbb{E}_{\tau \sim P_\pi} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t,a_t) \Big]$
- The policy $\pi$ is optimized to **maximize expected return**.
- The optimal policy often collapses to **deterministic actions** once learned.
###  2. Maximum Entropy RL Objective
MaxEnt RL augments the reward with an **entropy term**:
$J_{\text{MaxEnt}}(\pi) = \mathbb{E}_{\tau \sim P_\pi} \Big[ \sum_{t=0}^\infty \gamma^t \big( r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \big) \Big]$
where:
- $\mathcal{H}(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log \pi(a|s)$ = entropy of policy at state s.
- $\alpha > 0$ is the **temperature parameter**, trading off reward vs. exploration.
###  3. Key Differences from Regular RL
| **Regular RL**                                      | **MaxEnt RL**                                       |
| --------------------------------------------------- | --------------------------------------------------- |
| Maximize reward only                                | Maximize reward + entropy                           |
| Policies tend toward deterministic (greedy)         | Policies remain stochastic, explore more            |
| Exploitation-focused                                | Balances exploration + exploitation                 |
| May get stuck in suboptimal deterministic behaviors | Encourages robustness, avoids premature convergence |
###  4. Why Use MaxEnt RL?

#### Motivations
1. **Better Exploration**
    - Entropy term encourages diverse actions, preventing premature collapse.
2. **Robustness**
    - Policies don’t overfit to specific trajectories.
    - More robust to model errors, noise, or perturbations.
3. **Connections to Probabilistic Inference**
    - MaxEnt RL ≈ learning a distribution over trajectories via **energy-based models**.
    - Links to variational inference and control-as-inference frameworks.
4. **Stability**
    - Stochastic policies help smooth the optimization landscape.
###  5. Famous Algorithms in MaxEnt RL

#### **Soft Q-Learning (SQL)**
- Extends Q-learning by using **soft Bellman updates**:
    $Q(s,a) \leftarrow r(s,a) + \gamma \, \mathbb{E}_{s'} \big[ V(s') \big]$
    with $V(s) = \alpha \log \sum_a \exp\!\Big(\tfrac{1}{\alpha} Q(s,a)\Big)$
- Equivalent to a **log-sum-exp softmax backup** instead of $\max_a Q(s,a)$.
#### **Soft Actor-Critic (SAC)**
- The most popular MaxEnt RL algorithm.
- Actor: learns stochastic policy $\pi(a|s)$ by maximizing entropy-regularized Q-values:
    $J(\pi) = \mathbb{E}_{s \sim D, a \sim \pi} \Big[ Q(s,a) - \alpha \log \pi(a|s) \Big]$
- Critic: soft Bellman target with entropy bonus.
- Auto-tuning of α\alphaα is often used.
- Known for **sample efficiency, stability, robustness**.
#### **Maximum Entropy Policy Gradient**
- Directly extends REINFORCE / policy gradients with entropy regularization.
- Objective:
    $\nabla_\theta J(\pi_\theta) = \mathbb{E}\Big[ \nabla_\theta \log \pi_\theta(a|s) (R_t + \alpha \mathcal{H}(\pi(\cdot|s))) \Big]$
#### **Other Related Algorithms**
- **Path Consistency Learning (PCL)**: policy optimization via KL consistency equations.
- **Relative Entropy Policy Search (REPS)**: KL-constrained optimization that implicitly encourages entropy.
- **Soft DDPG / Soft TD3**: MaxEnt variants of deterministic actor-critic methods.
###  6. Advantages of MaxEnt RL
✅ **Exploration:** avoids premature deterministic collapse.  
✅ **Robustness:** stochasticity improves generalization.  
✅ **Better performance in continuous action tasks** (robotics, locomotion).  
✅ **Links to probabilistic inference** (control as inference view).
###  7. Disadvantages
❌ **Extra hyperparameter $\alpha$:** must balance reward vs entropy (though SAC auto-tunes it).  
❌ **Stochastic policies may be unnecessary** in very simple tasks (deterministic suffices).  
❌ **More complex objectives:** training can be computationally heavier.
###  8.Takeaway
- **Regular RL:** maximize expected reward → deterministic greedy policy.
- **MaxEnt RL:** maximize expected reward **+ entropy** → stochastic robust policy.
- Famous methods: **Soft Q-Learning, Soft Actor-Critic, REPS, PCL.**
- Widely used in modern deep RL (esp. SAC) due to stability and sample efficiency.

---

# Offline Actor Critic

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

---

# PPO

### 1. The Policy Gradient Foundation

The standard policy gradient theorem gives us:
$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$
where $A^{\pi_\theta}(s_t, a_t)$ is the advantage function under the current policy $\pi_\theta$.
### 2. From On-Policy to Off-Policy via Importance Sampling

PPO wants to reuse data collected from an old policy $\mu = \pi_{\theta_{old}}$ to update the current policy $\pi_\theta$. This requires converting the expectation under actions from $\pi$ to actions from $\mu$ via importance sampling:
$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \mu} \left[ \frac{\pi_\theta(a|s)}{\mu(a|s)} f(a) \right]$

Applying this to the policy gradient:
$\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \mu} \left[ \frac{\pi_\theta(a|s)}{\mu(a|s)} \nabla_\theta \log \pi_\theta(a|s) A^\mu(s, a) \right]$

This leads to the surrogate objective function:
$L^{CPI}(\theta) = \mathbb{E}_{(s,a) \sim \mu} \left[ \frac{\pi_\theta(a|s)}{\mu(a|s)} A^\mu(s, a) \right]$

where CPI stands for “Conservative Policy Iteration.” The ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)}$ is called the **importance sampling ratio**. 
### What Problem Does PPO Solve?
All we have changed is the order of $\pi$ and $\mu$ in the KL term. However, there is no closed-form solution in this case. Instead, we return to gradient-based optimization.

Traditional policy gradient methods directly optimize the expected return by updating the policy parameters $\theta$ in the direction of the gradient:

$\nabla_\theta \mathbb{E}_{s \sim d^{\pi}(s),\ a \sim \pi_\theta(\cdot|s)} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$

Now, instead we have $d^{\mu}(s)$ instead of $d^{\pi}(s)$, and an advantage term instead of the sum.
One of the ways we can approach this is to write down an explicit objective. We can approach this with importance sampling:
$\mathbb{E}_{x \sim p(x)} [f(x)] = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$
Using this, we change the expectation from under $\pi_\theta$ to $\mu$ and multiply by an importance weight:
$\mathbb{E}_{s \sim d^{\mu}(s),\ a \sim \mu(\cdot|s)} \left[ \frac{\pi_\theta(a|s)}{\mu(a|s)} A^{\mu}(s, a) \right] - \alpha D_{KL} \left( \mu(\cdot|s) \parallel \pi_\theta(\cdot|s) \right)$
Here, importance weights are less ‘finicky’, since they are for actions rather than states. For an Atari game, for instance, the state distribution is over every possible image in Atari, but the action space has a much lower dimensionality.

However, large updates to the policy can lead to performance collapse or instability. Trust Region Policy Optimization (TRPO) addressed this by constraining the policy update, but TRPO is complex and computationally expensive.

PPO solves this by introducing a simpler, more practical approach to limit policy updates, ensuring that each update is "proximal" to the previous policy.
### Intuition of Using KL Divergence
PPO uses the KL divergence $D_{KL}(\mu(\cdot|s) \parallel \pi_\theta(\cdot|s))$ to measure the difference between the old policy $\mu$ and the new policy $\pi_\theta$. By penalizing large KL divergence, PPO ensures that the new policy does not deviate too much from the old one, preventing destructive updates and encouraging stable learning: optimization function:
$\max_\theta\ \mathbb{E}_{s \sim d^\mu(s)} \left[ \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ A^\mu(s, a) \right] - \alpha D_{KL}(\mu(\cdot|s) \parallel \pi_\theta(\cdot|s)) \right]$
### Why Does PPO Use Policy Gradient?
PPO is based on the policy gradient framework because it directly optimizes the expected return with respect to the policy parameters. This allows PPO to handle high-dimensional, continuous action spaces and stochastic policies, which are challenging for value-based methods.
### Why Is There No Closed-Form Solution as in AWR?
Algorithms like AWR (Advantage-Weighted Regression) can sometimes derive a closed-form solution for the optimal policy update due to their specific loss structure. In contrast, PPO's objective involves expectations over both the state and action distributions, as well as the KL penalty, making the optimization nontrivial and not amenable to a closed-form solution. Instead, PPO relies on stochastic gradient ascent to iteratively improve the policy.
### Advantages of PPO
- **Stable and Reliable**: By limiting policy updates, PPO avoids large, destabilizing changes.
- **Simple to Implement**: PPO does not require complex second-order optimization or constraints like TRPO.
- **Sample Efficient**: Can reuse data from old policies via importance sampling.
- **Widely Applicable**: Works well in both discrete and continuous action spaces.
### Disadvantages of PPO
- **No Closed-Form Update**: Requires iterative optimization, which can be slower than algorithms with closed-form solutions (like AWR in some cases).
- **Sensitive to Hyperparameters**: Performance can depend on the choice of KL penalty coefficient $\alpha$ and other settings.
- **Still Approximate**: The surrogate objective is an approximation and may not always perfectly reflect the true improvement.
### Why and How PPO Uses the Clipped Estimate
PPO replaces the explicit KL penalty with a **clipped surrogate objective** (as shown in the image) to limit how much the new policy $\pi_\theta$ can deviate from the old policy $\mu$ in a single update. Instead of directly penalizing the KL divergence, PPO clips the importance sampling ratio: $r(\theta) = \frac{\pi_\theta(a|s)}{\mu(a|s)}$
The objective becomes: $\mathcal{L}_{\text{CLIP}}(\theta) = \mathbb{E}_{s \sim d^\mu(s),\ a \sim \mu(\cdot|s)} \left[ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A^\mu(s, a) \right]$
This means if $r(\theta)$ moves outside $[1-\epsilon, 1+\epsilon]$, the advantage is not further increased, preventing excessively large policy updates.
### How PPO Improves the Clip Term
PPO improves stability by ensuring that the policy update does not push $r(\theta)$ too far from 1. This is a simpler and more robust alternative to enforcing a hard KL constraint or penalty, as it directly restricts the update at the level of each sample.
![PPO](CMU_Notes/DRL/img/PPO.png)

### Gradient Behaviors Induced by Clipping
- **Within the Clip Range:** When $r(\theta)$ is within $[1-\epsilon, 1+\epsilon]$, the gradient behaves like the standard policy gradient.
- **Outside the Clip Range:** When $r(\theta)$ exceeds the range, the gradient is zero with respect to $r(\theta)$, so further updates in that direction are suppressed.
- **Effect:** This prevents the policy from changing too much in a single update, reducing the risk of performance collapse.
### Limitations of Clipping
- **Bias:** Clipping introduces bias, as it may prevent the policy from fully exploiting large advantages if $r(\theta)$ is clipped.
- **No Direct KL Control:** While clipping restricts large updates, it does not directly control the average KL divergence, so the actual divergence can still drift over time.
- **Hyperparameter Sensitivity:** The choice of $\epsilon$ is critical; too small slows learning, too large risks instability.
- **Gradient Saturation:** If many samples are clipped, the effective gradient can become small, slowing down learning.
- **Hurts exploration of action space**: want more flexibility on the positive side, use asymmetric clip $\epsilon_{high}$ and $\epsilon_{low}$, especially in LLM. (DAPO)
**Summary:**
PPO’s clipped objective provides a simple, effective way to limit policy updates, improving stability over vanilla policy gradients, but at the cost of some bias and less direct control over the policy divergence.
![PPO takeaway](CMU_Notes/DRL/img/PPO-takeaway.png)

---

# REINFORCE

### Policy Optimization (Lecture 3)

#### **Definitions**
- **Policy Gradient update**:  
    $\theta_{\text{new}} = \theta_{\text{old}} + \Delta\theta, \quad \Delta \theta = \alpha \nabla_{\theta} J(\theta)$  
    where $J(\theta)$ is the expected return objective, and $\alpha$ is the learning rate.
- **Advantages over value based methods**: 
	- Effective in high-dimensional or continuous action spaces
	- Can learn stochastic policies
- **Policy Functions**
    - **Deterministic continuous policy**: $a = \pi_\theta(s)$  Example: DDPG (Deterministic Policy Gradient).
    - **Stochastic continuous policy**:  $a \sim \mathcal{N}(\mu_\theta(s), \sigma^2_\theta(s))$  Example: PPO with Gaussian policies.
    - **Stochastic discrete policy**:  $∑a′exp⁡(hθ(s,a′))\pi_\theta(a \mid s) = \frac{\exp(h_\theta(s,a))}{\sum_{a'} \exp(h_\theta(s,a'))}$  (Softmax over action preferences).
- **Policy Objective**:  
    The goal is to maximize the expected return:  
    $\max _\theta \quad U(\theta)=\mathbb{E}_{\tau \sim P(\tau ; \theta)}[R(\tau)]=\sum_\tau P(\tau ; \theta) R(\tau)$
    $\max _\theta . \mathbb{E}_{x \sim P(x ; \theta)} f(x)$
- **Probability of trajectory:**
    $P(\tau ; \theta)=\prod_{t=0}^H \underbrace{P\left(s_{t+1} \mid s_t, a_t\right)}_{\text {dynamics }} \cdot \underbrace{\pi_\theta\left(a_t \mid s_t\right)}_{\text {policy }}$
#### **Derivations**
- **Gradient of expectations (likelihood ratio trick):**  
    For random variable $x \sim P_\theta$: $\nabla_\theta \mathbb{E}_x[f(x)] = \mathbb{E}_x[\nabla_\theta \log P_\theta(x) f(x)]$  In practice (Monte Carlo estimate):  
I can obtain an unbiased estimator for the gradient by sampling!
From the law of large numbers, it will converge to the right gradient with infinite number of samples.
	$\nabla_\theta \mathbb{E}_x f(x)=\mathbb{E}_{x \sim P_\theta(x)}\left[\nabla_\theta \log P_\theta(x) f(x)\right] \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log P_\theta\left(x^{(i)}\right) f\left(x^{(i)}\right)$
- **REINFORCE estimator**  
    $\nabla_\theta U(\theta) =\mathbb{E}_{\tau \sim P_\theta(\tau)}\left[\nabla_\theta \log P_\theta(\tau) R(\tau)\right]$ 
    $\approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log P_\theta\left(\tau^{(i)}\right) R\left(\tau^{(i)}\right)$
    $\approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(\alpha_t^{(i)} \mid s_t^{(i)}\right) R\left(\tau^{(i)}\right)$
- **The policy gradient estimator has an intuitive interpretation:**
	-  Increase the probability of actions in trajectories with positive rewards
	- Decrease the probability of actions in trajectories with negative rewards
	- The magnitude of the update is proportional to the trajectory reward
In each iteration of a reinforcement learning algorithm, we would (1) sample some trajectories, (2) calculate the rewards of those sampled trajectories, and (3) update the parameters 𝜃so as to move the mean of the distribution 𝜇 closer to the samples that give positive rewards, thus making those trajectories more probable.
#### **Gradient for specific policy classes**
- **Gradient for Gaussian Policy:**
	- univariate
	$\pi_\theta(a \mid s)=\mathcal{N}\left(a ; \mu_\theta(s), \sigma^2\right):\nabla_\theta \log \pi_\theta(a \mid s)=\frac{\left(a-\mu_\theta(s)\right)}{\sigma^2} \nabla_\theta \mu_\theta(s)$
	- multi-variate
	$\pi_\theta(a \mid s)=\mathcal{N}\left(a ; \mu_\theta(s), \Sigma\right): \nabla_\theta \log \pi_\theta(a \mid s)=\Sigma^{-1}\left(a-\mu_\theta(s)\right) \frac{\partial \mu_\theta(s)}{\partial \theta}$

- **Gradient for Softmax Policy for Discrete Actions**
	For discrete action space $A$:  
	$\pi_\theta(a \mid s) = \frac{\exp(\theta^\top \phi(s,a))}{\sum_{a'} \exp(\theta^\top \phi(s,a'))}$
	where $\phi(s,a)$ is a feature representation.
	- Ensures probabilities are non-negative and sum to 1.
	- Naturally differentiable → easy gradient computation.
$$
\begin{aligned}
\nabla_\theta \log \pi_\theta(a \mid s) &= \nabla_\theta\left[\log \frac{e^{h_\theta(s, a)}}{\sum_b e^{h_\theta(s, b)}}\right] \\
&= \nabla_\theta\left[\log e^{h_\theta(s, a)} - \log \sum_b e^{h_\theta(s, b)}\right] & \text{(log of fraction becomes difference)} \\
&= \nabla_\theta\left[h_\theta(s, a) - \log \sum_b e^{h_\theta(s, b)}\right] & \text{(log of exponential is the exponent)} \\
&= \nabla_\theta h_\theta(s, a) - \nabla_\theta \log \sum_b e^{h_\theta(s, b)} & \text{(distribute gradient)} \\
&= \nabla_\theta h_\theta(s, a) - \frac{1}{\sum_b e^{h_\theta(s, b)}} \nabla_\theta \sum_b e^{h_\theta(s, b)} & \text{(chain rule)} \\
&= \nabla_\theta h_\theta(s, a) - \frac{1}{\sum_b e^{h_\theta(s, b)}} \sum_b \nabla_\theta e^{h_\theta(s, b)} & \text{(gradient of sum is sum of gradients)} \\
&= \nabla_\theta h_\theta(s, a) - \frac{1}{\sum_b e^{h_\theta(s, b)}} \sum_b e^{h_\theta(s, b)} \nabla_\theta h_\theta(s, b) & \text{(chain rule)} \\
&= \nabla_\theta h_\theta(s, a) - \sum_b \frac{e^{h_\theta(s, b)}}{\sum_b e^{h_\theta(s, b)}} \nabla_\theta h_\theta(s, b) & \text{(push scalar sum into summation)} \\
&= \nabla_\theta h_\theta(s, a) - \sum_b \pi_\theta(s, b) \nabla_\theta h_\theta(s, b) & \text{(softmax policy definition)}
\end{aligned}
$$
#### **Temporal Structure**
So far, our derivation for the policy objective gradient assigns the blame (or reward) of the full trajectory 𝑅(𝜏(𝑖)) to each action in the trajectory. However, actions later in the trajectory do not impact rewards from earlier in the trajectory and should not be blamed as such.
- MC vs TD: Monte Carlo (MC) targets $Gt​=∑_{k≥0}​γkr_{t+k}$​ are (asymptotically) **unbiased but high-variance** and require the end of an episode; TD(0) uses the bootstrapped target $r_t​+γV(s_t+1​)$, introducing **bias but dramatically lowering variance** and enabling online updates.
- Policy gradients sum contributions over all time steps:  
    $\nabla_\theta J(\theta) = \mathbb{E}\Bigg[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t\Bigg]$
    $\begin{aligned} \hat{g} & =\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right) R\left(\tau^{(i)}\right) \\ & =\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right)\left(\sum_{k=0}^T R\left(s_k^{(i)}, a_k^{(i)}\right)\right) \\ & =\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right)\left(\sum_{k=0}^{t-1} R\left(s_k^{(i)}, a_k^{(i)}\right)+\sum_{k=t}^T R\left(s_k^{(i)}, a_k^{(i)}\right)\right) \\ & =\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right)\left(\sum_{k=t}^T R\left(s_k^{(i)}, a_k^{(i)}\right)\right)\end{aligned}$
	$G_t= \sum_{k=t}^T R\left(s_k^{(i)}, a_k^{(i)}\right)$
- Credit assignment problem: rewards at later times are attributed back to earlier actions.
#### **REINFORCE- Monte Carlo**
![REINFORCE Algorithm](CMU_Notes/DRL/img/REINFORCE.png)
#### **Variance Reduction**
- Gradient is unbiased, but needs a very large N.
- Monte Carlo estimators suffer from **high variance**.
    $\text{Var}(\hat{g}) = \text{tr}\left( \mathbb{E}\left[ (\hat{g} - \mathbb{E}[\hat{g}])(\hat{g} - \mathbb{E}[\hat{g}])^T \right] \right) = \sum_{k=1}^n \mathbb{E}\left[ (\hat{g}_k - \mathbb{E}[\hat{g}_k])^2 \right]$
- Solution: subtract a **baseline** $b(s_t)$ without changing expectation:  
    $\nabla_\theta J(\theta) = \mathbb{E}\Big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, (G_t - b(s_t))\Big]$
- **Choices of baseline**:
    - Constant baseline $\hat{b} = \mathbb{E}[R(\tau)]$
    - Time-dependent baseline $b_t = \sum_{i=1}^N G_t^{(i)}$
    - State-dependent baseline $b(s)$
     $b(s) = \mathbb{E}[r_t + r_{t+1} + r_{t+2} + \dots + r_{T-1} | s_t = s] = V_{\pi}(s)$
- **Optimal baseline** (to minimize variance): the **state-value function** $V^\pi(s)$.![REINFORCE Baseline](CMU_Notes/DRL/img/REINFORCE_baseline.png)

---

# SAC

### **What it is**
- An **off-policy, stochastic actor–critic algorithm**.
- Extends DDPG by adding **maximum entropy RL** (reward + entropy).
- Learns both a stochastic policy and Q-functions.
### **Key Equations**
- **Soft value function:**$V(s) = \mathbb{E}_{a \sim \pi_\theta} \big[ Q(s,a) - \alpha \log \pi_\theta(a|s) \big]$
- **Critic target:** $y = r + \gamma \, \mathbb{E}_{a' \sim \pi_\theta} \big[ Q^{tgt}(s',a') - \alpha \log \pi_\theta(a'|s') \big]$
- **Actor update:** maximize entropy-regularized objective: $J(\pi) = \mathbb{E}_{s \sim D, a \sim \pi} \big[ Q(s,a) - \alpha \log \pi(a|s) \big]$
### **Why use it**
- DDPG can overfit to narrow action regions → brittle policies.
- SAC maintains **stochasticity and entropy** → better exploration & robustness.
### **Problems it solves**
- Avoids **deterministic overfitting**.
- Learns **robust policies** that generalize better.
- More **stable** than DDPG.
### **SAC vs A2C**
- **SAC is off-policy** → sample efficient.
- Adds **entropy maximization** → better exploration.
- Handles **continuous control** easily.
- **More stable** than A2C (low variance actor update).
- **But** more computationally heavy.
##### ✅ **SAC Advantages**
- Stable, robust training (entropy regularization).
- Strong exploration, avoids premature convergence.
- Works very well on continuous control benchmarks.
- Off-policy → sample efficiency.
##### ❌ **SAC Disadvantages**
- More complex than DDPG/A2C.
- Extra hyperparameter $\alpha$ (though auto-tuning helps).
- Computationally heavier (learns multiple Qs + actor).

---

# TD3

![TD3](CMU_Notes/DRL/img/TD3.png)
###  1. What is TD3?
TD3 = **Twin Delayed Deep Deterministic Policy Gradient**
- An **off-policy, actor–critic algorithm** for **continuous action spaces**.
- Built as an improvement over **DDPG**, addressing its instability.
###  2. Why was TD3 introduced?
**DDPG Problems:**
1. **Overestimation bias** in Q-values (similar to DQN).
2. **Exploiting function approximation errors** (policy pushes toward overestimated actions).
3. **Unstable actor updates** (actor learns from noisy critic).
**TD3 Fixes:**
- Introduces three key tricks to stabilize training:
1. **Twin Critics (Clipped Double Q-learning)**
    - Maintain two Q-networks $Q_{\varphi_1}, Q_{\varphi_2}$​​.
    - Target uses the **minimum**: $y = r + \gamma \min_{i=1,2} Q^{tgt}_{\varphi_i}(s', \pi^{tgt}_\theta(s'))$
    - Reduces overestimation bias.
2. **Target Policy Smoothing**
    - Add noise to target policy actions in critic target: $a' = \pi^{tgt}_\theta(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0,\sigma), -c, c)$
    - Prevents critic from exploiting sharp Q-function peaks.
3. **Delayed Policy Updates**
    - Update actor less frequently than critics (e.g., 1 actor update for every 2 critic updates).
    - Actor only updates when critics are stable.
###  3. How TD3 Compares with DDPG
| Aspect            | **DDPG**                    | **TD3**                             |
| ----------------- | --------------------------- | ----------------------------------- |
| **Critic**        | Single Q                    | Twin Qs (min of two)                |
| **Target**        | Deterministic, no smoothing | Noisy target actions → smoother     |
| **Actor updates** | Every step                  | Delayed (less frequent)             |
| **Bias**          | Overestimation              | Mitigated via clipped double Q      |
| **Stability**     | Brittle, unstable           | Much more stable                    |
| **Performance**   | Weaker                      | State-of-the-art continuous control |
###  4. Advantages of TD3
✅ **More stable training** than DDPG.  
✅ **Reduces overestimation bias** with twin critics.  
✅ **Better performance** on continuous control benchmarks (MuJoCo, robotics).  
✅ **Sample efficient** (off-policy replay buffer).  
✅ **Simple extension of DDPG** (easy to implement once DDPG is in place).
###  5. Disadvantages of TD3
❌ Still **deterministic policy** → weak exploration compared to SAC.  
❌ More compute cost than DDPG (two critics instead of one).  
❌ Sensitive to hyperparameters (noise scale, policy delay).  
❌ Doesn’t explicitly encourage **policy stochasticity or robustness** (unlike MaxEnt methods).
###  6. Takeaway
- **TD3 = “DDPG but fixed.”**
- Solves DDPG’s overestimation and instability with **twin critics, target smoothing, delayed actor updates**.
- Works extremely well for continuous control benchmarks.
- But compared to **SAC**, TD3 has deterministic policies → less robust and exploratory.

---

# Value Optimization

Value Iteration - dynamic programming
1. $Q(s,a)=r(s,a)+\gamma Es'​[V(s')]$
2. $V(s)\leftarrow max_a ​Q(s,a)$
The entire purpose of Value Iteration is to calculate the optimal value function, $V^*(s)$. This represents the maximum possible cumulative reward an agent can achieve from any state $s$. To achieve this maximum reward, the agent must follow an **optimal policy ($\pi^*$)**, which is inherently **greedy**. The update step in your question, $V(s) \leftarrow \max_a Q(s,a)$, is the mathematical expression of this greedy choice. It directly implements the Bellman Optimality Equation
- $Q(s, a)$ tells you the value of taking action $a$ in state $s$.
- $\max_a Q(s, a)$ finds the value of the single best action and sets $V(s)$ to that maximum possible value. It calculates the value of a state under the assumption that you will _always_ act optimally from that point forward.
[[Fitted Q Iteration]] --> [[DQN]]