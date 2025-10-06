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
    