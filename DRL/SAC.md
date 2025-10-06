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