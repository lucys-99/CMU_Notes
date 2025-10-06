Value Iteration - dynamic programming
1. $Q(s,a)=r(s,a)+\gamma Es'​[V(s')]$
2. $V(s)\leftarrow max_a ​Q(s,a)$
The entire purpose of Value Iteration is to calculate the optimal value function, $V^*(s)$. This represents the maximum possible cumulative reward an agent can achieve from any state $s$. To achieve this maximum reward, the agent must follow an **optimal policy ($\pi^*$)**, which is inherently **greedy**. The update step in your question, $V(s) \leftarrow \max_a Q(s,a)$, is the mathematical expression of this greedy choice. It directly implements the Bellman Optimality Equation
- $Q(s, a)$ tells you the value of taking action $a$ in state $s$.
- $\max_a Q(s, a)$ finds the value of the single best action and sets $V(s)$ to that maximum possible value. It calculates the value of a state under the assumption that you will _always_ act optimally from that point forward.
[[Fitted Q Iteration]] --> [[DQN]]