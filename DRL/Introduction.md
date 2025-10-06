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


