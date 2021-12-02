## CMSC742 Course Project
### Adversarial Communication in Multi-Agent Reinforcement Learning System
<font size="12"> Chenghao Deng, Haowei Deng, Dehao Yuan</font> 

The goal of the project is to improve the robustness of deep Q-learning by adding a simple regularization term. The regularization term serves as a random smoother and has been proved effective in classification tasks. In this project, we adapt the regularization term to reinforcement learning tasks and shows the effectiveness by experimenting in a simple environment setting. The detail of the algorithm can be viewed in our [final report](https://www.google.com).

We use CartPole as the environment setting and show the randomized smoothing does not affect the performance of the agent, while improve the robustness against attack.

1. Install the required packages:
`pip install -r requirements.txt`
2. Train an agent without attacking:
`python train_DQN.py`
3. After training the agent, it will be stored in `models/Agent.pth`. To attack the agent:
`python train_attacker.py`
4. Train an agent with random smoothing:
`python train_DQN.py`
5. After training the agent, it will be stored in 'models/Agent_robust.pth'. To attack the robust agent:
`python train_attacker.py`
