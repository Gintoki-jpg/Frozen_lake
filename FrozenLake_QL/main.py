from FrozenLake import FrozenLake
from Q_Learning import QlearningAgent

def train():
    env = FrozenLake()
    agent = QlearningAgent(env)
    agent.Q_Learning()
    agent.print_results(['←', '↓', '→', '↑'],[5, 7, 11, 12], [15])
    return agent

def test(agent):
    print('------------------Game Beginning...------------------')
    agent.latest_iteration()
    print('------------------Game Ending...------------------')

if __name__ == '__main__':
    agent=train()
    test(agent)
