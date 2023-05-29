from FrozenLake_SI import PolicyIterationAgent
from FrozenLake_VI import ValueIterationAgent
import gymnasium as gym
import numpy as np
from copy import deepcopy

env = gym.make("FrozenLake-v1", is_slippery=False)  # 创建环境
action_meaning = ['←', '↓', '→', '↑']  # 这个动作意义是gym库中对FrozenLake这个环境事先规定好的
theta = 1e-5
gamma = 0.9
map = np.array([['S', 'F', 'F', 'F'], ['F', 'H', 'F', 'H'], ['F', 'F', 'F', 'H'], ['H', 'F', 'F', 'G']])

# 该函数返回map数组中环境当前状态的行和列索引
def get_indices_of_current_state(state):
    temp = 0 # 临时变量初始化为0
    for i in range(len(map)): # 在map的行和列中循环，每次迭代的时候将temp递增1
        for j in range(len(map[0])):
            if temp == state: # 当temp等于环境当前状态时，返回当前的行和列索引(i,j)
                return i, j
            temp += 1

def print_current_state(state):
    temp_map = deepcopy(map)
    row, column = get_indices_of_current_state(state)
    temp_map[row][column] = 'X'
    for r in temp_map:
        print(r[0], r[1], r[2], r[3])

def print_action(action):
    if action==0:
        print('←')
    elif action==1:
        print('↓')
    elif action==2:
        print('→')
    elif action==3:
        print('↑')

# 打印策略的函数，打印当前策略的每个状态下的价值以及采取的动作
def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("------------------OptimalValue------------------")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.V[i * agent.env.ncol + j]), end=' ')
        print()
    print('\n')

    print("------------------OptimalPolicy------------------")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 如果状态被标记为灾难或结束，函数不会打印出策略，而是打印特殊字符串（灾难和结束状态都是直接传参数）
            if (i * agent.env.ncol + j) in disaster:  # 冰窟状态
                print(f'State {4 * i + j} -> Optimal Action: Hole')
            elif (i * agent.env.ncol + j) in end:  # 终点
                print(f'State {4 * i + j} -> Optimal Action: End')
            else:
                # 策略存储在agent对象的pi属性中
                a = agent.policy[i * agent.env.ncol + j]
                for k in range(len(action_meaning)):
                    if a[k] > 0:
                        print(f'State {4 * i + j} -> Optimal Action:{action_meaning[k]}')
    print('\n')

def train(iteration_name):
    if iteration_name == 'PolicyIteration':
        # 策略迭代
        print('PolicyIteration Beginning...')
        agent = PolicyIterationAgent(env, theta, gamma)
        agent.policy_iteration()
    elif iteration_name == 'ValueIteration':
        # 价值迭代
        print('ValueIteration Beginning...')
        agent = ValueIterationAgent(env, theta, gamma)
        agent.value_iteration()
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
    return agent

def test(agent):
    print('------------------Game Beginning...------------------')
    # 使用agent.pi来通过游戏
    optimal_strategies = agent.policy
    # print(optimal_strategies)

    state = env.reset()
    # print(state)
    # print(type(state))
    # print(type(state[0]))

    done = False
    count = 0
    while not done:
        if count > 0:
            action = optimal_strategies[state].index(max(optimal_strategies[state]))
            # print('当前状态：'+str(state))
            print_current_state(int(state))
            #         print('当前动作：'+str(action))
            print('Current selection action:')
            print_action(action)
            print('\n')
            state, reward, t_1, t_2, info = env.step(action)
            done = t_1 or t_2
            # env.render()
            count = count + 1
        else:
            action = optimal_strategies[state[0]].index(max(optimal_strategies[state[0]]))
            # print('当前状态：'+str(state[0]))
            print_current_state(int(state[0]))
            #         print('当前动作：'+str(action)+'\n')
            print('Current selection action:')
            print_action(action)
            print('\n')
            state, reward, t_1, t_2, info = env.step(action)
            done = t_1 or t_2
            # env.render()
            count = count + 1
    print('Reach the goal!')
    print('------------------Game Ending...------------------')
    env.close()

if __name__ == '__main__':
    agent = train('ValueIteration') # 选择策略迭代或者价值迭代
    # agent = train('PolicyIteration')
    test(agent)