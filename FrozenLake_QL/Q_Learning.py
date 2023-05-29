import numpy as np
import random
import time

# Q-learning类使用Q-learning学习算法训练agent在grid世界中的navigate
class QlearningAgent:
    # 构造函数初始化环境、Q学习算法参数和Q表
    def __init__(self,env):
        self.env = env

        self.episodes = 10000# agent在训练期间运行的episode
        self.max_steps = 100# agent在每个episode中执行的最大步骤数
        self.learning_rate = 0.1# 学习率确定Q值更新步长
        self.discount_rate = 0.99# 折扣因子
        self.epsilon = 1.0# 根据当前Q值确定代理采取随机动作而不是最佳动作的概率
        self.epsilon_min = 0.01 # 探索速率下限
        self.epsilon_max = 1.0# 探索速率上限
        self.epsilon_decay = 0.001 # 确定探索速率随时间降低的速率
        # Q表被初始化为一个numpy数组，其维度等于环境中可能的状态和操作的数量，并且所有条目都被设置为零
        self.Q = np.zeros((len(env.get_state_space()), len(env.get_action_space())))
        # self.Q = np.zeros(len(self.env.get_state_space()),len(self.env.get_action_space()))
        # self.q_table = np.zeros((len(environment.get_state_space()), len(environment.get_action_space())))
        # all_epimode_rewards列表用于跟踪agent在训练期间每episode获得的总奖励
        self.rewards = []

    # 核心算法，在指定的episodes中运行Q-learning算法，在每个episode上迭代并根据当前state和Q表选择action，同时根据收到的rewards和下一状态的最大Q值更新Q表
    def Q_Learning(self):
        for episode in range(self.episodes):
            total_reward = 0# 初始化当前episode的reward为0
            state = self.env.reset()# 重置当前状态为起始状态
            for step in range(self.max_steps):# 在max_steps_per_set范围内的每个步骤上循环
                rand = random.uniform(0,1)
                # 如果随机生成的数字小于探索率，则使用环境的get_random_action（）方法选择一个随机操作
                if rand < self.epsilon:
                    action = self.env.get_random_action().value
                # 否则使用numpy的argmax（）方法为当前状态选择Q值最高的动作，以检索Q值最高动作的索引
                else:
                    action = np.argmax(self.Q[state,:])
                # 使用环境的step方法将所选action作用在环境上，step返回下一个状态、采取动作的奖励以及指示事件是否完成的布尔值
                new_state,reward,done = self.env.step(action)
                # 更新当前<状态，动作>对的Q值
                self.Q[state][action] = self.Q[state][action] * (1 - self.learning_rate) + \
                                                    self.learning_rate * (
                                                            reward + self.discount_rate * np.max(
                                                        self.Q[new_state, :]))
                # 更新当前状态
                state = new_state
                # 用从action中获得rewards更新当前episode的总奖励
                total_reward += reward
                if done:# 假如该episode已完成，则循环中断
                    break
            # 每次episode结束后，通过使用指数衰减函数将探索率从最大值逐渐降低到最小值来更新探索率
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)
            # 将当前eposide获得的总奖励存储在一个名为total_reward的列表中
            self.rewards.append(total_reward)

    # 打印Q表和每1000个episode的平均rewards
    def print_action(self,action):
        if action == 0:
            print('←')
        elif action == 1:
            print('↓')
        elif action == 2:
            print('→')
        elif action == 3:
            print('↑')

    def print_results(self, action_meaning, disaster=[], end=[]):
        print('------------------Q-Table------------------')
        print(self.Q)
        print('\n')

        print('------------------OptimalPolicy------------------')
        q_table = self.Q
        optimal_actions = np.argmax(q_table, axis=1)
        for i in range(4):
            for j in range(4):
                if (4 * i + j) in disaster:
                    print(f'State {4 * i + j} -> Optimal Action: Hole')
                elif (4 * i + j) in end:
                    print(f'State {4 * i + j} -> Optimal Action: End')
                else:
                    print(f'State {4 * i + j} -> Optimal Action:{action_meaning[optimal_actions[4 * i + j]]}')
        print('\n')

        # Calculate and print the average reward per thousand episodes
        rewards_per_thousand_episodes = np.split(np.array(self.rewards), self.episodes / 1000)
        count = 1000

        print("------------------AverageRewards------------------")
        for r in rewards_per_thousand_episodes:
            print(count, ": ", str(sum(r) / 1000))
            count += 1000
        print('\n')

    # 运行Q学习算法的最新迭代并打印grid世界的当前状态，代理的当前位置用“X”标记，如果代理达到目标，额外打印结束信息
    def latest_iteration(self):
        state= self.env.reset()
        for step in range(self.max_steps):
            self.env.print_current_state()
            time.sleep(1)
            action = np.argmax(self.Q[state,:])
            print('当前选择动作：')
            self.print_action(action)
            print('\n')
            new_state,reward,done = self.env.step(action)
            state = new_state
            if done:
                print('Reached the goal')
                break
