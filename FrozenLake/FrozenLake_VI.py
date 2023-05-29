import copy

class ValueIterationAgent: # 动态规划之价值迭代
    def __init__(self,env,theta,gamma):
        self.env = env
        self.theta = theta# 价值收敛阈值
        self.gamma = gamma# discounted折扣因子
        # 初始化价值为0，v是长度为env.ncol*env.nrow的列表，表示每个状态的值函数
        self.V = [0] * self.env.ncol * self.env.nrow
        # pi是长度为nv.ncol*nv.nrow的表，表示算法学习的最优策略
        self.policy = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]

    # 通过使用Bellman方程重复计算每个状态的期望值并找到所有可能动作的最大期望值来更新v列表，直到新旧值函数之间的差值小于self.theta
    def value_iteration(self):
        while 1:
            delta = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            # 对于每个状态s，函数计算每个动作a的期望值，并选择使该值最大化的动作
            for s in range(self.env.ncol * self.env.nrow):
                exp_list = []# 用于存储每个状态-动作对的Q值，Q值是通过在特定状态s中采取特定动作a而获得的未来奖励的预期总和
                for a in range(4):# 对于每个动作a
                    exp = 0
                    for res in self.env.P[s][a]:# 使用当前值函数计算采取该操作的预期值
                        exp += res[0] * (res[2] + self.gamma * self.V[res[1]]*(1-res[3]))
                    exp_list.append(exp)# 将此值添加到列表qsa_list中
                    # 一旦已经为状态s中的所有动作计算了Q值，该算法就将状态s的值设置为所有动作的最大Q值，这对应于贝尔曼最优方程
                new_v[s] = max(exp_list)
                # 将new_v中的状态值设置为qsa_list中的最大值 ，该方程指出，一个状态的最优值等于所有可能行动的未来回报的最大预期总和
                delta = max(delta, abs(new_v[s] - self.V[s]))
            self.V = copy.deepcopy(new_v)
            if delta < self.theta:
                break
        print('Value Iteration Done!')
        # 通过为每个状态选择具有最大期望值的操作，基于最终的v列表来计算最优策略
        # 根据价值函数导出一个贪心策略
        for s in range(self.env.ncol * self.env.nrow):
            exp_list = []
            for a in range(4):
                exp = 0
                for res in self.env.P[s][a]:
                    # 使用最优值函数self.v计算每个状态-动作对的Q值
                    exp += res[0] * (res[2] + self.gamma * self.V[res[1]]*(1-res[3]))
                exp_list.append(exp)
            max_exp = max(exp_list)
            cnt_max = exp_list.count(max_exp)
            # 将具有最大Q值的动作的概率设置为1，并将所有其他动作的概率设为0来提取每个状态的最优策略
            # 让相同的动作价值均分概率
            self.policy[s] = [1 / cnt_max if q == max_exp else 0 for q in exp_list]
        print('Policy Improvement Done!')
