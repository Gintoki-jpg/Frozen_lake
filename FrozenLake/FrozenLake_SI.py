import copy

class PolicyIterationAgent: # 动态规划之策略迭代
    def __init__(self,env,theta,gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.V = [0] * self.env.ncol * self.env.nrow  # 将状态值函数v初始化为零
        self.policy = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]  # 将策略pi初始化为统一随机策略
    def policy_evaluation(self):
        while 1:
            delta = 0 # 每次迭代开始时初始化为0，以跟踪值函数的最大变化
            new_v = [0] * self.env.ncol * self.env.nrow # 将状态值函数v初始化为零
            for s in range(self.env.ncol * self.env.nrow): # 在环境中所有状态上循环，实现在状态空间列表上迭代
                exp_list = []
                for a in range(4): # 对于每个状态s，迭代所有动作来计算每个状态的状态值函数
                    exp = 0 # 用于跟踪状态s中每个动作a的预期回报
                    for res in self.env.P[s][a]: # 预期回报是通过在状态s中对采取行动a的所有可能结果进行循环
                        # 预期回报=下一状态next_state的奖励和discounted value之和
                        exp += res[0] * (res[2] + self.gamma * self.V[res[1]]*(1-res[3]))
                    # 将所有操作的加权预期回报相加来计算状态s的新值
                    exp_list.append(self.policy[s][a] * exp)
                new_v[s] = sum(exp_list)
                # 计算新的和旧的状态值函数之间的最大差值
                delta = max(delta, abs(new_v[s] - self.V[s]))
            # self.v变量被更新为新的值函数new_v
            self.V = copy.deepcopy(new_v)
            if delta < self.theta:
                break
        print('Policy Evaluation Done!')

    def policy_improvement(self):# 策略提升，agent通过选择使每个状态的预期discounted累积奖励最大化的操作来更新其策略
        for s in range(self.env.ncol * self.env.nrow):# 在环境中所有可能的状态s上循环
            exp_list = []
            for a in range(4):
                exp = 0# 对于每个状态s，agent计算每个可能动作的预期siacounted积累奖励
                for res in self.env.P[s][a]:
                    # 贝尔曼方程 qsa = sum(p * (r + gamma * v(next_state)) for p, next_state, r, done in P[s][a])
                    # P[s][a]是一组可能的下一个状态、奖励和转换到下一个状态的概率
                    exp += res[0] * (res[2] + self.gamma * self.V[res[1]]*(1-res[3]))
                exp_list.append(exp)
            max_exp = max(exp_list)# 选择最大值maxq并计算具有该最大值的动作数量cntq
            cnt_max = exp_list.count(max_exp)
            # 所有具有最大动作值的动作被选中的概率相等，而所有其他动作的概率为0
            self.policy[s] = [1 / cnt_max if q == max_exp else 0 for q in exp_list]
        print('Policy Improvement Done!')
        return self.policy

    def policy_iteration(self):
        while 1:
            # self.policy_evaluation()
            # self.policy_improvement()
            # if self.policy == self.policy:
            #     break
            # self.policy = copy.deepcopy(self.policy)
            self.policy_evaluation() # 调用policy_evaluation函数，通过计算当前策略的状态值函数v来评估该策略
            old_policy = copy.deepcopy(self.policy)# 将列表进行深拷贝，方便接下来进行比较
            new_policy = self.policy_improvement()# 调用policy_improvement函数，通过选择最大化预期长期回报的行动（即行动值函数q）来改进当前策略
            if old_policy == new_policy:
                break
        print('Policy Iteration Done!')







