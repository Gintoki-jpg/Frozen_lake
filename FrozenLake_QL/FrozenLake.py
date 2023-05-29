import enum
import numpy as np
from copy import deepcopy

class Action(enum.Enum):
    Left = 0
    Down = 1
    Right = 2
    Up = 3

class FrozenLake:
    def __init__(self):
        self.map = np.array([['S', 'F', 'F', 'F'], ['F', 'H', 'F', 'H'], ['F', 'F', 'F', 'H'], ['H', 'F', 'F', 'G']])

        self.action_space = np.array([Action.Left, Action.Right, Action.Up, Action.Down])
        self.state_space = [i for i in range(np.array(self.map).size)]
        # 为每个操作定义一组无效状态（0,1,2,3）表示处于这些状态时向上移动无意义
        self.no_left_states = [4, 8, 0, 12]
        self.no_right_states = [7, 11, 3, 15]
        self.no_up_states = [1, 2, 0, 3]
        self.no_down_states = [13, 14, 12, 15]
        # 定义当前状态为0，也就是起始位置
        self.current_state = 0

    # 定义获取当前动作空间和状态空间的方法
    def get_action_space(self):
        return self.action_space

    def get_state_space(self):
        return self.state_space

    # 随机返回动作空间中的一个动作
    def get_random_action(self):
        return np.random.choice(self.action_space)

    # step将一个action作为参数，相应地更新当前状态
    def step(self, action_index):
        action = Action(action_index)
        # 调用invalid_action方法检查操作是否无效
        # 如果操作无效（如将agent从grid中删除），则返回当前状态、0奖励以及指示事件未结束False
        if self.invalid_action(action):
            return self.current_state, 0, False
        # 如果操作有效，该方法会根据action更新当前的state
        if action == Action.Left:
            self.current_state -= 1
        elif action == Action.Right:
            self.current_state += 1
        elif action == Action.Up:
            self.current_state -= 4
        else:
            self.current_state += 4
        # 调用get_indices_of_current_state方法获取新状态的行和列索引，并在映射中查找相应字母
        row, column = self.get_indices_of_current_state()
        letter = self.map[row][column]
        # 如果代理在冰块上则返回0奖励
        if letter == 'S' or letter == 'F':
            return self.current_state, 0, False
        # 如果代理到达终点则返回1奖励
        elif letter == 'G':
            return self.current_state, 1, True
        # 如果代理掉进Hole则返回0奖励（事件结束但agent尚未到达目标）
        else:
            return self.current_state, 0, True

        # 检查给定的操作对于环境当前状态是否有效，需要借助前面定义的操作对应的无效状态
        # 若不允许该操作则返回Ture，否则返回False表示该动作有效
    def invalid_action(self, action):
        if (action == Action.Left and self.current_state in self.no_left_states) or \
                (action == Action.Right and self.current_state in self.no_right_states) or \
                (action == Action.Up and self.current_state in self.no_up_states) or \
                (action == Action.Down and self.current_state in self.no_down_states):
            return True

        return False

        # 该函数返回map数组中环境当前状态的行和列索引
    def get_indices_of_current_state(self):
        temp = 0  # 临时变量初始化为0
        for i in range(len(self.map)):  # 在map的行和列中循环，每次迭代的时候将temp递增1
            for j in range(len(self.map[0])):
                if temp == self.current_state:  # 当temp等于环境当前状态时，返回当前的行和列索引(i,j)
                    return i, j
                temp += 1

        # 重置当前状态为0（即回到起点）
    def reset(self):
        self.current_state = 0
        return self.current_state

        # 通过显示map数组打印环境的当前状态X所在位置
    def print_current_state(self):
        temp_map = deepcopy(self.map)  # 首先使用copy模块中的deepcopy（）函数创建self.map数组的副本，这样对副本所做的任何修改都不会影响原始副本
        row, column = self.get_indices_of_current_state()
        temp_map[row][column] = 'X'  # 调用get_indices_of_current_state（）来获取当前状态的行和列索引，并在self.map数组的副本中用“X”标记该状态
        # 循环遍历self.map数组副本的行，将`每一行`打印为字符串，每个元素之间有一个空格字符
        for r in temp_map:
            print(r[0], r[1], r[2], r[3])