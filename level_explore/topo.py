import json
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

LABEL_NAMES = ["Text", "List Item", "Image", "Text Button", "Icon", "Toolbar",
               "Input", "Advertisement", "Card", "Web View", "Drawer",
               "Background Image", "Radio Button", "Modal", "Multi-Tab",
               "Pager Indicator", "Slider", "On/Off Switch", "Map View",
               "Bottom Navigation", "Video", "Checkbox", "Button Bar",
               "Number Stepper", "Date Picker"]
ID_TO_LABEL = dict(
    {i: v for (i, v) in enumerate(LABEL_NAMES)})
LABEL_TO_ID_ = dict(
    {l: i for i, l in ID_TO_LABEL.items()})
level_graph = np.load("level_graph_one.npy")
print(level_graph)
graph = level_graph > 0
# 记录每个结点的访问状态：未访问（0），正在访问（1），已完成（2）
visited = [0] * len(graph)

# 记录当前正在访问的路径
path = []

# 记录所有找到的环
cycles = []


# 定义一个函数来判断是否是新环（即没有被标记过）
def is_new_cycle(cycle):
    # 对于每个已经找到的环
    for c in cycles:
        # 如果新环和旧环长度相同，并且包含相同的元素，则认为是同一个环
        if len(cycle) == len(c) and set(cycle) == set(c):
            return False  # 不是新环
    return True  # 是新环


# 定义一个函数来进行深度优先搜索
def dfs(v):
    global graph, visited, path, cycles  # 引用全局变量

    # 将当前结点标记为正在访问，并加入当前路径
    visited[v] = 1
    path.append(v)

    # 遍历当前结点的邻接结点
    for i in range(len(graph[v])):
        if graph[v][i] == 1:  # 如果存在边 (v,i)
            if visited[i] == 0:  # 如果邻接结点未被访问过，则继续递归搜索
                dfs(i)
            elif visited[i] == 1:  # 如果邻接结点正在被访问，则说明找到了一个环

                # 获取当前路径中从邻接结点开始到当前结点结束的部分作为一个环，并判断是否是新环（即没有被标记过）
                cycle = path[path.index(i):]
                if is_new_cycle(cycle):
                    cycles.append(cycle)  # 将新环加入结果集合

    # 将当前结点标记为已完成，并从当前路径中移除
    visited[v] = 2
    path.pop()


# 遍历所有未被访问过的顶点，进行深度优先搜索
for i in range(len(graph)):
    if visited[i] == 0:
        dfs(i)

# 输出所有找到的环
for cycle in cycles:
    print([ID_TO_LABEL[i] for i in cycle])
