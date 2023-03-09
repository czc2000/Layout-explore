## 探索单个布局中元素的层级关系，并尝试建立出一种树形的结构
import collections
import json
import math
from pathlib import Path
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean
import numpy
from tqdm import tqdm
import level_vis as vis
import numpy as np

LABEL_NAMES = ("Text", "List Item", "Image", "Text Button", "Icon", "Toolbar",
               "Input", "Advertisement", "Card", "Web View", "Drawer",
               "Background Image", "Radio Button", "Modal", "Multi-Tab",
               "Pager Indicator", "Slider", "On/Off Switch", "Map View",
               "Bottom Navigation", "Video", "Checkbox", "Button Bar",
               "Number Stepper", "Date Picker")
ID_TO_LABEL = dict(
    {i: v for (i, v) in enumerate(LABEL_NAMES)})
LABEL_TO_ID_ = dict(
    {l: i for i, l in ID_TO_LABEL.items()})
level_graph = np.load("level_graph_one.npy")


def load(path=""):
    with open(path, "r", encoding='utf-8') as file:
        layout = json.load(file)
        return layout


def find_parent(part, parent):
    part["parent"] = parent
    if "children" not in part.keys():
        return
    for child in part["children"]:
        find_parent(child, part)


def preorder(part, graph):
    if "children" not in part.keys():
        return
    for child in part["children"]:
        parents_statistics(child, graph)
        preorder(child, graph)


def parents_statistics(part, graph):
    child_n = part["number"]
    parent = part["parent"]
    while parent is not None and "componentLabel" in parent.keys():
        parent_n = parent["number"]
        graph[parent_n][child_n] += 1
        parent = parent["parent"]


def get_graph_full(layout):
    find_parent(layout, parent=None)
    boxs_nums = count_nodes(layout, 0)
    single_graph = np.zeros([boxs_nums, boxs_nums])
    preorder(layout, single_graph)
    return single_graph


def test_judgement(layout):
    graph = get_graph_full(layout)
    flat_layout = append_child(layout, [])
    if len(flat_layout) == 1:
        return numpy.zeros(1), numpy.zeros(1)
    # rels = numpy.zeros((len(flat_layout) * (len(flat_layout) - 1)) // 2)
    # juds = numpy.zeros((len(flat_layout) * (len(flat_layout) - 1)) // 2)
    rels = numpy.zeros((len(flat_layout) * len(flat_layout)))
    juds = numpy.zeros((len(flat_layout) * len(flat_layout)))
    total = 0
    for i in range(len(flat_layout)):
        for j in range(len(flat_layout)):
            box1 = flat_layout[i]
            box2 = flat_layout[j]
            n1 = box1[5]
            n2 = box2[5]
            jud = judge_relation(box1, box2)
            rel = 1 if graph[n1][n2] else 0
            rels[total] = rel
            juds[total] = jud
            total += 1
    return rels, juds


def count_nodes(layout, number):
    ## 这里希望在计算node数的同时，给布局的元素编上号
    layout["number"] = number
    if "children" not in layout.keys():
        return 1
    total = 1
    for i, child in enumerate(layout["children"]):
        num = count_nodes(child, number + 1)
        number += num
        total += num
    return total


def level_order(layout, nums):
    levels = []
    relation = np.zeros([nums, nums])
    queue = [layout]
    while queue:
        size = len(queue)
        temp = []
        for i in range(size):
            node = queue.pop(0)
            ## class,x1,y1,x2,y2,number
            cxy = [0] * 6
            cxy[0] = LABEL_TO_ID_[node["componentLabel"]] if "componentLabel" in node.keys() else -1
            cxy[1], cxy[2], cxy[3], cxy[4] = node["bounds"]
            cxy[5] = node["number"]
            temp.append(cxy)
            if "children" not in node.keys():
                continue
            for child in node["children"]:
                relation[node["number"]][child["number"]] = 1
                queue.append(child)
        levels.append(temp)
    return levels, relation


def matrix_to_list(relation):
    """把邻接矩阵转换成邻接表的形式"""
    graph = {}
    for i in range(relation.shape[0]):
        for j in range(i + 1, relation.shape[0]):
            if relation[i][j] == 1:
                if i not in graph.keys():
                    graph[i] = {}
                    graph[i]['children'] = set()
                    graph[i]['parent'] = 0 if i != 0 else None
                if j not in graph.keys():
                    graph[j] = {}
                    graph[j]['children'] = set()
                    graph[j]['parent'] = 0
                graph[i]['children'].add(j)
                graph[j]['parent'] = i
    return graph


def flat_levels(levels):
    """把层级结构的布局展平成层级增强算法需要的格式"""
    d_tot = len(levels) - 1
    layout = {}
    for i, level in enumerate(levels):
        for box in level:
            box.append(i)
            layout[box[5]] = box
    return layout, d_tot


def build_level_with_annotation(layout):
    """我们希望能把布局表示成多层次的结构，同时需要获取元素与元素之间的父子关系。元素与元素之间的关系用邻接矩阵记录，层级用数组记录,这种记录方式虽然有较高的花费，但是好处是取用的时候非常方便"""

    boxs_nums = count_nodes(layout, 0)
    levels, relation = level_order(layout, boxs_nums)
    return levels, relation


def layout_hierarchy_extraction(layout):
    """
    输入有：树（最好是一个双向字典），bbox，深度
    输出是：增强层级信息后的树T
    算法主要思路：
    首先是用原来的树进行初始化
    反序遍历每个层：
        获取这个层的所有节点
        遍历两种对齐方式（这里要注意的是这个对齐方式的先后是不确定的，事实上可能是两个都试一遍按照论文提到的标准取一个较好的先进行。
            创建图，包含这一层的所有节点
            二重遍历，遍历所有节点对，检查出所有符合条件的节点对，连一条边
            获取所有连通分量
            遍历连通分量，对每个连通分量，创建新节点p，类型取决于对齐的方式
            p的深度为j
            所有在连通分量里面的节点深度+1

    """

    levels, relation = build_level_with_annotation(layout)
    ## 初始化树结构
    boxs, d_tot = flat_levels(levels)
    edges = matrix_to_list(relation)
    vis.vis_tree(edges)

    def judge_edge(node1, node2, align="VCA"):

        box1 = boxs[node1]
        box2 = boxs[node2]
        par = (edges[node1]["parent"] == edges[node2]["parent"])
        spa_tsh = 50
        ali_tsh = 10
        x1, y1, x2, y2 = box1[1:5]
        u1, v1, u2, v2 = box2[1:5]
        dis = abs(min(y2, v2) - max(y1, v1)) if align == "VCA" else abs(min(x2, u2) - max(x1, u1))
        spa = dis <= spa_tsh
        ali = False
        if align == "VCA":
            xc = (x1 + x2) / 2
            uc = (u1 + u2) / 2
            ali = abs(xc - uc) <= ali_tsh
        elif align == "HCA":
            yc = (y1 + y2) / 2
            vc = (v1 + v2) / 2
            ali = abs(yc - vc) <= ali_tsh
        return par and ali and spa

    def get_connected_components(graph):
        """求图的所有连通分量"""
        result = []
        visited = set()
        for v in graph.keys():
            if v not in visited:
                component = []
                queue = collections.deque()
                queue.append(v)
                visited.add(v)
                while queue:
                    u = queue.popleft()
                    component.append(u)
                    for w in graph[u]:
                        if w not in visited:
                            queue.append(w)
                            visited.add(w)
                result.append(component)
        return result

    def build(nodes, align="VCA"):
        ## 初始化图
        graph = {}
        ## 按照规则建图
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                v1 = nodes[i]
                v2 = nodes[j]
                if not judge_edge(v1, v2, align):
                    continue
                if v1 not in graph.keys():
                    graph[v1] = []
                if v2 not in graph.keys():
                    graph[v2] = []
                graph[v1].append(v2)
                graph[v2].append(v1)
        ## 求出连通分量
        components = get_connected_components(graph)
        ## 计算损失
        if len(components) == 0:
            return components, float("inf")
        cost = 0
        for component in components:
            for bbox in component:
                cost += area(boxs[bbox][1:5])
        cost /= len(components)
        return components, cost

    def get_level_boxs(d):
        v_j = []
        for v, box in boxs.items():
            if box[6] == d:
                v_j.append(v)
        return v_j

    def add_level(node, num=1):
        boxs[node][6] += num
        if len(edges[node]["children"]) == 0:
            return
        for child in edges[node]["children"]:
            add_level(child, num)

    def insert_node(component, depth, align="VCA"):
        """创建新节点，插入新节点,修改旧节点，这是最重要的函数"""
        ## 定义新节点
        new_box = [0] * 7
        new_box[0] = len(LABEL_NAMES) if align == "VCA" else len(LABEL_NAMES) + 1
        bboxs = np.array([boxs[v] for v in component])
        border = 10
        new_box[1] = min(bboxs[:, 1]) - border
        new_box[2] = min(bboxs[:, 2]) - border
        new_box[3] = max(bboxs[:, 3]) + border
        new_box[4] = max(bboxs[:, 4]) + border
        new_v = max(boxs.keys()) + 1
        new_box[5] = new_v
        new_box[6] = depth
        ## 插入新节点
        boxs[new_box[5]] = new_box
        ## 修改旧节点和旧节点的子节点的层数+1
        for v in component:
            add_level(v, 1)
        ## 修改树状图的边关系
        edges[new_v] = {"parent": 0, "children": set()}
        par = edges[component[0]]["parent"]
        edges[new_v]["parent"] = par
        edges[par]["children"].add(new_v)
        for v in component:
            edges[v]["parent"] = new_v
            edges[new_v]["children"].add(v)
            edges[par]["children"].remove(v)

    ## 反序遍历每个层
    for d_j in range(d_tot, -1, -1):
        ## 获取该层的所有节点
        v_j = get_level_boxs(d_j)
        ## 两种对齐方式存在顺序先后问题，通过比较他们的cost来决定先后
        if len(v_j) < 2:
            continue
        components1, cost1 = build(v_j, align="VCA")
        components2, cost2 = build(v_j, align="HCA")
        order = 1 if cost1 <= cost2 else 0
        if max(len(components1), len(components2)) == 0:
            continue
        components = components1 if order else components2
        for component in components:
            insert_node(component, d_j, align="VCA" if order else "HCA")

        ## 插入之后重新获取该层的所有节点，进行另外一种对齐方式
        v_j = get_level_boxs(d_j)
        if len(v_j) < 2:
            continue
        components, _ = build(v_j, align="HCA" if order else "VCA")
        if len(components) == 0:
            continue
        for component in components:
            insert_node(component, d_j, align="HCA" if order else "VCA")
    print(boxs)
    print(edges)

    return [boxs[v] for v in boxs.keys()], edges


def judge_relation(box1, box2):
    """用于判断两个box是否会存在父子关系,这里注意存在方向性，如果1->2则返回1,否则返回0
        判断原则主要依赖overlap和coverage关系，大致原则如下：
        1. 首先考虑coverage关系，如果一个box完全cover了另外一个box，则认为存在父子关系
        2. 考虑重叠关系，重叠不一定会存在父子关系，因为同层级的box也可能出现overlap，可以规定一个设定一个重叠面积与box面积之比的阈值，
        大于某个阈值则可以认为存在父子关系，面积大的作为父，面积小的作为子，如果面积接近，则根据之前统计的label之间的父子关系频率来规定指向

        因为coverage本质上是一个特殊的overlap，所以只需要计算overlap就行了

    """
    global level_graph
    coverage_threshold = 0.95
    overlap_threshold = 0.9
    size_threshold = 0

    overlap = get_intersection_area(box1[:5], box2[:5])
    area1 = area(box1[1:5])
    area2 = area(box2[1:5])
    if area1 == 0 or area2 == 0:
        return 0
    overlap1 = overlap / area1
    overlap2 = overlap / area2
    if overlap_threshold > max(overlap1, overlap2):
        return 0
    if size_threshold >= abs(overlap2 - overlap1):
        c1 = box1[0]
        c2 = box2[0]
        if level_graph[c1][c2] == 0 and level_graph[c2][c1] == 0 and coverage_threshold > max(overlap1, overlap2):
            return 0
        return 1 if level_graph[c1][c2] > level_graph[c2][c1] else 0
    return 1 if overlap1 < overlap2 else 0


def build_level_without_annotation(layout):
    """这里我们希望能够在无标注的情况下，也能够通过几何关系大致获得层级关系，这里的输入是cltrb格式的layout"""
    boxs_nums = len(layout)

    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            box1 = layout[i]
            box2 = layout[j]


def append_child(element, elements):
    if 'children' in element.keys():
        for child in element['children']:
            cxy = [0] * 6
            cxy[0] = LABEL_TO_ID_[child["componentLabel"]]
            cxy[1], cxy[2], cxy[3], cxy[4] = child["bounds"]
            cxy[5] = child["number"]
            elements.append(cxy)
            elements = append_child(child, elements)
    return elements


def get_intersection_area(bb0, bb1):
    """Computes the intersection area between two elements."""
    _, x_0, y_0, x_1, y_1 = bb0
    _, u_0, v_0, u_1, v_1 = bb1

    intersection_x_0 = max(x_0, u_0)
    intersection_y_0 = max(y_0, v_0)
    intersection_x_1 = min(x_1, u_1)
    intersection_y_1 = min(y_1, v_1)
    intersection_area = area(
        [intersection_x_0, intersection_y_0, intersection_x_1, intersection_y_1])
    return intersection_area


def area(bounding_box):
    """Computes the area of a bounding box."""

    x_0, y_0, x_1, y_1 = bounding_box

    return max(0., x_1 - x_0) * max(0., y_1 - y_0)


# def find_overlap(layout):
#     global overlap_graph, label_graph
#     layout = append_child(layout, [])
#     graph = np.zeros([25, 25])
#     label = np.zeros([25, 25])
#     for i in range(len(layout)):
#         for j in range(i + 1, len(layout)):
#             box1 = layout[i]
#             box2 = layout[j]
#             c1 = box1[0]
#             c2 = box2[0]
#             label[c1][c2] += 1
#             label[c2][c1] += 1
#             intersection_area = get_intersection_area(box1, box2)
#             if intersection_area > 0.:
#                 graph[c1][c2] += 1
#                 graph[c2][c1] += 1
#     overlap_graph += (graph > 0)
#     label_graph += (label > 0)


if __name__ == '__main__':
    path = "../rico_dataset_v0.1_semantic_annotations/semantic_annotations/14.json"
    layout = load(path)
    b, e = layout_hierarchy_extraction(layout)
    vis.render_level(b)
    vis.vis_tree(e)
    # t, c = test_judgement(layout)
    # pre = precision_score(t, c, average=None)
    # recall = recall_score(t, c, average=None)
    #
    # print(pre)
    # print(recall)
    # path = "../rico_dataset_v0.1_semantic_annotations/semantic_annotations/"
    # acc_score = []
    # pre_score = []
    # re_score = []
    # for json_path in tqdm(sorted(Path(path).glob('*.json'))):
    #     layout = load(json_path)
    #     # build_level_with_annotation(layout)
    #     t, c = test_judgement(layout)
    #     if t.shape[0] == 0:
    #         acc_score.append(1)
    #         pre_score.append(1)
    #         re_score.append(1)
    #         continue
    #     acc_score.append(accuracy_score(t, c))
    #     pre = precision_score(t, c, zero_division=1, average=None)
    #     re = recall_score(t, c, zero_division=1, average=None)
    #     if len(pre) == 1:
    #         pre_score.append(1)
    #         re_score.append(1)
    #         continue
    #     pre_score.append(pre[1])
    #     re_score.append(re[1])
    # print(np.mean(acc_score), np.mean(pre_score), np.mean(re_score))
