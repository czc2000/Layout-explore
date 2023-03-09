## 探究一下不同种类之间的父子关系和重叠关系
import json
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

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
level_graph = np.zeros([25, 25])
overlap_graph = np.zeros([25, 25])
label_graph = np.zeros([25, 25])
layout_num = 0
find_layout = [[""] * 25 for i in range(25)]


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
    child_c = LABEL_TO_ID_[part["componentLabel"]]
    parent = part["parent"]
    while parent is not None and "componentLabel" in parent.keys():
        parent_c = LABEL_TO_ID_[parent["componentLabel"]]
        graph[parent_c][child_c] += 1
        # graph[child_c][parent_c] += 1
        parent = parent["parent"]


def level_statistics():
    global level_graph
    path = "../rico_dataset_v0.1_semantic_annotations/semantic_annotations/"
    for json_path in tqdm(sorted(Path(path).glob('*.json'))):
        layout = load(json_path)
        find_parent(layout, parent=None)
        single_graph = np.zeros([25, 25])
        preorder(layout, single_graph)
        level_graph = level_graph + (single_graph > 0)
        # for i in range(25):
        #     for j in range(25):
        #         if single_graph[i][j] > 0:
        #             find_layout[i][j] = json_path.name


def append_child(element, elements):
    if 'children' in element.keys():
        for child in element['children']:
            cxy = [0] * 5
            cxy[0] = LABEL_TO_ID_[child["componentLabel"]]
            cxy[1], cxy[2], cxy[3], cxy[4] = child["bounds"]
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


def find_overlap(layout):
    global overlap_graph, label_graph
    layout = append_child(layout, [])
    graph = np.zeros([25, 25])
    label = np.zeros([25, 25])
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            box1 = layout[i]
            box2 = layout[j]
            c1 = box1[0]
            c2 = box2[0]
            label[c1][c2] += 1
            label[c2][c1] += 1
            intersection_area = get_intersection_area(box1, box2)
            if intersection_area > 0.:
                graph[c1][c2] += 1
                graph[c2][c1] += 1
    overlap_graph += (graph > 0)
    label_graph += (label > 0)


def overlap_statistics():
    path = "../rico_dataset_v0.1_semantic_annotations/semantic_annotations/"
    for json_path in tqdm(sorted(Path(path).glob('*.json'))):
        layout = load(json_path)
        find_overlap(layout)


if __name__ == '__main__':
    overlap_statistics()
    level_statistics()
    for i in range(25):
        for j in range(25):
            # level_graph[i][j] /= max(label_graph[i][j], 1)
            ## 修改一个归一化的方法，计算重叠的时候有父子关系的频率，如果没有重叠关系直接记为0
            level_graph[i][j] = level_graph[i][j] / overlap_graph[i][j] if overlap_graph[i][j] > 0 else 0
    for i in range(25):
        for j in range(25):
            overlap_graph[i][j] /= max(label_graph[i][j], 1)
    np.save("overlap_graph.npy", overlap_graph)
    np.save("level_graph_one.npy", level_graph)
