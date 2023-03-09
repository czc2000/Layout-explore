import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
import seaborn as sns
import json
from tqdm import tqdm
from treelib import Node, Tree
import copy

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


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in palette]
    return rgb_triples


def vis_tree(dict_):
    dict_ = copy.deepcopy(dict_)
    added = set()
    tree = Tree()
    while dict_:
        for key, value in dict_.items():
            if value['parent'] in added:
                tree.create_node(key, key, parent=value['parent'])
                added.add(key)
                dict_.pop(key)
                break
            elif value['parent'] is None:
                tree.create_node(key, key)
                added.add(key)
                dict_.pop(key)
                break

    tree.show()


def render(layout):
    colors = gen_colors(len(LABEL_NAMES))
    width = 1440
    height = 2560
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')
    # layout = layout.reshape(-1)
    # layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)

    # box[:, [0, 2]] = box[:, [0, 2]] / (size - 1) * (width - 1)
    # box[:, [1, 3]] = box[:, [1, 3]] / size * height
    # box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

    for i in range(len(layout)):
        x1, y1, x2, y2 = layout[i][1], layout[i][2], layout[i][3], layout[i][4]
        bbox = [x1, y1, x2, y2]
        cat = layout[i][0]
        if 0 <= cat < len(colors):
            col = colors[cat]
            class_name = ID_TO_LABEL[cat]
            font = ImageFont.truetype('arial.ttf', size=35)
            text_w, text_h = draw.textsize(class_name, font)
            text_bbox = [x1, y1]
            draw.rectangle(bbox,
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)
            draw.text(text_bbox, class_name, fill=tuple(col) + (200,), font=font)
    # Add border around image
    img = ImageOps.expand(img, border=2)
    return img


def render_level(layout):
    """注意这里需要把dict先转成list再送进来"""
    layout = sorted(layout, key=lambda x: x[6])
    colors = gen_colors(len(LABEL_NAMES))
    width = 1440
    height = 2560
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    for i in range(len(layout)):
        x1, y1, x2, y2 = layout[i][1], layout[i][2], layout[i][3], layout[i][4]
        bbox = [x1, y1, x2, y2]
        cat = layout[i][0]
        num = layout[i][5]
        font = ImageFont.truetype('arial.ttf', size=20)
        if 0 <= cat < len(colors):
            col = colors[cat]
            class_name = ID_TO_LABEL[cat]
            text_bbox = [x1, y1]
            draw.rectangle(bbox,
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)
            draw.text(text_bbox, class_name + str(num), fill=tuple(col) + (200,), font=font)
        elif cat == 25 or cat == 26:
            col = "red" if cat == 25 else "blue"
            # class_name = "VCA" if cat == 25 else "HCA"
            text_bbox = [x2, y2]
            draw.rectangle(bbox,
                           outline=col,
                           width=5)
            draw.text(text_bbox, str(num), fill=col, font=font)

            # Add border around image
    img = ImageOps.expand(img, border=2)
    img.show()
    return img


if __name__ == '__main__':
    pass
