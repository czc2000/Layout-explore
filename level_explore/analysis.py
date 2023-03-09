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
# overlap_graph = np.load("overlap_graph.npy")
# overlap_weight = (1 - overlap_graph) ** 2
# df = pd.DataFrame(overlap_graph, index=LABEL_NAMES, columns=LABEL_NAMES)
df2 = pd.DataFrame(level_graph, index=LABEL_NAMES, columns=LABEL_NAMES)
# plt.figure(figsize=(14, 14))
# sns.heatmap(df, annot=True, annot_kws={'size': 8, 'weight': 'bold', 'color': 'w'}, fmt='.2f')
# plt.savefig("./overlap.png")
# plt.show()
plt.figure(figsize=(14, 14))
sns.heatmap(df2, annot=True, annot_kws={'size': 8, 'weight': 'bold', 'color': 'w'}, fmt='.2f')
plt.savefig("./level_one.png")
# plt.show()
# print(level_graph > 0.05)
# np.save("level.npy", level_graph > 0.05)
