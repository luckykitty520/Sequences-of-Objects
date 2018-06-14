import os
import os.path as osp
from xml.etree import ElementTree

path = "/home/hadoop/datasets/Painting/Annotations"
files = os.listdir(path)
files = list(map(lambda item: osp.join(path, item), files))
print(files)
print(len(files))

labels = {}
for file in files:
    root = ElementTree.parse(file).getroot()
    objects = [item for item in root.getchildren() if item.tag == "object"]
    for obj in objects:
        label = obj.getchildren()[0].text
        if label not in labels.keys():
            labels[label] = 1
        else:
            labels[label] = labels[label] + 1

print(labels)
