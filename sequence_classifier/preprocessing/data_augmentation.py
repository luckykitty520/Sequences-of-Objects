"""
    flip horizontally
"""


from PIL import Image
import os
import numpy as np


def get_folder_path(path):

    folder_path = []

    for folder in os.listdir(path):
        folder_path.append(os.path.join(path, folder))

    return folder_path


def get_image_path(folder_path):

    image_path = dict()

    for folder in folder_path:

        key = folder.split('/')[-1]
        image_path[key] = []

        for image in os.listdir(folder):
            image_path[key].append(os.path.join(folder, image))

    return image_path


"""
    get image path
"""
# get folders
path = './data/paint_276'
folder_path = get_folder_path(path)
print(folder_path)
print(len(folder_path))
print('\n\n')

# get images
# dictionary,   key = class,    value = list of image_path
image_path = get_image_path(folder_path)

total_count = 0
for a_class in image_path:
    print(a_class)
    print(len(image_path[a_class]))
    print(image_path[a_class])
    print('\n')

    total_count += len(image_path[a_class])

print('total count:', total_count)
print('\n\n')


"""
    flip horizontally
"""
results_path = './data/paint_augmentation'

for a_class in image_path:

    # make dir
    folder_path = os.path.join(results_path, a_class)
    if not (os.path.exists(folder_path)):
        os.mkdir(folder_path)

    # flip horizontally & save
    for image in image_path[a_class]:

        img = Image.open(image)
        img_name = image.split('/')[-1]
        save_path = os.path.join(folder_path, img_name)
        img.save(save_path, quality='keep')

        # flip horizontally
        flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_name = img_name[:-4] + "_f.jpg"
        save_path = os.path.join(folder_path, flip_name)
        flip.save(save_path)


















