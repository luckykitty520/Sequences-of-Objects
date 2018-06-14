import os
import csv
from PIL import Image


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
    input image path
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
    write title label
    format: (class, name, width, height)
"""
# get results
results = []

for a_class in image_path:

    for image in image_path[a_class]:

        img = Image.open(image)
        img_name = image.split('/')[-1]

        temp = []
        temp.append(a_class)
        temp.append(img_name)
        temp.append(img.size[0])
        temp.append(img.size[1])

        results.append(temp)


# write
path = './data/title_label.csv'
with open(path, 'w', newline='') as out:
    writer = csv.writer(out)

    for item in results:
        writer.writerow(item)








