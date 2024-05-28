import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

"""
# 获取 photos 文件夹下除 .txt 文件以外所有文件夹名（即3种分类的类名）
file_path = 'data'
flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

# 创建 训练集train 文件夹，并由3种类名在其目录下创建3个子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('flower_data/train/' + cla)

# 创建 验证集val 文件夹，并由3种类名在其目录下创建3个子目录
mkfile('flower_data/val')
for cla in flower_class:
    mkfile('flower_data/val/' + cla)
"""

file_path='data'
class_list=['0', '1']

# 划分比例，训练集 : 验证集 = 9 : 1
split_rate = 0.5


for cla in class_list:
    cla_path = file_path + '/' + cla + '/'  # 某一类别动作的子目录
    images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
    images.sort()
    num = len(images)
    #eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        #if image in eval_index:
        if index%10<(10*split_rate):
            image_path = cla_path + image
            new_path = 'data/val_888/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径

        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train_888/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")

