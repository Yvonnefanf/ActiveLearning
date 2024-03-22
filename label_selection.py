import torch
import json
import numpy as np

file_path = '/home/yiming/cophi/training_dynamic/image/AL_dataset/Training_data/training_dataset_label.pth'

# 使用torch.load()加载.pth文件
data = torch.load(file_path)

# 从CUDA设备中将标签数据转移到CPU上
labels = data.cpu().numpy()


# 需要每个类别包含的点数
points_per_class = 500

# 初始化类别计数器和结果列表
class_counts = np.zeros(10, dtype=int)
selected_index = []

# 遍历每个索引和标签
for index, label in enumerate(labels):
    label = int(label) 
    # 检查当前类别的计数是否已满
    if class_counts[label] < points_per_class:
        # 添加当前索引到结果列表
        selected_index.append(index)
        
        # 更新当前类别的计数
        class_counts[label] += 1
        
    # 检查所有类别的计数是否已满
    if np.all(class_counts >= points_per_class):
        break

# 存储选定的索引到index.json文件
output_file = '/home/yiming/cophi/training_dynamic/image/AL_dataset/Model/index.json'
with open(output_file, 'w') as file:
    json.dump(selected_index, file)

# 获取类别0的索引
class_0_indexes = np.where(labels == 0)[0]

# 从类别0的索引中随机选择500个不重叠的索引
selected_class_0_indexes = np.setdiff1d(class_0_indexes, selected_index, assume_unique=True)[:1000]

# 添加额外选择的索引到结果列表中
selected_index.extend(selected_class_0_indexes.tolist())

# 存储额外选择的索引到new_index.json文件
output_file_extra = '/home/yiming/cophi/training_dynamic/image/AL_dataset/Model/new_index.json'
with open(output_file_extra, 'w') as file:
    json.dump(selected_index, file)
