# 图像聚类任务

本项目实现了对图像数据集的无监督聚类任务。

## 任务要求

1. **问题的形式化描述** (5%)
2. **如何处理图像特征** (5%)
3. **选择合适的聚类算法** (10%)
4. **评估聚类效果** (5%)

## 数据集

- 数据位置: `dataset/`
- 类别数: 6
- 每类样本数: 100
- 总样本数: 600
- 类别: cable, tile, bottle, pill, leather, transistor

## 环境要求

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python clustering_task.py
```

## 实现方案

### 1. 问题的形式化描述

将图像数据集划分为k个不相交的簇，使得同一簇内的图像相似度最大化，不同簇之间的图像相异度最大化。

### 2. 图像特征处理

使用预训练的ResNet50模型提取图像特征:
- 将图像调整为224×224
- 使用ImageNet预训练权重提取深度特征
- 特征维度: 2048维
- 使用PCA降维到128维以提高聚类效率

### 3. 聚类算法

实现了两种聚类算法进行对比:
- **K-Means**: 基于距离的划分聚类，适合球形簇
- **层次聚类 (Agglomerative Clustering)**: 基于Ward链接的层次聚类

### 4. 评估指标

使用多种评估指标评估聚类效果:

**内部评估指标:**
- Silhouette Score (轮廓系数)

**外部评估指标:**
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneity (同质性)
- Completeness (完整性)
- V-measure

## 输出结果

运行脚本后，会生成 `clustering_results.json` 文件，包含:
- 每个算法的聚类标签
- 各种评估指标的值
- 标签映射关系

## 文件说明

- `clustering_task.py`: 主程序文件
- `cluster_labels.json`: 真实标签文件(用于评估)
- `dataset/`: 图像数据集目录
- `requirements.txt`: 依赖包列表

