"""
任务1: 聚类任务
使用合适的聚类算法,对所给的图像数据集进行聚类。

数据集信息:
- 数据位于dataset/内,共6个类别,每类图像100张
- 类别: cable, tile, bottle, pill, leather, transistor
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1.0 问题的形式化描述
# ============================================================================
"""
问题形式化描述:

给定一个图像数据集 D = {I₁, I₂, ..., Iₙ}, 其中 n = 600,
每个图像 Iᵢ ∈ R^(H×W×C) 表示一个高为H、宽为W、通道数为C的图像。

目标: 将数据集D划分为k个不相交的簇 {C₁, C₂, ..., Cₖ}, 使得:
1. ∪ᵢ₌₁ᵏ Cᵢ = D (所有图像都被分配到某个簇)
2. Cᵢ ∩ Cⱼ = ∅ (簇之间不相交)
3. 同一簇内的图像应该具有相似的特征(相似度最大化)
4. 不同簇之间的图像应该具有不同的特征(相异度最大化)

对于无监督聚类任务，我们需要:
- 将原始图像空间映射到特征空间: f: R^(H×W×C) → R^d
- 在特征空间中执行聚类算法: g: R^d → {1, 2, ..., k}
- 评估聚类质量: E: {C₁, ..., Cₖ} → R (评估指标)

在本任务中，k = 6 (已知类别数)
"""


# ============================================================================
# 1.1 如何处理图像特征
# ============================================================================
class ImageFeatureExtractor:
    """
    图像特征提取器
    
    使用预训练的深度卷积神经网络(ResNet)提取图像特征。
    ResNet在ImageNet上预训练，能够提取通用的视觉特征表示。
    """
    
    def __init__(self, model_name='resnet50', device=None):
        """
        初始化特征提取器
        
        Args:
            model_name: 使用的模型名称 ('resnet18', 'resnet50', 'resnet101')
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # 加载预训练的ResNet模型
        # 兼容新旧版本的torchvision API
        try:
            # 新版本使用weights参数
            if model_name == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.feature_dim = 512
            elif model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.feature_dim = 2048
            else:
                model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
                self.feature_dim = 2048
        except AttributeError:
            # 旧版本使用pretrained参数
            if model_name == 'resnet18':
                model = models.resnet18(pretrained=True)
                self.feature_dim = 512
            elif model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
                self.feature_dim = 2048
            else:
                model = models.resnet101(pretrained=True)
                self.feature_dim = 2048
        
        # 移除最后的全连接层，只保留特征提取部分
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # 图像预处理: 将图像调整为224x224，归一化到[0,1]，然后标准化
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_paths, batch_size=32):
        """
        批量提取图像特征
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
        
        Returns:
            features: numpy数组，形状为 (n_samples, feature_dim)
        """
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.transform(img)
                        batch_images.append(img_tensor)
                    except Exception as e:
                        print(f"处理图像 {img_path} 时出错: {e}")
                        # 如果图像损坏，使用零张量
                        batch_images.append(torch.zeros(3, 224, 224))
                
                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_features = self.model(batch_tensor)
                # 全局平均池化并展平
                batch_features = batch_features.squeeze().cpu().numpy()
                
                # 处理单个样本的情况
                if len(batch_images) == 1:
                    batch_features = batch_features.reshape(1, -1)
                
                features.append(batch_features)
        
        features = np.vstack(features)
        return features


# ============================================================================
# 1.2 选择合适的聚类算法
# ============================================================================
class ClusteringPipeline:
    """
    聚类 pipeline
    
    支持多种聚类算法:
    1. K-Means: 基于距离的划分聚类，适合球形簇，需要指定簇数
    2. DBSCAN: 基于密度的聚类，能发现任意形状的簇，不需要指定簇数
    3. Agglomerative Clustering: 层次聚类，可以构建簇的层次结构
    """
    
    def __init__(self, n_clusters=6):
        """
        初始化聚类pipeline
        
        Args:
            n_clusters: 簇的数量 (已知为6个类别)
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = None
    
    def preprocess_features(self, features, use_pca=True, n_components=128):
        """
        特征预处理
        
        Args:
            features: 原始特征
            use_pca: 是否使用PCA降维
            n_components: PCA降维后的维度
        
        Returns:
            处理后的特征
        """
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        if use_pca and features_scaled.shape[1] > n_components:
            # 使用PCA降维以加速聚类并去除冗余信息
            self.pca = PCA(n_components=n_components, random_state=42)
            features_scaled = self.pca.fit_transform(features_scaled)
            print(f"PCA降维: {features.shape[1]} -> {n_components} 维")
        
        return features_scaled
    
    def kmeans_clustering(self, features, random_state=42):
        """
        K-Means聚类
        
        优点:
        - 计算效率高
        - 适合球形簇
        - 结果易于解释
        
        缺点:
        - 需要预先指定簇数
        - 对初始值敏感
        - 假设簇是球形的
        """
        print("\n使用 K-Means 算法进行聚类...")
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        labels = kmeans.fit_predict(features)
        return labels, kmeans
    
    def dbscan_clustering(self, features, eps=0.5, min_samples=5):
        """
        DBSCAN聚类
        
        优点:
        - 能发现任意形状的簇
        - 能识别噪声点
        - 不需要预先指定簇数
        
        缺点:
        - 对参数敏感(eps, min_samples)
        - 对高维数据效果不好
        """
        print("\n使用 DBSCAN 算法进行聚类...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        return labels, dbscan
    
    def hierarchical_clustering(self, features, linkage='ward'):
        """
        层次聚类
        
        优点:
        - 可以构建簇的层次结构
        - 结果可视化直观
        - 不需要预先指定簇数(但我们可以截取到k个簇)
        
        缺点:
        - 计算复杂度高 O(n² log n)
        - 对噪声敏感
        """
        print("\n使用层次聚类算法进行聚类...")
        hierarchical = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=linkage
        )
        labels = hierarchical.fit_predict(features)
        return labels, hierarchical


# ============================================================================
# 1.3 评估聚类效果
# ============================================================================
class ClusteringEvaluator:
    """
    聚类效果评估器
    
    使用多种评估指标来评估聚类质量:
    
    1. 内部评估指标(不需要真实标签):
       - Silhouette Score: 轮廓系数，衡量簇内紧密度和簇间分离度
    
    2. 外部评估指标(需要真实标签):
       - Adjusted Rand Index (ARI): 调整兰德指数
       - Normalized Mutual Information (NMI): 标准化互信息
       - Homogeneity: 同质性
       - Completeness: 完整性
       - V-measure: V度量(同质性和完整性的调和平均)
    """
    
    def __init__(self, true_labels):
        """
        初始化评估器
        
        Args:
            true_labels: 真实标签列表
        """
        self.true_labels = np.array(true_labels)
    
    def evaluate(self, predicted_labels, features=None):
        """
        评估聚类结果
        
        Args:
            predicted_labels: 预测的聚类标签
            features: 特征向量(用于计算轮廓系数)
        
        Returns:
            dict: 包含各种评估指标的字典
        """
        predicted_labels = np.array(predicted_labels)
        results = {}
        
        # 内部评估指标
        if features is not None:
            # 如果有噪声点(标签为-1)，需要过滤
            mask = predicted_labels != -1
            if mask.sum() > 1:
                try:
                    silhouette = silhouette_score(features[mask], predicted_labels[mask])
                    results['Silhouette Score'] = silhouette
                except:
                    results['Silhouette Score'] = None
        
        # 外部评估指标
        # 过滤噪声点
        mask = predicted_labels != -1
        if mask.sum() > 0:
            pred_filtered = predicted_labels[mask]
            true_filtered = self.true_labels[mask]
            
            ari = adjusted_rand_score(true_filtered, pred_filtered)
            nmi = normalized_mutual_info_score(true_filtered, pred_filtered)
            homogeneity = homogeneity_score(true_filtered, pred_filtered)
            completeness = completeness_score(true_filtered, pred_filtered)
            v_measure = v_measure_score(true_filtered, pred_filtered)
            
            results['Adjusted Rand Index (ARI)'] = ari
            results['Normalized Mutual Information (NMI)'] = nmi
            results['Homogeneity'] = homogeneity
            results['Completeness'] = completeness
            results['V-measure'] = v_measure
            
            # 计算簇的数量
            n_clusters_pred = len(np.unique(pred_filtered))
            results['Number of Clusters (predicted)'] = n_clusters_pred
        
        return results
    
    def print_results(self, results, algorithm_name):
        """
        打印评估结果
        """
        print(f"\n{'='*60}")
        print(f"{algorithm_name} 聚类评估结果:")
        print(f"{'='*60}")
        for key, value in results.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"{key:35s}: {value:.4f}")
                else:
                    print(f"{key:35s}: {value}")
        print(f"{'='*60}\n")


def load_dataset(data_dir, label_file):
    """
    加载数据集
    
    Args:
        data_dir: 数据集目录
        label_file: 标签文件路径
    
    Returns:
        image_paths: 图像路径列表
        labels: 标签列表
        label_to_id: 标签到ID的映射
    """
    # 加载标签
    with open(label_file, 'r') as f:
        label_dict = json.load(f)
    
    # 获取所有图像路径和标签
    image_paths = []
    labels = []
    
    for filename, label in sorted(label_dict.items()):
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(label)
    
    # 创建标签到ID的映射
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    label_ids = [label_to_id[label] for label in labels]
    
    print(f"数据集信息:")
    print(f"  总图像数: {len(image_paths)}")
    print(f"  类别数: {len(unique_labels)}")
    print(f"  类别: {unique_labels}")
    print(f"  每类样本数: {[labels.count(l) for l in unique_labels]}")
    
    return image_paths, label_ids, label_to_id


def main():
    """
    主函数: 执行完整的聚类流程
    """
    # 配置
    DATA_DIR = 'dataset'
    LABEL_FILE = 'cluster_labels.json'
    N_CLUSTERS = 6
    
    print("="*60)
    print("图像聚类任务")
    print("="*60)
    
    # 1. 加载数据集
    print("\n[步骤1] 加载数据集...")
    image_paths, true_labels, label_to_id = load_dataset(DATA_DIR, LABEL_FILE)
    
    # 2. 提取图像特征
    print("\n[步骤2] 提取图像特征...")
    feature_extractor = ImageFeatureExtractor(model_name='resnet50')
    features = feature_extractor.extract_features(image_paths)
    print(f"特征维度: {features.shape}")
    
    # 3. 特征预处理
    print("\n[步骤3] 特征预处理...")
    clustering_pipeline = ClusteringPipeline(n_clusters=N_CLUSTERS)
    features_processed = clustering_pipeline.preprocess_features(
        features, 
        use_pca=True, 
        n_components=128
    )
    
    # 4. 执行聚类
    print("\n[步骤4] 执行聚类算法...")
    
    # 4.1 K-Means聚类
    labels_kmeans, kmeans_model = clustering_pipeline.kmeans_clustering(
        features_processed
    )
    
    # 4.2 DBSCAN聚类 (可选，用于对比)
    # labels_dbscan, dbscan_model = clustering_pipeline.dbscan_clustering(
    #     features_processed, eps=0.5, min_samples=5
    # )
    
    # 4.3 层次聚类
    labels_hierarchical, hierarchical_model = clustering_pipeline.hierarchical_clustering(
        features_processed, linkage='ward'
    )
    
    # 5. 评估聚类效果
    print("\n[步骤5] 评估聚类效果...")
    evaluator = ClusteringEvaluator(true_labels)
    
    # 评估K-Means
    results_kmeans = evaluator.evaluate(labels_kmeans, features_processed)
    evaluator.print_results(results_kmeans, "K-Means")
    
    # 评估层次聚类
    results_hierarchical = evaluator.evaluate(labels_hierarchical, features_processed)
    evaluator.print_results(results_hierarchical, "层次聚类")
    
    # 6. 保存结果
    print("\n[步骤6] 保存结果...")
    results = {
        'kmeans': {
            'labels': labels_kmeans.tolist(),
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in results_kmeans.items()}
        },
        'hierarchical': {
            'labels': labels_hierarchical.tolist(),
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in results_hierarchical.items()}
        },
        'label_mapping': label_to_id
    }
    
    with open('clustering_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("结果已保存到 clustering_results.json")
    
    # 7. 选择最佳算法
    print("\n[步骤7] 算法对比总结:")
    print(f"K-Means - ARI: {results_kmeans.get('Adjusted Rand Index (ARI)', 0):.4f}, "
          f"NMI: {results_kmeans.get('Normalized Mutual Information (NMI)', 0):.4f}")
    print(f"层次聚类 - ARI: {results_hierarchical.get('Adjusted Rand Index (ARI)', 0):.4f}, "
          f"NMI: {results_hierarchical.get('Normalized Mutual Information (NMI)', 0):.4f}")
    
    # 选择ARI最高的算法作为最佳结果
    ari_kmeans = results_kmeans.get('Adjusted Rand Index (ARI)', 0)
    ari_hierarchical = results_hierarchical.get('Adjusted Rand Index (ARI)', 0)
    
    if ari_kmeans >= ari_hierarchical:
        best_labels = labels_kmeans
        best_algorithm = "K-Means"
    else:
        best_labels = labels_hierarchical
        best_algorithm = "层次聚类"
    
    print(f"\n最佳算法: {best_algorithm}")
    
    print("\n" + "="*60)
    print("聚类任务完成!")
    print("="*60)


if __name__ == '__main__':
    main()

