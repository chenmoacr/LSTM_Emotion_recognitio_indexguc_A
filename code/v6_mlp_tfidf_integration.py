# code/v6_mlp_tfidf_integration.py # <-- 重命名文件以反映变化
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel # <-- 移除 BERT 导入
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer # <-- 引入 TF-IDF
import pandas as pd
import numpy as np
import os
import time
import jieba
import joblib
import os
# 如果使用了 jieba
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm import tqdm # 如果不预先计算所有 embedding，可能不需要 tqdm



# 定义一个全局的 tokenizer 函数
def jieba_tokenizer(text):
  """使用 jieba.lcut 进行分词"""
  return jieba.lcut(text)


# --- 参数 ---
# 数据和模型路径
data_path = os.path.join('..', 'dataset', 'demo.csv')
# --- 移除本地 BERT 路径 ---
# local_bert_path = os.path.join('..', 'bert-base-chinese')
# --------------------------------------
code_folder = '.'
# !!! 修改模型文件名以反映 TF-IDF !!!
model_save_path = os.path.join(code_folder, 'best_mlp_on_demo_data_v6_tfidf.pt')

# --- TF-IDF 相关参数 ---
# 可以调整这些参数来优化 TF-IDF 特征
tfidf_max_features = 5000 # 限制 TF-IDF 特征的维度，None 表示不限制
tfidf_ngram_range = (1, 1) # 可以尝试 (1, 2) 来包含二元语法

# MLP 模型参数
# !!! input_dim 将由 TF-IDF 决定，后面动态获取 !!!
# bert_embedding_dim = 768 # <-- 移除
mlp_hidden_dim = 512     # MLP 隐藏层维度 (保持不变或根据需要调整)
num_classes = 7          # demo.csv 数据集有 7 个类别
dropout_prob = 0.5
learning_rate = 1e-3     # 对于 TF-IDF + MLP，可以尝试稍高一点的学习率

# 训练参数
batch_size = 64          # TF-IDF 通常比 BERT 快，可以适当增大 batch_size
num_epochs = 30          # 可能需要更多或更少的轮数
use_class_weight = True # 是否使用类别权重
early_stopping_patience = 5 # 早停耐心

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# --- 移除加载 BERT 模型和 Tokenizer 的代码 ---
# print(f"加载本地 BERT 模型用于特征提取: {local_bert_path}")
# try:
#     ... (BERT 加载代码块整个删除) ...
# except Exception as e:
#     print(f"加载本地 BERT 模型或 Tokenizer 时出错: {e}")
#     exit()

# --- 加载新数据并处理标签 (这部分基本不变) ---
print(f"加载新数据: {data_path}")
try:
    # 调整路径以适应在 code 目录运行
    data_path_adjusted = data_path
    if not os.path.exists(data_path_adjusted) and not data_path.startswith('..'):
        data_path_adjusted = os.path.join('..', data_path)
    if not os.path.exists(data_path_adjusted):
        raise FileNotFoundError(f"在预期位置找不到数据文件: {data_path} 或 {os.path.join('..', data_path)}")

    df = pd.read_csv(data_path_adjusted)
    print(f"成功加载 {len(df)} 条数据")

    # 提取文本
    texts = df['posts'].fillna("").astype(str).tolist() # 确保文本是字符串且处理 NaN

    # --- 提取主要情绪标签 (保持不变) ---
    emotion_cols = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    if not all(col in df.columns for col in emotion_cols):
        raise ValueError(f"错误：CSV 文件缺少必要的情绪列: {emotion_cols}")
    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)
    derived_labels = df['dominant_emotion'].tolist()
    print("已根据最高得分提取主要情绪标签。")
    print("\n新标签分布情况:")
    print(df['dominant_emotion'].value_counts())
    # -----------------------------------------

    # 创建标签映射 (保持不变)
    unique_labels = sorted(df['dominant_emotion'].unique())
    if len(unique_labels) != num_classes:
        print(f"警告: 数据中唯一标签数量 ({len(unique_labels)}) 与预设类别数 ({num_classes}) 不符，将使用数据中的实际类别数。")
        num_classes = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    numeric_labels = [label_to_id[label] for label in derived_labels]
    print(f"标签映射: {label_to_id}, 类别数量: {num_classes}")

except FileNotFoundError:
    print(f"错误: 数据文件未找到。")
    exit()
except Exception as e:
    print(f"加载或处理新数据时出错: {e}")
    exit()


# --- 划分训练集、验证集、测试集 (仍然是文本和标签) ---
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, numeric_labels, test_size=0.20, random_state=42, stratify=numeric_labels
)
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_train_text, y_train, test_size=0.15, random_state=42, stratify=y_train # 0.8 * 0.15 = 0.12 validation
)
print(f"\n数据集划分 (文本):")
print(f"训练集大小: {len(X_train_text)}")
print(f"验证集大小: {len(X_val_text)}")
print(f"测试集大小: {len(X_test_text)}")


# --- 使用 TF-IDF 提取特征 ---print("\n使用 TF-IDF 提取特征...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=tfidf_max_features,
    ngram_range=tfidf_ngram_range,
    tokenizer=jieba_tokenizer,  # <-- 使用定义的函数名
    token_pattern=None          # <-- 明确设置 token_pattern 为 None
)

# 在训练集上拟合 TF-IDF
print("在训练集上拟合 TF-IDF...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)

# --- 保存 TF-IDF Vectorizer 和特征维度 ---
tfidf_save_path = os.path.join(code_folder, 'tfidf_vectorizer_v6.joblib')
try:
    joblib.dump(tfidf_vectorizer, tfidf_save_path)
    print(f"TF-IDF Vectorizer 已保存到: {tfidf_save_path}")
except Exception as e:
    print(f"错误：无法保存 TF-IDF Vectorizer 到 {tfidf_save_path}: {e}")

tfidf_feature_dim = X_train_tfidf.shape[1]
tfidf_dim_save_path = os.path.join(code_folder, 'tfidf_feature_dim_v6.txt')
try:
    with open(tfidf_dim_save_path, 'w') as f:
        f.write(str(tfidf_feature_dim))
    print(f"TF-IDF 特征维度 ({tfidf_feature_dim}) 已保存到: {tfidf_dim_save_path}")
except IOError as e:
    print(f"错误：无法写入 TF-IDF 维度文件 {tfidf_dim_save_path}: {e}")
# --- 保存结束 ---

# 在验证集和测试集上转换 TF-IDF
print("转换验证集和测试集...")
X_val_tfidf = tfidf_vectorizer.transform(X_val_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"确认 TF-IDF 特征维度: {tfidf_feature_dim}")

# --- 后续代码：将稀疏矩阵转换为 PyTorch Tensor ---
print("将 TF-IDF 特征转换为稠密 Tensor...")
try:
    X_train_features = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float)
    X_val_features = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float)
    X_test_features = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float)
except MemoryError:
    print("错误：将 TF-IDF 转换为稠密矩阵时内存不足。...")
    exit()
except Exception as e:
    print(f"错误：转换 TF-IDF 特征到 Tensor 时出错: {e}")
    exit()


# --- 再往后是创建 FeatureDataset 和 DataLoader 的代码 ---

# 将稀疏矩阵转换为 PyTorch Tensor (如果内存允许，转为稠密；否则需处理稀疏Tensor)
# 注意：.toarray() 会消耗大量内存，如果特征维度非常大或数据量很大，可能需要优化
print("将 TF-IDF 特征转换为稠密 Tensor...")
try:
    X_train_features = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float)
    X_val_features = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float)
    X_test_features = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float)
except MemoryError:
    print("错误：将 TF-IDF 转换为稠密矩阵时内存不足。可能需要减小 max_features 或处理稀疏张量。")
    # 这里可以考虑使用 torch.sparse 相关的操作，但会更复杂
    exit()


# --- PyTorch Dataset (现在接受特征向量和标签) ---
class FeatureDataset(Dataset): # <-- 重命名 Dataset
    def __init__(self, features, labels):
        self.features = features # 输入是 Tensor
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels) # 或者 self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx] # 返回特征向量和标签

# 使用转换后的特征创建 Dataset
train_dataset = FeatureDataset(X_train_features, y_train)
val_dataset = FeatureDataset(X_val_features, y_val)
test_dataset = FeatureDataset(X_test_features, y_test)

# DataLoader (保持不变)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- MLP 模型定义 (保持不变) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = self.fc1(x); out = self.relu(out); out = self.dropout(out)
        logits = self.fc2(out)
        return logits

# 实例化 MLP 模型 (使用 TF-IDF 特征维度)
mlp_model = SimpleMLP(
    input_dim=tfidf_feature_dim, # <-- 使用 TF-IDF 特征维度
    hidden_dim=mlp_hidden_dim,
    num_classes=num_classes,
    dropout_prob=dropout_prob
).to(device)

print("\n模型结构 (MLP - 输入为 TF-IDF):")
print(mlp_model)
total_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
print(f"\nMLP 模型总参数量: {total_params:,}")


# --- 训练设置 (计算类别权重部分保持不变) ---
if use_class_weight:
    print("\n计算类别权重...")
    label_counts = np.bincount(y_train)
    if len(label_counts) < num_classes:
        padded_counts = np.zeros(num_classes, dtype=label_counts.dtype)
        padded_counts[:len(label_counts)] = label_counts
        label_counts = padded_counts
    # 处理训练集中可能不存在的类别，防止除零
    label_counts = np.where(label_counts == 0, 1, label_counts)
    class_weights = len(y_train) / (num_classes * label_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"计算得到的类别权重: {class_weights_tensor.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print("已启用带权重的 CrossEntropyLoss。")
else:
    criterion = nn.CrossEntropyLoss()
    print("未使用类别权重。")

# 优化器只优化 MLP 的参数 (保持不变)
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

# --- 移除 get_embeddings_batch 函数 ---
# def get_embeddings_batch(...):
#     ... (整个函数删除) ...

# --- 简化后的训练与验证函数 ---
print("--- 定义简化后的训练与验证函数 ---")
def train_epoch(model, data_loader, criterion, optimizer, device): # <-- 移除 bert 相关参数
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for features_batch, labels_batch in data_loader: # 直接获取特征和标签
        features_batch = features_batch.to(device) # <-- 移动特征到设备
        labels_batch = labels_batch.to(device)

        # 使用 MLP 进行预测
        optimizer.zero_grad()
        outputs = model(features_batch) # <-- 直接输入特征到 MLP
        loss = criterion(outputs, labels_batch)

        # 反向传播和更新 MLP 的权重
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels_batch).sum().item()
        total_samples += labels_batch.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device): # <-- 移除 bert 相关参数
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features_batch, labels_batch in data_loader: # 直接获取特征和标签
            features_batch = features_batch.to(device) # <-- 移动特征到设备
            labels_batch = labels_batch.to(device)

            # MLP 预测
            outputs = model(features_batch) # <-- 直接输入特征到 MLP
            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels_batch).sum().item()
            total_samples += labels_batch.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, all_preds, all_labels

# --- 开始训练 ---
print(f"\n开始训练 MLP (on TF-IDF features) 模型 (v6_tfidf)...")
best_val_accuracy = 0.0
training_start_time = time.time()
epochs_no_improve = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    # 使用简化后的训练函数
    train_loss, train_acc = train_epoch(
        mlp_model, train_loader, criterion, optimizer, device
    )
    # 使用简化后的评估函数
    val_loss, val_acc, _, _ = evaluate(
        mlp_model, val_loader, criterion, device
    )
    epoch_duration = time.time() - epoch_start_time
    print("-" * 50)
    print(f'Epoch {epoch+1} 结束 | 时间: {epoch_duration:.2f}s') # 时间会比 BERT 版本快很多
    print(f'\t训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}')
    print(f'\t验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}')
    print("-" * 50)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        # 只保存 MLP 模型的权重
        torch.save(mlp_model.state_dict(), model_save_path)
        print(f'*** 在验证集上获得更好准确率 ({best_val_accuracy:.4f})，MLP 模型已保存到 {model_save_path} ***')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"\n验证准确率已连续 {early_stopping_patience} 个 epochs 未提升，触发早停机制。")
        break
    print("\n")

total_training_time = time.time() - training_start_time
print(f"训练完成! 总耗时: {total_training_time // 60:.0f} 分 {total_training_time % 60:.0f} 秒")

# --- 评估最佳模型 ---
print("\n在测试集上评估最佳模型 (v6_tfidf)...")
try:
    # 实例化 MLP 结构来加载权重
    model_to_load = SimpleMLP(tfidf_feature_dim, mlp_hidden_dim, num_classes, dropout_prob).to(device)
    model_to_load.load_state_dict(torch.load(model_save_path))
    model_to_load.eval()
    print(f"已加载最佳 MLP 模型: {model_save_path}")

    # 评估时不需要 BERT 了
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor if use_class_weight else None)
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model_to_load, test_loader, eval_criterion, device
    )
except FileNotFoundError:
    print(f"错误: 找不到保存的最佳模型 {model_save_path}。")
    exit()
except Exception as e:
    print(f"加载模型权重或评估时出错: {e}")
    exit()

print(f'\n测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}')
print("\n测试集分类报告 (v6_tfidf - 基于 demo.csv TF-IDF 特征):")
if id_to_label:
    target_names = [id_to_label[i] for i in range(num_classes)]
    # 确保 test_labels 和 test_preds 中的所有值都在 target_names 的索引范围内
    valid_labels = set(range(num_classes))
    filtered_indices = [i for i, (l, p) in enumerate(zip(test_labels, test_preds)) if l in valid_labels and p in valid_labels]
    if len(filtered_indices) < len(test_labels):
        print(f"警告: 过滤掉 {len(test_labels) - len(filtered_indices)} 个标签/预测对，因为它们超出了预期类别范围 {num_classes}。")
    filtered_labels = [test_labels[i] for i in filtered_indices]
    filtered_preds = [test_preds[i] for i in filtered_indices]

    if not filtered_labels:
        print("错误：没有有效的标签和预测对可用于生成报告。")
        report = "无法生成报告"
    else:
        report = classification_report(filtered_labels, filtered_preds, target_names=target_names, digits=4, zero_division=0)

else:
    print("无法加载标签名称，报告将只显示类别 ID。")
    report = classification_report(test_labels, test_preds, digits=4, zero_division=0)
print(report)
print("\n重要提示：此结果基于 TF-IDF 特征，通常训练速度更快，但精度可能低于使用 BERT Embedding 的模型。")
print("\n脚本 v6_tfidf (MLP on TF-IDF features from demo.csv) 执行完毕。")