# code/app_v6_tfidf_api.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
# from transformers import BertTokenizer, BertModel # <-- 不再需要 BERT
from sklearn.feature_extraction.text import TfidfVectorizer # <-- 可能需要导入基类，但 joblib 加载通常不需要
import joblib # <-- 用于加载 Vectorizer
import numpy as np
import os
import time
from flask import Flask, request, jsonify
import jieba # <-- 确保 API 环境也安装并可以导入 jieba

# --- 1. 配置和全局变量 ---

# 模型和 Vectorizer 路径
MLP_MODEL_PATH = 'best_mlp_on_demo_data_v6_tfidf.pt' # TF-IDF 模型的权重
TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer_v6.joblib' # 保存的 TF-IDF Vectorizer
TFIDF_DIM_PATH = 'tfidf_feature_dim_v6.txt' # 保存的 TF-IDF 维度文件

# MLP 模型参数 (必须与训练时一致)
# TFIDF_FEATURE_DIM = 5000 # 可以硬编码，但从文件读取更安全
MLP_HIDDEN_DIM_1 = 512   # MLP 隐藏层维度 (与训练时一致)
NUM_CLASSES = 6          # 类别数量 (与训练时一致)
DROPOUT_PROB = 0.5       # Dropout 概率 (与训练时一致)

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API 将使用设备: {DEVICE}")

# --- !!! 新增：添加与训练脚本相同的 tokenizer 函数定义 !!! ---
def jieba_tokenizer(text):
    """使用 jieba.lcut 进行分词"""
    return jieba.lcut(text)
# --- 函数定义结束 ---




# --- 读取 TF-IDF 特征维度 ---
try:
    with open(TFIDF_DIM_PATH, 'r') as f:
        TFIDF_FEATURE_DIM = int(f.read().strip())
    print(f"从文件加载 TF-IDF 特征维度: {TFIDF_FEATURE_DIM}")
except FileNotFoundError:
    print(f"错误: 找不到 TF-IDF 维度文件 {TFIDF_DIM_PATH}，请先运行训练脚本生成该文件。")
    TFIDF_FEATURE_DIM = None # 标记为失败
except ValueError:
     print(f"错误: TF-IDF 维度文件 {TFIDF_DIM_PATH} 内容不是有效的整数。")
     TFIDF_FEATURE_DIM = None # 标记为失败
except Exception as e:
    print(f"读取 TF-IDF 维度文件时发生未知错误: {e}")
    TFIDF_FEATURE_DIM = None # 标记为失败


# --- 2. 定义模型结构 (必须与训练时一致) ---
# 使用与 v6_mlp_tfidf_integration.py 训练时相同的单层 MLP 定义
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

# --- 3. 加载 TF-IDF Vectorizer 和 MLP 模型 (在 Flask 启动时加载一次) ---

# 加载 TF-IDF Vectorizer
print(f"加载 TF-IDF Vectorizer: {TFIDF_VECTORIZER_PATH}")
tfidf_vectorizer = None
try:
    # 检查 jieba 是否可用，因为 Vectorizer 可能依赖它
    _ = jieba.lcut("测试")
    print("jieba 分词器可用。")

    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    print("TF-IDF Vectorizer 加载完成。")
    # 可以在这里打印 vectorizer 的一些信息确认，例如：
    # print(f"Vectorizer Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
except FileNotFoundError:
    print(f"错误: 找不到 TF-IDF Vectorizer 文件 {TFIDF_VECTORIZER_PATH}。请确保已运行训练脚本并生成该文件。")
except ImportError:
     print("错误: jieba 未安装或无法导入。TF-IDF Vectorizer 可能依赖它。请运行 'pip install jieba'")
except Exception as e:
    print(f"加载 TF-IDF Vectorizer 时出错: {e}")

# 加载训练好的 MLP
print(f"加载 MLP 模型权重: {MLP_MODEL_PATH}")
mlp_model = None
if tfidf_vectorizer and TFIDF_FEATURE_DIM: # 只有 Vectorizer 和维度加载成功才继续
    try:
        # --- !!! 关键改动：使用 TFIDF_FEATURE_DIM 实例化 MLP !!! ---
        mlp_model = SimpleMLP(
            input_dim=TFIDF_FEATURE_DIM, # 使用正确的 TF-IDF 维度
            hidden_dim=MLP_HIDDEN_DIM_1,
            num_classes=NUM_CLASSES,
            dropout_prob=DROPOUT_PROB
        )
        # 加载权重
        mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))
        # 移动模型到设备
        mlp_model.to(DEVICE)
        # 设置为评估模式
        mlp_model.eval()
        print("MLP 模型权重加载完成并移至设备。")
    except FileNotFoundError:
        print(f"错误: 找不到 MLP 模型权重文件 {MLP_MODEL_PATH}。API 将无法工作。")
        mlp_model = None # 标记为加载失败
    except RuntimeError as e:
         print(f"加载 MLP 模型权重时发生运行时错误 (可能是维度不匹配): {e}")
         print("请确认 TFIDF_FEATURE_DIM 与模型权重文件中的维度一致。")
         mlp_model = None # 标记为加载失败
    except Exception as e:
        print(f"加载 MLP 模型权重时发生未知错误: {e}")
        mlp_model = None # 标记为加载失败

# 定义标签映射 (与训练时一致)
ID_TO_LABEL = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
print(f"使用的标签映射: {ID_TO_LABEL}")

# --- 4. Flask 应用定义 ---
app = Flask(__name__)

# --- 移除 get_embeddings_for_inference 函数 ---

# --- API 端点 ---
@app.route('/emo', methods=['POST'])
def predict_emotion_api():
    global tfidf_vectorizer, mlp_model, ID_TO_LABEL, DEVICE

    # 检查 Vectorizer 和 MLP 模型是否加载成功
    if not tfidf_vectorizer or not mlp_model:
        return jsonify({"code": 500, "message": "错误：TF-IDF Vectorizer 或 MLP 模型未能成功加载，无法处理请求。"}), 500

    # ... (检查 JSON 输入部分保持不变) ...
    if not request.is_json:
        return jsonify({"code": 0, "message": "请求错误：需要 JSON 格式的数据。"}), 400
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({"code": 0, "message": "请求错误：JSON 数据中缺少 'sentence' 键。"}), 400
    sentence = data['sentence']
    if not isinstance(sentence, str) or not sentence.strip():
        return jsonify({"code": 0, "message": "请求错误：'sentence' 不能为空字符串。"}), 400

    try:
        start_time = time.time()

        # --- !!! 核心改动：使用 TF-IDF Vectorizer 提取特征 !!! ---
        # 1. 使用加载的 Vectorizer 转换输入文本
        # transform 期望一个列表或可迭代对象
        features_sparse = tfidf_vectorizer.transform([sentence])

        # 2. 将稀疏特征转换为稠密 PyTorch Tensor
        # 注意: 对于非常大的词汇表，toarray() 可能消耗较多内存，但对于单个句子通常没问题
        features_tensor = torch.tensor(features_sparse.toarray(), dtype=torch.float).to(DEVICE)
        # ----------------------------------------------------------

        # 3. MLP 预测 (输入为 TF-IDF 特征)
        with torch.no_grad():
            logits = mlp_model(features_tensor) # 将 TF-IDF 特征输入 MLP
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_id = torch.max(probabilities, dim=1)

        # 4. 获取结果 (与之前相同)
        predicted_label = ID_TO_LABEL.get(predicted_id.item(), "未知标签")
        # conf = confidence.item() # 如果需要返回置信度

        end_time = time.time()
        print(f"预测完成: '{sentence}' -> {predicted_label}, 耗时: {end_time - start_time:.4f} 秒")

        # 5. 返回成功结果 (与之前相同)
        response_data = {
            "code": "200",
            "data": predicted_label,
            "msg": "Success"
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        # 返回通用服务器错误
        return jsonify({"code": 500, "message": f"服务器内部错误: {e}"}), 500


# --- 5. 启动 Flask 应用 ---
if __name__ == '__main__':
    # 检查模型是否已加载，提供更明确的启动信息
    if tfidf_vectorizer and mlp_model:
         print("TF-IDF Vectorizer 和 MLP 模型已成功加载。")
    else:
         print("警告：TF-IDF Vectorizer 或 MLP 模型未能完全加载，API 可能无法正常工作。")

    print("启动 Flask API 服务器 (TF-IDF 版本)，监听 http://0.0.0.0:8080/emo")
    # 运行 Flask 应用
    app.run(host='0.0.0.0', port=8080, debug=False)