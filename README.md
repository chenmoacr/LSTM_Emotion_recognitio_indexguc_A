# 中文情绪分类模型 V6 (TF-IDF+MLP) 及 API

本项目包含一个基于 TF-IDF 特征和 MLP（多层感知器）训练的中文文本情绪分类模型（V6-TFIDF版本），以及一个用于提供预测服务的 Flask API。

## 文件结构

```
/
├── code/                      # 主要代码目录
│   ├── app_v6_tfidf_api.py     # Flask API 服务脚本
│   ├── best_mlp_on_demo_data_v6_tfidf.pt  # 训练好的 MLP 模型权重
│   ├── tfidf_feature_dim_v6.txt # TF-IDF 特征维度记录文件
│   ├── tfidf_vectorizer_v6.joblib # 训练好的 TF-IDF Vectorizer
│   └── v6_mlp_tfidf_integration.py  # 模型训练脚本
├── dataset/                   # 数据集目录
│   └── demo.csv              # 用于训练和评估的数据集
├── requirements.txt           # Python 依赖列表
└── README.md                  # 本说明文件
```

## 环境设置

1.  **创建虚拟环境 (推荐):**
    建议使用 Conda 或 Python `venv` 创建一个独立的 Python 环境（例如，使用 Python 3.8 或更高版本）。
    ```bash
    # 使用 Conda (示例)
    conda create -n emotion_tfidf python=3.9
    conda activate emotion_tfidf
    ```

2.  **安装 PyTorch (重要 - GPU 版本):**
    * 你的电脑配备了 RTX 3050 4GB GPU，请务必安装支持 CUDA 的 PyTorch 版本以利用 GPU 加速。
    * 访问 PyTorch 官方网站获取安装命令：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    * 根据你系统安装的 CUDA 版本（例如 CUDA 11.8 或 12.1）选择对应的命令。例如 (请务必在官网确认！)：
        ```bash
        # 示例 (CUDA 11.8)
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
        # 或者 示例 (CUDA 12.1)
        # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```
    * **请在安装其他依赖之前，先单独执行适合你环境的 PyTorch 安装命令。**

3.  **安装其他依赖:**
    * 在安装好 PyTorch 后，进入包含 `requirements.txt` 的项目根目录，运行：
        ```bash
        pip install -r requirements.txt
        ```

## 如何运行

### 1. 运行 API 服务 (直接使用预训练模型)

进入 `code` 目录，运行 API 脚本：

```bash
cd code
python app_v6_tfidf_api.py
```

如果一切顺利，你会看到类似以下的输出，表示 Flask 服务器已启动并监听在 `http://0.0.0.0:8080/emo`：

```
API 将使用设备: cuda
从文件加载 TF-IDF 特征维度: 5000
加载 TF-IDF Vectorizer: tfidf_vectorizer_v6.joblib
... (jieba 加载信息) ...
jieba 分词器可用。
TF-IDF Vectorizer 加载完成。
加载 MLP 模型权重: best_mlp_on_demo_data_v6_tfidf.pt
MLP 模型权重加载完成并移至设备。
使用的标签映射: {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
TF-IDF Vectorizer 和 MLP 模型已成功加载。
启动 Flask API 服务器 (TF-IDF 版本)，监听 [http://0.0.0.0:8080/emo](http://0.0.0.0:8080/emo)
 * Serving Flask app 'app_v6_tfidf_api'
 * Debug mode: off
...
 * Running on [http://127.0.0.1:8080](http://127.0.0.1:8080)
 * Running on http://[你的IP地址]:8080
Press CTRL+C to quit
```

### 2. 测试 API 服务

你可以使用 `curl` (命令行工具) 或编写一个简单的 Python 脚本来发送 POST 请求。

**Curl 示例:**

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"sentence\": \"今天阳光明媚，真是开心的一天！\"}" [http://127.0.0.1:8080/emo](http://127.0.0.1:8080/emo)
```

**Python `requests` 示例:**

```python
import requests
import json

url = "[http://127.0.0.1:8080/emo](http://127.0.0.1:8080/emo)"
# 更换你要测试的句子
data = {"sentence": "这个电影太无聊了，有点失望。"}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf-8')) # 确保正确编码中文
    response.raise_for_status() # 检查请求是否成功
    print("响应内容:", response.json())
except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")
    # 如果服务器返回错误，尝试打印文本内容
    if response:
         print("服务器原始响应:", response.text)

```

### 3. 运行训练 (可选)

如果你想基于 `dataset/demo.csv` 重新训练模型（注意：这会覆盖 `code` 目录下的 `.pt`, `.joblib`, `.txt` 文件）：

```bash
cd code
python v6_mlp_tfidf_integration.py
```

训练过程会显示在控制台，结束后会生成新的模型文件和评估报告。

## 注意事项

* 请确保运行脚本时，所有必需的文件都在正确的相对路径下。
* 模型的表现在很大程度上依赖于训练数据的质量和特性。基于 `demo.csv` 测试集的准确率约为 63.5%，对稀有类别（如 'fear'）的识别能力有限。
* RTX 3050 的 4GB 显存对于运行这个 TF-IDF+MLP 模型（包括训练和推理）应该是足够的。