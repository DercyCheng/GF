# 项目说明

## utils.py

`utils.py` 包含了一系列实用函数，用于数据预处理、可视化以及模型解释。主要功能包括：

- **确保目录存在**

  - `ensure_dir(directory)`: 检查指定目录是否存在，不存在则创建。
- **清理文件名中的非法字符**

  - `sanitize_filename(filename)`: 移除文件名中的非法字符，确保文件名安全。
- **绘制实际值与预测值的散点图**

  - `plot_results(y_true, y_pred, title, model_type, target_column)`: 生成并保存实际值与预测值的散点图，包括拟合线和评估指标。
- **进行 SHAP 和 LIME 分析以解释模型**

  - `shap_analysis(model, X, feature_names, target_column, model_type, attention_type, dataset_name)`: 使用 SHAP 进行模型解释并保存结果图。
  - `lime_analysis(model, X, y, feature_names, target_column, model_type, attention_type, dataset_name)`: 使用 LIME 进行模型解释并保存结果图。
- **数据增强和预处理**

  - `augment_data(X, y)`: 通过添加高斯噪声进行数据增强。
  - `load_data(file_path, target_columns)`: 从 Excel 文件加载数据，处理缺失值并分离特征与目标。
  - `preprocess_data(X)`: 对数据进行填补缺失值和标准化处理。
- **设置随机种子**

  - `set_seed(seed=42)`: 设置各种随机数生成器的种子以确保结果可复现。

## DL.py

`DL.py` 负责训练和评估深度学习模型。主要功能包括：

- **定义和导入多种深度学习模型**

  - 支持的模型包括 CNN、ResNet18、VGG7、LSTM、SSLTransformer。
- **实现训练循环**

  - 包括数据加载、优化器设置、学习率调度等。
- **使用 K-Fold 交叉验证评估模型性能**

  - 通过 K-Fold 交叉验证提高模型的泛化能力。
- **记录和显示训练进度**

  - 使用 `rich` 库记录和显示训练的进度和结果。
- **生成并保存模型评估结果和可视化图表**

  - 评估模型的 R²、RMSE、RPD，并生成对应的可视化图表。

### 模型详解

本项目实现了多种深度学习模型，每种模型具有不同的架构和参数设置，以适应不同的数据和任务需求。

#### 1. CNN (卷积神经网络)

- **架构**：

  - 包含多个卷积层和池化层，用于提取数据的局部特征。
  - 最后连接全连接层进行回归预测。
- **关键参数**：

  - `input_dim`：输入数据的维度。
  - `attention_type`：注意力机制类型（如 SE、ECA、CBAM），用于增强特征表示。

#### 2. ResNet18

- **架构**：

  - 基于残差网络（ResNet）架构，具有18层深度。
  - 使用残差块以缓解深层网络中的梯度消失问题。
- **关键参数**：

  - 无需特定输定维度设置，默认适应输入数据形状。
  - `device`：模型运行的设备（CPU 或 GPU）。

#### 3. VGG7

- **架构**：

  - 基于 VGG 网络的简化版本，包含7层深度。
  - 主要由连续的卷积层和池化层组成，后接全连接层。
- **关键参数**：

  - `input_dim`：输入数据的维度。
  - 通过调整卷积核数量和全连接层大小以适应不同任务。

#### 4. LSTM (长短期记忆网络)

- **架构**：

  - 基于循环神经网络（RNN）的 LSTM 单元，适用于处理序列数据。
  - 包含多个 LSTM 层，可捕捉数据中的时间依赖关系。
- **关键参数**：

  - `input_dim`：输入数据的特征维度。
  - 序列长度设定为10，以适应时间序列分析。

#### 5. SSLTransformer

- **架构**：

  - 基于 Transformer 架构，包含自注意力机制。
  - 包含多层编码器以捕捉全局特征关系。
- **关键参数**：

  - `input_dim`：输入数据的特征维度。
  - `embed_dim`：嵌入维度，需保证能被 `num_heads` 整除（默认128）。
  - `num_heads`：多头注意力的头数（默认8）。
  - `num_layers`：Transformer 编码器层数（默认6）。
  - `num_classes`：输出类别数（回归任务设为1）。
