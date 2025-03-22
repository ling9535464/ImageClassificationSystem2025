# ImageClassificationSystem2025

2025.3.22

---

### **环境配置**
以下是运行该代码所需的软件和硬件环境：

#### **硬件要求**
- **GPU**：建议使用NVIDIA GPU（支持CUDA），显存至少4GB以上（取决于批量大小）。
- **内存**：建议16GB以上。

#### **软件依赖**
1. **操作系统**：支持Linux、Windows或macOS（需配置GPU驱动）。
2. **Python版本**：Python 3.8+。
3. **核心库**：
   ```bash
   # 安装PyTorch及相关库（根据CUDA版本选择）
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   pip install Pillow scikit-learn tensorboard
   ```

#### **关键库版本**
- `torch >= 1.10.0`
- `torchvision >= 0.11.0`
- `scikit-learn >= 1.0.0`
- `Pillow >= 8.4.0`

---

### **数据处理流程**
#### **数据集结构**
- **根目录**：
  - `all_data_dir`：包含所有正常和异常图像的文件夹（如 `2021-2023all`）。
  - `abnormal_data_dir`：仅包含异常图像的文件夹（如 `2021-2023abnormal`）。

#### **数据加载与预处理**
1. **去重与完整性检查**：
   - 使用SHA-256哈希值对异常图像去重，并缓存哈希值到`hashes.csv`。
   - 验证图像完整性（避免损坏文件）。

2. **数据增强**：
   - **正常样本**：Resize到224x224 + 随机水平翻转 + 归一化。
   - **异常样本**：额外应用随机旋转（30度）和对比度调整（`ColorJitter`），增强异常特征。

3. **类别平衡**：
   - 使用`WeightedRandomSampler`，为异常样本分配更高权重（15:1），缓解类别不平衡问题。

#### **代码关键类**
- `CustomDataset`：自定义数据集类，处理数据加载、哈希去重、动态增强。

---

### **模型构建**
#### **模型架构**
1. **主干网络**：  
   - 使用预训练的ResNet18（`torchvision.models.resnet18`）。
   - **微调策略**：解冻所有层参数（`param.requires_grad = True`），允许全网络微调。

2. **分类头**：  
   - 替换原始全连接层（`self.model.fc`），输出维度为2（二分类）。

#### **损失函数**
- **RecallLoss**：自定义损失函数，结合交叉熵损失和召回率惩罚项。
  ```python
  loss = CrossEntropyLoss + beta * (1 - recall)
  ```
  - `beta=2.0`：召回率优化权重。
  - 直接最大化异常类别的召回率。

#### **优化策略**
- **优化器**：AdamW（学习率 `0.0001`）。
- **学习率调整**：`ReduceLROnPlateau`（根据验证损失动态调整）。
- **早停机制**：连续10个epoch验证损失未改善则提前终止训练。

---

### **测试与评估**
#### **运行流程**
1. **交叉验证**：
   - 使用`StratifiedKFold`进行5折分层交叉验证。
   - 每折将数据分为训练集（50%）、验证集（25%）、测试集（25%）。

2. **训练与验证**：
   - 每epoch记录训练损失、验证损失、召回率等指标。
   - 使用TensorBoard可视化指标（日志保存在`runs/improved_experiment`）。

3. **测试阶段**：
   - 在测试集上计算以下指标：
     - **准确率**（Accuracy）
     - **精确率**（Precision）、**召回率**（Recall）
     - **F1分数**、**F2分数**（公式：`F2 = 5*(P*R)/(4P + R)`）

#### **关键代码片段**
```python
# 测试阶段计算F2分数
precision_abnormal = test_report['abnormal']['precision']
recall_abnormal = test_report['abnormal']['recall']
f2_score_abnormal = 5 * (precision_abnormal * recall_abnormal) / (4 * precision_abnormal + recall_abnormal)
```

#### **运行命令**
```bash
python classification5.py
```

---


