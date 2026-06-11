# MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid

**会议**：ACM MM 2023  
**作者**：Zhuo Chen, Jiaoyan Chen, Wen Zhang, Lingbing Guo, Yin Fang, Yufeng Huang, Yichi Zhang, Yuxia Geng, Jeff Z. Pan, Wenting Song, Huajun Chen  
**单位**：浙江大学  
**代码**：https://github.com/zjukg/MEAformer  
**arXiv**：https://arxiv.org/abs/2212.14454

---

## 一句话总结

通过 Transformer 跨模态交互和元模态权重预测，动态融合视觉/文本/关系三种模态，实现多模态实体对齐。

---

## 动机

现有多模态 EA 方法存在两个问题：
1. **静态融合权重**：对所有实体使用相同的模态权重，但不同实体对模态的依赖程度不同（"蒙娜丽莎"高度依赖视觉，"微积分"几乎无图像信息）
2. **模态交互不足**：多数方法只做 early/late fusion，缺乏模态间的深度交互

---

## 方法

### 架构图

```
        [图像] ──► ResNet/ViT ──► V_emb
        [文本] ──► BERT ──────► T_emb    ──► Transformer 跨模态层 ──► 融合表示
        [关系] ──► RGAT ──────► R_emb
                                              ↑
                                   元模态权重预测
                                   (softmax over 3 modalities)
```

### 核心组件

**1. 模态编码器**
- 视觉：ViT-B/16 或 ResNet50，多图取平均
- 文本：BERT-base，取 [CLS] token
- 关系：RGAT（Relational Graph Attention Network），2层

**2. 跨模态 Transformer**
- 将三种模态表示拼接为序列，输入 Transformer
- 用注意力机制建模模态间互补性

**3. 元模态权重预测（Meta Modality Hybrid）**
- 对每个实体预测一个 3 维权重向量（对应 V/T/R）
- 加权融合三种模态的 Transformer 输出
- 关键洞察：权重是实体级别的，不是全局共享的

**4. 训练损失**
- 主要：基于马氏距离的对齐损失（Margin-based ranking loss）
- 辅助：模态预测任务（预测被 mask 掉的某个模态）

---

## 实验结果

| 数据集 | Hits@1 | Hits@10 | MRR |
|--------|--------|---------|-----|
| FB15K-DB15K | 72.0 | 90.6 | 0.789 |
| FB15K-YG15K | 71.3 | 89.9 | 0.783 |

超过之前所有方法约 5 个 Hits@1 点。

---

## 我的分析

### 优点
- 元模态权重的设计直觉清晰，可解释性强
- 实验结果扎实，消融实验充分
- 代码质量高，便于复现

### 局限（值得攻克）
1. **图像噪声**：多图平均策略粗糙，未区分有效/噪声图像
2. **单层 Transformer**：跨模态交互深度有限
3. **固定文本编码器**：BERT 未针对 KG 微调
4. **随机负采样**：未使用困难负样本
5. **模态完整假设**：未处理模态缺失
6. **元权重预测独立于图结构**：权重预测时未利用邻居信息

### 与 UMAEA 的关系
UMAEA（同组，ISWC 2023）可看作 MEAformer 的改进版，引入了不确定性建模，进一步提升约 2 个点。

---

## 复现要点

```bash
git clone https://github.com/zjukg/MEAformer
cd MEAformer
pip install -r requirements.txt
# 数据预处理
python preprocess.py --dataset FB15K-DB15K
# 训练
python main.py --dataset FB15K-DB15K --model MEAformer
```

**关键超参**：
- `lr`: 5e-4
- `batch_size`: 1024  
- `neg_num`: 25
- `gnn_layers`: 2
- `transformer_layers`: 1
- `hidden_dim`: 256

---

## 参考阅读

- UMAEA（改进版，同组）
- MCLEA（对比学习路线，同期竞品）
