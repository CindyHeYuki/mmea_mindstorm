# MMEA 方法性能对比

> 标准评测协议：30% 训练种子对，Hits@1 / Hits@10 / MRR  
> 数据集：FB15K-DB15K（F-D）、FB15K-YG15K（F-Y），以及跨语言 DBP15K

---

## 完整方法列表

| 方法 | 会议/年 | 核心创新 | 代码 |
|------|---------|---------|------|
| EVA | IJCAI 2021 | 视觉近邻增强关系 EA | [link](https://github.com/cambridgeltl/eva) |
| MMEA | KSEM 2022 | 三模态联合 GNN + 奠基 benchmark | [link](https://github.com/liyichen-cly/MMEA) |
| MCLEA | COLING 2022 | 多粒度模态对比学习 | [link](https://github.com/lzxlin/MCLEA) |
| MSNEA | KDD 2022 | 语义感知多模态网络 | — |
| **MEAformer** | ACM MM 2023 | Transformer 跨模态 + 实体级元权重 | [link](https://github.com/zjukg/MEAformer) |
| UMAEA | ISWC 2023 | 不确定性加权 + 缺失模态 benchmark | [link](https://github.com/zjukg/UMAEA) |
| ACK-MMEA | WWW 2023 | 属性一致性知识图谱表示 | [link](https://arxiv.org/abs/2304.01563) |
| PathFusion | ISWC 2023 | 路径为模态载体，迭代融合 | [link](https://arxiv.org/abs/2310.05364) |
| MoAlign | EMNLP 2023 | 少样本鲁棒多模态 Transformer | [link](https://aclanthology.org/2023.findings-emnlp.70/) |
| PCMEA | AAAI 2024 | 互信息最大化 + 伪标签半监督 | [link](https://arxiv.org/abs/2403.01203) |
| DESAlign | ICDE 2024 | Dirichlet 能量语义一致性 + 缺失处理 | [link](https://arxiv.org/abs/2401.17859) |
| PMF | ACL 2024 | 渐进模态冻结，9 数据集 SOTA | [link](https://arxiv.org/abs/2407.16168) |
| LoginMEA | ECAI 2024 | 实体级局部融合 → 全局 GNN | [link](https://arxiv.org/abs/2407.19625) |
| M-VAE/GEEA | ICLR 2024 | 生成式 VAE，KG 间实体转换 | [link](https://arxiv.org/abs/2305.14651) |
| MCSFF | IEEE 2024 | 一致性+特异性分离 + 迭代去噪 | [link](https://arxiv.org/abs/2410.14584) |
| CDMEA | SIGIR 2025 | 因果视角反事实去除视觉偏置 | [link](https://arxiv.org/abs/2504.19458) |
| SE-GNN | IEEE TKDE 2025 | 条件过滤伪种子 + 迭代半监督 | [link](https://arxiv.org/abs/2503.20801) |
| AMF2SEA | COLING 2025 | 自适应融合策略系统性分析 | [link](https://aclanthology.org/2025.coling-main.522/) |
| LLM-Align | arXiv 2024 | 零样本 LLM 对齐（多轮投票）| [link](https://arxiv.org/abs/2412.04690) |
| EA-Agent | arXiv 2025 | 多步推理 LLM Agent | [link](https://arxiv.org/abs/2604.11686) |

---

## 性能对比：FB15K-DB15K

| 方法 | Hits@1 | Hits@5 | Hits@10 | MRR |
|------|--------|--------|---------|-----|
| EVA | 52.6 | 74.1 | 80.3 | 0.619 |
| MMEA | 57.1 | 77.8 | 83.3 | 0.655 |
| MSNEA | 65.4 | 83.0 | 87.6 | 0.731 |
| MCLEA | 67.3 | 84.7 | 88.8 | 0.745 |
| **MEAformer** | **72.0** | **87.1** | **90.6** | **0.789** |
| UMAEA | 74.2 | 88.3 | 91.5 | 0.804 |
| PathFusion | ~74–76 | — | ~91–92 | — |
| PMF | SOTA† | — | — | — |

†PMF 在 ACL 2024 中报告 9 个数据集 SOTA，具体数字见原论文。

---

## 性能对比：FB15K-YG15K

| 方法 | Hits@1 | Hits@5 | Hits@10 | MRR |
|------|--------|--------|---------|-----|
| EVA | 53.6 | 74.3 | 80.1 | 0.625 |
| MMEA | 54.0 | 75.3 | 81.0 | 0.630 |
| MSNEA | 64.5 | 82.1 | 86.5 | 0.725 |
| MCLEA | 67.1 | 83.2 | 87.4 | 0.742 |
| **MEAformer** | **71.3** | **86.2** | **89.9** | **0.783** |
| UMAEA | 73.5 | 87.5 | 91.0 | 0.798 |

---

## 数据集详情

### FB15K-DB15K / FB15K-YG15K（英语内，标准 MMEA benchmark）

| 属性 | FB15K | DB15K | YG15K |
|------|-------|-------|-------|
| 实体数 | 14,951 | 12,842 | 15,000 |
| 关系数 | 1,345 | 279 | 32 |
| 三元组数 | 592,213 | 89,197 | 175,767 |
| 训练种子对 | 4,500 (30%) | — | — |
| 测试对 | 10,500 | — | — |

### DBP15K（跨语言 benchmark）
- ZH-EN, JA-EN, FR-EN 三对
- 每对约 15,000 对齐实体
- 有图像/属性/描述（部分方法使用）

### MMEA-UMVM（模态缺失鲁棒性 benchmark，UMAEA 引入）
- 基于 FB15K-DB15K/YG15K
- 60 个评测拆分（不同缺失比例）

### M3（多图像 benchmark，CIKM 2024）
- 每个实体有来自多个搜索引擎的多张图像
- 基于 DBP15K，更贴近真实场景

---

## MEAformer 详解（我们的 Baseline）

**论文**：*MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid*  
**代码**：https://github.com/zjukg/MEAformer

### 架构
```
实体图像 ──► ViT/ResNet ──►┐
实体文本 ──► BERT ─────────┼──► Transformer 跨模态 ──► 元权重预测 ──► 融合表示 ──► 对齐
关系结构 ──► RGAT ─────────┘
```

### 关键设计
1. **元模态权重**：每个实体有独立的 3 维模态权重（学到"依赖哪个模态"）
2. **跨模态 Transformer**：[V, T, R] 作为 3-token 序列，自注意力建模互补
3. **辅助任务**：模态预测（masked modality prediction）

### 训练配置
- Adam, lr=5e-4, batch=1024, epochs=1000, neg=25, margin=1.0

### MEAformer 的局限（我们的攻击点）
1. 图像质量噪声未处理（多图平均）
2. 模态缺失未处理
3. 随机负采样
4. 文本编码器固定（未 KG-aware 微调）
5. 元权重预测不感知图结构
6. 跨模态交互仅单层 Transformer

---

## 论文阅读优先级

| 优先级 | 论文 | 原因 |
|--------|------|------|
| ★★★ | MEAformer | Baseline，必须完全理解代码 |
| ★★★ | UMAEA | 不确定性建模 + 缺失模态，最相关上游工作 |
| ★★★ | CDMEA (SIGIR 2025) | 因果去偏，2025 最新方向，可能和我们的方向重叠 |
| ★★★ | PMF (ACL 2024) | 9 数据集 SOTA，了解当前天花板 |
| ★★ | PCMEA (AAAI 2024) | 半监督 + 对比，了解伪标签路线 |
| ★★ | PathFusion (ISWC 2023) | 路径融合，了解迭代路线 |
| ★★ | MCLEA | 对比学习基础 |
| ★ | LLM-Align / EA-Agent | 了解 LLM 方向趋势 |
| ★ | M-VAE/GEEA (ICLR 2024) | 了解生成式路线 |
