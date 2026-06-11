# MMEA Mindstorm

**多模态知识图谱实体对齐 (Multimodal Knowledge Graph Entity Alignment)**

> Research workspace: survey → ideation → implementation → iteration

---

## 项目目标

本仓库用于系统性地调研多模态知识图谱实体对齐（MMEA）领域，并在此基础上设计、实现和验证新方法。

- **Baseline**：[MEAformer](https://arxiv.org/abs/2212.14454)（Multi-modal Entity Alignment Transformer, ACM MM 2023）
- **数据集**：FB15K-DB15K、FB15K-YG15K（标准 MMEA benchmark）
- **目标**：在现有方法基础上找到可突破的切入点，发表顶会论文

---

## 仓库结构

```
mmea_mindstorm/
├── survey/                        # 文献调研
│   ├── papers/                    # 逐篇论文笔记（Markdown）
│   ├── taxonomy.md                # 方法分类体系
│   ├── comparison_table.md        # 各方法性能对比表格
│   └── open_problems.md           # 开放问题与研究机会
├── src/                           # 核心代码
│   ├── data/                      # 数据加载、预处理
│   ├── models/                    # 模型定义
│   │   ├── meaformer.py           # MEAformer baseline
│   │   └── ours.py                # 我们自己的方法
│   ├── losses/                    # 对齐损失函数
│   ├── trainer/                   # 训练循环
│   └── utils/                     # 工具函数
├── experiments/                   # 实验管理
│   ├── configs/                   # YAML 配置文件
│   └── results/                   # 实验结果（自动生成）
│       └── baseline/
├── data/                          # 数据集（见下载说明）
├── notebooks/                     # 探索性分析 Jupyter Notebooks
├── scripts/                       # 下载/预处理脚本
└── brainstorm/                    # 头脑风暴记录（方法想法）
```

---

## 快速开始

### 环境配置

```bash
conda create -n mmea python=3.9
conda activate mmea
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric transformers timm scikit-learn pandas
```

### 数据准备

```bash
bash scripts/download_data.sh
```

数据集来源：[MMEA 标准数据集](https://github.com/liyichen-cly/MMEA)

### 运行 MEAformer Baseline

```bash
python src/trainer/train.py --config experiments/configs/meaformer_fb15k_db15k.yaml
```

---

## 调研进展

见 [`survey/`](survey/) 目录：

| 文档 | 内容 |
|------|------|
| [`taxonomy.md`](survey/taxonomy.md) | 方法分类体系 |
| [`comparison_table.md`](survey/comparison_table.md) | 各方法性能对比 |
| [`open_problems.md`](survey/open_problems.md) | 开放问题列表 |
| `papers/` | 逐篇论文笔记 |

---

## 头脑风暴记录

见 [`brainstorm/`](brainstorm/) 目录，记录每次讨论的想法、假设和方向。

---

## 实验结果追踪

| 方法 | 数据集 | Hits@1 | Hits@10 | MRR | 备注 |
|------|--------|--------|---------|-----|------|
| MEAformer | FB15K-DB15K | - | - | - | 待复现 |
| MEAformer | FB15K-YG15K | - | - | - | 待复现 |
| **Ours** | FB15K-DB15K | - | - | - | TBD |
| **Ours** | FB15K-YG15K | - | - | - | TBD |

---

## 参考文献

调研论文详见 [`survey/comparison_table.md`](survey/comparison_table.md)。

关键 baseline：
- **MEAformer**: Chen et al., ACM MM 2023. [[paper]](https://arxiv.org/abs/2212.14454) [[code]](https://github.com/zjukg/MEAformer)
