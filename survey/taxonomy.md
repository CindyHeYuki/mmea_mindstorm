# MMEA 方法分类体系

> 基于文献系统梳理（覆盖至 2026 年初），多维度分类

---

## 维度一：模态组合

| 类别 | 描述 | 代表方法 |
|------|------|---------|
| 结构/关系 | 仅 KG 三元组 | MTransE, BootEA, MRAEA |
| 结构 + 属性 | + 字面量属性 | JAPE, GCN-Align |
| 结构 + 视觉 | + 图像 | EVA (IJCAI 2021) |
| 结构 + 视觉 + 属性 | 三模态 | MCLEA, MEAformer, UMAEA, PMF |
| 结构 + 视觉 + 属性 + 文本描述 | 四模态 | ACK-MMEA, FMEA-TD (2025), PMF |

---

## 维度二：跨模态融合策略

| 策略 | 机制 | 优缺点 | 代表 |
|------|------|--------|------|
| **早期融合**（Early） | Concat/Add 后统一建模 | 简单，缺深度交互 | MMEA (KSEM 2020) |
| **晚期融合**（Late） | 各模态独立打分，最后线性组合 | 可模块化 | EVA |
| **跨模态注意力**（Cross-modal Attention）| Transformer 建模模态间互补 | 强表达，计算重 | MEAformer, MoAlign, PathFusion |
| **自适应/实体级**（Adaptive） | 每个实体动态学习模态权重 | 最灵活 | MEAformer, AMF2SEA (COLING 2025) |
| **一致性+特异性分离** | 分离共享信号与模态独有信号 | 理论清晰 | MCSFF (IEEE 2024) |

---

## 维度三：训练目标

| 目标类型 | 代表 |
|---------|------|
| Margin/Triplet loss | MMEA (2020) |
| 模态内对比学习（InfoNCE） | MCLEA |
| 跨模态对比 | MCLEA (IAL loss), PCMEA (AAAI 2024) |
| 互信息最大化 | PCMEA |
| 生成式（VAE/GAN）| GEEA/M-VAE (ICLR 2024) |
| 跨模态一致性损失 | PMF (ACL 2024) |
| Dirichlet 能量语义插值 | DESAlign (ICDE 2024) |
| 因果反事实去偏 | CDMEA (SIGIR 2025) |

---

## 维度四：监督类型

| 类型 | 代表 |
|------|------|
| 有监督（固定种子对）| MEAformer 等大多数方法 |
| 半监督 / 伪标签扩展 | PCMEA (AAAI 2024), SE-GNN (TKDE 2025) |
| 迭代自举（bootstrapping）| PathFusion, MCSFF, SE-GNN |
| 无监督 / 自监督 | OPICE (2026), UMEAD (2025), PSQE (2026) |
| 少样本 / 低资源 | MEAformer（低资源设置 SOTA）, MoAlign |

---

## 维度五：模态缺失处理能力

| 类型 | 代表 |
|------|------|
| 假设模态完整（大多数方法）| EVA, MMEA, MCLEA, MEAformer |
| 不确定性建模 | UMAEA (ISWC 2023) |
| 能量引导传播 | DESAlign (ICDE 2024) |
| 高斯噪声掩码统一框架 | SnAg (arXiv 2024) |
| 因果去偏（视觉噪声）| CDMEA (SIGIR 2025) |
| 渐进模态冻结 | PMF (ACL 2024) |

---

## 时间轴

```
2021  EVA (IJCAI) ─────────────── 视觉+关系，图像打分辅助
2022  MMEA (KSEM) ──────────────  奠基三模态框架 + benchmark
      MCLEA (COLING) ──────────── 对比学习引入 MMEA
      MSNEA (KDD) ─────────────── 语义感知多模态网络
2023  MEAformer (ACM MM) ───────── Transformer 跨模态 + 元权重（当前主流 baseline）
      UMAEA (ISWC) ────────────── 不确定性感知 + 缺失模态 benchmark
      ACK-MMEA (WWW) ──────────── 属性一致性
      PathFusion (ISWC) ───────── 图路径统一多模态迭代融合
      MoAlign (EMNLP Findings) ── 少样本鲁棒 Transformer
2024  PCMEA (AAAI) ────────────── 互信息 + 伪标签半监督
      DESAlign (ICDE) ─────────── Dirichlet 能量语义一致性
      PMF (ACL) ───────────────── 渐进模态冻结，9 数据集 SOTA
      LoginMEA (ECAI) ─────────── 实体级多模态 → 全局 GNN
      MCSFF (IEEE) ────────────── 一致性+特异性分离
      GEEA/M-VAE (ICLR) ──────── 生成式 KG 实体转换
      LLM-Align (arXiv) ──────── 零样本 LLM 对齐
      M3 Dataset (CIKM) ──────── 多图像 benchmark
2025  CDMEA (SIGIR) ───────────── 因果视角去除视觉噪声偏置
      SE-GNN (TKDE) ───────────── 种子扩展半监督 GNN
      AMF2SEA (COLING) ────────── 自适应融合策略分析
      EA-Agent/AgentEA (arXiv) ── LLM 多智能体推理
2026  PSQE (arXiv) ────────────── 无监督伪种子质量增强
      OPICE (Electronics) ──────── 本体引导无监督 MMEA
```

---

## 分类总结矩阵

| 方法 | 年份 | 会议 | 模态 | 融合策略 | 训练目标 | 监督 | 缺失处理 |
|------|------|------|------|---------|---------|------|---------|
| EVA | 2021 | IJCAI | V+R | 晚期 | Margin | 有监督 | ✗ |
| MMEA | 2022 | KSEM | V+T+R | 早期 | Margin | 有监督 | ✗ |
| MCLEA | 2022 | COLING | V+T+R | 跨模态注意力 | 对比 | 有监督 | ✗ |
| MSNEA | 2022 | KDD | V+T+R | 语义网络 | Margin | 有监督 | ✗ |
| MEAformer | 2023 | ACM MM | V+T+R | 自适应 Transformer | Margin+辅助 | 有/少样本 | ✗ |
| UMAEA | 2023 | ISWC | V+T+R | 不确定性加权 | 对比 | 有监督 | ✓ |
| PathFusion | 2023 | ISWC | V+T+R | 图路径迭代 | 对比 | 迭代 | ✗ |
| PCMEA | 2024 | AAAI | V+T+R | 跨模态 | MI+对比 | 半监督 | ✗ |
| DESAlign | 2024 | ICDE | V+T+R | 能量引导 | Dirichlet能量 | 有监督 | ✓ |
| PMF | 2024 | ACL | V+T+R | 渐进冻结 | 一致性对比 | 半监督 | ✓ |
| LoginMEA | 2024 | ECAI | V+T+R | 局部→全局 | 对比 | 有监督 | ✗ |
| CDMEA | 2025 | SIGIR | V+T+R | 因果去偏 | 反事实 | 有监督 | ✓（噪声） |
| M-VAE/GEEA | 2024 | ICLR | V+T+R | 生成 | VAE | 无监督 | ✓ |
