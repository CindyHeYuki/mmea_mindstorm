# MCLEA: Multi-modal Contrastive Learning for Entity Alignment

**会议**：EMNLP 2022  
**作者**：Zhenxi Lin, Ziheng Zhang, Meng Wang, Yinghui Shi, Xian Wu, Yefeng Zheng  
**代码**：https://github.com/lzxlin/MCLEA

---

## 一句话总结

将对比学习引入多模态实体对齐，通过模态内和跨模态对比损失同时对齐实体表示和各模态表示。

---

## 动机

- 之前方法（MMEA）用 margin loss，负样本利用不充分
- 对比学习在视觉-语言预训练（CLIP）中效果显著
- 如何设计 EA 任务专用的多粒度对比学习？

---

## 方法

### 关键设计

**1. 多粒度对比学习**
- **模态内对比**：同一实体的对齐表示在同一模态空间内靠近
- **跨模态对比**：同一实体的不同模态表示互相靠近（模态一致性）
- **跨图对比**：不同 KG 中对齐实体对的对比

**2. 联合嵌入空间**
- 将所有模态投影到统一空间
- 用 InfoNCE 损失优化

### 损失函数

```
L = λ1 * L_intra + λ2 * L_inter + λ3 * L_cross_graph
```

---

## 实验结果

| 数据集 | Hits@1 | Hits@10 | MRR |
|--------|--------|---------|-----|
| FB15K-DB15K | 67.3 | 88.8 | 0.745 |
| FB15K-YG15K | 67.1 | 87.4 | 0.742 |

---

## 对我们的启示

1. 对比学习框架是 MEAformer margin loss 之外的有效替代
2. 跨模态对比可以作为辅助任务加入我们的方法
3. 困难负样本挖掘在对比框架下更自然

---

## 局限

- 缺乏模态间的深度交互（MEAformer 的 Transformer 跨模态更强）
- 未处理模态噪声和缺失
