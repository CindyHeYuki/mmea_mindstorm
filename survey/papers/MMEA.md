# MMEA: Multi-Modal Entity Alignment

**会议**：EMNLP 2022  
**代码**：https://github.com/liyichen-cly/MMEA

---

## 一句话总结

第一个系统性地将视觉、文本、关系三种模态统一引入实体对齐的工作，构建了标准 benchmark（FB15K-DB15K、FB15K-YG15K）。

---

## 贡献

1. **数据集**：构建并发布了 MMEA 标准 benchmark（沿用至今）
2. **三模态框架**：V+T+R 联合 GNN 嵌入
3. **基础 baseline**：后续所有方法均以此为比较基准

---

## 方法

```
视觉: ResNet → 平均池化 → V_emb
文本: fastText/BERT → T_emb  
关系: GCN → R_emb
融合: concat([V_emb, T_emb, R_emb]) → MLP → 对齐
```

---

## 局限

- 早期融合（concat），模态间无深度交互
- 用 margin loss，负样本利用不充分
- 无模态权重学习

这些局限奠定了 MEAformer、MCLEA 等后续工作的出发点。
