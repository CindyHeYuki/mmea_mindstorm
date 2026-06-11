# EVA: Visual Entity Alignment via Image-guided Graph Inference

**会议**：IJCAI 2021  
**代码**：https://github.com/cambridgeltl/eva

---

## 一句话总结

将实体图像作为辅助信号，通过视觉近邻聚合增强基于关系结构的实体对齐。

---

## 动机

- 传统 EA 方法只用关系结构，对于结构稀疏的实体表现差
- 图像提供了结构无关的跨图语义桥梁（"同一事物长得像"）

---

## 方法

```
视觉编码: ResNet → 图像特征
邻居聚合: 找视觉最近邻实体，聚合其图嵌入
打分融合: score = α * relation_score + (1-α) * visual_score
```

## 对我们的启示

- 奠基性工作，说明了图像对 EA 的价值
- 融合策略过于简单（线性组合），被后续方法大幅超越
- 理解该方法有助于把握多模态 EA 的演进脉络
