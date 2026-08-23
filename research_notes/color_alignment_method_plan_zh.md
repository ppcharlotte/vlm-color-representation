# VLM 颜色跨模态对齐与分类表现：方法操作构思

本文档将项目方法压缩为可执行方案，用于后续实验设计、分析脚本规划和论文方法部分起草。

## 1. 核心问题与假设

研究问题：

视觉-语言模型中，颜色图像表征与颜色文本表征的跨模态对齐程度，是否能够解释并改善颜色分类表现？

两个层次的假设：

- 描述性假设：颜色图像表征与颜色文本表征越对齐，颜色分类准确率越高。
- 因果性假设：通过干预改善图像-文本颜色对齐，会提高 held-out 颜色分类准确率。

谨慎表述：

Cross-modal alignment is one mechanism underlying color classification performance, rather than the sole determinant of performance.

也就是说，颜色分类还可能受到视觉编码质量、文本原型质量、语言先验、prompt、物体语义、类别边界、照明条件和解码过程影响。本项目要检验的是：在控制这些因素后，alignment 是否仍然有独立贡献。

## 2. Alignment 的操作化定义

不要只使用匹配图像和文本之间的平均 cosine similarity。该指标容易受 embedding anisotropy、模态整体偏移和向量范数影响。建议同时报告三类 alignment。

### 2.1 配对对齐：instance-level alignment

对每个颜色图像样本 `x_i` 和正确颜色词 `t_yi`，计算正确文本原型相对于错误文本原型的 margin：

```text
A_pair = (1 / N) sum_i [
  cos(v_i, t_yi) - (1 / (K - 1)) sum_{j != yi} cos(v_i, t_j)
]
```

解释：

- 该指标直接对应 zero-shot text-prototype classification。
- 如果正确颜色词与图像 embedding 的相似度没有高于其他颜色词，模型就很难通过 cosine-based zero-shot 方法正确分类。

### 2.2 几何结构对齐：relational alignment

分别构建图像颜色空间和文本颜色空间的 Representational Dissimilarity Matrix：

```text
D^V_ij = 1 - cos(vbar_i, vbar_j)
D^T_ij = 1 - cos(t_i, t_j)
```

其中：

- `vbar_i` 是第 `i` 个颜色类别的图像 centroid。
- `t_i` 是第 `i` 个颜色名称的文本 embedding。

建议报告：

- Spearman RSA
- Kendall's tau
- CKA
- Procrustes alignment error
- Mantel 或 permutation test

解释：

该分析回答的问题是：如果 red 在视觉空间中接近 orange、远离 blue，那么文本空间是否也保留了相似的关系结构？

### 2.3 局部类别对齐：class-wise alignment

全局 RSA 可能掩盖局部类别问题。因此为每个颜色类别 `c` 计算：

```text
A_c = corr(D^V_c,., D^T_c,.)
```

然后检验：

```text
A_c -> Accuracy_c
```

解释：

如果 yellow 的视觉邻域和文本邻域不一致，它是否更容易被误判成 orange、green 或 white？这个类别级分析尤其重要，因为基础颜色类别通常只有约 11 个，单次 global RSA 的统计功效较低。

## 3. 数据设计

建议使用三个层级的数据，避免结论只适用于纯色 patch。

### Dataset A：受控纯色刺激

来源可包括：

- Munsell color chips
- CIELAB 均匀采样
- World Color Survey 或类似人类颜色命名数据
- 11 个 basic color categories

控制变量：

- patch 大小
- 背景颜色
- 亮度和显示条件
- 色彩管理
- 图像编码格式

用途：

测量模型是否真正编码颜色，而不是依赖物体语义。

### Dataset B：合成 object-color 数据

构造同一物体在不同颜色下的反事实样本，例如：

- red car
- blue car
- green car

实验要求：

- object 固定
- shape 固定
- background 固定
- lighting 固定
- 只改变目标物体颜色

用途：

检验模型是在识别颜色，还是依赖 banana-yellow、grass-green 这类语义共现。

### Dataset C：自然图像

包含真实场景中的颜色目标，例如：

- 衣服
- 汽车
- 玩具
- 食物
- 家居产品

标注字段：

- 主色
- 目标区域或 mask
- illumination
- saturation
- object category
- background complexity

用途：

检验方法是否能从受控颜色刺激泛化到真实图像。

## 4. 模型与层级设计

不要只研究一个 VLM 的最后一层。需要制造足够多的模型间和层级间 variation。

模型范围：

- CLIP / OpenCLIP 变体
- 不同 vision encoder
- 不同 text encoder
- generative VLM，例如 LLaVA 类模型
- 不同模型规模
- 不同训练数据来源

每个模型建议提取：

- vision encoder 各层 hidden states
- projection 前后的 visual embedding
- text encoder 各层 hidden states
- multimodal projector 输出
- LLM 中与 image tokens 对应的 hidden states
- 输出颜色词前一层 hidden state

需要区分四个问题：

1. 视觉空间自身是否有颜色可分性。
2. 视觉空间与文本颜色空间是否对齐。
3. multimodal projector 是否改善或破坏这种对齐。
4. 最终语言输出是否忠实使用视觉信息。

## 5. 颜色分类任务

至少分成三种任务，以区分 alignment 问题和 visual representation 问题。

### 5.1 Zero-shot text-prototype classification

分类规则：

```text
y_hat = argmax_c cos(v(x), t(c))
```

prompt 条件：

- `red`
- `the color red`
- `a red color patch`
- `this image is red`
- prompt ensemble

用途：

该任务对跨模态 alignment 依赖最大，因此假设应在这里表现最明显。

### 5.2 Image-only linear probe

冻结 visual embedding，训练线性分类器：

```text
y_hat = W v(x)
```

解释逻辑：

- linear probe 高、zero-shot 低、alignment 低：视觉空间包含颜色信息，但没有正确对齐到颜色词。
- linear probe 低、zero-shot 低：问题更可能在视觉编码本身，而不是 alignment。
- linear probe 高、zero-shot 高：视觉编码和跨模态读取都相对有效。

### 5.3 Generative color naming

给 VLM 输入图像，要求只回答颜色名称。

评估指标：

- exact match
- 11-class normalized accuracy
- top-k accuracy
- macro-F1
- confusion matrix
- CIELAB error
- semantic distance error

边界颜色不要只使用硬标签。若有人类颜色命名分布 `p_human(y | x)`，同时报告：

- KL divergence
- cross-entropy
- expected CIELAB distance

## 6. 核心统计检验

### Study 1：跨模型相关分析

对每个模型 `m` 计算：

- `A_m`：alignment
- `S_m`：visual separability
- `T_m`：text separability
- `Acc_m`：color accuracy

回归模型：

```text
Acc_m =
  beta_0
  + beta_1 A_m
  + beta_2 S_m
  + beta_3 T_m
  + beta_4 ModelSize_m
  + beta_5 Dataset_m
  + epsilon_m
```

关键检验：

`beta_1` 在控制视觉可分性、文本可分性、模型规模和数据集后是否仍然显著。

### Study 2：类别层面的 mixed-effects model

建议使用 logistic mixed-effects regression：

```text
Correct_imc ~
  Alignment_mc
  + VisualSeparability_mc
  + ColorBoundary_i
  + Lightness_i
  + Saturation_i
  + (1 | Model)
  + (1 | Color)
```

变量含义：

- `i`：样本
- `m`：模型
- `c`：颜色类别

目的：

避免只在 11 个颜色类别上做低功效 Pearson correlation。

### Study 3：误差结构分析

检验图像-文本几何关系是否预测 confusion matrix：

```text
P(y_hat = j | y = i) proportional to exp(-alpha D^T_ij)
```

或使用图像空间距离、文本空间距离及二者交互项共同预测具体错误方向。

核心问题：

模型把 purple 错认成 blue 的频率，是否能由 purple-blue 在视觉空间和文本空间中的邻近关系预测？

## 7. 因果干预实验

这是项目从 association 走向 causal evidence 的关键。

### 7.1 Intervention A：只调整文本空间

固定视觉编码，只学习 text prototypes 的线性映射：

```text
W* = argmin_W || V W - T ||^2
```

或使用 orthogonal Procrustes：

```text
W* = argmin_{W^T W = I} || V W - T ||_F^2
```

然后使用变换后的文本原型分类：

```text
y_hat = argmax_c cos(v(x), W t_c)
```

比较：

- 原始 alignment
- 调整后 alignment
- 原始 accuracy
- 调整后 accuracy

必要控制：

- train/test color chips 分离
- held-out shades
- leave-one-color-family-out
- 防止映射直接记住标签

### 7.2 Intervention B：correct versus sham transformation

设置三个条件：

1. Aligned transformation：优化正确 image-text correspondence。
2. Random orthogonal transformation：保持距离结构但破坏跨模态对应。
3. Label-permuted alignment：把 red 对齐到 blue、blue 对齐到 green 等错误对应。

强证据模式：

- correct alignment 提高 accuracy。
- random mapping 不提高 accuracy。
- permuted mapping 导致可预测的类别交换。

如果把 red visual centroid 对齐到 blue text prototype 后，模型更频繁输出 blue，则说明分类行为确实由跨模态 correspondence 驱动。

### 7.3 Intervention C：fine-tuning alignment loss

在少量训练数据上只训练 projection layer：

```text
L = L_contrastive + lambda L_geometry
L_geometry = || D^V - D^T ||_F^2
```

比较条件：

- classification-only tuning
- contrastive alignment tuning
- structural alignment tuning
- random 或 shuffled-pair tuning

关键结果：

如果 structural alignment 在 held-out shades 和 natural images 上带来更好泛化，说明改进不只是记住训练类别标签。

## 8. 替代解释与控制变量

必须显式控制以下因素。

视觉可分性：

- silhouette score
- Fisher discriminant ratio
- inter/intra-class distance
- image-only linear probe accuracy

文本原型质量：

- 单一颜色词
- prompt ensemble
- 同义词
- gray/grey normalization
- basic term versus descriptive prompt

类别原型性：

- 到类别 centroid 的 CIEDE2000 distance
- saturation
- lightness
- human naming entropy

物体语义偏差：

- pure patches
- color-swapped objects
- canonical object-color pairs
- counterfactual pairs，例如 blue banana、green strawberry

语言先验：

- image removed
- grayscale image
- shuffled image
- text-only baseline

## 9. 支持假设的预注册式判据

建议预先规定：以下四项至少满足三项，才认为结果支持 alignment hypothesis。

1. 跨模型证据：alignment 显著预测 color accuracy，并且在控制 visual/text separability 后仍成立。
2. 类别级证据：alignment 较差的颜色类别具有更低 accuracy 和更高 confusion。
3. 误差结构证据：image-text geometry 能预测具体会错成哪个颜色。
4. 干预证据：正确提高 alignment 能提高 held-out accuracy；shuffled 或错误 alignment 不提高，甚至产生可预测错误。

其中第 4 项最重要。

## 10. 从当前 pilot 到完整项目的执行顺序

当前 pilot 条件：

- 11 个 basic colors
- 每类约 5 个 patches
- 已有 image embeddings
- 已有 text prototypes
- 已有 RSA
- 已有 per-color accuracy

当前局限：

- 图像总量约 55，统计功效低。
- 每类样本太少。
- 颜色类别只有 11 个。
- accuracy 可能接近 ceiling。
- global RSA 只有一次观测。
- 难以区分 alignment 与 visual separability。

下一步执行顺序：

1. 扩展到至少 500-1000 个 CIELAB chips。
2. 为每个 chip 获取硬标签或人类颜色命名软标签。
3. 计算 instance-level、global relational、class-wise alignment。
4. 比较多个 VLM、多个层、projection 前后表征。
5. 同时运行 zero-shot text-prototype classification 和 image-only linear probe。
6. 加入 generative color naming，并建立 normalized 11-class evaluation。
7. 计算 visual separability、text separability、prototype typicality 和 language prior baselines。
8. 运行 Procrustes correct、random orthogonal、label-permuted 三种干预。
9. 使用 mixed-effects regression 检验 alignment 的独立贡献。
10. 用 confusion matrix 分析验证错误方向是否由跨模态几何预测。

## 11. 最小可行实验版本

如果时间有限，优先完成以下最小版本：

1. Dataset A：500-1000 个 CIELAB 或 Munsell chips。
2. 模型：至少 4-6 个 CLIP/OpenCLIP 变体。
3. 任务：zero-shot text-prototype classification + image-only linear probe。
4. Alignment：pairwise margin + RSA + class-wise RSA。
5. 控制：visual separability + prompt ensemble。
6. 干预：orthogonal Procrustes correct/random/permuted。
7. 统计：跨模型回归 + 类别级 mixed-effects model。

这个版本已经能够回答核心问题：颜色分类差异是否只是视觉可分性差异，还是跨模态颜色结构对齐也有独立作用。

## 12. 论文方法部分的核心论证句

可作为论文引言或方法动机的基础表述：

VLM 的 zero-shot color classification depends not only on whether visual embeddings encode chromatic distinctions, but also on whether those distinctions are mapped consistently onto the relational structure of color terms in the text space. Therefore, stronger cross-modal structural alignment should facilitate the correct association of visual color representations with linguistic color labels.

更谨慎版本：

We hypothesize that cross-modal alignment is one mechanism underlying color classification performance, rather than the sole determinant of performance.

## 13. 推荐输出物

实验完成后建议整理以下结果表和图：

- 模型 × 数据集 × 任务的 accuracy 表。
- 模型 × 层级的 alignment heatmap。
- visual separability、alignment、accuracy 的 partial regression plot。
- class-wise alignment 与 class-wise accuracy 的散点图。
- zero-shot 与 linear probe 的对照表。
- correct/random/permuted Procrustes 干预前后 accuracy 图。
- confusion matrix 与颜色空间距离的预测分析。
- held-out shades 与 natural images 的泛化结果。

