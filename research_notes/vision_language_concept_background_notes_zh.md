# 视觉-语言模型概念共享与感知结构：背景研究笔记

这份笔记整理两篇与 thesis 背景密切相关的论文，重点记忆其研究问题、方法、实验设计、主要结论、局限，以及可用于 related work 的连接点。

原始 PDF：
- `Do Vision and Language Models Share Concepts? A Vector Space Alignment Study`
- `Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color`

全文抽取文本：
- `research_notes/pdf_extracts/do_vision_and_language_models_share_concepts.txt`
- `research_notes/pdf_extracts/can_language_models_encode_perceptual_structure_color.txt`

## 1. Do Vision and Language Models Share Concepts? A Vector Space Alignment Study

### 核心问题

这篇论文问的是：纯语言模型和纯视觉模型是否会在各自的表征空间中学到相似的“概念几何结构”？如果语言模型没有直接 grounding，却能和视觉模型的表征空间线性对齐，那么这说明文本训练可能也能诱导某种关于世界结构的内部模型。

它直接回应 Bender and Koller 关于语言模型不能把 utterance 和世界连接起来的论点。作者的逻辑是：语言模型和视觉模型训练在独立模态和数据源上，如果二者表征空间出现结构相似性，那么共同来源很可能是它们都捕捉到了世界中的规律。

### 方法设计

模型范围很广：
- 视觉模型：14 个视觉模型，覆盖 SegFormer、MAE、ResNet 三类。
- 语言模型：14 个 Transformer 语言模型，覆盖 BERT、GPT-2、OPT、LLaMA-2 四类。
- CLIP 也做了附录实验，但正文主要强调纯视觉模型，因为 CLIP 预训练已经用过图文配对数据，不适合作为“独立模态”的强证据。

视觉概念表征：
- 每个概念对应多张图片。
- 用视觉模型 encoder 抽取每张图片的向量。
- 对同一概念的图片向量取平均，得到该概念的视觉表征。
- SegFormer 使用最后 hidden state 的 patch 平均；MAE 使用倒数第二层；ResNet 使用 average pooling 层。

语言概念表征：
- 不是直接把单词孤立输入语言模型，而是在 Wikipedia 句子中找到包含该概念的上下文。
- 提取概念对应 token 的 contextual representation。
- 对多个句子中的表示做平均，得到 decontextualized word representation。
- masked LM 对构成概念的 token 平均；自回归 LM 使用概念中的最后一个 token。

对齐方法：
- 使用 Procrustes analysis 学习从视觉空间到语言空间的正交线性映射。
- 给定视觉矩阵 A 和语言矩阵 B，求正交矩阵 Omega，使 RA 最接近 B。
- 因为关注空间几何是否近似同构，所以作者刻意使用线性/正交映射，而不是复杂非线性映射。
- 如果两个空间维度不同，用 PCA 降维到相同维度。

数据与双模态词典：
- 主要基于 ImageNet-21K 和 CLDI。
- ImageNet-21K 中每个概念类别有多张图片和一个或多个 alias。
- 过滤规则包括：类别有超过 100 张图片，alias 在 Wikipedia 中至少出现 5 次，去除跨类别共享 alias 和有 hyponym 冲突的 alias。
- 重点报告 Exclude-1K，即排除 ImageNet-1K 概念，避免视觉模型预训练类别带来的偏置。
- EN-CLDI 用于测试 nouns 之外的 adjectives 和 verbs。
- 训练/测试按照 class ID 做 70/30 切分，避免同一概念泄漏；做 5 个随机 split 后平均。

评估：
- 学到从视觉到语言的映射后，在 held-out 图像-文本概念对上做 retrieval。
- 目标语言空间被扩展到 79,059 个候选词，包括 ImageNet aliases 和英文 wordlist。
- 主要指标是 P@1、P@10、P@100。
- P@k 是保守指标：P@1 对同义词、变体、粗粒度类别不友好；P@10 和 P@100 更能反映是否落入正确语义邻域。

基线：
- 随机检索：P@1 约 0.0015%，P@100 约 0.1531%。
- length-frequency baseline：用词长和频率构成二维词表征，对齐视觉向量，控制浅层统计信号；P@100 约 0.6053%。
- non-isomorphic baseline：打乱 OPT30B 词和表征的对应关系，制造非同构空间；P@100 约 0.1105%。
- 所有基线 P@100 都低于 1%，而真实视觉-语言映射最高可达约 64%。

### 主要结果

总体发现：
- 语言模型和视觉模型的表征空间几何显著可线性对齐。
- 最小的语言模型也远高于基线。
- 最好结果来自 ResNet152 与 OPT30B，P@100 达到 64.1%。
- 对 P@10，最多约五分之一视觉概念能映射到语言空间中非常接近的邻域。
- P@100 可看作粗粒度语义邻域，结果表明大量视觉概念能被映射到正确语义区域。

模型规模效应：
- 同一语言模型家族中，模型越大，与视觉模型的几何结构越相似。
- scaling 呈近似 log-linear，没有明显饱和。
- 视觉模型规模也有影响，但弱于语言模型规模。

不同视觉模型：
- ResNet 整体最好，其次 SegFormer，再次 MAE。
- 作者认为 ResNet 的分类目标可能引入类别结构 bias。
- MAE 是自监督重建 masked patches，结果虽然不最高，但作为“纯视觉无监督信号也能和语言空间对齐”的证据更强。

词性泛化：
- ImageNet 主要是名词，所以作者用 EN-CLDI 测 adjectives 和 verbs。
- 如果训练和测试都混合 nouns/adjectives/verbs，ResNet152 + OPT30B 的 P@100 可达 81.9%。
- 如果只用 nouns 训练、在 adjectives 或 verbs 上测试，结果下降但仍显著高于基线，说明共享结构不只存在于名词。

影响因素：
- image dispersion：同一概念的图像越分散，越难对齐。低 dispersion 概念通常更容易映射。
- polysemy：词义越多，对齐精度越低。OPT30B + ResNet152 中，单义 alias 的 P@100 为 57.2%，4 个以上词义的 alias 只有 25.4%。
- language dispersion：同一词在不同上下文中的语言表征越分散，对齐越差。
- frequency：低频 alias 反而更容易对齐。作者发现 OPT30B + ResNet152 中，频率排名 100k 以后的 alias P@100 为 64.0%，top 10k 频率词只有 24.7%。这可能是因为高频词更常见、更抽象、更多义，语言表征更混杂。
- dictionary size：用很少的 seed pairs 也能超过基线。仅用 Exclude-1K 词典的 5%，约 297 个训练对，就能学到高于基线的映射。

### 讨论与局限

理论意义：
- 论文认为结果挑战了“语言模型完全不能连接世界”的强说法。
- 如果视觉和语言表征空间已经结构相似，那么只需一个简单线性映射，就可以把图像概念投到语言语义区域。
- 作者称这提示语言模型可能从文本中诱导出某种 internal world model 或 referential semantics 的雏形。

局限：
- 核心对齐仍然是 supervised alignment，需要少量图像-文本对应词典。
- 作者尝试过无监督对齐，但出现 degenerate solution 问题，目前没有可靠算法能完全无监督跨模态对齐。
- 主要证据来自具体概念，尤其 noun-heavy 的视觉概念；verbs 和 adjectives 有扩展但更弱。
- 线性可对齐不等于完整理解，也不等于所有概念子空间都可线性对齐。
- CLIP 附录结果更高，但因为 CLIP 用图文数据预训练，所以不能作为 text-only grounding 争论中的干净证据。

### 对 thesis 的启发

这篇可作为“跨模态概念空间可能共享结构”的强背景：
- 方法上可借鉴 vector space alignment、Procrustes、P@k retrieval、dispersion/polysemy/frequency 分析。
- 写 related work 时可强调它从图像-文本概念层面证明 LM representations 与 VM representations 存在可恢复的结构相似性。
- 如果 thesis 研究 perceptual/conceptual grounding，可以把它放在“语言模型是否能从纯文本中学到世界结构”的核心证据链里。
- 注意措辞要谨慎：它证明的是 alignability 和 structural similarity，不是证明语言模型拥有完整 embodied understanding。

## 2. Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color

### 核心问题

这篇论文问的是：没有视觉 grounding 的语言模型，是否能在颜色词表征中编码人类感知颜色空间的拓扑结构？

颜色是一个很好的 case study，因为：
- 颜色具有明确的感知空间结构。
- CIELAB 空间中的距离与人类感知差异相关。
- 颜色词既是语言对象，又与视觉感知高度相关。
- 如果语言模型能仅从文本中学到接近 CIELAB 的结构，说明感知结构可能部分反映在语言使用中，并可被分布式表征学习到。

### 数据与表征

颜色数据：
- 使用 Color Lexicon of American English。
- 数据包含 330 个 Munsell color chips。
- 每个 chip 有 51 个单词颜色命名判断。
- free-naming task 产生 122 个 monolexemic color terms。

感知颜色空间：
- 每个颜色 chip 映射到 3D CIELAB 空间。
- L 维表示 lightness。
- A 维表示 red-green 方向。
- B 维表示 blue-yellow 方向。
- CIELAB 距离用于近似颜色之间的人类感知差异。

语言模型：
- BERT large uncased。
- RoBERTa large。
- ELECTRA large discriminator。

基线：
- PMI statistics：在 Common Crawl 上用窗口大小 1、2、3 计算颜色词共现向量。
- FastText word-type embeddings trained on Common Crawl。

颜色词表征抽取：
- NC：no context，把颜色词单独输入模型，例如 `[CLS] red [SEP]`。
- RC：random context，从 Common Crawl 随机抽包含颜色词的句子，编码后平均。
- CC：controlled context，用模板生成相同结构的句子，控制上下文变化。

controlled context 模板：
- COPULA: `the <obj> is <col>`
- POSSESSION: `i have a <col> <obj>`
- SPATIAL: `the <col> <obj> is there`

对象词来自 McRae 数据集中可出现在多种颜色中的对象，覆盖 13 个类别。最终组合为 3 x 122 x 18 = 6588 个句子，每个颜色词 366 个句子。

### 方法一：Representation Similarity Analysis

RSA 用来比较两个空间的整体拓扑结构，而不是训练一个 probe。

步骤：
- 对每个颜色词，计算其在 CIELAB 空间中的 centroid：所有被该词命名的 color chips 的平均 CIELAB 坐标。
- 只保留在 color lexicon 中频率超过 100 的颜色词，最终 18 个常用颜色词。
- 分别构建语言表征空间和 CIELAB 空间的 N x N representational similarity matrix。
- 语言空间中使用 Pearson correlation 衡量颜色词 embedding 之间相似性。
- CIELAB 空间中使用基于 CIELAB distance 的心理学相似度函数。
- 计算两个 RSM 之间的 Kendall rank correlation tau。
- 也计算每个颜色词对应 row 的 tau，分析单个颜色的对齐差异。

RSA 的优点：
- 非参数。
- 关注整体关系结构。
- 避免部分 probing 方法可能通过复杂模型学到任务而非表征信息的问题。

### 方法二：线性映射

线性映射用于测试颜色词 embedding 是否能通过简单变换预测 CIELAB 坐标。

步骤：
- 训练 regularized linear regression，从颜色词 embedding 空间 X 映射到 CIELAB 空间 Y。
- 目标函数包含 L2 预测误差和 L1 lasso regularization。
- 通过改变 alpha 控制 probe complexity。
- 用 nuclear norm 衡量投影矩阵复杂度。
- 6-fold cross-validation 评估。
- 对每个 Munsell chip，在语言空间中根据 51 个命名词的加权平均得到 chip 的 embedding centroid。
- 评估 explained variance 和 predicted color code 的 ranking。

控制任务：
- 随机交换每个 color chip 的 CIELAB code，破坏颜色词和感知坐标的真实对应。
- 线性映射控制任务做 10 次 random remapping。
- 报告 selectivity，即真实任务 explained variance 减去控制任务 explained variance。
- RSA 也做随机 shuffle CIELAB centroid 的控制。

### 主要结果

RSA：
- 多数 LM 层在各种配置下都与 CIELAB 有显著拓扑对齐。
- 随机打乱 CIELAB centroids 后，RSA 相关都不显著，支持对齐不是随机产生的。
- static embeddings 和 PMI baseline 也有一定对齐，说明颜色感知结构的一部分确实反映在分布统计里。
- LM 表征通常比简单 PMI 和 FastText 更强。
- NC 的跨层方差较大，平均相关接近 0；这符合预期，因为模型很少在预训练中看到孤立颜色 token。
- CC 通常表现更稳定，因为控制了上下文噪声。

线性映射：
- 所有模型在真实任务上都有高 selectivity，说明 probe 的拟合不是靠记忆随机标签。
- CC 设置在线性映射中最高，说明控制上下文能更清晰地暴露颜色结构。
- BERT/RoBERTa/ELECTRA 的线性映射 selectivity 大致在 0.73-0.78 左右。
- baseline 的 selectivity 也有 0.70 左右，但 LM 尤其 controlled context 配置更高。

颜色子空间：
- 预测 CIELAB 的信息存在于较低维子空间中。
- 约 10 个 embedding 维度就可解释约 0.4 的 CIELAB variance。
- 30-40 个维度可解释约 0.7 的 variance，接近完整表征表现。

模型规模：
- BERT-mini、small、medium、base 的实验显示，模型越大，对齐越强。
- 这与第一篇论文中的 scaling 发现相呼应：更大语言模型的表征几何更接近外部/感知结构。

颜色温度差异：
- warm colors，例如 yellow、orange、red，平均比 cool colors，例如 blue、green，更好地与 CIELAB 对齐。
- 作者将其连接到 color naming communication efficiency 文献：warm colors 往往是行为相关的 salient objects 的颜色，cool colors 更常作为背景色出现，因此 warm colors 在语言中可能被更有效地交流。

surprisal 分析：
- 使用 Gibson et al. 的 listener surprisal 框架。
- 一个 chip 的 surprisal 越高，说明说话者对它的命名共识越低，听者需要更多猜测。
- 论文发现更高 surprisal 与更差的预测 ranking 显著相关。
- 这说明语言模型中的颜色结构与人类颜色命名的一致性和交流效率有关。

语料统计分析：
- 作者提取颜色词在 Common Crawl 中的 log frequency、POS、dependency relation、head lemma、PMI entropy 等特征。
- 用 Linear Mixed Effects Model 预测 RSA score，model type 作为 random effect。
- pmi-col 显著改善模型拟合，且系数为负：颜色词越常出现在固定 collocation 中，越不对齐感知空间。例子如 Red Army、Black Death，这类用法常是隐喻或专名，不是颜色描述。
- dependency relation entropy 和 head entropy 为正：颜色词出现在更多样的句法关系中、修饰更多样的 head 时，对齐更好。
- amod-prop 为正：颜色词越常作为 adjectival modifier，对齐越强。
- copula construction 比例不显著，因此类似 `banana is yellow` 的断言式句子频率并没有明显带来额外优势。

多模态模型：
- 作者初步测试 VisualBERT 和 VideoBERT，没有发现相比 text-only 模型的明显优势。
- 这说明对颜色词的拓扑编码不一定需要显式图文预训练，至少在该实验设置下如此。

### 结论与局限

主要结论：
- text-derived color term representations 与人类感知颜色空间 CIELAB 之间存在可测的拓扑对应。
- 简单共现统计已包含一部分信号，语言模型表征更强。
- 语言中的使用模式和上下文统计可能携带感知结构。
- 颜色词的 alignment 不是均匀的：warm/cool、surprisal、collocation、句法功能都会影响对齐程度。

局限：
- 研究只覆盖颜色领域，不能直接推广到所有感知结构。
- RSA 中常用颜色词只有 18 个，样本量较小。
- CIELAB 是感知空间的近似，不等于完整人类颜色经验。
- text-only 对齐不等于 embodied grounding；它说明感知结构可在语言统计中留下痕迹。
- 完全同构不太可能，因为颜色词并非纯指称，语言还受文化、隐喻、语境和使用习惯影响。

### 对 thesis 的启发

这篇可作为“语言模型能否编码感知结构”的直接背景：
- 如果 thesis 涉及 color、perception、grounding、distributional semantics，它是核心引用。
- 方法上可借鉴 RSA、linear mapping、controlled/random/no context 表征抽取、control task、selectivity、surprisal 和 corpus usage analysis。
- 它比一般 probing 更关注空间结构，而不是单点属性分类，因此适合支撑“conceptual geometry / perceptual topology”相关论点。
- 写作时可强调：语言模型对感知结构的编码可能不是神秘出现的，而是来自语言中反复沉积的感知和交际规律。

## 两篇论文的共同主线

共同研究问题：
- 纯文本训练的语言模型是否能学到超出形式统计的世界/感知结构？
- 如果能，这种结构是否表现为 embedding space 的几何相似性？

共同方法思想：
- 不只问“模型知道某个事实吗”，而是问“模型表征空间中的关系结构是否与外部结构同构或可对齐”。
- 使用几何对齐而非单纯分类 probing。
- 强调 linear mapping / Procrustes / RSA，因为简单映射更能说明原始表征本身已经包含结构。
- 使用控制实验或基线排除随机、浅层统计、复杂 probe 记忆等解释。

共同发现：
- 语言模型表征确实与外部结构存在部分对齐。
- 对齐随模型规模增强。
- 对齐程度受概念本身的歧义、分散度、频率、上下文使用方式影响。
- 语言中的统计结构可能携带由人类感知和世界规律投射出来的信息。

重要区别：
- 第一篇比较语言模型和视觉模型，外部参照是视觉模型学到的概念空间。
- 第二篇比较语言模型和人类感知颜色空间，外部参照是 CIELAB。
- 第一篇使用 image-text concept dictionary + Procrustes + retrieval P@k。
- 第二篇使用 color lexicon + RSA + regularized linear mapping + selectivity。
- 第一篇重点是跨模态概念共享；第二篇重点是 text-only 表征中的感知拓扑。

可用于 related work 的综述句：

Recent work has shifted from asking whether language models encode isolated facts or attributes to asking whether their representational geometries align with independently defined perceptual or visual spaces. Abdou et al. show that text-derived color term representations partially recover the topology of human perceptual color space, while Li et al. demonstrate that language model representations can be linearly aligned with visual model representations across a broad range of concepts. Together, these studies suggest that distributional language learning may capture structured traces of perceptual and world knowledge, although such alignment remains partial, domain-dependent, and sensitive to ambiguity, frequency, and contextual usage.
