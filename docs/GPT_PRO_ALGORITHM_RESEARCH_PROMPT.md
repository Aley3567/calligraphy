# GPT Pro Prompt - Chinese Calligraphy Generation Algorithm Diagnosis

你现在不要把自己当成普通聊天助手，也不要把自己当成只会推荐模型的工程顾问。请你扮演一个同时懂计算机视觉、生成模型、图像到图像翻译、字体生成、中文汉字结构、深度学习优化、损失函数设计和科研论文方法论的算法 professor。

我需要你对我的项目做一次完整的算法会诊。重点不是给我一个“下一步计划”，而是从第一性原理出发，把这个任务到底应该如何建模、为什么当前模型会失败、数学上该怎么约束、训练链路应该怎么设计、评价体系应该怎么定义、哪些路线理论上不成立、哪些路线才有研究价值，完整推导出来。

请你不要迎合我已有想法。你需要像导师答辩前审查学生项目一样，严厉指出错误假设、伪进展、指标陷阱、训练幻觉和工程上看似成功但算法上不成立的地方。

我最关心的是：

```text
完整链路
清晰思路
算法推理
数学建模
loss 设计依据
训练逻辑
评价逻辑
为什么这么做而不是那么做
```

不要只给“可以试试 U-Net / Pix2Pix / diffusion / GAN”。每一个结论都必须回答：

```text
它在数学上约束了什么？
它解决当前哪个具体失败模式？
它可能引入什么副作用？
需要什么数据假设？
怎么验证？
什么情况下必须判失败？
```

---

## 0. 你必须先建立任务本质

请先从数学和任务定义角度回答：

```text
我们到底在学习什么函数？
输入空间 X 是什么？
输出空间 Y 是什么？
条件变量是什么？
风格变量是否显式存在？
汉字结构变量是否显式存在？
训练样本 (x_i, y_i) 的监督关系是否真的成立？
这个任务是 paired image-to-image translation、style transfer、font generation、few-shot generation，还是一个混合问题？
```

当前我们的临时定义是：

```text
x = 标准字体渲染出的汉字 content glyph，256x256 灰度图
y = 数据集中的目标书法字 calligraphy glyph，256x256 灰度图
模型学习 f_theta(x) -> y
```

请你判断这个定义是否成立。如果不成立，请重新定义问题。

你需要明确区分：

```text
content / structure / glyph topology / stroke skeleton / style / ink texture / layout
```

并解释这些变量在当前训练中哪些是显式建模的，哪些只是被模型隐式学习，哪些根本没有被约束。

---

## 1. 项目目标

我要做一个中文书法/字迹仿生生成系统。

阶段 1 老师要求：

```text
先只考虑中文，不考虑英文、数字、标点
先做中文书法体/手写体仿真生成
```

最终系统目标：

```text
用户输入中文文本
系统生成某个书法家/风格下的仿生书法图像
可以在前端展示、保存、后续扩展多风格
```

但当前算法阶段只讨论：

```text
中文单字生成
```

也就是：

```text
给定一个汉字的标准字形图，生成对应风格的书法字图
```

项目最重要的质量要求不是“看起来像一团书法”，而是：

```text
字必须可辨认
结构必须正确
笔画不能被吞
内部留白不能被黑块吃掉
复杂字不能糊成一团
同时要有目标书法风格
```

请把这个目标转成算法约束。

---

## 2. 数据集情况

数据集：

```text
zhuojg/chinese-calligraphy-dataset
```

地址：

```text
https://github.com/zhuojg/chinese-calligraphy-dataset
```

我们使用的是带书法家分类的数据压缩包：

```text
chinese-calligraphy-dataset-with-calligrapher-221030.zip
```

目录结构：

```text
书法家或风格目录 / 汉字目录 / 数字.gif
```

注意：

```text
最底层数字文件名不是标签，只是样本编号
真正的汉字标签来自汉字目录名
```

当前第一阶段只选了一个风格：

```text
楷-赵孟俯三门记
```

审计结果：

```text
约 7200 张图
约 6388 个不同汉字
每个图像对应一个目标书法字
处理后统一成 256x256 灰度图
```

训练 pair 构造：

```text
content 图：用系统字体渲染汉字，例如宋体/明体
target 图：数据集中的真实书法字，经过裁切、灰度化、居中、归一化
```

即：

```text
(standard glyph image, calligraphy glyph image)
```

请你从数据分布角度审查：

```text
这种 pair 是否真的语义对齐？
一个标准字体 glyph 到书法 glyph 是否是一对一映射？
同一汉字只有一个或少量样本时，模型能学到什么？
content 字体和书法 target 的结构差距会造成什么偏差？
如果 target 本身有异体、扫描噪声、裁切偏差、浓墨粘连，监督信号会怎样误导模型？
```

---

## 3. 之前为什么放弃 FontDiffuser

之前尝试过 FontDiffuser 类路线，但出现了严重问题：

```text
缺笔
多笔
偏旁错位
部件变形
拓扑断裂
复杂字不稳
看起来像书法但不一定是那个字
输出变黑、变厚、变脏
多风格互相干扰
背景纸纹污染
英文数字标点不适配
content fallback 一旦出错就崩
refiner/selector 只能修表面，不能保证结构正确
```

我的理解是：

```text
FontDiffuser 更像图像风格生成器，不是汉字结构可验证生成器
```

请你判断这个理解是否正确。

请从 diffusion 的条件控制、随机性、loss、结构先验、汉字拓扑约束角度解释：

```text
为什么普通 diffusion / FontDiffuser 可能在中文书法生成里出现结构不稳？
是否可以通过强 content conditioning 或 ControlNet 类方法解决？
如果可以，需要什么条件？
如果不建议，原因是什么？
```

---

## 4. 已完成实验事实

### 4.1 U-Net + L1 baseline

模型：

```text
U-Net Generator
输入：1 通道 content glyph
输出：1 通道 calligraphy glyph
loss：L1 reconstruction loss
```

训练：

```text
数据：楷-赵孟俯三门记
样本：7200 pair
分辨率：256
epoch：100
batch size：16
GPU：Modal A10G
```

结果：

```text
epoch100:
train_l1 = 0.06861742266902217
val_l1 = 0.1305038692222701
```

但 best.pt 来自更早的 epoch36：

```text
best val_l1 = 0.12746000836292903
```

视觉观察：

```text
大部分简单字能认
有一定毛笔风格
比 FontDiffuser 稳定
但笔画偏黑、偏厚
内部留白被吃
复杂字细节糊成黑块
笔锋不自然
```

请你从 U-Net 的结构、skip connection、L1 优化目标解释：

```text
为什么它能保持基本字形？
为什么它会黑块化？
为什么训练越久可能更黑？
为什么 best 在 epoch36 而不是 epoch100？
这是否说明过拟合、目标偏差、还是 loss 与视觉质量不一致？
```

请写出 L1 loss 的数学形式，并解释其对多模态目标、灰度边缘、墨色变化、结构细节的影响。

---

### 4.2 Pix2Pix 实验

模型：

```text
Generator：U-Net
Discriminator：PatchGAN
Loss：GAN loss + L1 loss
```

目的：

```text
用 PatchGAN 提升局部笔触、边缘和墨色真实感
```

结果：

```text
100 epoch 完成
最终 val_l1 = 0.19153394765324064
```

视觉观察：

```text
明显比 U-Net baseline 差
字形结构破坏
部分字不可辨认
局部更像书法，但整体不像字
墨色更脏
笔画更糊
```

请你从 PatchGAN 的目标函数解释：

```text
D 看到的是局部 patch 真假
它如何推动 G 生成局部更真实的纹理？
为什么这不保证整体汉字结构正确？
为什么局部真实感可能和全局拓扑保真冲突？
```

请写出 Pix2Pix 的典型目标：

```text
G* = arg min_G max_D L_cGAN(G,D) + λ L_L1(G)
```

然后解释为什么这个目标在中文书法任务里可能不够。

请判断 Pix2Pix 当前失败是：

```text
算法本身不适合？
loss 权重不合适？
训练策略不合适？
数据规模/目标不适合？
还是缺少结构约束？
```

如果 Pix2Pix 还有救，请给出严谨改法，不要泛泛说“调参”。

---

### 4.3 U-Net resume 实验

从 U-Net baseline best.pt 继续训练：

```text
起点：epoch36 best.pt
继续训练：50 epoch
实际 epoch：37 -> 86
学习率：5e-5
```

最终：

```text
epoch86
train_l1 = 0.07452412473419567
val_l1 = 0.12922164450089138
```

它没有超过原 baseline best：

```text
baseline best val_l1 = 0.12746000836292903
resume final val_l1 = 0.12922164450089138
```

视觉观察：

```text
继续出现笔画偏黑、内部细节被吞、墨块变重的问题
```

我们已经停止继续训练。

请你判断：

```text
这个实验说明了什么？
为什么继续训练不一定带来质量提升？
为什么“修修补补”会把模型带坏？
如何建立训练 stop rule，避免重蹈 FontDiffuser 式越修越差？
```

---

## 5. 当前最核心的失败模式

请围绕下面失败模式做完整分析：

```text
黑色墨块过重
笔画变厚
内部留白被吃
复杂字细节消失
日、月、田、国、民、夜、耀、翔、龟、鬱、齋 等高风险字容易结构糊掉
val_l1 不能可靠反映结构质量
GAN 提升局部真实感但破坏全局结构
```

请把这些失败模式映射到算法原因：

```text
数据原因
模型结构原因
loss 原因
训练动态原因
评价指标原因
```

---

## 6. 请你给出真正的数学建模方案

请不要只说“加 skeleton loss”。请你完整推导一个合理目标。

我希望你定义：

```text
x: content glyph
y: target calligraphy glyph
y_hat = f_theta(x)
S(.): skeleton or soft skeleton operator
E(.): edge operator
D(.): density or ink statistic
B(.): bbox/shape statistic
C(.): component/topology proxy
```

然后讨论可能的目标函数：

```text
L_total =
  λ_rec L_rec(y_hat, y)
  + λ_edge L_edge(...)
  + λ_skel L_skel(...)
  + λ_density L_density(...)
  + λ_structure L_structure(...)
  + optional λ_adv L_adv(...)
```

你必须说明：

```text
每个 loss 的数学形式
每个 loss 的 target 应该是 y 还是 x
为什么
是否可微
如果不可微如何近似
它解决哪个失败模式
它的副作用是什么
权重初值怎么选
如何做 ablation
```

特别是：

```text
1. skeleton loss 应该约束生成图接近 content skeleton，还是 target skeleton？
2. 如果约束 content skeleton，会不会让结果像普通字体，不像书法？
3. 如果约束 target skeleton，target 本身噪声/书法变形会不会误导？
4. density loss 是全局约束还是局部 patch 约束？
5. edge loss 用 Sobel、Laplacian、Canny、distance transform 哪个更合理？
6. 如何约束内部留白不被吃掉？
7. 如何防止约束过强导致风格消失？
```

请尽量给出公式，而不只是文字。

---

## 7. 请你重新思考算法路线

请比较以下路线，并给出 professor 级别判断：

```text
A. 继续 U-Net，但加入 density/edge/skeleton/structure loss
B. Structure-preserving Pix2Pix
C. Content-skeleton constrained generator
D. Two-stage: skeleton/structure first, style rendering second
E. Diffusion with strong structure conditioning
F. Stroke extraction/vectorization + neural renderer
G. Few-shot font generation / font style transfer methods
H. 基于传统图像处理的骨架保真 + 神经风格化
```

对每条路线请分析：

```text
适合当前数据吗？
需要新增什么标注或预处理？
数学目标清不清楚？
训练难度如何？
能否解释给导师？
能否做本科/硕士项目？
失败风险是什么？
是否能解决黑块/吞笔画问题？
```

最后请你明确推荐：

```text
主路线 1
备选路线 2
明确不建议的路线
```

---

## 8. 数据预处理必须重新审查

请系统分析预处理是否应该改：

```text
target 是否应该二值化？
是否应该保留灰度墨色？
是否应该做 contrast normalization？
是否应该做 stroke thinning？
是否应该提取 skeleton 作为辅助监督？
是否应该用 distance transform 表示结构？
是否应该使用多种 content font 增强？
content font 用宋体、楷体、黑体还是多字体？
target 裁切居中是否破坏比例？
如何处理过黑/过淡/噪声样本？
是否应该过滤不适合训练的 target？
```

请给出一个严谨的数据预处理 pipeline：

```text
raw gif -> normalized target -> auxiliary maps -> train pair
```

每一步说明：

```text
目的
算法
风险
是否保留可逆信息
如何记录 metadata
```

---

## 9. 评价体系必须重建

请设计一个固定评价协议。

不能只看：

```text
val_l1
单张 preview
肉眼感觉
```

请设计：

```text
1. 固定测试字集
2. 训练内字符 vs 未见字符
3. 结构类型分类
4. 高风险字集合
5. 自动指标
6. 人工评价表
7. 升级主模型的门槛
8. 停止训练的规则
```

固定测试字集至少覆盖：

```text
简单字：一、二、三、人、大、口
框结构：日、田、回、国
左右结构：明、湖、海、翔、耀
上下结构：空、夢、意
半包围/包围：闹、建、延、開
复杂字：龟、鬱、齋、鹤
高风险字：民、夜、寝、恢、投、影、制、物
```

自动指标可以考虑：

```text
ink ratio
local ink density
bbox
edge sharpness
skeleton recall
skeleton precision
component count proxy
hole/white-space preservation
SSIM against content structure
distance-transform similarity
```

但请你判断每个指标是否真的可靠。

请设计人工评分维度：

```text
是否可读
是否缺笔/多笔
结构是否正确
内部留白是否保留
风格是否像目标书法
墨色是否过黑
复杂字是否可辨
```

并给出：

```text
什么情况下即使 loss 变好也必须判失败
什么情况下可以升主线
```

---

## 10. 训练策略和科研路线

请给出严谨训练策略，不要给“多试几个”。

要求：

```text
每次只改一个核心变量
每个实验必须有假设
每个实验必须有成功/失败标准
先做小规模验证，再做完整训练
不超过 3 个实验分支
```

请你给出：

```text
实验 1：目的、模型、loss、数据、epoch、成功标准、失败标准
实验 2：目的、模型、loss、数据、epoch、成功标准、失败标准
实验 3：目的、模型、loss、数据、epoch、成功标准、失败标准
```

但注意：不要只给计划。你必须解释为什么这 3 个实验是从前面数学分析自然推出的。

---

## 11. 最终工程链路如何设计

如果算法第一阶段成立，系统最终要落地。

请给出工程链路：

```text
用户输入中文
字符拆分
content glyph 渲染
模型推理
质量检测
失败拒绝或 fallback
单字缓存
文本排版
前端展示
结果保存
```

请说明：

```text
哪些应该由模型做
哪些应该由传统图像处理做
哪些应该由规则系统做
哪些不应该交给生成模型
```

当前老师说第一阶段不考虑英文、数字、标点。  
但后续如果要支持：

```text
英文用衡水体
数字和标点跟随整体毛笔风格
中文标点单独适配
```

请你说明这应该是同一个模型解决，还是 adapter 分流解决。

---

## 12. 请明确告诉我哪些事情不要再做

请列出禁止继续做的方向，例如：

```text
继续无目标加 epoch
继续单纯 Pix2Pix
继续用 val_l1 作为唯一指标
继续凭 preview 决定模型好坏
继续 refiner/patch 一层层补
没有结构评价就启动大训练
没有固定测试集就比较模型
```

请解释为什么这些会导致项目变糟。

---

## 13. 输出格式要求

请按下面结构输出，必须详细：

```text
1. 任务本质和数学建模
2. 数据假设是否成立
3. 已有实验的算法诊断
4. U-Net baseline 的价值和上限
5. Pix2Pix 失败的数学原因
6. 继续训练为什么会黑块化
7. 推荐的主算法路线
8. 具体 loss 公式和每项解释
9. 数据预处理 pipeline
10. 固定评价协议
11. 训练实验设计，不超过 3 个分支
12. 工程落地链路
13. 禁止继续做的方向
14. 给导师汇报的专业表述
```

请你务必给出完整推理链路，不要只给结论。  
请你把重点放在算法理解和数学论证，而不是项目管理。

最后，请给出一个明确判断：

```text
如果你是这个项目的算法导师，你会让学生下一步做什么？
你会禁止学生做什么？
为什么？
```
