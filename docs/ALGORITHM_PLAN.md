# Algorithm Plan

## Stage 1 Scope

老师已确认第一阶段先做中文书法/手写体仿真生成，不先考虑英文、数字和标点。

当前任务定义：

```text
weakly paired, single-style, structure-preserving Chinese calligraphy glyph generation
```

即：

```text
标准字体 content glyph -> 单风格书法 glyph
```

但这个定义只是工程入口，不是完整算法目标。完整目标必须显式保护：

```text
汉字结构
笔画骨架
内部留白
墨色密度
复杂字可读性
```

## Current Model Status

保留为当前主基线：

```text
U-Net L1 baseline best.pt
baseline best val_l1 = 0.12746000836292903
```

不进入主线：

```text
Pix2Pix:
  val_l1 = 0.19153394765324064
  reason = local realism damaged global glyph structure

U-Net resume:
  val_l1 = 0.12922164450089138
  reason = did not beat baseline best and continued dark/thick stroke risk
```

## Why Training Is Paused

继续裸 U-Net + L1、继续加 epoch、继续 Pix2Pix 都已经暴露同一个问题：

```text
loss 优化不等于汉字结构变好
```

当前最危险的失败模式：

```text
笔画变黑变厚
内部留白被吃
复杂字糊成黑块
局部像书法但整体不像字
```

因此后续不能再以 `val_l1` 或单张 preview 作为主判断依据。

## Next Valid Algorithm Direction

下一阶段只允许做结构化底座：

```text
1. preprocessing:
   clean mask
   skeleton
   distance transform
   hole map
   metadata

2. evaluation:
   fixed high-risk character board
   seen / unseen split
   structure-type groups
   ink density
   hole preservation
   skeleton recall / precision

3. model:
   structure-aware U-Net
   mask head + ink head
   gray + mask + edge + density + hole + bbox loss
```

## Forbidden For Now

```text
do not continue U-Net L1 epochs
do not run plain Pix2Pix
do not retry FontDiffuser / diffusion as mainline
do not stack refiner or selector patches
do not compare models without fixed evaluation
do not start full training before smoke validation
do not mix multiple styles before single-style structure is stable
```

## Frontend Boundary

前端可以展示用户样本预处理：

```text
导入样本
去噪
二值化
倾斜校正
字符区域定位
字符分割预览
特征提取
```

训练数据预处理和模型训练不放普通用户前端：

```text
数据集解压
train/val split
loss
epoch
batch size
checkpoint
CUDA logs
```

系统最终可以把云端训练包装成：

```text
模型版本
风格库
评估图
训练状态
```

而不是暴露底层训练细节。
