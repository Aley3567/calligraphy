# Algorithm Plan

## Stage 1 Scope

老师已确认第一阶段范围是中文书法/手写体仿真生成。

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

## Why Evaluation Comes First

当前关键问题是：

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

因此下一步先建设固定结构评估和结构辅助图，再接入结构保真模型。

## Next Valid Algorithm Direction

下一阶段主线是结构化底座：

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
