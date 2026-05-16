# Algorithm Plan

## Stage 1 Scope

老师已确认第一步先做中文手写体/书法体仿真生成，不先考虑英文、数字和各种标点符号。

因此第一阶段只做：

- 中文字符
- 单书法家
- 标准字形 content 到书法风格 target
- 结构清楚优先
- 能训练、能推理、能生成评估板

## Why Not Classification

分类模型只能回答一张字像哪个书法家，不能根据用户输入的新字生成对应风格的图像。

本项目需要的是生成，不是识别。

## Why Baseline First

先用 U-Net / Pix2Pix 思路跑通 baseline，不是因为它最终一定最强，而是因为它最适合先验证工程链路：

- 数据能否清洗成 paired samples。
- 标准 content 字形能否稳定渲染。
- 模型能否学到基本风格转换。
- 输出是否结构清楚。
- 训练结果能否接入后端。

## Frontend and Preprocessing Boundary

前端可以植入“用户样本预处理”，但不要把“训练数据集预处理”原样暴露给普通用户。

两条流程必须分开：

- 用户样本预处理：前端可展示，属于计划书功能。包括导入样本、去噪、二值化、倾斜校正、字符区域定位、字符分割预览、特征提取。
- 训练数据预处理：后台/云端流程，不放到普通用户前端。包括解压数据集、按书法家统计、生成 content-target pair、train/val split、epoch、batch size、loss、checkpoint、CUDA 日志。

前端普通用户流程建议保持为：

- 导入样本
- 预处理
- 字符分割
- 特征提取
- 输入中文文本
- 选择风格
- 生成笔迹
- 查看结果

后台或实验端可以展示：

- 内置书法样本库
- 书法家风格库
- 样本数量
- 覆盖字符
- 模型版本
- 训练状态
- 固定测试字评估图

远端训练不算“假”。真实系统可以表现为：提交训练任务、云端训练中、训练完成后同步模型。普通用户界面不要出现 `manifest.csv`、`prepare_pix2pix_pairs.py`、`CUDA`、`checkpoint` 这类开发痕迹。

## Known Risk

新算法不是银弹。VQ-Font、MX-Font、U-Net、Pix2Pix 也可能出现复杂字泛化差、风格弱、缺笔多笔、笔画粘连等问题。

降低风险靠的是一整套方法：

- 数据清洗
- 标准字形 content 强约束
- 单风格先跑通
- 固定测试字评估板
- 骨架、轮廓、连通区域、墨迹密度等结构指标
- 坏结果拒绝机制

## Next Algorithms

baseline 跑通后，再把数据迁移到：

- VQ-Font: few-shot font generation, PyTorch, 适合做正式模型候选。
- MX-Font: few-shot font generation, 可作为对照或备选。
- fewshot-font-generation: 统一仓库，可参考 FUNIT、DM-Font、LF-Font、MX-Font 的数据和评估组织。
