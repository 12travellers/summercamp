# Learning-based Motion Generating

## 前置任务

本任务假设大家已经熟悉`BVH`文件格式，以及完成一些基本的动作编辑任务。
> 如果不会请参考[GAMES105](https://games-105.github.io/)和lab1/lab2 的part1，注意本任务基本只会使用part1,其余part可以根据兴趣完成。课程视频在[此链接](https://www.bilibili.com/video/BV1GG4y1p7fF)

## 任务目标

你的任务是训练一个可以生成随机动作的神经网络。

具体来讲，我们会提供一个长BVH(walk1_subject5.bvh)，其中包含了许多走路的动作，你需要利用此数据训练一个能生成不同走路动作的网络。

基本要求：
- 生成的动作连续，自然
- 实现某种程度的随机(生成的动作不能完全是原始动作的子序列)

可以进一步实现的:
- 让根节点位移更加合理，减少人的漂浮感
- 使用IK或者加训练loss的方法，让脚和地面的接触更加合理，减少打滑感
- 交互式生成，可以利用手柄控制角色的前进方向

可以参考的文章与方法:
- MotionVAE:Character Controllers Using Motion VAEs
  - github: https://github.com/electronicarts/character-motion-vaes
- PFNN: Phase-Functioned Neural Networks for Character Control
  - github: https://github.com/sreyafrancis/PFNN
- MDM: Human Motion Diffusion Model
  - github: https://github.com/GuyTevet/motion-diffusion-model
- Learned Motion Matching:
  - github: https://github.com/pau1o-hs/Learned-Motion-Matching

以上方法都可以达到所需目标(MDM较难实现交互控制)。从实现角度讲MotionVAE方便进行随机生成，PFNN更容易进行交互控制，MDM需要一定数学基础但是容易随机生成，Learned Motion Matching适合已经完成GAMES105 Lab2 的Motion Matching部分的同学。

## 辅助工具

我们提供一个辅助工具包`pymotionlib`，它能够对BVH数据进行：
- 读入与输出
- 简单编辑(改变采样频率，计算速度角速度等等)
- 可视化

用法简单介绍可以参考pymotionlib的readme



