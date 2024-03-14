# PhyGrasp: Generalizing Robotic Grasping with Physics-informed Large Multimodal Models
2024

LLM + 3d 点云做输入，使得模型能够理解一物体易碎部分，避免机器人损坏易碎物品

相关工作
- **physical reasoning**
  - 使用interaction data做输入，预测physical property
    - [2015 NIPS] Galileo: Perceiving physical object properties by integrating a physics engine with deep learning
    - [2016] Physics 101: Learning physical object properties from unlabeled videos
    - [2020] Visual grounding of learned physical models
  - 学习包含物理信息的特征，不直接预测物理信息
    - [2019 ICLR] Reasoning about physical interactions with objectoriented prediction and planning
    - [2019 RSS] Densephysnet: Learning dense physical object representations via multi-step dynamic interactions
    - [2021 NIPS] Dynamic visual reasoning by learning differentiable physics models from video and language
    - [2020] Learning long-term visual dynamics with region proposal interaction networks
    - [2022] Comphy: Compositional physical reasoning of objects and events from videos
    - [2021] Physion: Evaluating physical prediction from vision in humans and machines
    - [2024] Physion++: Evaluating physical scene understanding that requires online inference of different physical properties
  - 使用llm或vlm进行physical reasoning
    - [2023 ICLR] Mind’s eye: Grounded language model reasoning through simulation
    - [2023] Can language models understand physical concepts?
    - [2023] Physically grounded vision-language models for robotic manipulation

模型
- PhyPartNet
  - ![](./note_images/PhyPartNet.png)
  - 得到一物体点云，对每一位置预测材料 易碎性 质量 密度 摩擦力信息
  - 使用以上信息对一物体生成Affordance Map，代表每一物体部分可以用于抓取的几率
    - 使用analytical方法，得到物体物理属性的gt，条件为限制一point上的压强等，生成Affordance Map
      - 图中analytical输入的文本 即给定的物理属性gt，为
  - vision encoder：
    - ![](./note_images/PhyPointNext-vision.png)
    - 使用预训练PointNeXt抽点云全局特征(1024维度) 并对每一point抽局部特征(64维度)
    - mlp得到每一point特征为[nlp文本全局特征, 点云全局特征, point 64维特征, point坐标]，对每一point预测 [特征, affordance值]
      - 对应的affordance pair特征应该相近
      - affordance值和analytical 得到的affordance map应相近
- 使用llm理解指令，生成期望的robot pose，通过analytical方法得到robot每一joint应旋转的角度

# Visual Grounding of Learned Physical Models
2020

模型
- 输入为多帧连续图像$O_t$，模型预测几何结构，预测后续帧的场景变化
- cnn + linear layer模型，$f_V(O) = (X, G)$
  - $X = \{(x, y, z)\}^{N}$ 为一帧图像中point的三维坐标
  - $G = \{m\}^{N}$ 为每一point的类别标签，标明point属于哪一物体
  - 模型对每一帧输入图像分别预测(X, G)
- dynamic guided inference：$f_I(X, G) = (P, Q, \Delta X)$
  - P为物理参数
  - Q为binary mask，代表每一point是否属于rigid物体
  - $\Delta X$为对坐标位置X的refinement
  - 三变量分别使用gnn + 序列处理模型进行预测
- dynamic prior：$f_D(X^T, G, P, Q) = X^{T+1}$
  - 预测后续point坐标
  - 使用gnn处理每一point
    - $h_i^t = \phi_{gnn}(v_i^t, \sum_j g_{ij}^t)$
      - 从一point i特征$v_i^t$ 和 临近point的特征和$\sum_j g_{ij}^t$预测point i新特征$h_i^t$
    - 每一point输入特征为坐标X和物理参数P，物理参数P为给定参数
  - 使用DPInet，得到每一node i以往T时刻的gnn特征，预测下一时刻坐标
    - $X_i^{T+1} = \phi(\{h_i^t\}^{1..T} | G_i, Q_i)$
    - 根据Q，属于同一rigid物体的point共用一transformation，保证一rigid物体不同时刻形状不变

# Physically grounded vision-language models for robotic manipulation
2024

vlm + physics进行robot planning，"通过视觉实现reasoning"
- 提出PhysObjects数据集

相关工作
- physical reasoning
  - 从interaction data学物体物理属性
    - 15–17
  - 学物理特征representation
    - VEC 21：用vlm/llm进行physical reasoning，reasoning部分通过文本
  - openscene 14：使用clip检测物体物理属性

