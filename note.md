# Aug-NeRF: Training Stronger Neural Radiance Fields with Triple-Level Physically-Grounded Augmentations
2022

regularize nerf 模型和训练过程，减少outlier
- 考虑相机位置预测中的noise，考虑输入图片中的noise
- regularize使得学习的物体表面平滑
- 没有使用物理公式


# Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos 
2023

场景音频生成，输入包含物理参数

相关工作
- 音频生成
  - 手工方法：50：使用linear model生成rigid body音频
  - 5：conditional gan生成音频
  - 62：sampleRNN based method
  - vision转乐器声音：1- 47 48

模型
- 1.使用signal processing得到物理parameter，使用nn预测residual parameter
  - 使用音频样本，将音频样本分为D个damped oscillation，得到每一oscillation参数
- 2.使用vision + 1.中输出，通过DDPM生成音频spectrogram

# DeepONet
## DeepONet：Learning nonlinear operators via deeponet based on the universal approximation theorem of operators
2020

提出deeponet的论文，学习的operator针对dynamic system (ode)和pde

相关工作
- operator针对dynamic system中的ode
  - 22 33：使用nn学习可用difference equation描述的dynamic system
  - 模拟一特定dynamic system的evolution
    - 20：使用rnn + reservior computing
    - 23：residual net
    - 9：autoencoder
    - 24：FNN
    - 6：neural ordinary differential equation
    - 12：neural jump stochastic differential equation

## Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets

physics-informed DeepOnets论文

## Long-time integration of parametric evolution equations with physics-informed deeponets 
2021

physics informed deepONet的variant，进行long term dynamic system
- 在loss中添加由pde定义的residual项，deepOnet在$<\Delta t$时间步的样本上训练，在$>\Delta t$时间上eval
- eval时 branch模型得到$\{u(t) | t \in [0, \Delta t)\}$，trunk模型得到$(x, t = \Delta t)$，模型输出作为$\Delta x$时刻$u(x)$和x加入$\{u(t) | t \in [0, \Delta t)\}$作为branch模型下一次预测输入
- 即autoregressive预测


# Unsupervised physics-informed disentanglement of multimodal data for high-throughput scientific discovery
2022

提出vae结构，发现signal中fingerprint。
- 针对mnist数据集，分析embedding分布的论文

# Residual-based attention in physics-informed neural networks
2024 https://github.com/soanagno/rba-pinns

不使用grad的pinn模型

# NOMAD: Nonlinear Manifold Decoders for Operator Learning
2022

使用non linear deeponet decoder
- 即不同于deeponet使用branch trunk模型的输出点乘计算结果，而是将branch输出和每一yconcat通过mlp计算结果

相关工作
- 23：fourier neural operator，使用fourier transform theorem计算integral
- 2：deeponet + pca based representation
- 36：random feature approach + deeponet
- 15：使用wavelet approximation to integral transform
- 23：attention based + deeponet

# DeepM&Mnet: Inferring the electroconvection multiphysics fields based on operator approximation by neural networks
2020

仅使用样本 不用pde做loss 训练多个deeponet。随后训练mlp模型预测deeponet输入，训练mlp时deeponet固定不参与训练
- 共有$\phi, c^+, c^-, u, v$ 5变量，都为关于$(x, y)$的函数
- 第一deeponet：branch模型得到$c^+, c^-$，trunk模型得到xy，预测$\phi(x, y)$
- 其余4 deeponet都得到$\phi(x, y)$的值做branch模型输入，分别预测$c^+(x, y), c^-(x, y), u(x, y), v(x, y)$
- 第二阶段训练的mlp得到$(x, y)$，预测$\phi(x, y), c^+(x, y), c^-(x, y), u(x, y), v(x, y)$值

# Optimal control of PDEs using physics-informed neural networks
2022

得到governing pde，求control variable最小化loss
- control variable为pde或boundary condition中一参数
- control variable作为可学习参数参与反向传播，和pinn同时从头开始训练

定义
- control variable 为$c_v(x, t), c_b(x, t), c_0(x)$
- pde函数depend on control variable：
  - 令x domain 为$\Omega \in R^d$
  - $F(u(x, t), c_v(x, t)) = 0, x \in \Omega, t \in [0, T]$
  - boundary condition $B(u(x, t), c_b(x, t)) = 0, x \in \delta\Omega, t \in [0, T]$
  - initial condition $I(u(x, t), c_0(x)) = 0, x \in \Omega$
- 用user define loss $J(u, c)$，gradient descent得到一满足pde的state value u和condition variable c

模型
- $u(x, t)$和三个$c(x, t)$都使用nn近似，即u为一pinn
- 代价函数为u的PINN loss + J(u, c)，即u和c同时参与优化。J(u, c)项有loss weight，即pinn和control variable的nn有不同learning rate

# ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation

# Interpretable Intuitive Physics Model

# Ben Moseley PhD thesis

能否使用deeponet branch输出做hypernet参数 y值通过hypernet进行预测
- 由于branch模型即预测一func参数，使得输出func为complex func可能可以预测更复杂问题

能否使用branch net预测一func沿时间的转变，如nerf随时间变化即通过多次branch模型
- branch模型本身为对一func进行操作的模型，并且能够generalize到不同basis func上，

使用多个pde先学多个operator，合并和下游任务pde相关的branch模型 对下游任务学习 
