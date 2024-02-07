# physics based deep learning note
2022

分类
- Supervised：使用physics模拟生成数据，随后模型训练中不使用模拟器生成额外数据
- Loss-terms：使用可微分physics模拟器做loss项
- interleaved：可微分模拟器，模型输出和可微分模拟器输出共同决定loss值

分类
- forward simulation：预测state改变
- inverse simulation：得到物理公式中的参数

loss term中使用物理信息
- 定义：需要模拟的函数为u(x, t)，$u_x$即函数u对x求一阶导，$u_{xx}$为二阶导
  - u即一关于坐标x和时间t的state
  - 根据一关于$u(x, t)$的pde，可将u写为$u = F(u_x, u_{xx}, ..., u_{xx...x})$

[physical loss term 两varient]

differentiable physics reconstruction：使用simulator。simulate至一特定时刻，使用样本计算loss，back prop得到simulator的参数。例：优化simulator参数为initial state，使得simulator经过一定时间步后生成指定数据

# pytorch based physics informed machine learning youtube video
https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets/tree/main

inverse problem
- 在burgurs equation例中，两pde参数参与模型grad descent和模型同时训练

# deepONet实现
- branch模型输入为多个function，每一function u在$[x_1, ..., x_m]$位置取值 用于代表u函数特征
- trunk模型输入为多个(x, t)坐标，称y collocation point
- loss中pde的loss称residual loss
- ？对x求导时是针对(x, t)求导还是包含branch net的x
  - pde loss计算时应当使用同一x做trunk模型输入 和 得到branch模型的u(x)输入
- **？branch和trunk 模型的x是否相同**，trunk 模型的(x, t)对应branch 函数的一个u还是多个u
  - pde loss中branch trunk x相同，同上一问
  - boundary 和initial condition loss中两模型x不相关。即设计boundary 和initial condition的数据时只需保证x t s覆盖足够样本范围即可
  - 可以有多个u，每一u有collocation point number个(x, t)
- **？branch模型的输入u是多个不同函数**
  - long term integration of Pi deeponet 使用输出s作为下一时刻的u，所以1.每一u为一时刻不同x位置的状态 2.可将initial condition的取值做u的取值 3.initial condition 的loss可为训练模型输出s = u
  - **u仅为一basis func，可使用radial basis function。测试时可替换为测试数据使用的initial func**
    - 如：使用RBF做u训练模型，用sin做u得到测试时模型的输入
- ？为什么branch的输入为2维，而trunk输入为1维，理应trunk得到x t而branch得到u
  - pendulum例子假设y仅为t，u为2维由于有两个state需要学习
- ？u代表函数，但loss定义只需pde和boundary condition，如何使用u
  - u仅作为定义函数的点，类似few shot样本
- **取u的位置为linear？，否则模型如何知道u的变化快慢**
  - 在diffusion reaction例子中取u的x为linear分布，布满整个u domain 即布满整个x可取值的范围
  - **要求每次采样u使用同一集合x位置**
- 看https://github.com/PredictiveIntelligenceLab/Long-time-Integration-PI-DeepONets/blob/main/pendulum/PI_DeepONet_pendulum.ipynb
  - ？operator net为什么输出两个向量分别点乘：由于物理公式针对两值进行pde定义

看Differential Equations as a Pytorch Neural Network Layer 是做什么，如何实现，是否能够令模型得到pde输入
- 对应代码https://gist.github.com/khannay/4cd75e2ff9aecd335f0e770e9d04415c

看deeponet写好的module https://deepxde.readthedocs.io/en/latest/modules/deepxde.nn.pytorch.html
- 使用for loop + torch.autograd.grad

尝试vmap和for loop的时间差
- torch版本只能使用两层for loop，看写好的module怎么加快时间

看jax版本怎么使用gpu
- colab上没成功

尝试将1 batchsize的代码跑通
- 跑通，使用batch的方式进行训练，不然训练时间过长

尝试将tensor分为list，通过grad计算
- grad不支持

尝试使用jacobian方式计算grad，观察是否和grad相同
- grad为jacobian沿一dim求和，jacobian求两向量间所有元素间的斜率，不适用batch情况

physics informed deeponet
- loss中添加物理pde loss项

modified pi deeponet
- 输出的s即u，进行autoregressive预测

