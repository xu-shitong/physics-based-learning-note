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
- 1.模型得到输入$u_x, u_{xx}, ...$，预测$u(t)$