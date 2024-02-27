# Physics-Informed Machine Learning: A Survey on Problems, Methods and Applications
2023

以往survey
- Physics-informed machine learning 2021 Nature Reviews Physics：PIML的历史发展
- Scientific machine learning through physicsinformed neural networks: Where we are and what’s next 2022：PIML的算法和应用
- An overview on deep learning-based approximation methods for partial differential equations 2020：使用nn解pde的theoretical result
- PIML sub domain的survey
  - Physicsinformed neural networks (pinns) for fluid mechanics: A review 2021：fluid mechanics
  - Uncertainty quantification in scientific machine learning: Methods, metrics, and comparisons 2022：uncertainty quantification
  - Combining machine learning and domain decomposition methods for the solution of partial differential equations—a review GAMM 2021：domain decomposition
  - Physics-guided deep learning for dynamical systems: A survey 2021：dynamic system

物理prior 分类
- differential equations做模型prior
  - 定义
    - x：spatial temporal coordinate，即[坐标向量, timestep]
    - $u = u(x)$：state vector，为pde希望求解的函数，得到x预测系统的state
    - pde/ode函数称governing equation
  - condition：
    - partial differential equation：$F(u, \theta)(x) = F(u, \theta, x) = 0$
    - initial condition：$I(u, \theta)(x, t_0) = 0$
    - boundary condition：$B(u, \theta)(x, t) = 0$
- symmetry constraints
  - 设计网络使得模型有invariance，或添加regularize term使得模型学习此性质
  - 包括 旋转 移动 permutation topology 等invariance
- intuitive physical constraints
  - 令模型预测时运用常识信息，难以在模型设计时同时考虑所有constraint
  - 例：物体在视野外时仍存在场景中 不会消失，有重力存在，能量 动量守恒，牛顿力学定率

模型使用物理prior方法
- 将物理信息prior加入模型
  - 数据：augment数据使得数据具有对应物理特性，如symmetric invariance
  - model：使用"inductive biases guided by the physical prior into the hypothesis space"，令模型有prior性质
  - objective：将pde作为loss func
  - optimizer：修改gradient descent方法
  - inference：预测时在backbone模型外增加一模型

使用PIML的任务
- neural simulation：预测物理system后续状态，survey针对ode/pde求解任务
  - 传统方法问题：
    - curse of dimensionality：complexity为$O(n^r)$，n为求解时sample的点数，r常为3
      - 计算时间可达一周
    - 仅知道部分pde无法求解，无法使用额外数据generalize到新场景
  - PINN：将$F, I, B$三函数做loss项，训练模型满足governing equation
    - 模型优化每一governing equation loss的速度不同
      - [每一governing loss term使用不同权重]
        - Understanding and mitigating gradient flow pathologies in physics-informed neural networks
      - [resampling样本]
    - 提出新的loss func
      - novel Optimization Objectives：解决直接使用boundary condition做loss训练效果不好
        - Dgm: A deep learning algorithm for solving partial differential equations 2018 Journal of computational physics：使用monte carlo方法代替gradient backward prop，解决当输入坐标维度大时计算量过大问题
        - CAN-PINN：Canpinn: A fast physics-informed neural network based on coupledautomatic–numerical differentiation method 2022：使用back prop需要大量collocation points做样本，提出修改back prop计算方法
        - cvPINNs Thermodynamically consistent physicsinformed neural networks for hyperbolic systems 2022：使用finite volumn method解pde
      - 使用variational formulation降低model对模拟的环境的smoothness的要求
        - 一集合test function和pde相乘，通过integration by part
        - The deep ritz method: a deep learning-based numerical algorithm for solving variational problems 2018：降低loss function中的求导阶数，针对self-adjoint differential operators
        - VPINN：Variational physics-informed neural networks for solving partial differential equations 2019：Petrov-Galerkin formulation，针对更广泛类型pde
        - Pfnn: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries 2021：使用两nn，一学习boundary condition，另一nn从variational formulation学习
        - Weak Adversarial Networks WAN：Weak adversarial networks for high-dimensional partial differential equations 2020：选择最差test function，训练模型满足pde，类似gan训练
        - finite element method常用
      - Binet: learning to solve partial differential equations with boundary integral networks 2021：使用bondary integration method + nn解pde
      - 添加regularization term
        - gradient enhanced regularization/Soblev training：将对pde求更高阶，做regularization term
          - Inverse dirichlet weighting enables reliable training of physics informed neural networks 2022
          - Sobolev training for the neural network solutions of pdes 2021
    - 修改网络结构
      - 修改activation function
        - 由于代价函数对模型求高阶导，需要smooth activation function
        - Adaptive activation functions accelerate convergence in deep and physicsinformed neural networks 2020：使用adaptive activation function，不同层将输入乘以不同scale后通过activation
        - Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks 2020：使用layerwise + neuronwise adaptive activation
      - 修改输入feature
        - 使用sinusodial embedding
        - Multi-scale deep neural network (mscalednn) for solving poisson-boltzmann equation in complex domains 2020：使用multiscale feature embedding
        - Accelerating physicsinformed neural network training with prior dictionaries 2020：记录一集合feature preprocessing function，根据prior选择使用哪一func生成输入feature
      - multi nn 和 boundary encoding
        - 模型输出state vector包含不同物理意义的部分，所以PINN训练为multitask任务。使用不同mlp预测各state vector部分
        - Physics-informed deep learning for incompressible laminar flows 2020：使用variable substitution，可降低pde求导阶数
        - 用hard constraint将pde encode进nn
          - 即人工解pde，使用nn模拟解中未知的函数，使得设计的nn满足pde要求
          - 对没有analytical solution的nn无法使用
        - 使用多个nn将boundary condition作为hard constraint进行训练
          - 在nn后添加post processing layer用于encode boundary constraint
            - 139：1998年论文，140：2009论文
          - Physics-informed neural networks with hard constraints for inverse design 2021：encode dirichlet boundary：nn输出向量做fourier basis的系数，针对矩形domain
          - 将encode boundary condition的方法generalize进Theorey of Functional Connections
            - Deep theory of functional connections: A new method for estimating the solutions of partial differential equations 2020
            - Extreme theory of functional connections: A fast physics-informed neural network method for solving ordinary and partial differential equations 2021
          - 针对非矩形domain的dirichlet boundary condition encode
            - Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data 2020
            - Physics-informed graph neural galerkin networks: A unified framework for solving pdegoverned forward and inverse problems 2022
          - A unified hard-constraint framework for solving geometrically complex pdes 2022：在geometry complex domain上encode Dirichlet + Neumann + Robin boundary conditions
          - [介绍 + 使用distance funcition论文]
      - 使用rnn模拟时序模型
        - 有loss为两连续时刻间的差 和 differential function求出的差 之间的loss
        - 设计nn遵循hamiltonian equation和lagrangian equation：
          - Hamiltonian neural networks 2019
          - Deep lagrangian networks: Using physics as model prior for deep learning 2019
          - On learning hamiltonian systems from data 2019
          - Hamiltonian generative networks 2019：将vae和hamiltonian nn结合模拟time dependent system with uncertainty
      - conv based nn
        - conv based PIML
          - Theory-guided hard constraint projection (hcp): A knowledge-based data-driven scientific machine learning method 2021
          - Phycrnet: Physics-informed convolutional-recurrent network for solving spatiotemporal pdes 2022
          - Physics-informed multi-lstm networks for metamodeling of nonlinear structures 2020
          - Phygeonet: Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state pdes on irregular domain 2021：使用bijective feature transformation使得非矩形输入特征可传入cnn
        - Predicting parametric spatiotemporal dynamics by multi-resolution pde structurepreserved deep learning 2022：使用unet 抽特征，包含pde preserving part，可解forward + inverse problem
      - 使用gnn
        - 使用gnn预测下一timestep simulation result
          - Learning to simulate complex physics with graph networks 2020
          - Physics-informed graph neural galerkin networks: A unified framework for solving pdegoverned forward and inverse problems 2022
          - Learning mesh-based simulation with graph networks 2020
        - Predicting physics in mesh-reduced space with temporal attention 2022：在temporal维度使用attention layer
        - Phygnnet: Solving spatiotemporal pdes with physics-informed graph neural network 2022
        - Learning the solution operator of boundary value problems using graph neural networks 2022
      - 使用domain decomposition用多个模型模拟场景
        - 代价函数为 每一模型在分配的domain上的loss + 相邻两domain的模型模拟交接处的point的loss
        - hp-vpinns: Variational physics-informed neural networks with domain decomposition 2021
        - cPINN：Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems 2020：针对满足conservation law的system，loss包含 对domain交界处两模型 预测差别 + flux差别
        - Extended physics-informed neural networks (xpinns): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations 2020：无需场景满足conservation law，flux loss改为general PDE residuals
        - 使用multi-scale feature/multiscale nn + domain decomposition based variational formulation
          - CENN：Varnet: Variational neural networks for the solution of partial differential equations 2020
          - D3m: A deep domain decomposition method for partial differential equations 2019
        - Ppinn: Parareal physics-informed neural network for time-dependent pdes 2020：进行temporal domain decomposition
        - Large-scale neural solvers for partial differential equations 2020：使用soft gated mixture of expert模拟不同domain
    - 不足
      - 现有optimizer + loss function对PINN 训练非optimal
      - 新ml模型做PINN的效果未知
      - 使用nn解决pde中state维数过高，造成curse of dim问题
        - The deep ritz method: a deep learning-based numerical algorithm for solving variational problems 2018
        - Solving high-dimensional partial differential equations using deep learning 2018
  - neural operator
    - 解differential equation，将control function/参数 map到state
    - direct methods represented
      - DeepONet：Learning nonlinear operators via deeponet based on the universal approximation theorem of operators 2021
        - deepONet输出$G(\theta)(x) = b_0 + \sum_{k=1}^p b_k(\theta)t_k(x)$
        - $b$称branch network，$t$称trunk network。$\theta$为parameter，x为coordinate
        - 两network输出p维向量，点乘后和bias$b_0$相加做模型输出
        - 训练时直接使用ground truth $G(\theta)(x)$值 计算l2 loss训练模型，**loss中没有使用pde的residual项**
      - physics informed deepONet
        - 解决 原deepONet为data driven 但样本数据难以生成 问题
        - Long-time integration of parametric evolution equations with physics-informed deeponets 2021：physics informed deepONet的variant，进行long term dynamic system
        - 增加loss reweighting和data resampling
          - On the influence of over-parameterization in manifold based surrogates and deep neural operators 2022
          - Improved architectures and training algorithms for deep operator networks 2022：使用gated mlp做模型
      - 修改deepONet：
        - Meta-auto-decoder for solving parametric partial differential equations 2021：使用audo decoder模型
        - multiple input deepONet：模型得到多个$\theta$进行预测
          - Mionet: Learning multiple-input operators via tensor product 2022
          - Physics-informed neural networks for high-speed flows 2020
      - DeepM&Mnets：预训练deepONet模拟多种物理场景
        - Deepm&mnet for hypersonics: Predicting the coupled flow and finite-rate chemistry behind a normal shock using neuralnetwork approximation of operators 2021
        - Deepm&mnet: Inferring the electroconvection multiphysics fields based on operator approximation by neural networks 2021
      - Accelerated replica exchange stochastic gradient langevin diffusion enhanced bayesian deeponet for solving noisy parametric pdes 2021：使用bayesian deepONet
      - multifidelity deepONet：输入为multifidelity data
        - Multifidelity deep operator networks 2022
        - Multifidelity deep neural operators for efficient learning of partial differential equations with application to fast inverse design of nanoscale heat transport 2022
      - Multiauto-deeponet: A multiresolution autoencoder deeponet for nonlinear dimension reduction, uncertainty quantification and operator learning of forward and inverse stochastic problems 2022：输入为multi-resolution，针对高维stochastic problem
    - [Green’s function learning]
    - grid-based operator learning
      - 得到pde参数$\theta$，预测一gird位置的N个state $\{u(x)\}_{n=1}^N$。当$\theta$同样为一grid特征时，模型可使用image-image mapping模型，如unet
      - FNO：Fourier neural operator for parametric partial differential equations 2020：针对image-image mapping
        - 输入通过1 conv层 + 多个fourier layer。每一fourier layer有两分支，一分支为 fft + 可学习transformation + ifft，另一分支为linear模型。分支输出按元素相加做layer输出
        - A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data 2022：将FNO extend到geometric complex场景中
      - 使用attention
        - Choose a transformer: Fourier or galerkin 2021：cnn抽特征，通过attention
    - graph-based operator learning
      - Neural operator: Graph kernel network for partial differential equations 2020：每一坐标x抽特征 [x本身, x的fixed Borel measure, x的fixed Borel measure通过Gaussian smooth的结果]
      - Message passing neural pde solvers 2022：使用autoregressive + graph operator learning，针对使用pde定义的operator
    - 后续工作
      - 速度和准确率低于使用finite element method解pde
      - physics simulation：在chaotic 和long term prediction中 模型难以准确预测
      - 创建预训练模型
  - PINN theory 基础
    - 1.不限维度的单层deepONet可以表达所有continuous operator
    - 2.层数较少的deepONet需要exponential多个neuron才能达到和较深但教窄的deepONet相同的expressability
    - [error estimation]
- inverse problem：
  - 优化$\theta$，即更改部分场景参数，最小化一loss 并 满足物理场景条件
    - 即$min_{\theta} J(u(\theta, x)), s.t. P(u, \theta)(x) = 0$
      - $P(u, \theta)(x)$代表一集合pde constraint
  - Neural Surrogate Models：模型模拟物理场景，避免使用numerical solver从$\theta$预测模型是否符合pde constraint
    - Optimal control of pdes using physicsinformed neural networks 2021：使用PINN解inverse problem，直接将pde和优化目标$J$作为代价函数项，针对optimal control problem
    - hPINN：Physics-informed neural networks with hard constraints for inverse design 2021：提出使用hard constraint
    - Bilevel physics-informed neural networks for pde constrained optimization using broyden’s hypergradients 2022：使用PINN解pde，使用hypergradient优化$\theta$
    - PINC：Physics-informed neural nets for control of dynamical systems 2021：Physics-Informed Neural Nets for Control：针对control task，能够进行long term prediction
  - neural operator
    - Fast pde-constrained optimization via self-supervised operator learning 2021：使用deepONet，通过gradient-based methods优化$\theta$，使用self supervised method
    - [其余使用neural operator进行inverse problem的论文]
  - neural simulator
    - Iterative surrogate model optimization (ismo): an active learning algorithm for pde constrained optimization with deep neural networks 2021：使用quasi-Newtonian optimizer + nn
    - 物理场景学习复杂物理场景
      - Designing an adhesive pillar shape with deep learning-based optimization 2020：使用cnn + linear层预测一表面压力分布
      - Scalable deep-learning-accelerated topology optimization for additively manufactured materials 2020
  - "surrogate model仅能使用supervised训练方式，需要simulated data"
  - 应用
    - 解inverse problem，发现新材料/研究材料特性
      - A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics 2021
      - Extraction of mechanical properties of materials through deep learning from instrumented indentation 2020
    - 从flow data得到难以测量的物理属性，如速度
      - Physics-informed neural networks for high-speed flows 2020
      - Physics-informed neural networks for solving forward and inverse flow problems via the boltzmann-bgk formulation 2021
      - Nsfnets (navierstokes flow nets): Physics-informed neural networks for the incompressible navier-stokes equations 2021
      - Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations 2020
    - 使用PINN/neural operator得到物理公式参数/发现新公式
      - Physics-integrated variational autoencoders for robust and interpretable generative modeling 2021
      - Model inversion for spatio-temporal processes using the fourier neural operator
      - Physics-informed learning of governing equations from scarce data 2021
      - Deep hidden physics models: Deep learning of nonlinear partial differential equations 2018：补全governing function并求解

cv中应用
- 使用PIML进行pose estimation和tracking
  - Physical inertial poser (pip): Physicsaware real-time human motion tracking from sparse inertial sensors 2022
  - Trajectory optimization for physics-based reconstruction of 3d human pose from monocular video 2022：在inference中加入物理信息，如摩擦力
- Hamiltonian generative networks 2019：restrict trajectory
- Learning to see physics via visual de-animation 2017：explicitely estimate object state
- Interaction networks for learning about objects, relations and physics 2016：使用graph nn学物体间交互
- Reblur2Deblur：Reblur2deblur: Deblurring videos via self-supervised learning 2018：进行deblur
- Neural mocon: Neural motion control for physically plausible human motion capture 2022：使用physics simulator做supervisor进行human motion capture
- Geo-supervised visual depth prediction 2019：depth estimation
- Feature forwarding for efficient single image dehazing 2019：image dehazing
- Aug-nerf: Training stronger neural radiance fields with triple-level physicallygrounded augmentations 2022：3D reconstruction
- 数据集
  - 包含多种物体性质
    - Objectfolder: A dataset of objects with implicit visual, auditory, and tactile representations 2021
    - Objectfolder 2.0: A multisensory object dataset for sim2real transfer 2022
  - Visual Newtonian Dynamics (VIND)：Newtonian scene understanding: Unfolding the dynamics of objects in static images 2016：Newtonian场景，使用游戏引擎生成
  - Physics 101: Learning physical object properties from unlabeled videos 2016：撞击场景下的物体运动视频数据集

# When Physics Meets Machine Learning: A Survey Of Physics-Informed Machine Learning
2022

motivation
- 使用ml解物理domain task
- 物理principle + ml针对real world task

ml解物理task
- simulation：simulation参数通过学习得到，不需要人工调参数
  - 湍流预测
    - Meshfreeflownet: a physics-constrained deep continuous space-time super-resolution framework 2020 SC20
    - **Towards physics-informed deep learning for turbulent flow prediction** 2020 ACM SIGKDD
    - Enforcing physical constraints in cnns through differentiable pde layer 2020 ICLR workshop
  - particle system预测
    - **Learning to simulate complex physics with graph networks** 2020 PMLR：particle system，可用于rigid和fluid和deformable物体
    - **Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids** 2019 ICLR
    - **Lagrangian fluid simulation with continuous convolutions** 2019 ICLR
    - Learning mesh-based simulation with graph networks 2021 ICLR
  - Combining differentiable pde solvers and graph neural networks for fluid flow prediction 2020 PMLR：fluid flow预测，效果好，并比ground truth快60倍
    - 一gnn得到低维mesh，一cfd solver得到高维coarse mesh。cfd solver输出upsample到和gnn相同维度后concat
- pde solver：
  - Deep hidden physics models: Deep learning of nonlinear partial differential equations 2018
    - 两模型。一模型作为solution prior 避免需要计算differentiation。一模型包含spatial temporal数据在时序上的变化信息
  - 36 2003论文：multi scale analysis，得到低维microscopic信息，预测coarse macroscopic信息
  - Datadriven solutions of nonlinear partial differential equations 2017：data driven pde solver
  - **Neural operator: Graph kernel network for partial differential equations** 2020：使用spatial domain message passing模拟从initial condition到pde solution的mapping
  - Fourier neural operator for parametric partial differential equations 2021 ICLR：同上一论文，使用frequency domain message passing
- downsampling
  - Downscaling satellite precipitation estimates with multiple linear regression, artificial neural networks, and spline interpolation techniques 2019：从卫星 precipitation 预测云层的optical 和 microphysical性
  - Coarse-scale pdes from fine-scale observations via machine learning 2020 Chaos：从显微镜数据得到general level信息
- parameterization：使用parameterized process替代dynamic systen，模拟复杂物理现象
  - **Prognostic validation of a neural network unified physics parameterization** 2018：模拟一类似地球的水体行星的热源分布
  - **Could machine learning break the convection parameterization deadlock**? 2018：模拟一水体行星上的对流
- reduce order model ROM
  - **A reduced order model for turbulent flows in the urban environment using machine learning** 2019：模拟城市楼房间的空气湍流
  - A deep learning based approach to reduced order modeling for turbulent flow control using lstm neural networks 2018：使用LSTM进行湍流预测
- causality
  - **Causal network reconstruction from time series: From theoretical assumptions to practical estimation** 2018 Chaos：提出方法 1.区分direct和indirect causal，2.得到多个time series间共同的cause
  - Detecting and quantifying causal associations in large nonlinear time series datasets 2019：non linear time series中的causal association
  - **Causal discovery with attention-based convolutional neural networks** 2019 ：从time series进行causal discovery
- **Counterfactual Analysis of Physical Dynamics观察不同intervension的影响**
  - physics law intervension
    - Interaction networks for learning about objects, relations and physics 2016 NIPS
    - A compositional object-based approach to learning physical dynamics 2016
    - Galileo: Perceiving physical object properties by integrating a physics engine with deep learning 2016 NIPS
  - 将physics feature和其余feature 分离进行conterfactual analysis
    - Decomposing motion and content for natural video sequence prediction 2017 ICLR 
    - Unsupervised learning of disentangled representations from video 2017 NIPS
  - Learning to generate long-term future via hierarchical prediction 2017 ICML：使用额外环境prior进行conterfactual analysis
  - PhyDNet：Disentangling physical dynamics from unknown factors for unsupervised video prediction 2020 CVPR：将pde和额外信息分离
  - Interpretable intuitive physics model 2018 ECCV：进行后续帧collision prediction，encoder预测摩擦力 质量，decoder得到encoder输出预测optic flow
  - PIP：Pip: Physical interaction prediction via mental imagery with span selection 2021
  - CWM：Causal discovery in physical systems from videos 2020 NIPS：无监督学latent confounding factor，得到intervension和后续帧间关系
  - Counterfactual learning of physical dynamics 2020 ICLR：通过预测latent confounding factor预测后续帧
  - Filteredcophy: Unsupervised learning of counterfactual physics in pixel space 2022 ICLR：不同帧取dense feature + key point + 每一key point feature作为输入/输出

物理principle + ml解real world problem
- 处理object centric data
  - 处理sensor得到的位置速度信息
    - Neural relational inference for interacting systems 2018 ICML
    - **Spatial temporal graph convolutional networks for skeleton-based action recognition** 2018 AAAI
  - 53：处理molecule data
  - 使用GNN处理object centric data的sota
    - Interaction networks for learning about objects, relations and physics 2016 NIPS
    - Relational inductive biases, deep learning, and graph networks 2018
    - Graph networks as learnable physics engines for inference and control 2018 ICML
  - **Schnet: A continuous-filter convolutional neural network for modeling quantum interactions**. 2017 NIPS：计算total energy和分子间力，使得预测符合能量守恒
  - **Hamiltonian neural networks** 2019 NIPS：预测n体问题，无监督学习，考虑Hamiltonian mechanics conservation law
  - **Lagrangian Neural Networks** 118：不使用坐标系，parameterize任意Lagrangians
- 处理time series
  - **Disentangling physical dynamics from unknown factors for unsupervised video prediction** 2020 CVPR：将pde和未知factor disentangle，在latent space进行pde constraint prediction
  - **Advectivenet: An eulerian-lagrangian fluidic reservoir for point cloud processing** 2020 ICLR：point cloud flow simulation
  - Learning continuous-time pdes from sparse data with graph neural networks 2021 ICLR：提出continuous time differentiable model
- 处理manifoid：当物体坐标系不在平面上时，使用物理公式对cnn提供坐标间临近关系
  - Geodesic convolutional neural networks on riemannian manifolds 2015：cnn处理geodesic polar坐标系数据

合并的物理信息
- Newtonian mechanics
- Lagrangian Mechanics
  - lagrangian function：$L(q, \dot{q})$，q为generalized coordinate向量，$\dot{q}$为坐标q关于时间的求导
    - 常用的L为动能和势能间的差值 $L(q, \dot{q}) = K(q, \dot{q}) - V(q)$
    - 使用nn模拟lagrangian func
  - "合理的physical trajectory为当$\delta \int_{t_1}^{t_2} L dt = 0$的trajectory"
    - 从上式可得到 Euler-Lagrange equation：$\frac{d \Delta_{\dot{q}}L}{d t} = \Delta_q L$
    - 从chain rule：$(\Delta_q \Delta_{\dot{q}}^T L) \ddot{q} + (\Delta_{\dot{q}} \Delta_{\dot{q}}^T L) \ddot{q} = \Delta_q L$
    - $\ddot{q} = (\Delta_{\dot{q}} \Delta_{\dot{q}}^T L)^{-1} [\Delta_q L - (\Delta_q \Delta_{\dot{q}}^T L) \dot{q}]$
- Hamiltonian Mechanics
  - 令q为包含所有物体的spatial coordinate向量，p为所有物体的momentum向量，$\dot{q}, \dot{p}$为qp关于时间的求导
    - qp应满足canonical condition：$p = \Delta_{\dot{p}} L$
  - Hamiltonian func为(q, p)两向量的函数：$H(q, p) = \dot{q} \dot p - L$
  - 从Euler-Lagrange equation + canonical condition + Hamiltonian func：$\dot{p} = - \Delta_q H$, $\dot{q} = \Delta_p H$
  - 从上式：$\frac{d}{dt} H = \dot{q} \Delta_q H + \dot{p} \Delta_p H = 0$
- symmetricity：针对输入数据的symmetry性质设计网络，多为gnn模型

合并物理信息方法
- transfer learning
  - Training deep networks with synthetic data: Bridging the reality gap by domain randomization 2018 CVPR workshow：随机使用光线 pose 物体texture模拟场景，生成负样本。使得模型学正确场景的特征
  - Physics-guided machine learning for scientific discovery: An application in simulating lake temperature profiles 2021：使用不精确的simulator生成数据预训练模型
- multi-task/meta learning
  - Adversarial multi-task learning enhanced physics-informed neural networks for solving partial differential equations 2021：使用随机参数生成pde，使用multitask learning学多个pde的shared representation
  - Physics-aware spatiotemporal modules with auxiliary tasks for meta-learning 2021 IJCAI：spatialtemporal 预测模型，分离spatial和temporal模型。spatial 模型为pde independent，用MAML方式训练。temporal 模型为task dependent，对每一task单独训练
- physics informed computation graph：将已有physics based solution中复杂的计算替换为nn
  - DeLaN：Deep lagrangian networks: Using physics as model prior for deep learning 2019 ICLR：lagrange mechanics + DeLaN
  - LNN：Lagrangian neural networks 2020 ICLR workshop：提出lagrangian neural network模拟任意lagrangian function
  - HNN：Diffusion convolutional recurrent neural network: Datadriven traffic forecasting 2018 ICLR：hamiltonian function + nn
- dl和physics based模型fusion
  - Combining generative and discriminative models for hybrid inference 2019 NIPS：使用equation of motion得到gnn，autoregressive预测下一场景state
  - **Hybridnet: integrating model-based and data-driven learning to predict evolution of dynamical systems** 2018 Conference on Robot Learning：使用convLSTM做data driven模型，预测系统得到的下一external input。cellular nn CeNN预测物理参数
  - **Pde-net: Learning pdes from data** 2018 ICML：使用constrained comvolution kernel得到spatial derivative，通过后续层预测场景dynamics

# Explainable Machine Learning for Scientific Insights and Discoveries


- [直接预测物理现象，无法输出现象发生原因]
  - PhysNet：39：预测一积木塔坍塌的trajectory
- 预测物理现象参数/property
  - A compositional object-based approach to learning physical dynamics 2017 ICLR
  - **53**：判断视频中场景是否遵守给定物理公式
  - 55：从视频中提取物理参数。
    - 使用constrained least square method预测物体位置和朝向，constrain为conservation of momentum

todo

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
