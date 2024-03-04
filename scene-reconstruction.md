# UniSDF: Unifying Neural Representations for High-Fidelity 3D Reconstruction of Complex Scenes with Reflections
2023

3d重建，考虑reflection

相关工作
- nerf + reflectance
  - 37：Ref-Nerf：使用物体表面norm
  - 18：ENVIDR：使用物体材质信息重建有光滑表面的物体，但导致物体表面细节缺失
  - 44：BakedSDF：使用模型类似Ref-Nerf模型 + VolSDF模型，重建光滑物体表面，但导致物体表面细节缺失
    - 43：VolSDF

模型
- ![](/note_images/UniSDF.png)
- 坐标x通过iNGP得到position embedding $\gamma(x)$
  - iNGP在不同pyramid level的特征$\gamma_l(x)$ concat得到特征$\gamma(x)$
- SDF mlp预测位置x的sdf $d(x)$
  - 使用contracted space：当$\|x\| \leq 1$时 $contract(x) = x$，当$\|x\| > 1$时 $contract(x) = (2 - \frac{1}{\x|\|})(\frac{x}{\|x\|})$
  - 得到$\gamma(x)$，预测(函数输出$d(x)$, 法线方向n', 颜色context b)
  - 计算一位置的法线方向为$n = \nabla d(x) / \| \nabla d(x) \|$
    - 即对$d(x)$关于x求导，得到一sdf值关于临近位置的导数
    - 此法线方向用于后续计算，模型预测的n'仅用于近似$n$，（？使得模型学习法线信息
  - 一位置的density为$\sigma(x) = \alpha\delta(d(x))$
    - $\alpha$为参数，$\delta$为0 mean的spike distribution，即当$d(x) = 0$时density最大
- $f_{ref}$得到(x位置, 反射光线方向$w_r$, 颜色context b, 法线 n)，预测反射光线路径上的颜色
- $f_{cam}$得到(x位置, 视线方向$\bf d$ 颜色context b, 法线 n)，预测直射的颜色
- weight mlp预测反射和直射颜色的相对权重

代价函数
- 颜色loss
- eikonal loss：
  - $L = E_{x}[(\|\nabla d(x)\| - 1)^2]$
- norm n n'间差别loss
  - $L = \sum \|n - n'\|^2$
- norm n和d方向差别loss：
  - $L = \sum max(n \dot \bf d)^2$
  - 即 一点如果为可见点，则此点的norm n不应和和视线方向$\bf d$相反

# NeRRF: 3D Reconstruction and View Synthesis for Transparent and Specular Objects with Neural Refractive-Reflective Fields
2023

考虑重建场景中的折射和反射，并且通过物体反射折射重建背景environment radiance

相关工作
- reflective 场景重建
  - 88：使用物体轮廓重建反光/透明物体
  - 44 45 46 79：重建表面非漫反射的物体
  - 52：将一反光物体表面图像分为view dependent和view independent部分
  - 33：使用parametric BRDF模拟反射
  - 59 23：将一图像分为直射颜色和反射颜色两部分
  - [2023 CVPR] **Nerf-ds**: Neural radiance fields for dynamic specular objects：重建动态的反光物体
  - 64 24：使用物理公式模拟物体表面view direction改变
- refraction 场景重建
  - 针对一特定背景环境，如假设背景为全灰
    - [2020 ACM TOG] Differentiable refraction-tracing for mesh reconstruction of transparent objects
    - [2023] **Neto**: Neural reconstruction of transparent objects with self-occlusion aware refraction-tracing
    - [2018] Full 3d reconstruction of transparent objects
    - [2022] **Hybrid mesh-neural representation for 3d transparent object reconstruction**
  - 使用Eikonal rendering中的ray equation模拟光路
    - [2023 SIGGRAPH] Sampling neural radiance fields for refractive objects
    - [2022 SIGGRAPH] Eikonal fields for refractive novel-view synthesis
  - [2023] **Nemto**: Neural environment matting for novel view and relighting synthesis of transparent objects：使用ray bending net预测光路

模型
- 1.得到物体轮廓mask，直接从mask预测物体几何结构
  - 避免由于物体表面的反光导致对物体表面norm/depth预测出错
  - 使用预训练segmentation模型预测mask
  - 使用可微分的场景重建模型 Deep Marching Tetrahedra DMTet 得到物体模型
    - 每一vertice位置$v_i$包含一sdf值$s_i$和一deformation向量$\Delta v_i$
  - 使用progressive encoding：随训练步数增加 增加position embedding的维度。避免从一开始使用高维度position embedding导致mesh表面noise
- 2.使用物理公式计算折射-反射光线方向 和 折射-反射相对权重，得到一视线的颜色。学nerf mlp预测每一段光线的颜色

# Neto: Neural reconstruction of transparent objects with self-occlusion aware refraction-tracing
2023

使用grid背景，可得到一视线通过折射后在背景上的坐标o。优化sdf 使得通过物体折射后 视线在背景的落点o'和o相同

# NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects
2023

重建反光的移动物体

相关工作
- 34：HyperNerf：使用hyper coordinate input模拟移动场景，而非使用物体template + 在随后帧的deform预测物体移动
- 使用translation field 模拟位移/deform
  - 23 36 50
- 使用special euclidean SE(3) field
  - 33 34

预测物体mask，用于重建物体结构。

使用SE(3)，即对每一时刻预测一[R | t]矩阵，将world coordinate的坐标x/norm转为相机coordinate中的坐标x/norm。预测deform

使用物体表面norm作模型输入，便于预测反射颜色





# Ref-Nerf


# 3D Reconstruction of Transparent Objects with Position-Normal Consistency
2016

假设refraction 光路仅两条
