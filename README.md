# 百度网盘AI大赛——图像处理挑战赛：文档图像摩尔纹消除第8名方案

## 项目描述
# 任务
对比赛给定的带有摩尔纹的图片进行处理，消除屏摄产生的摩尔纹噪声，还原图片原本的样子
# 评价指标
![](https://ai-studio-static-online.cdn.bcebos.com/47963fc72fb84278a46c51b2e852dd15fadd9b15b14f4e5ba21b5c9bc2690f03)

# 数据
训练集1000张，测试集A、B各200张

![](https://ai-studio-static-online.cdn.bcebos.com/b561143cbef843c2ae94251d8a1e8d97ad3e7714d06242d88b828aec459ba381)

# 难点
1、摩尔纹形态有多种，没有统一的特征

2、摩尔纹在整张图上分布不均匀
![](https://ai-studio-static-online.cdn.bcebos.com/d14f646643864c37821a67c58f26a702bd7cc1a91a75430d851e3c1f522e0ae1)

# 数据处理
1000张训练数据使用900张训练，100张验证
# 模型实现
![](https://ai-studio-static-online.cdn.bcebos.com/1116e0f0f175465c8378b7609c770423e9795099225e459e8ec4dd3ac829169d)

Unet,降采样4次，使用MaxPool,升采样4次，使用ConvTranspose,层间有跨连接

优点：参数量少、速度快、对均匀分布的摩尔纹效果好

缺点：对于非均匀分布的摩尔纹，处理效果不佳

![](https://ai-studio-static-online.cdn.bcebos.com/f46c99639b4d407995c28a269f673e990d344ac4f0cf41a0b1b5628cc63ac1fa)

# 训练
1、训练时数据增强：裁剪为512x512大小的patch进行训练，以0.5的概率随机旋转、翻转

2、损失函数:直接使用评分指标做为loss

3、学习率：使用stepLR，从1e-4降低到1e-5

4、优化器：Adam

# 测试细节
数据增强：将原图旋转、翻转后的预测结果取平均，PSNR可提升0.36db

# 主要提分点
1、unet结构中使用maxpool作为下采样

2、使用score作为loss

3、使用大patch进行训练

4、测试增强

## 项目结构
-|data
-|work
-README.MD
-main.ipynb

## 使用方式  
在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/3439039)  
