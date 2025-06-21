# 说明文档
### 题目一：
1. **cGAN_Generator.py**:
> 这个文件是生成器模型的训练脚本，运行该脚本能够生成所需要的生成器
> 该生成器命名为generator_final.path

**说明**：会去跑测试集，然后生成一个result的文件夹，在该文件夹下保存了test里面每一个E对应的原始图以及我用训练的模型去生成的图片

2. **Generate_single_image.py**:
> 这个文件主要用于加载我们生成的生成器，在终端的提示下输入所需要的E值我们可以在interactive_generated_images文件夹中得到我们保存的生成的图片


### 题目二：
1. **regression.py**:
> 采用了一个名为ResNet18的预训练模型，生成的回归分析模型保存在同目录下的best_regression_model.pth中

**说明**： 我没做可视化的交互，仅仅提供两个评估指标

### 题目三：

**主要验证逻辑** ： 因为我观察到是验证生成模型的正确性，所以，采用了如下逻辑?
> 输入一个E值，然后用生成器去生成对应的图片，再用回归模型去预测我们生成后图片的E值，这样我们就可以同时检验我们回归和生成两个模型的正确性了

**过程**： 我去跑了一下test集，里面挑了10个test的E值，用E去生成了图像，再回归，将对比图保存在了validation_results文件夹下

![image](https://github.com/zhaolin191544/wink/blob/main/image-20250621230856874.png)
