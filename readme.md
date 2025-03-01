# 这是处理光谱（类）的 机器学习算法

## 主要内容

- [classfy] : 用于分类
  - svm
  - cnn
  - mlp

- cluster : 用于聚类

- proprecess : 数据处理相关

- examples  : 示例

## 使用说明

1. 创建环境
下载 miniconda 

2. 创建虚拟环境 
（虚拟环境指的是一个python 执行环境，为什么要用虚拟环境？因为不同的项目对需要的外部依赖库版本不同，所以每一个项目可以有一个独立的环境，叫做虚拟环境）

```
conda create --name food python=3.10
// 创建food 虚拟环境，python 版本为3.10

conda activate food
// 进入虚拟环境 food 

pip install ./
// 安装

``` 
3. 使用


## 主要方法介绍

### 分类算法

```
使用决策树
from food.classify.function import decision_tree

if __name__ == "__main__":
    file_path = 'food/data/data.csv'  # 替换为你的文件路径
    print ("使用decision_tree")
    decision_tree(file_path)

```
```
使用cnn
from food.classify.function import cnn
if __name__ == "__main__":
    file_path = 'food/data/data.csv'  # 替换为你的文件路径
    print ("使用CNN")
    cnn(file_path)
```
```
使用 svm
from food.classify.function import svm
  if __name__ == "__main__":
    file_path = 'food/data/data.csv'  # 替换为你的文件路径
    print ("使用svm")
    svm(file_path)
```



## SVM 介绍
支持向量机（Support Vector Machine，简称SVM）是一种监督学习算法，它的原理是通过最大间隔分类来实现。下面用符号和公式来介绍 SVM 的原理：

### 问题定义

假设我们有一个二元分类问题，数据集 $(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$，其中 $x_i \in R^d$ 是特征向量，$y_i \in {-1,1}$ 是标签。我们的目标是找到一个决策函数 $f(x)=sign(w^Tx+b)$，使得这个决策函数可以将所有的训练数据正确分类。

### Soft Margin

我们首先引入一个假设，存在一个超平面，使得这个超平面尽量远离训练数据，同时也要使得这个超平面能够将所有的训练数据正确分类。这个假设称为软间隔（Soft Margin）。

$$\min_{w,b,\xi} \frac{1}{2}||w||^2+C\sum_{i=1}^{n}\xi_i$$

$$s.t.\quad y_i(w^Tx_i+b) \ge 1-\xi_i,\quad i=1,2,...,n$$

其中 $w$ 是超平面的法向量，$b$ 是超平面的截距，$\xi_i$ 是松弛变量，$C$ 是 penalty parameter。

### Hard Margin

如果没有松弛变量，那么我们就可以引入一个新的假设，存在一个超平面，使得这个超平面尽量远离训练数据，同时也要使得这个超平面能够将所有的训练数据正确分类。这称为硬间隔（Hard Margin）。

$$\min_{w,b} \frac{1}{2}||w||^2$$

$$s.t.\quad y_i(w^Tx_i+b) \ge 1,\quad i=1,2,...,n$$

但是，硬间隔假设很难满足，因为很多数据点无法被正确分类。

### Dual Problem

我们可以将原始问题转换为双变量优化问题（Dual Problem），这样可以避免硬间隔问题。

$$\min_{\alpha} \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_jx_i^Tx_j+C\sum_{i=1}^{n}\alpha_i$$

$$s.t.\quad 0 \le \alpha_i \le C,\quad i=1,2,...,n$$

$$\sum_{i=1}^{n}\alpha_i y_i = 0$$

其中 $\alpha_i$ 是 Lagrange multiplier。

### KKT Condition

我们可以使用 Karush-Kuhn-Tucker 条件（KKT Condition）来简化优化问题。

$$w=\sum_{i=1}^{n}\alpha_iy_ix_i$$

$$\alpha_i(C-\xi_i)=0,\quad i=1,2,...,n$$

$$\xi_i \ge 0,\quad i=1,2,...,n$$

### Kernel Trick

为了使得 SVM 可以处理高维空间，我们可以使用核技巧（Kernel Trick）。

$$k(x,x')=\phi(x)^T\phi(x')$$

$$f(x)=sign(\sum_{i=1}^{n}\alpha_iy_ik(x,x_i)+b$$

其中 $\phi(x)$ 是特征映射函数，$k(x,x')$ 是核函数。

### SVM 的优化算法

SVM 的优化算法是使用 Sequential Minimal Optimization（SMO）算法，通过迭代计算来找到最优解。

### 优化步骤

1. 初始化 $\alpha_i$ 和 $\xi_i$
2. 计算 $w$ 和 $b$
3. 更新 $\alpha_i$ 和 $\xi_i$
4. 重复步骤 2 和 3，直到收敛


SVM 的原理是通过最大间隔分类来实现，软间隔和硬间隔都是假设。我们可以使用双变量优化问题和 KKT 条件来简化优化问题，然后使用核技巧来处理高维空间最后使用 SMO 算法来实现优化。

## CNN 介绍

 convolutional neural network（CNN）是一种深度学习算法，用于图像识别和分类。下面用公式来介绍 CNN 的原理：

### 卷积层的公式

卷积层是 CNN 中最重要的一层，它的公式如下：

$$y[i,j]=\sigma(b+\sum_{k=1}^{N_c}\sum_{m=1}^{H_f}\sum_{n=1}^{W_f}x[m,n] \cdot w[k,m,n]$$

其中：

* $y[i,j]$ 是输出结果
* $b$ 是偏移量
* $x[m,n]$ 是输入图像
* $w[k,m,n]$ 是权重参数
* $N_c$ 是通道数
* $H_f$ 和 $W_f$ 是过滤器的高度和宽度
* $\sigma$ 是激活函数，通常是 ReLU 或 Sigmoid 函数

### 卷积操作

卷积操作可以看作是一种窗口滑动的过程，每个窗口都计算一个特征图像。公式如下：

$$y[i,j]=\sigma(b+\sum_{k=1}^{N_c}\sum_{m=1}^{H_f}\sum_{n=1}^{W_f}x[m+i-1,n+j-1] \cdot w[k,m,n]$$

其中 $i$ 和 $j$ 是窗口的坐标，$m$ 和 $n$ 是图像的坐标。

### 池化层的公式

池化层是用来降低特征图像的维度，提高计算效率。公式如下：

$$y[i,j]=\sigma(f(x[m+i-1,n+j-1])$$

其中 $f(x)$ 是池化函数，通常是最大值或平均值。

### 全连接层的公式

全连接层是将卷积和池化后的特征图像组合成一个向量。公式如下：

$$y[i]=\sigma(b+\sum_{j=1}^{N_c}x[j] \cdot w[j,i]$$

其中 $w[j,i]$ 是权重参数。

### Softmax 层的公式

softmax 层是用于多分类问题，公式如下：

$$P(y=c)=\frac{e^{z_c}}{\sum_{i=1}^{N_c}e^{z_i}}$$

其中 $z_i$ 是输出结果，$N_c$ 是类别数。

### CNN 的优化目标

CNN 的优化目标是最小化损失函数。公式如下：

$$L=\frac{1}{n}\sum_{i=1}^n L(y_i,o_i)$$

其中 $y_i$ 是真实标签，$o_i$ 是预测结果，$n$ 是样本数。

### CNN 的优化算法

CNN 的优化算法是使用 Stochastic Gradient Descent（SGD）或 Adam 算法。公式如下：

$$w_{t+1}=w_t-\alpha \cdot \frac{\partial L}{\partial w}$$

其中 $w_t$ 是权重参数，$\alpha$ 是学习率。


CNN 的原理是卷积、池化和全连接的组合。卷积层用于提取特征，池化层用于降低维度，全连接层用于组合特征。优化目标是最小化损失函数，然后使用 SGD 或 Adam 算法来优化权重参数。

