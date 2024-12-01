"""
从零开始实现线性回归，包括数据流水线、模型、损失函数、小批量随机梯度下降优化器，以及训练过程中的损失曲线绘制。
Linear regression implementation from scratch, including data pipeline, model,
loss function, mini-batch stochastic gradient descent optimizer, and loss curve plotting.
"""

# 导入必要的包 / Import necessary packages
import random  # 用于生成随机数 / For generating random numbers
import torch  # 导入 PyTorch，进行张量计算与深度学习 / For tensor computations and deep learning
from d2l import torch as d2l  # 可视化数据与结果 / For data visualization and result plotting
import matplotlib.pyplot as plt  # 绘制损失曲线 / For plotting the loss curve


# 生成数据集 / Generate dataset
def create_dataset(input_w, input_b, num_examples):
    """
    生成 y = xw + b + 噪声 / Generate y = xw + b + noise.

    Args:
        input_w (torch.Tensor): 权重参数 / Weight parameter.
        input_b (float): 偏差参数 / Bias parameter.
        num_examples (int): 样本数 / Number of examples.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 输入特征和输出标签 / Features and labels.
    """
    input_x = torch.normal(0, 1, (num_examples, len(input_w)))  # 随机生成输入数据 / Generate random input data
    output_y = torch.matmul(input_x, input_w) + input_b  # 计算输出数据 / Compute output data
    output_y += torch.normal(0, 0.01, output_y.shape)  # 加入噪声 / Add noise
    return input_x, output_y.reshape((-1, 1))  # 转为列向量 / Reshape to column vector


# 真实参数 / True parameters
true_w = torch.tensor([2, -3.4])  # 权重 / Weights
true_b = 4.2  # 偏差 / Bias
features, labels = create_dataset(true_w, true_b, 1000)  # 生成数据集 / Generate dataset

# 绘制数据集散点图 / Plot dataset scatter plot
d2l.set_figsize()  # 设置图像大小 / Set figure size
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)  # 绘制散点图 / Plot scatter points
d2l.plt.show()


# 生成批量数据 / Generate mini-batch data
def data_iter(input_batch_size, input_features, input_labels):
    """
    生成批量数据迭代器 / Generate mini-batch data iterator.

    Args:
        input_batch_size (int): 批量大小 / Batch size.
        input_features (torch.Tensor): 输入特征 / Input features.
        input_labels (torch.Tensor): 输出标签 / Output labels.

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: 批量特征和标签 / Batch features and labels.
    """
    num_examples = len(input_features)  # 样本数 / Number of samples
    indices = list(range(num_examples))  # 样本索引 / Sample indices
    random.shuffle(indices)  # 随机打乱顺序 / Shuffle the order randomly
    for i in range(0, num_examples, input_batch_size):  # 按批量大小分割样本 / Split samples by batch size
        batch_indices = torch.tensor(
            indices[i: min(i + input_batch_size, num_examples)]
        )  # 获取当前批次的索引 / Get indices for the current batch
        yield input_features[batch_indices], input_labels[batch_indices]  # 返回批量数据 / Return batch data


# 初始化模型参数 / Initialize model parameters
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 权重参数 / Weight parameter
b = torch.zeros(1, requires_grad=True)  # 偏差参数 / Bias parameter


# 定义线性回归模型 / Define linear regression model
def linreg(input_x, input_w, input_b):
    """
    线性回归模型 / Linear regression model.

    Args:
        input_x (torch.Tensor): 输入特征 / Input features.
        input_w (torch.Tensor): 权重参数 / Weight parameter.
        input_b (torch.Tensor): 偏差参数 / Bias parameter.

    Returns:
        torch.Tensor: 模型输出 / Model output.
    """
    return torch.matmul(input_x, input_w) + input_b


# 定义损失函数 / Define loss function
def squared_loss(y_hat, output_y):
    """
    均方损失函数 / Squared loss function.

    Args:
        y_hat (torch.Tensor): 预测值 / Predicted values.
        output_y (torch.Tensor): 实际值 / Actual values.

    Returns:
        torch.Tensor: 均方误差 / Squared error.
    """
    return (y_hat - output_y.reshape(y_hat.shape)) ** 2 / 2


# 定义随机梯度下降优化器 / Define stochastic gradient descent optimizer
def sgd(params, learning_rate, input_batch_size):
    """
    小批量随机梯度下降优化器 / Mini-batch stochastic gradient descent optimizer.

    Args:
        params (List[torch.Tensor]): 模型参数 / Model parameters.
        learning_rate (float): 学习率 / Learning rate.
        input_batch_size (int): 批量大小 / Batch size.
    """
    with torch.no_grad():  # 禁用梯度跟踪 / Disable gradient tracking
        for param in params:  # 遍历参数 / Iterate over parameters
            param -= learning_rate * param.grad / input_batch_size  # 更新参数 / Update parameters
            param.grad.zero_()  # 清空梯度 / Clear gradients


# 绘制损失曲线 / Plot loss curve
def plot_loss_curve(input_losses):
    """
    绘制损失曲线 / Plot the loss curve.

    Args:
        input_losses (list): 每轮的平均损失 / List of average losses per epoch.
    """
    plt.figure(figsize=(8, 6))  # 设置图像大小 / Set figure size
    plt.plot(range(1, len(input_losses) + 1), input_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch', fontsize=14)  # 设置x轴标签 / Set x-axis label
    plt.ylabel('Loss', fontsize=14)  # 设置y轴标签 / Set y-axis label
    plt.title('Training Loss Curve', fontsize=16)  # 设置标题 / Set title
    plt.grid(True)  # 显示网格 / Display grid
    plt.show()  # 显示图像 / Show plot


# 训练模型并记录损失 / Train the model and record the loss
lr = 0.03  # 学习率 / Learning rate
num_epochs = 50  # 迭代次数 / Number of epochs
batch_size = 32  # 批量大小 / Batch size
net = linreg  # 选择模型 / Select model
loss = squared_loss  # 选择损失函数 / Select loss function

losses = []  # 初始化损失列表 / Initialize list to store losses

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        batch_loss = loss(net(x, w, b), y)  # 计算损失 / Compute loss
        batch_loss.sum().backward()  # 反向传播 / Backward propagation
        sgd([w, b], lr, batch_size)  # 更新参数 / Update parameters using SGD
    with torch.no_grad():  # 禁用梯度跟踪 / Disable gradient tracking
        train_l = loss(net(features, w, b), labels)  # 计算训练集损失 / Compute training loss
        losses.append(float(train_l.mean()))  # 记录当前轮的平均损失 / Record current epoch's average loss
        print(f"epoch {epoch + 1}, loss {losses[-1]:f}")  # 打印当前轮损失 / Print current epoch loss

# 绘制损失曲线 / Plot the loss curve
plot_loss_curve(losses)

# 打印估计误差 / Print estimation errors
print(f"w的估计误差为：{true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差为：{true_b - b}")
