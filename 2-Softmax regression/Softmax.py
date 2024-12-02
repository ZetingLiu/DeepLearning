import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_data_fashion_mnist(input_batch_size):
    """
    加载FashionMNIST数据集并返回数据迭代器
    Load FashionMNIST dataset and return data iterators.

    Args:
        input_batch_size (int): 批量大小 / Batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: 训练数据迭代器和测试数据迭代器 / Training and test data iterators.
    """
    transform = transforms.ToTensor()
    train_dataset = FashionMNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = FashionMNIST(
        root='./data', train=False, transform=transform, download=True
    )
    output_train_iter = DataLoader(
        train_dataset, batch_size=input_batch_size, shuffle=True
    )
    input_test_iter = DataLoader(
        test_dataset, batch_size=input_batch_size, shuffle=False
    )
    return output_train_iter, input_test_iter


# 设置全局参数 / Set global parameters
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 初始化模型参数 / Initialize model parameters
num_inputs, num_outputs = 784, 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(x):
    """
    计算输入x的softmax / Compute the softmax of input x.

    Args:
        x (torch.Tensor): 输入张量 / Input tensor.

    Returns:
        torch.Tensor: softmax后的张量 / Softmax tensor.
    """
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


def net(x):
    """
    定义网络 / Define the network.

    Args:
        x (torch.Tensor): 输入张量 / Input tensor.

    Returns:
        torch.Tensor: 网络输出 / Network output.
    """
    return softmax(torch.matmul(x.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, input_y):
    """
    交叉熵损失函数 / Cross-entropy loss function.

    Args:
        y_hat (torch.Tensor): 模型预测值 / Predicted values.
        input_y (torch.Tensor): 实际标签 / Actual labels.

    Returns:
        torch.Tensor: 交叉熵损失值 / Cross-entropy loss.
    """
    return -torch.log(y_hat[range(len(y_hat)), input_y] + 1e-9)


def accuracy(y_hat, input_y):
    """
    计算准确率 / Compute accuracy.

    Args:
        y_hat (torch.Tensor): 模型预测值 / Predicted values.
        input_y (torch.Tensor): 实际标签 / Actual labels.

    Returns:
        float: 准确率 / Accuracy.
    """
    if y_hat.ndimension() > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(input_y.dtype) == input_y
    return float(cmp.sum())


class Accumulator:
    """
    用于累加数据的类 / Class for accumulating data.
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        """
        累加多个参数的值 / Add values of multiple arguments.

        Args:
            *args: 要累加的值 / Values to accumulate.
        """
        self.data = [a + float(bia) for a, bia in zip(self.data, args)]

    def reset(self):
        """重置累加器 / Reset the accumulator."""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的值 / Get value at specified index.

        Args:
            idx (int): 索引值 / Index.

        Returns:
            float: 累加的值 / Accumulated value.
        """
        return self.data[idx]


def evaluate_accuracy(input_net, data_iter):
    """
    评估模型在数据集上的准确率 / Evaluate model accuracy on a dataset.

    Args:
        input_net (callable): 网络函数 / Network function.
        data_iter (DataLoader): 数据迭代器 / Data iterator.

    Returns:
        float: 模型准确率 / Model accuracy.
    """
    metric = Accumulator(2)
    for input_X, input_y in data_iter:
        metric.add(accuracy(input_net(input_X), input_y), input_y.numel())
    return metric[0] / metric[1] if metric[1] > 0 else 0


def train_epoch_ch3(input_net, input_train_iter, loss, input_updater):
    """
    训练模型一个epoch / Train the model for one epoch.

    Args:
        input_net (callable): 网络函数 / Network function.
        input_train_iter (DataLoader): 训练数据迭代器 / Training data iterator.
        loss (callable): 损失函数 / Loss function.
        input_updater (callable): 参数更新函数 / Parameter updater.

    Returns:
        Tuple[float, float]: 平均损失和准确率 / Average loss and accuracy.
    """
    metric = Accumulator(3)
    for input_X, input_y in input_train_iter:
        y_hat = input_net(input_X)
        loss_value = loss(y_hat, input_y).sum()
        loss_value.backward()
        input_updater()
        W.grad.zero_()
        b.grad.zero_()
        metric.add(float(loss_value), accuracy(y_hat, input_y), input_y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


lr = 2e-4  # 学习率 / Learning rate


def updater():
    """更新模型参数 / Update model parameters."""
    global W, b
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad


def train_ch3(input_net, input_train_iter, input_test_iter, loss, input_num_epochs, input_updater):
    """
    训练模型 / Train the model.

    Args:
        input_net (callable): 网络函数 / Network function.
        input_train_iter (DataLoader): 训练数据迭代器 / Training data iterator.
        input_test_iter (DataLoader): 测试数据迭代器 / Test data iterator.
        loss (callable): 损失函数 / Loss function.
        input_num_epochs (int): 训练轮数 / Number of training epochs.
        input_updater (callable): 参数更新函数 / Parameter updater.
    """
    for epoch in range(input_num_epochs):
        train_metrics = train_epoch_ch3(
            input_net, input_train_iter, loss, input_updater
        )
        test_acc = evaluate_accuracy(input_net, input_test_iter)
        print(
            f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, '
            f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}'
        )


def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    """
    显示图片 / Display images.

    Args:
        images (List[torch.Tensor]): 图片列表 / List of images.
        num_rows (int): 行数 / Number of rows.
        num_cols (int): 列数 / Number of columns.
        titles (List[str], optional): 图片标题 / Titles of images. Defaults to None.
        scale (float, optional): 图片缩放比例 / Scale factor. Defaults to 1.5.
    """
    figure_size = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figure_size)
    axes = axes.flatten()
    for i, (img, ax) in enumerate(zip(images, axes)):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


# 全局变量初始化
X = None
y = None


def predict_ch3(input_net, input_test_iter, n=6):
    """
    预测并显示结果 / Predict and display results.

    Args:
        input_net (callable): 网络函数 / Network function.
        input_test_iter (DataLoader): 测试数据迭代器 / Test data iterator.
        n (int, optional): 显示图片数量 / Number of images to display. Defaults to 6.
    """
    global X, y
    for X, y in input_test_iter:
        break
    trues = [str(y[i].item()) for i in range(n)]
    predictions = [
        str(input_net(X).argmax(dim=1)[i].item()) for i in range(n)
    ]
    titles = [true + '\n' + pred for true, pred in zip(trues, predictions)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles)


def train_ch3_with_metrics(input_net, input_train_iter, input_test_iter, loss, input_num_epochs, input_updater):
    """
    训练模型并记录损失与准确率 / Train the model and record loss and accuracy.

    Args:
        input_net (callable): 网络函数 / Network function.
        input_train_iter (DataLoader): 训练数据迭代器 / Training data iterator.
        input_test_iter (DataLoader): 测试数据迭代器 / Test data iterator.
        loss (callable): 损失函数 / Loss function.
        input_num_epochs (int): 训练轮数 / Number of training epochs.
        input_updater (callable): 参数更新函数 / Parameter updater.

    Returns:
        Tuple[List[float], List[float], List[float]]:
        训练损失、训练准确率、测试准确率 / Training losses, training accuracies, test accuracies.
    """
    train_losses = []  # 保存每轮的训练损失 / To store training losses
    train_accuracies = []  # 保存每轮的训练准确率 / To store training accuracies
    test_accuracies = []  # 保存每轮的测试准确率 / To store test accuracies

    for epoch in range(input_num_epochs):
        metric = Accumulator(3)  # 损失值、正确预测数、样本总数 / Loss, correct predictions, total samples

        for X, y in input_train_iter:
            y_hat = input_net(X)
            loss_value = loss(y_hat, y).sum()
            loss_value.backward()
            input_updater()
            W.grad.zero_()
            b.grad.zero_()
            metric.add(float(loss_value), accuracy(y_hat, y), y.numel())

        train_loss = metric[0] / metric[2]  # 平均损失 / Average loss
        train_acc = metric[1] / metric[2]  # 训练准确率 / Training accuracy
        test_acc = evaluate_accuracy(input_net, input_test_iter)  # 测试准确率 / Test accuracy

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f'epoch {epoch + 1}, loss {train_loss:.4f}, '
              f'train acc {train_acc:.4f}, test acc {test_acc:.4f}')

    return train_losses, train_accuracies, test_accuracies


def plot_metrics(train_losses, train_accuracies, test_accuracies):
    """
    绘制损失曲线、训练准确率和测试准确率曲线
    Plot loss curve, training accuracy, and test accuracy.

    Args:
        train_losses (list): 训练损失值列表 / List of training losses.
        train_accuracies (list): 训练准确率列表 / List of training accuracies.
        test_accuracies (list): 测试准确率列表 / List of test accuracies.
    """
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线 / Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制训练和测试准确率曲线 / Plot training and test accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


# 设置训练参数 / Set training parameters
num_epochs = 50
train_losses, train_accuracies, test_accuracies = train_ch3_with_metrics(
    net, train_iter, test_iter, cross_entropy, num_epochs, updater
)

# 绘制训练曲线 / Plot training curves
plot_metrics(train_losses, train_accuracies, test_accuracies)

# 预测并显示结果 / Predict and display results
predict_ch3(net, test_iter)


