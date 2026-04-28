"""
PINN (Physics-Informed Neural Networks) 求解一维对流扩散方程
方程: ∂u/∂t + a * ∂u/∂x = ν * ∂²u/∂x²
精确解: u(x,t) = exp(-νπ²t) * sin(π(x - a*t))
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from timeit import default_timer

# ==================== 1. 环境设置 ====================
os.makedirs("Results", exist_ok=True)
print("📁 创建Results文件夹")

# 设置默认数据类型
torch.set_default_dtype(torch.float32)

# 设备检测 (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 随机种子设置 (确保可重复性)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 物理参数 (对流扩散方程系数)
a_true = 1.0      # 对流速度
v_true = 0.01     # 扩散系数

# 训练数据批量大小
batch_size_pde = 1000  # PDE内部点
batch_size_ic = 50     # 初始条件点
batch_size_bc = 100    # 边界条件点


# ==================== 2. 神经网络模型定义 ====================
class PINN(nn.Module):
    """物理信息神经网络 (PINN)"""

    def __init__(self, input_dim=2, output_dim=1, hidden_dim=50, num_hidden=3, activation='tanh'):
        """
        初始化PINN模型
        参数:
            input_dim: 输入维度 (x, t)
            output_dim: 输出维度 (u)
            hidden_dim: 隐藏层神经元数量
            num_hidden: 隐藏层数量
            activation: 激活函数 ('tanh' 或 'sin')
        """
        super(PINN, self).__init__()

        # 构建神经网络层
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # 设置激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        else:
            raise ValueError("Activation function should be 'tanh' or 'sin'")

    def forward(self, x, t):
        """
        前向传播
        参数:
            x: 空间坐标张量 [batch_size, 1]
            t: 时间坐标张量 [batch_size, 1]
        返回:
            u: 预测值 [batch_size, 1]
        """
        input_combined = torch.cat([x, t], dim=-1)  # 拼接输入: [x, t]
        out = input_combined

        # 前向传播 (隐藏层使用激活函数，输出层不使用)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)

        return out

    def calculate_derivatives(self, x, t):
        """
        计算偏导数 (自动微分)
        参数:
            x: 空间坐标张量
            t: 时间坐标张量
        返回:
            u_pred: 预测值
            u_xx: 二阶空间导数 (∂²u/∂x²)
            u_t: 时间导数 (∂u/∂t)
            u_x: 一阶空间导数 (∂u/∂x)
        """
        # 启用梯度计算
        x = x.requires_grad_()
        t = t.requires_grad_()

        # 计算预测值
        u_pred = self.forward(x, t)

        # 计算一阶空间导数 ∂u/∂x
        u_x = grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

        # 计算二阶空间导数 ∂²u/∂x²
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # 计算时间导数 ∂u/∂t
        u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

        return u_pred, u_xx, u_t, u_x

    def losses(self, x, t):
        """
        计算总损失函数 (PDE残差 + 初始条件 + 边界条件)
        参数:
            x: 字典，包含 'pde', 'ic', 'bc' 的坐标数据
            t: 字典，包含 'pde', 'ic', 'bc' 的时间数据
        返回:
            total_loss: 总损失
            loss_pde: PDE残差损失
            loss_ic: 初始条件损失
            loss_bc: 边界条件损失
        """
        # 1. PDE损失: 强制满足对流扩散方程
        u_pde, u_xx, u_t, u_x = self.calculate_derivatives(x['pde'], t['pde'])
        residual = u_t + a_true * u_x - v_true * u_xx  # ∂u/∂t + a*∂u/∂x - ν*∂²u/∂x²
        loss_pde = torch.mean(residual**2)

        # 2. 初始条件损失: 强制满足 u(x,0) = sin(πx)
        x_ic = x['ic'].requires_grad_()
        t_ic = t['ic'].requires_grad_()
        u_ic = self.forward(x_ic, t_ic)
        loss_ic = torch.mean((u_ic - torch.sin(torch.pi * x['ic'])) ** 2)

        # 3. 边界条件损失: 强制满足边界条件
        u_bc_pred = self.forward(x['bc'], t['bc'])
        # 边界条件公式: u(0,t) = -exp(-νπ²t)*sin(π(1-a*t)), u(1,t) = +exp(-νπ²t)*sin(π(1-a*t))
        # 使用 (2x-1) 技巧: x=0时=-1, x=1时=+1
        u_bc_true = (2 * x['bc'] - 1) * torch.exp(-v_true * (torch.pi ** 2) * t['bc']) * torch.sin(
            torch.pi * (1 - a_true * t['bc']))
        loss_bc = torch.mean((u_bc_pred - u_bc_true) ** 2)

        # 4. 总损失 (加权和)
        total_loss = loss_pde + 100 * loss_ic + 100 * loss_bc

        # 记录损失历史
        if not hasattr(self, 'history'):
            self.history = {
                'total_loss': [],
                'loss_pde': [],
                'loss_ic': [],
                'loss_bc': []
            }

        self.history['total_loss'].append(total_loss.item())
        self.history['loss_pde'].append(loss_pde.item())
        self.history['loss_ic'].append(loss_ic.item())
        self.history['loss_bc'].append(loss_bc.item())

        return total_loss, loss_pde, loss_ic, loss_bc


# ==================== 3. 数据生成函数 ====================
def get_data(a, v, batch_size_pde, batch_size_ic, batch_size_bc):
    """
    生成训练和测试数据
    参数:
        a: 对流速度
        v: 扩散系数
        batch_size_pde: PDE点采样数量
        batch_size_ic: 初始条件点采样数量
        batch_size_bc: 边界条件点采样数量
    返回:
        X, T: 网格坐标 (用于可视化)
        x_test, t_test, u_test: 测试数据 (完整网格)
        x_train, t_train: 训练数据坐标
        U: 精确解网格 (用于误差计算)
    """
    # 创建空间和时间网格
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)

    # 精确解: u(x,t) = exp(-νπ²t) * sin(π(x - a*t))
    U = np.exp(-v * (np.pi**2) * T) * np.sin(np.pi * (X - a * T))

    # ===== 1. 生成PDE训练点 (内部点) =====
    x_pde = x[1:-1]  # 排除边界
    t_pde = t[1:-1]  # 排除初始时刻
    X_pde, T_pde = np.meshgrid(x_pde, t_pde)
    X_pde_flat = X_pde.flatten()[:, None]  # 展平为列向量
    T_pde_flat = T_pde.flatten()[:, None]
    U_pde_flat = U[1:-1, 1:-1].flatten()[:, None]

    # 随机选择PDE点
    indices_pde = np.random.choice(len(X_pde_flat), batch_size_pde, replace=False)
    x_pde_train = X_pde_flat[indices_pde]
    t_pde_train = T_pde_flat[indices_pde]

    # ===== 2. 生成初始条件训练点 (t=0) =====
    x_ic = x[1:-1]  # 排除边界
    t_ic = np.zeros_like(x_ic)  # t=0
    u_ic = U[0, 1:-1]  # t=0时的精确解

    indices_ic = np.random.choice(len(x_ic), batch_size_ic, replace=False)
    x_ic_train = x_ic[indices_ic][:, None]
    t_ic_train = t_ic[indices_ic][:, None]

    # ===== 3. 生成边界条件训练点 (x=0 和 x=1) =====
    t_bc = t[1:]  # 排除t=0 (已在初始条件中)

    # 左边界 (x=0)
    x_bc_left = np.zeros_like(t_bc)

    # 右边界 (x=1)
    x_bc_right = np.ones_like(t_bc)

    # 合并左右边界
    x_bc = np.concatenate((x_bc_left, x_bc_right))[:, None]
    t_bc_concat = np.concatenate((t_bc, t_bc))[:, None]

    # 随机选择边界点
    indices_bc = np.random.choice(len(x_bc), batch_size_bc, replace=False)
    x_bc_train = x_bc[indices_bc]
    t_bc_train = t_bc_concat[indices_bc]

    # ===== 4. 组织训练数据 =====
    x_train = {'pde': x_pde_train, 'ic': x_ic_train, 'bc': x_bc_train}
    t_train = {'pde': t_pde_train, 'ic': t_ic_train, 'bc': t_bc_train}

    # 转换为PyTorch张量并移动到设备
    for key in x_train.keys():
        x_train[key] = torch.from_numpy(x_train[key]).float().to(device)
        t_train[key] = torch.from_numpy(t_train[key]).float().to(device)

    # ===== 5. 生成测试数据 (完整网格，用于评估) =====
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    u_test = U.flatten()[:, None]

    x_test = torch.from_numpy(x_test).float().to(device)
    t_test = torch.from_numpy(t_test).float().to(device)
    u_test = torch.from_numpy(u_test).float().to(device)

    return X, T, x_test, t_test, u_test, x_train, t_train, U


# ==================== 4. 训练函数 ====================
def train_pinn_adam(pinn, x_train, t_train, epochs=5000, lr=1e-3):
    """
    使用Adam优化器训练PINN
    参数:
        pinn: PINN模型实例
        x_train: 训练数据x坐标
        t_train: 训练数据t坐标
        epochs: 训练轮次
        lr: 学习率
    """
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    print("开始Adam训练")

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, loss_pde, loss_ic, loss_bc = pinn.losses(x_train, t_train)
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"轮次 {epoch}: 总损失 = {total_loss.item():.6e}, "
                  f"PDE损失 = {loss_pde.item():.6e}, "
                  f"IC损失 = {loss_ic.item():.6e}, "
                  f"BC损失 = {loss_bc.item():.6e}")

    print("Adam训练完成")


def train_pinn_lbfgs(pinn, x_train, t_train, max_iter=500, lr=0.1,
                     tolerance_grad=1e-7, tolerance_change=1e-9):
    """
    使用L-BFGS优化器训练PINN (二阶优化，收敛更快)
    参数:
        pinn: PINN模型实例
        x_train: 训练数据x坐标
        t_train: 训练数据t坐标
        max_iter: 最大迭代次数
        lr: 学习率
        tolerance_grad: 梯度容差
        tolerance_change: 变化容差
    """
    optimizer = torch.optim.LBFGS(
        pinn.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
        history_size=100
    )

    def closure():
        """L-BFGS需要的闭包函数"""
        optimizer.zero_grad()
        total_loss, _, _, _ = pinn.losses(x_train, t_train)
        total_loss.backward(retain_graph=True)
        return total_loss

    print("开始L-BFGS训练...")
    optimizer.step(closure)
    print("L-BFGS训练完成。")


# ==================== 5. 可视化函数 ====================
def plot_sampling_points(x_train, t_train):
    """
    可视化采样点分布
    参数:
        x_train: 训练数据x坐标
        t_train: 训练数据t坐标
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train['pde'].cpu().numpy(), t_train['pde'].cpu().numpy(),
                s=10, c='b', label='PDE Points')
    plt.scatter(x_train['ic'].cpu().numpy(), t_train['ic'].cpu().numpy(),
                s=10, c='r', label='Initial Condition Points')
    plt.scatter(x_train['bc'].cpu().numpy(), t_train['bc'].cpu().numpy(),
                s=10, c='g', label='Boundary Condition Points')
    plt.title('Distribution of Sampling Points: PDE, Initial Condition, and Boundary Condition')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_history(pinn):
    """
    绘制训练损失历史
    参数:
        pinn: PINN模型实例 (需有history属性)
    """
    if not hasattr(pinn, 'history'):
        print("No loss history available.")
        return

    epochs = range(len(pinn.history['total_loss']))

    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, pinn.history['total_loss'], label='Total Loss')
    plt.semilogy(epochs, pinn.history['loss_pde'], label='PDE Residual Loss')
    plt.semilogy(epochs, pinn.history['loss_ic'], label='Initial Condition Loss')
    plt.semilogy(epochs, pinn.history['loss_bc'], label='Boundary Condition Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Results/loss_history.png")
    plt.show()


def relative_l2_error(pred, true):
    """
    计算相对L2误差
    参数:
        pred: 预测值
        true: 真实值
    返回:
        相对L2误差
    """
    return torch.norm(pred - true) / torch.norm(true)


# ==================== 6. 主程序 ====================
if __name__ == "__main__":
    # 生成数据
    print("生成训练和测试数据...")
    X, T, x_test, t_test, u_test, x_train, t_train, U = get_data(
        a_true, v_true, batch_size_pde, batch_size_ic, batch_size_bc
    )

    # 可视化采样点分布
    plot_sampling_points(x_train, t_train)

    # 创建PINN模型
    print("初始化PINN模型...")
    model = PINN(
        input_dim=2,
        output_dim=1,
        hidden_dim=100,
        num_hidden=3,
        activation='tanh'
    ).to(device)

    # 训练模型 (Adam + L-BFGS组合训练)
    print("开始训练...")
    train_pinn_adam(model, x_train, t_train, epochs=2000, lr=1e-3)
    train_pinn_lbfgs(model, x_train, t_train, max_iter=500, lr=0.1)

    # 绘制损失历史
    plot_loss_history(model)

    # 评估模型
    print("评估模型性能...")
    u_pred = model(x_test, t_test).cpu().detach().numpy()
    u_error = relative_l2_error(torch.tensor(u_pred), u_test.cpu())
    print(f"相对L2误差: {u_error.item():.6e}")

    # 重塑预测结果为网格形状
    u_pred_reshaped = u_pred.reshape(101, 101)

    # 计算绝对误差
    abs_error = np.abs(U - u_pred_reshaped)

    # ==================== 7. 结果可视化 ====================
    # 创建可视化图形 (真实解、预测解、误差)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 绘制真实解
    contour_gt = axs[0].contourf(X, T, U, levels=50, cmap="viridis",
                                 vmin=np.min(U), vmax=np.max(U))
    axs[0].set_title("Ground Truth")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    plt.colorbar(contour_gt, ax=axs[0], label="True u(x, t)")

    # 绘制预测解
    contour_pred = axs[1].contourf(X, T, u_pred_reshaped, levels=50, cmap="viridis",
                                   vmin=np.min(U), vmax=np.max(U))
    axs[1].set_title("PINN Prediction")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")
    plt.colorbar(contour_pred, ax=axs[1], label="Predicted u(x, t)")

    # 绘制绝对误差
    contour_error = axs[2].contourf(X, T, abs_error, levels=50, cmap="inferno",
                                    vmin=0, vmax=np.max(abs_error))
    axs[2].set_title("Absolute Error")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("t")
    plt.colorbar(contour_error, ax=axs[2], label="Absolute Error")

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("Results/ground_truth_prediction_error.png")
    plt.show()

    print("程序执行完成！结果保存在Results文件夹中。")

