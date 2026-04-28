import os
import time
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm, trange
from pyDOE import lhs


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

use_gpu = torch.cuda.is_available()
print('GPU:', use_gpu)


def random_fun(num):
    temp = torch.from_numpy(lb + (ub - lb) * lhs(3, num)).float()
    if use_gpu:
        temp = temp.cuda()
    return temp


def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


class Net_attention(nn.Module):
    def __init__(self, layers):
        super(Net_attention, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
        nn.init.xavier_normal_(self.attention1.weight.data, gain=1.0)
        nn.init.zeros_(self.attention1.bias.data)
        nn.init.xavier_normal_(self.attention2.weight.data, gain=1.0)
        nn.init.zeros_(self.attention2.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        a = a * encoder_1 + (1 - a) * encoder_2
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        a = self.linear[-1](a)
        return a


class Model:
    def __init__(self, net, x_bc_x_left, x_bc_x_right, x_bc_y_bottom, x_bc_y_top,
                 x_ic, u_ic, x_f_loss_fun, x_test, x_test_exact):
        
        self.net = net
        self.optimizer_LBGFS = None

        # 接收 2D 的 4 面周期性边界
        self.x_bc_x_left = x_bc_x_left.requires_grad_(True) 
        self.x_bc_x_right = x_bc_x_right.requires_grad_(True)
        self.x_bc_y_bottom = x_bc_y_bottom.requires_grad_(True)
        self.x_bc_y_top = x_bc_y_top.requires_grad_(True)
        
        self.x_ic = x_ic
        self.u_ic = u_ic
        self.x_f_loss_fun = x_f_loss_fun
        self.x_test = x_test
        self.x_test_exact = x_test_exact
        
        self.x_f_N = None
        self.x_f_M = None

        # ====== Task 2: AMAW 自适应权重参数初始化 ======
        self.x_f_s = nn.Parameter(is_cuda(torch.tensor(0.).float()), requires_grad=True)
        # IC权重设为100 (e^-(-ln(100)) = 100)，防止网络直接掉入全0陷阱
        self.x_ic_s = nn.Parameter(is_cuda(-torch.log(torch.tensor(100.).float())), requires_grad=True)
        self.x_bc_s = nn.Parameter(is_cuda(torch.tensor(0.).float()), requires_grad=True)

        # ====== 数据收集器 (为了画 Task 1 和 Task 2 的图) ======
        self.loss_e_collect = []    # 收集 PDE Loss
        self.loss_ic_collect = []   # 收集 IC Loss
        self.loss_bc_collect = []   # 收集 BC Loss
        self.loss_pure_collect = [] # 收集 纯总和 Loss
        
        self.s_collect = []         # 收集 自适应权重 s 的变化
        self.iter = 0
        

    def train_U(self, x):
        return self.net(x)

    def predict_U(self, x):
        return self.train_U(x)

    def likelihood_loss(self, loss_e, loss_ic, loss_bc):
        # 更新权重 s：阻断网络梯度的反向传播
        loss = torch.exp(-self.x_f_s) * loss_e.detach() + self.x_f_s \
               + torch.exp(-self.x_ic_s) * loss_ic.detach() + self.x_ic_s \
               + torch.exp(-self.x_bc_s) * loss_bc.detach() + self.x_bc_s
        return loss

    def true_loss(self, loss_e, loss_ic, loss_bc):
        # 更新网络参数：阻断权重 s 的梯度
        return torch.exp(-self.x_f_s.detach()) * loss_e \
               + torch.exp(-self.x_ic_s.detach()) * loss_ic \
               + torch.exp(-self.x_bc_s.detach()) * loss_bc
    
    def pure_loss(self, loss_e, loss_ic, loss_bc):
        # 纯净版 Loss，用于画图监控真实物理误差
        return loss_e + loss_ic + loss_bc
    
    # computer backward loss
    def epoch_loss(self):
        x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
        loss_equation = torch.mean(self.x_f_loss_fun(x_f, self.train_U) ** 2)
        loss_ic = torch.mean((self.train_U(self.x_ic) - self.u_ic) ** 2)

        # 1. X方向周期性 (x=0 vs x=1)
        x_l = Variable(self.x_bc_x_left, requires_grad=True)
        u_l = self.train_U(x_l)
        u_x_l = torch.autograd.grad(u_l, x_l, grad_outputs=torch.ones_like(u_l), create_graph=True)[0][:, 1].unsqueeze(-1)
        
        x_r = Variable(self.x_bc_x_right, requires_grad=True)
        u_r = self.train_U(x_r)
        u_x_r = torch.autograd.grad(u_r, x_r, grad_outputs=torch.ones_like(u_r), create_graph=True)[0][:, 1].unsqueeze(-1)
        
        loss_bc_x = torch.mean((u_l - u_r) ** 2) + torch.mean((u_x_l - u_x_r) ** 2)

        # 2. Y方向周期性 (y=0 vs y=1)
        y_b = Variable(self.x_bc_y_bottom, requires_grad=True)
        u_b = self.train_U(y_b)
        u_y_b = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0][:, 2].unsqueeze(-1)
        
        y_t = Variable(self.x_bc_y_top, requires_grad=True)
        u_t = self.train_U(y_t)
        u_y_t = torch.autograd.grad(u_t, y_t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 2].unsqueeze(-1)
        
        loss_bc_y = torch.mean((u_b - u_t) ** 2) + torch.mean((u_y_b - u_y_t) ** 2)

        loss_bc = loss_bc_x + loss_bc_y
        return loss_equation, loss_ic, loss_bc

    # computer backward loss
    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        
        x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
        loss_equation = torch.mean(self.x_f_loss_fun(x_f, self.train_U) ** 2)
        loss_ic = torch.mean((self.train_U(self.x_ic) - self.u_ic) ** 2)

        # 1. X方向周期性 (x=0 vs x=1)
        x_l = Variable(self.x_bc_x_left, requires_grad=True)
        u_l = self.train_U(x_l)
        u_x_l = torch.autograd.grad(u_l, x_l, grad_outputs=torch.ones_like(u_l), create_graph=True)[0][:, 1].unsqueeze(-1)
        
        x_r = Variable(self.x_bc_x_right, requires_grad=True)
        u_r = self.train_U(x_r)
        u_x_r = torch.autograd.grad(u_r, x_r, grad_outputs=torch.ones_like(u_r), create_graph=True)[0][:, 1].unsqueeze(-1)
        
        loss_bc_x = torch.mean((u_l - u_r) ** 2) + torch.mean((u_x_l - u_x_r) ** 2)

        # 2. Y方向周期性 (y=0 vs y=1)
        y_b = Variable(self.x_bc_y_bottom, requires_grad=True)
        u_b = self.train_U(y_b)
        u_y_b = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0][:, 2].unsqueeze(-1)
        
        y_t = Variable(self.x_bc_y_top, requires_grad=True)
        u_t = self.train_U(y_t)
        u_y_t = torch.autograd.grad(u_t, y_t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 2].unsqueeze(-1)
        
        loss_bc_y = torch.mean((u_b - u_t) ** 2) + torch.mean((u_y_b - u_y_t) ** 2)

        loss_bc = loss_bc_x + loss_bc_y

        loss = self.true_loss(loss_equation, loss_ic, loss_bc)
        
        loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            print(f'L-BFGS Iter: {self.iter}, Loss: {loss.item():.4e}')
            
        return loss

    def evaluate(self):
        pred = self.train_U(self.x_test).cpu().detach().numpy()
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        return error

    def run_baseline(self):
        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)
        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            optimizer_adam.zero_grad()
            loss_e, loss_ic, loss_bc = self.epoch_loss()
            loss = self.true_loss(loss_e, loss_ic, loss_bc)
            loss.backward()
            optimizer_adam.step()
            self.net.iter += 1
            pbar.set_postfix({'Iter': self.net.iter,
                              'Loss': '{0:.2e}'.format(loss.item())
                              })

        print('Adam done!')
        self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        print('LBGFS done!')

        error = self.evaluate()
        print('Test_L2error:', '{0:.2e}'.format(error))

    def run_AW(self):
        print("=== 开始执行 Task 2 标准版: AW (仅自适应权重，不挪点) ===")
        
        # 初始化双优化器
        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        optimizer_adam_weight = torch.optim.Adam([self.x_f_s, self.x_ic_s, self.x_bc_s], lr=AW_lr)
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr, max_iter=lbgfs_iter)

        # --- Adam 交替训练阶段 ---
        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            loss_e, loss_ic, loss_bc = self.epoch_loss()

            # 记录画图数据
            with torch.no_grad():
                self.loss_e_collect.append([self.iter, loss_e.item()])
                self.loss_ic_collect.append([self.iter, loss_ic.item()])
                self.loss_bc_collect.append([self.iter, loss_bc.item()])
                self.loss_pure_collect.append([self.iter, self.pure_loss(loss_e, loss_ic, loss_bc).item()])
                self.s_collect.append([self.iter, self.x_f_s.item(), self.x_ic_s.item(), self.x_bc_s.item()])

            # 1. 优化网络参数 (True Loss)
            optimizer_adam.zero_grad()
            loss_net = self.true_loss(loss_e, loss_ic, loss_bc)
            loss_net.backward()
            optimizer_adam.step()

            # 2. 优化自适应权重 (Likelihood Loss)
            optimizer_adam_weight.zero_grad()
            loss_weight = self.likelihood_loss(loss_e, loss_ic, loss_bc)
            loss_weight.backward()
            optimizer_adam_weight.step()

            self.iter += 1
            pbar.set_postfix({'True_Loss': f'{loss_net.item():.2e}'})

        # --- L-BFGS 阶段 ---
        print('Adam 阶段完成，进入 L-BFGS...')
        self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        print('AW 训练彻底完成！')

    def run_AMAW(self):
        print("=== 开始执行 Task 2 完全体: AMAW (自适应权重 + 2D 挪点) ===")
        for move_count in range(AM_count):
            print(f"\n--- 第 {move_count+1}/{AM_count} 轮 AMAW 训练 ---")
            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s, self.x_ic_s, self.x_bc_s], lr=AW_lr)
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr, max_iter=lbgfs_iter)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                loss_e, loss_ic, loss_bc = self.epoch_loss()

                # --- 记录画图数据 ---
                with torch.no_grad():
                    self.loss_e_collect.append([self.iter, loss_e.item()])
                    self.loss_ic_collect.append([self.iter, loss_ic.item()])
                    self.loss_bc_collect.append([self.iter, loss_bc.item()])
                    self.loss_pure_collect.append([self.iter, self.pure_loss(loss_e, loss_ic, loss_bc).item()])
                    self.s_collect.append([self.iter, self.x_f_s.item(), self.x_ic_s.item(), self.x_bc_s.item()])

                # --- 优化网络参数 (True Loss) ---
                optimizer_adam.zero_grad()
                loss_net = self.true_loss(loss_e, loss_ic, loss_bc)
                loss_net.backward()
                optimizer_adam.step()

                # --- 优化自适应权重 (Likelihood Loss) ---
                optimizer_adam_weight.zero_grad()
                loss_weight = self.likelihood_loss(loss_e, loss_ic, loss_bc)
                loss_weight.backward()
                optimizer_adam_weight.step()

                self.iter += 1
                pbar.set_postfix({'True_Loss': f'{loss_net.item():.2e}'})

            print('Adam 阶段完成，进入 L-BFGS...')
            self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)

            # --- 2D WAM (基于梯度的挪点) ---
            if move_count < AM_count - 1:
                print(">>> 正在执行 2D 空间梯度挪点 (WAM)...")
                x_init = is_cuda(torch.from_numpy(lb + (ub - lb) * lhs(3, 30000)).float())
                x_var = Variable(x_init, requires_grad=True)
                u = self.train_U(x_var)
                
                dx = torch.autograd.grad(u, x_var, grad_outputs=torch.ones_like(u), create_graph=False)[0]
                grad_x = dx[:, 1].squeeze() 
                grad_y = dx[:, 2].squeeze() 
                
                grad_mag = torch.sqrt(1 + grad_x**2 + grad_y**2).cpu().detach().numpy()
                err_dx = np.power(grad_mag, AM_K) / np.power(grad_mag, AM_K).mean()
                p = (err_dx / sum(err_dx))
                X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=p)
                self.x_f_M = x_init[X_ids]

    def train(self):
        # 初始全局撒点 (N 个固定点，M 个移动点。跑 AW 时相当于 N+M 个固定点)
        self.x_f_N = is_cuda(torch.from_numpy(lb + (ub - lb) * lhs(3, N)).float())
        self.x_f_M = is_cuda(torch.from_numpy(lb + (ub - lb) * lhs(3, M)).float())

        start_time = time.time()
        
        # 0: Baseline, 1: AW (纯自适应权重), 2: AMAW (权重+挪点)
        if model_type == 0:
            self.run_baseline()
        elif model_type == 1:
            self.run_AW()
        elif model_type == 2:
            self.run_AMAW()
            
        elapsed = time.time() - start_time
        print('Training time: %.2f seconds' % elapsed)

def x_f_loss_fun(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    u = train_U(x)
    
    # 这里的 x 包含了三列：[t, x, y]
    d = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = d[:, 0].unsqueeze(-1)
    u_x = d[:, 1].unsqueeze(-1)
    u_y = d[:, 2].unsqueeze(-1)  # 新增：对 y 的一阶导数
    
    # 求 x 的二阶导
    dd_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xx = dd_x[:, 1].unsqueeze(-1)
    
    # 求 y 的二阶导
    dd_y = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_yy = dd_y[:, 2].unsqueeze(-1)
    
    # Ag3 的物理参数
    epsilon = 0.025
    lambda_param = 10.0
    
    # 2D Allen-Cahn PDE: u_t - lambda * (epsilon^2 * (u_xx + u_yy) - u^3 + u) = 0
    f = u_t - lambda_param * (epsilon**2 * (u_xx + u_yy) - u**3 + u)
    return f


# def draw_exact():
#     predict_np = model.predict_U(x_test).cpu().detach().numpy()
#     u_test_np = x_test_exact.cpu().detach().numpy()
#     TT, XX = np.meshgrid(t, x)
#     e = np.reshape(u_test_np, (TT.shape[0], TT.shape[1]))
#     plt.pcolor(TT, XX, e, cmap='jet')
#     plt.colorbar()
#     plt.xlabel('$t$', fontsize=20)
#     plt.ylabel('$x$', fontsize=20)
#     plt.title(r'Exact $u(t,x)$', fontsize=20)
#     plt.tight_layout()
#     plt.savefig('burgers_exact.pdf')
#     plt.show()

#     e = np.reshape(predict_np, (TT.shape[0], TT.shape[1]))
#     plt.pcolor(TT, XX, e, cmap='jet')
#     plt.colorbar()
#     plt.xlabel('$t$', fontsize=20)
#     plt.ylabel('$x$', fontsize=20)
#     plt.title(r'PINN $u(t,x)$', fontsize=20)
#     plt.tight_layout()
#     plt.savefig('burgers_pred_PINN.pdf')
#     plt.show()


# def draw_exact_points(points, N_points=None, show_exact=True):
#     if show_exact:
#         u_test_np = x_test_exact.cpu().detach().numpy()
#         XX1, XX2 = np.meshgrid(t, x)
#         e = np.reshape(u_test_np, (XX1.shape[0], XX1.shape[1]))
#         plt.pcolor(XX1, XX2, e, shading='auto', cmap='YlGnBu')
#         plt.colorbar()
#         plt.title(r'Exact $u(x)$')
#     if N_points is not None:
#         adds = N_points.cpu().detach().numpy()
#         plt.plot(adds[:, [0]], adds[:, [1]], 'kx', markersize=4, clip_on=False)

#     points = points.cpu().detach().numpy()
#     plt.plot(points[:, [0]], points[:, [1]], 'rx', markersize=4)
#     plt.xlim(-0.1, 1.1)
#     plt.ylim(-1.1, 1.1)
#     plt.xlabel('$t$', fontsize=20)
#     plt.ylabel('$x$', fontsize=20)
#     plt.savefig('burgers_xnm-PINN.pdf')
#     plt.show()


# def draw_residual():
#     f = x_f_loss_fun(x_test, model.train_U)
#     f = f.cpu().detach().numpy()
#     XX1, XX2 = np.meshgrid(t, x)
#     e = np.reshape(abs(f), (XX1.shape[0], XX1.shape[1]))
#     plt.pcolor(XX1, XX2, e, shading='auto', cmap='jet')
#     plt.colorbar()
#     plt.xlabel('$t$', fontsize=20)
#     plt.ylabel('$x$', fontsize=20)
#     plt.title('$Residual$', fontsize=20)
#     plt.tight_layout()
#     plt.savefig('burgers_residual-PINN.pdf')
#     plt.show()


# def draw_error():
#     predict_np = model.predict_U(x_test).cpu().detach().numpy()
#     u_test = x_test_exact.cpu().detach().numpy()
#     XX1, XX2 = np.meshgrid(t, x)
#     e = np.reshape(abs(predict_np - u_test), (XX1.shape[0], XX1.shape[1]))
#     plt.pcolor(XX1, XX2, e, shading='auto', cmap='jet')
#     plt.colorbar()
#     plt.xlabel('$t$', fontsize=20)
#     plt.ylabel('$x$', fontsize=20)
#     plt.title('$Error$', fontsize=20)
#     plt.tight_layout()
#     plt.savefig('burgers_error-PINN.pdf')
#     plt.show()


# def draw_epoch_loss():
#     x_label_loss_collect = np.array(model.x_label_loss_collect)
#     x_f_loss_collect = np.array(model.x_f_loss_collect)
#     plt.subplot(2, 1, 1)
#     plt.yscale('log')
#     plt.plot(x_label_loss_collect[:, 0], x_label_loss_collect[:, 1], 'b-', label='Label_loss')
#     plt.xlabel('$Epoch$')
#     plt.ylabel('$Loss$')
#     plt.legend()
#     plt.subplot(2, 1, 2)
#     plt.yscale('log')
#     plt.plot(x_f_loss_collect[:, 0], x_f_loss_collect[:, 1], 'r-', label='PDE_loss')
#     plt.xlabel('$Epoch$')
#     plt.ylabel('$Loss$')
#     plt.legend()
#     plt.show()


# def draw_epoch_w():
#     s_collect = np.array(model.s_collect)
#     np.savetxt('s_RAM-AW.npy', s_collect)
#     plt.rc('legend', fontsize=16)
#     plt.yscale('log')
#     plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label='$e^{-s_{r}}$')
#     plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label='$e^{-s_{i}}$')
#     plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 3]), 'g-', label='$e^{-s_{b}}$')
#     plt.xlabel('$Iters$', fontsize=20)
#     plt.ylabel('$\lambda$', fontsize=20)
#     plt.legend()
#     #plt.savefig('burgers_S_RAM-AW.pdf', fontsize=20)
#     plt.savefig('burgers_S_RAM-AW.pdf')
#     plt.show()


# def draw_some_t():
#     predict_np = model.predict_U(x_test).cpu().detach().numpy()
#     u_test_np = x_test_exact.cpu().detach().numpy()
#     XX1, XX2 = np.meshgrid(t, x)
#     u_pred = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]))
#     u_test = np.reshape(u_test_np, (XX1.shape[0], XX1.shape[1]))
#     gs1 = gridspec.GridSpec(1, 3)
#     gs1.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
#     plt.rc('legend', fontsize=16)
#     ax = plt.subplot(gs1[0, 0])

#     ax.plot(x, u_test.T[100, :], 'b-', linewidth=2, label='Exact')
#     ax.plot(x, u_pred.T[100, :], 'r--', linewidth=2, label='Prediction')
#     ax.set_xlabel('$x$', fontsize=20)
#     ax.set_ylabel('$u$', fontsize=20)
#     # ax.set_title('$t = %.2f$' % (t[100]), fontsize=20)
#     ax.set_title('$t = %.2f$' % (t[100].item()), fontsize=20)
#     ax.axis('square')
#     ax.set_xlim([-1.1, 1.1])
#     ax.set_ylim([-1.1, 1.1])

#     ax = plt.subplot(gs1[0, 1])
#     ax.plot(x, u_test.T[150, :], 'b-', linewidth=2, label='Exact')
#     ax.plot(x, u_pred.T[150, :], 'r--', linewidth=2, label='Prediction')
#     ax.set_xlabel('$x$', fontsize=20)
#     # ax.set_ylabel('$u$', fontsize=20)
#     ax.axis('square')
#     ax.set_xlim([-1.1, 1.1])
#     ax.set_ylim([-1.1, 1.1])
#     # ax.set_title('$t = %.2f$' % (t[150]), fontsize=20)
#     ax.set_title('$t = %.2f$' % (t[150].item()), fontsize=20)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

#     ax = plt.subplot(gs1[0, 2])
#     ax.plot(x, u_test.T[200, :], 'b-', linewidth=2, label='Exact')
#     ax.plot(x, u_pred.T[200, :], 'r--', linewidth=2, label='Prediction')
#     ax.set_xlabel('$x$', fontsize=20)
#     # ax.set_ylabel('$u$', fontsize=20)
#     ax.axis('square')
#     ax.set_xlim([-1.1, 1.1])
#     ax.set_ylim([-1.1, 1.1])
#     # ax.set_title('$t = %.2f$' % (t[200]), fontsize=20)
#     ax.set_title('$t = %.2f$' % (t[200].item()), fontsize=20)

#     # plt.tight_layout()

#     plt.savefig('burgers_qie_PINN.pdf')
#     plt.show()

# def draw_amaw_losses():
#     t_loss = np.array(model.true_loss_collect)
#     l_loss = np.array(model.likelihood_loss_collect)
#     p_loss = np.array(model.pure_loss_collect)  # 获取纯净版 Loss
    
#     plt.figure(figsize=(8, 6))
#     plt.yscale('log')
#     plt.plot(t_loss[:, 0], t_loss[:, 1], 'b-', linewidth=2, label='True Loss (Optimizing Network)')
#     plt.plot(l_loss[:, 0], np.abs(l_loss[:, 1]), 'r--', linewidth=2, label='Abs(Likelihood Loss)')
#     # 新加一条绿色的半透明纯净版曲线
#     plt.plot(p_loss[:, 0], p_loss[:, 1], 'g-', alpha=0.7, linewidth=2, label='Unweighted Pure Loss')
    
#     plt.xlabel('Adam Iterations', fontsize=20)
#     plt.ylabel('Loss Value', fontsize=20)
#     plt.title('AMAW Decoupled Losses', fontsize=20)
#     plt.legend(fontsize=14, frameon=False)
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.tight_layout()
#     plt.savefig('amaw_decoupled_losses.pdf')
#     plt.show()

# def draw_l2_error_curve():
#     # model.x_test_estimate_collect 里记录了每次挪点后的 L2 Error
#     data = np.array(model.x_test_estimate_collect)
#     iters = data[:, 0].astype(int)      # X 轴：挪点的次数 (0, 1, 2, ...)
#     errors = data[:, 1].astype(float)   # Y 轴：L2 误差
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(iters, errors, 'k-o', linewidth=2, markersize=8) # 黑色带圆点的实线
#     plt.yscale('log')
#     plt.xlabel('AM Iteration (Resampling Count)', fontsize=20)
#     plt.ylabel('Relative $L_2$ Error', fontsize=20)
#     plt.title('Error Convergence over Adaptive Resampling', fontsize=20)
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.xticks(iters) # 让 X 轴只显示整数刻度
#     plt.tight_layout()
#     plt.savefig('amaw_l2_error_curve.pdf')
#     plt.show()


if __name__ == '__main__':
    # ================= 1. 域的边界改成 3D =================
    # 根据 Ag3 的要求，(x, y) 属于 [0,1]^2，t 属于 [0,1]
    lb = np.array([0.0, 0.0, 0.0])  # t, x, y 的下界
    ub = np.array([1.0, 1.0, 1.0])  # t, x, y 的上界

    layers = [3, 64, 64, 64, 64, 1] # 确认输入层为 3
    net = is_cuda(Net_attention(layers)) 

    N = 8000
    M = 2000
    Nbc = 2000 # 2D 边界变长了，点数稍微加一点
    Nic = 5000 # 2D 初始面积变大了，点数加一点

    adam_iter, lbgfs_iter = 2000, 3000
    adam_lr, lbgfs_lr = 0.001, 0.5

    # Task 1 要求：先不带自适应权重，跑个 baseline，所以我们设为 0
    model_type = 2
    AM_type = 1  
    AM_K = 1
    AM_count = 5  
    AW_lr = 0.001 

    # ================= 2. 读取 2D 高保真 FEM 测试集 =================
    print("正在加载高保真 FEM 数据...")
    # 请确保路径和你上一条测试代码里的路径一模一样
    with h5py.File('Data/circle.mat', 'r') as f:
        t_star = np.array(f['t01']).flatten()   # 形状 (101,)
        x_star = np.array(f['x']).flatten()     # 形状 (129,)
        y_star = np.array(f['y']).flatten()     # 形状 (129,)
        U_star = np.array(f['FAI_0_1_data_M'])  # 形状 (101, 129, 129)

    # 用 meshgrid 构造出与 U_star 严丝合缝的 (t, x, y) 坐标点阵
    # indexing='ij' 保证生成的网格顺序就是 (101, 129, 129)
    TT, XX, YY = np.meshgrid(t_star, x_star, y_star, indexing='ij')

    # 把 3D 矩阵无情地展平，变成 PINN 喜欢的 N x 3 输入矩阵和 N x 1 标签矩阵
    x_test_np = np.hstack((TT.reshape(-1, 1), XX.reshape(-1, 1), YY.reshape(-1, 1)))
    u_test_np = U_star.reshape(-1, 1)

    # 转化为 Tensor 并推入显卡 GPU
    x_test = is_cuda(torch.from_numpy(x_test_np).float())
    x_test_exact = is_cuda(torch.from_numpy(u_test_np).float())
    
    print(f"数据加载完成！共有 {x_test.shape[0]} 个测试数据点。")

    # ================= 3. 生成 2D 双圆初始条件 (t=0) =================
    x1, y1, r1 = 0.35, 0.50, 0.14
    x2, y2, r2 = 0.65, 0.50, 0.14
    epsilon = 0.025

    x_ic_pts = torch.rand([Nic, 2])
    t_ic_pts = torch.zeros(Nic, 1)  
    x_initial = torch.cat((t_ic_pts, x_ic_pts), dim=1) # 拼接为 [t, x, y]

    X_ic = x_ic_pts[:, 0:1]
    Y_ic = x_ic_pts[:, 1:2]
    term1 = (r1 - torch.sqrt((X_ic - x1)**2 + (Y_ic - y1)**2)) / (1.5 * epsilon)
    term2 = (r2 - torch.sqrt((X_ic - x2)**2 + (Y_ic - y2)**2)) / (1.5 * epsilon)
    x_initial_label = 1.0 + torch.tanh(term1) + torch.tanh(term2)

    x_ic = is_cuda(x_initial)
    u_ic = is_cuda(x_initial_label)

    # ================= 4. 生成 2D 周期性四面边界 =================
    t_bc = torch.rand([Nbc, 1]) # 时间 t 在 [0, 1]
    s_bc = torch.rand([Nbc, 1]) # 共享的空间坐标 s 在 [0, 1]

    # X方向边界：左墙 (x=0) 和 右墙 (x=1)，此时 y 是随机的 s_bc
    x_bc_x_left = is_cuda(torch.cat((t_bc, torch.zeros(Nbc, 1), s_bc), dim=1))
    x_bc_x_right = is_cuda(torch.cat((t_bc, torch.ones(Nbc, 1), s_bc), dim=1))

    # Y方向边界：地板 (y=0) 和 天花板 (y=1)，此时 x 是随机的 s_bc
    x_bc_y_bottom = is_cuda(torch.cat((t_bc, s_bc, torch.zeros(Nbc, 1)), dim=1))
    x_bc_y_top = is_cuda(torch.cat((t_bc, s_bc, torch.ones(Nbc, 1)), dim=1))

    # ================= 5. 初始化模型并开始测试训练 =================
    model = Model(
        net=net,
        x_bc_x_left=x_bc_x_left,
        x_bc_x_right=x_bc_x_right,
        x_bc_y_bottom=x_bc_y_bottom,
        x_bc_y_top=x_bc_y_top,
        x_ic=x_ic,
        u_ic=u_ic,
        x_f_loss_fun=x_f_loss_fun,
        x_test=x_test,
        x_test_exact=x_test_exact,
    )
    
    print("模型初始化成功，准备开始训练...")

    model.train()

    print("训练测试完成！")

    # ================= 6. Task 3: 绘制 Side-by-side 热力图 =================
    print("正在绘制时间切片的热力图对比...")
    # model.train_U.eval() # 切换到测试模式
    
    # 我们抽取 t=0, t=0.5, t=1.0 三个时间切片 (对应索引 0, 50, 100)
    time_indices = [0, 50, 100]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    with h5py.File('Data/circle.mat', 'r') as f:
        t_star = np.array(f['t01']).flatten()
        x_star = np.array(f['x']).flatten()
        y_star = np.array(f['y']).flatten()
        U_star = np.array(f['FAI_0_1_data_M'])
        
    for i, t_idx in enumerate(time_indices):
        t_val = t_star[t_idx]
        
        # 1. 提取当前时刻的精确解
        u_exact = U_star[t_idx, :, :]
        
        # 2. 生成当前时刻的网格点阵让 PINN 预测
        XX, YY = np.meshgrid(x_star, y_star, indexing='ij')
        TT = np.full_like(XX, t_val)
        
        x_test_t = np.hstack((TT.reshape(-1, 1), XX.reshape(-1, 1), YY.reshape(-1, 1)))
        x_test_t_tensor = is_cuda(torch.from_numpy(x_test_t).float())
        
        with torch.no_grad():
            u_pred = model.train_U(x_test_t_tensor).cpu().numpy().reshape(len(x_star), len(y_star))
            
        # 3. 画 PINN 预测图 (左列)
        ax1 = axes[i, 0]
        im1 = ax1.imshow(u_pred.T, origin='lower', extent=[0, 1, 0, 1], cmap='jet', vmin=-1, vmax=1)
        # 改成动态标题
        ax1.set_title(f'AW/AMAW Predict (t={t_val:.2f})', fontsize=12)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # 4. 画 FEM 精确图 (右列)
        ax2 = axes[i, 1]
        im2 = ax2.imshow(u_exact.T, origin='lower', extent=[0, 1, 0, 1], cmap='jet', vmin=-1, vmax=1)
        ax2.set_title(f'FEM Exact (t={t_val:.2f})', fontsize=12)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

    # 添加全局 Colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('Task1_Baseline_Heatmap.png', dpi=300)
    print("热力图已保存为 Task1_Baseline_Heatmap.png")
    plt.show()


    