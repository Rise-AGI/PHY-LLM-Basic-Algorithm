import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
class VerySimpleCar(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(7,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Tanh()
        )
    def getForce(self,y):
        yy=torch.stack([y[0],torch.sin(y[1]),torch.cos(y[1]),torch.sin(2*y[1]),torch.cos(2*y[1]),y[2],y[3]])
        return 20.0*self.net(yy)


class Actor(nn.Module):
    def __init__(self,indim):
        super().__init__()
        self.rootnet=nn.Sequential(
                nn.Linear(indim,256),
                nn.LeakyReLU(),
                nn.Linear(256,256),
                nn.LeakyReLU()
        )
        self.actor=nn.Sequential(
            nn.Linear(256,1),
            nn.Tanh()
        )
        # self.critic=nn.Sequential(
        #     nn.Linear(256,1)
        # )
        self.log_std=nn.Parameter(torch.zeros(1))
    def forward(self,state)-> tuple[distributions.Normal, torch.Tensor]:
        feat=self.rootnet(state)
        mu=10*self.actor(feat)
        # v=self.critic(feat)
        # std=torch.exp(self.log_std)
        # dist=distributions.Normal(mu,std)
        return mu,self.log_std

k=2
def getDerivative(y,f):
    print(y)
    costh=torch.cos(y[...,1])
    sinth=torch.sin(y[...,1])
    D = 4/3*(1+k)-costh**2
    xdot=(4/3*y[...,2]-costh*y[...,3])/D
    thdot=((1+k)*y[...,3]-costh*y[...,2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    return torch.stack([xdot,thdot,pxdot,pthdot],dim=-1)
def rk2solver(y0,dt,f):
    ymid=y0+dt/2*getDerivative(y0,f)
    return y0+dt*getDerivative(ymid,f)
# net = VerySimpleCar().to(device)
# net.load_state_dict(torch.load("pendulum_controller_perfect.pth"))
# net.eval()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. 准备模型 ---
device = torch.device("cpu") # 可视化通常在 CPU 上跑就够了
# net = VerySimpleCar().to(device)
net=Actor(7).to(device)
net.load_state_dict(torch.load("pendulum_controller_ppo.pth"))
net.eval()

# --- 2. 收集物理轨迹 ---
# 故意给一个有挑战性的初始状态，比如偏离中心点，且稍微倾斜
# 看看它能不能自己跑回原点并立稳！
y0 = torch.tensor([0.2, 0.2, 0, 0.0], dtype=torch.float32, device=device)
dt = 0.05
steps = 200  # 模拟 200 步 (10秒)

history_x = []
history_theta =[]

# 不需要计算梯度了，省内存加速
with torch.no_grad():
    for _ in range(steps):
        # F = net.getForce(y0)
        y00=y0.unsqueeze(-1).transpose(0,1)
        yy=torch.stack([y00[:,0],torch.sin(y00[:,1]),torch.cos(y00[:,1]),torch.sin(2*y00[:,1]),torch.cos(2*y00[:,1]),y00[:,2],y00[:,3]],dim=-1)
        mu,lgstd=net.forward(yy)
        # F=dist.sample().squeeze(-1)
        F=torch.clamp(mu+torch.randn_like(mu)*torch.exp(lgstd),-15,15)
        print(y0.unsqueeze(-1).transpose(0,1))
        y = rk2solver(y0.unsqueeze(-1).transpose(0,1), dt, F.squeeze(-1))
        
        # 记录轨迹用于画图
        y=y.reshape(-1)
        history_x.append(y[0].item())
        history_theta.append(y[1].item())
        y0 = y

# --- 3. 画布初始化 ---
fig, ax = plt.subplots(figsize=(15, 4))
ax.set_xlim(-10.0, 10.0)  # 车道范围
ax.set_ylim(-1.5, 1.5)  # 上下空间
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Neural Network Inverted Pendulum Control")

# 划一根横线代表地面
ax.plot([-2, 2], [0, 0], color='black', linewidth=1)

# 初始化小车 (画一个矩形) 和摆杆 (画一条线)
cart_width, cart_height = 0.4, 0.2
cart_rect = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, fc='blue')
ax.add_patch(cart_rect)
pole_line, = ax.plot([],[], 'r-', lw=4) # 红色的摆杆

# --- 4. 动画更新函数 ---
L = 1.0 # 摆杆长度

def update(frame):
    x = history_x[frame]
    theta = history_theta[frame]
    
    # 1. 更新小车位置
    cart_rect.set_xy((x - cart_width/2, -cart_height/2))
    
    # 2. 更新摆杆端点位置 (物理几何关系: theta=0是正上方)
    pole_x = [x, x + L * np.sin(theta)]
    pole_y =[0, L * np.cos(theta)]
    pole_line.set_data(pole_x, pole_y)
    
    return cart_rect, pole_line

# --- 5. 播放动画 ---
ani = animation.FuncAnimation(fig, update, frames=steps, interval=dt*1000, blit=True)

plt.show()