import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
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
mu=1
def getDerivative(y,f):
    costh=torch.cos(y[1])
    sinth=torch.sin(y[1])
    D = 4/3*(1+mu)-costh**2
    xdot=(4/3*y[2]-costh*y[3])/D
    thdot=((1+mu)*y[3]-costh*y[2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    return torch.stack([xdot,thdot,pxdot,pthdot])
def rk2solver(y0,dt,f):
    ymid=y0+dt/2*getDerivative(y0,f)
    return y0+dt*getDerivative(ymid,f)





def loss(y,f):
    e=0.015
    # return torch.sqrt(y[0]**2+e**2)*0.1+torch.relu(y[0]**2-16)+(1-torch.cos(y[1]))*2+0.001*y[2]**2+0.001*y[3]**2+0.001*f**2
    return y[0]**2*0.5+torch.relu(y[0]**2-16)+(1-torch.cos(y[1]))*5+0.05*y[2]**2+0.6*y[3]**2+0.005*f**2




net = VerySimpleCar().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
dt = 0.05  # 时间步长

steps=60
times=60


# y0 = torch.tensor([0.8, 0.7, 0.0, 0.0], dtype=torch.float32, requires_grad=False,device=device)




class CurriculumManager:
    def __init__(self):
        self.level = 0
        self.max_level = 10
        self.loss_window = []

    def get_init_state(self,device,debug=-1):
        a=self.level
        if debug!=-1:
            self.level=debug
        x_scale = 0.1 + (self.level / self.max_level) * 2.0
        x_pos = (torch.rand(1, device=device) - 0.5) * 2.0 * x_scale
        if self.level<=3:
            th_scale = 0.1 + (self.level / self.max_level) * 3.04
            th_pos = (torch.rand(1, device=device) - 0.5) * 2.0 * th_scale
        elif self.level<=8:
            min_th = 1.0 + ((self.level - 5) / 3.0) * 1.0
            max_th = 2.0 + ((self.level - 5) / 3.0) * 1.0
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            th_pos = sign * (min_th + torch.rand(1, device=device) * (max_th - min_th))
        else:
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            th_pos = sign * 3.14 + (torch.rand(1, device=device) - 0.5) * 0.2

        v_noise = (torch.rand(2, device=device) - 0.5) * 0.2
        
        self.level = a
        
        return torch.tensor([
            x_pos.item(), 
            th_pos.item(), 
            v_noise[0].item(), 
            v_noise[1].item()
        ], device=device)


    def update(self, current_loss,u,times):
        if u>times//2:
            self.loss_window.append(current_loss)
            if len(self.loss_window) > 20:
                self.loss_window.pop(0)
                avg_loss = sum(self.loss_window) / 20
            
                # 如果平均误差足够小，且还没到最高级，则升级
                if avg_loss < 0.01 and self.level < self.max_level:
                    self.level += 1
                    self.loss_window = [] # 清空窗口重新评估新难度
                    print(f"!!!!!!!!!!!!! 难度升级！当前等级: {self.level} !!!!!!!!!!!!!")

                # 如果误差突然变得巨大（学崩了），可以考虑降级保护
                elif avg_loss > 5.0 and self.level > 0:
                    self.level -= 1
                    self.loss_window = []
                    print(f"!!!!!!!!!!!!! 难度降级保护! 当前等级: {self.level} !!!!!!!!!!!!!")
        elif u==0:
            self.loss_window=[]




manager=CurriculumManager()



def sim_step(y0,dt,step):
    ttlos = torch.tensor(0.0, device=y0.device)
    for i in range(step):
        F=net.getForce(y0)
        y=rk2solver(y0,dt,F[0])
        ttlos+=loss(y,F[0])
        y0=y
    return y,ttlos/step
compiled_step = torch.compile(sim_step, mode="reduce-overhead")
u=0
y0=manager.get_init_state(device)
t=0

for epoch in range(2000):
    optimizer.zero_grad()
    ttloss=0
    y,ttloss=compiled_step(y0,dt,steps)
    # y,ttloss=sim_step(y0,dt,steps)
    u+=1
    y0=y.clone().detach()
    if ttloss>20 or abs(y[1])>10 or abs(y[0])>10 or abs(y[2])>10:
        y0=manager.get_init_state(device)
        u=0
        t+=1
        print('---OutBoundReset---')
    if u>=times:
        y0=manager.get_init_state(device)
        u=0
        t+=1
        print('---Reset---')
    if torch.isnan(ttloss) or torch.isinf(ttloss) or torch.isnan(y).any():
        y0=manager.get_init_state(device)
        u=0
        print('---Force Reset---')
        continue
    ttloss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    # if epoch%50==0:
    #     print(y)
    # print(F)
    print(epoch)
    print(f"Level: {manager.level}")
    print(f'u={u}')
    print(f't={t}')
    print(ttloss.item())
    print(y)
    manager.update(ttloss.item(),u,times)
# 将模型的权重保存为一个 .pth 文件
torch.save(net.state_dict(), "pendulum_controller_curriculum.pth")
print("权重已成功保存！")