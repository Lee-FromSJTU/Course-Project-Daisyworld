import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

gamma = 0.1
k = 17.5 ** (-2)
T_opt = 295.5
aw = 0.9
ag = 0.5
ab = 0.1
q = 2.06e9
SL = 917
sigma = 5.67e-8


def beta(temp):
    if abs(temp - T_opt) < k ** (-0.5):
        return 1 - k * (temp - T_opt) ** 2
    else:
        return 0


def f(data, t):
    [alpha_w, alpha_b] = data[:]
    alpha_g = 1 - alpha_w - alpha_b
    AAlbedo = alpha_w * aw + alpha_b * ab + alpha_g * ag
    T_4 = SL * (1 - AAlbedo) / sigma
    temp_w = (q * (AAlbedo - aw) + T_4) ** 0.25
    T_b = (q * (AAlbedo - ab) + T_4) ** 0.25
    alpha_w1 = alpha_w * (alpha_g * beta(temp_w) - gamma)
    alpha_b1 = alpha_b * (alpha_g * beta(T_b) - gamma)
    return [alpha_w1, alpha_b1]


t = np.linspace(0, 25, 1000)

fig = plt.figure(figsize=(10, 6))
for i in range(5):
    tmp1 = random.uniform(0.1, 0.9)
    tmp2 = random.uniform(0, 1 - tmp1)
    data0 = [tmp1, tmp2]
    sol = odeint(f, data0, t)
    w = sol[:, 0]
    b = sol[:, 1]
    plt.plot(t, w, color='r', lw=3, alpha=-0.1 * i + 0.5)
    plt.plot(t, b, color='b', lw=3, alpha=-0.1 * i + 0.5)
    plt.grid()
    fig.tight_layout()
plt.xlabel('t')
plt.ylabel(r'$\alpha$')
plt.ylim([0, 1])
plt.legend([r'$\alpha_w$', r'$\alpha_b$'])
plt.show()


def alpha_to_temp(alpha_w, alpha_b):
    alpha_g = 1 - alpha_w - alpha_b
    Albedo = alpha_w * aw + alpha_b * ab + alpha_g * ag
    T_4 = SL * (1 - Albedo) / sigma
    Temp = T_4 ** 0.25
    T_w = (q * (Albedo - aw) + T_4) ** 0.25
    T_b = (q * (Albedo - ab) + T_4) ** 0.25
    T_g = (q * (Albedo - ag) + T_4) ** 0.25
    return T_w, T_b, T_g, Temp


alpha = [[], []]
Tw_list = []
Tb_list = []
Tg_list = []
T_list = []
t1 = np.linspace(0, 100, 10000)

SLrange = np.linspace(500, 1700, 1200)
data0 = [0.5, 0.5]
for i in SLrange:
    SL = i
    tmp = odeint(f, data0, t1)
    alpha[0].append(tmp[:, 0][-1])
    alpha[1].append(tmp[:, 1][-1])
    Tw, Tb, Tg, T = alpha_to_temp(alpha[0][-1], alpha[1][-1])
    Tw_list.append(Tw)
    Tb_list.append(Tb)
    Tg_list.append(Tg)
    T_list.append(T)

SL = 917
total = np.array(alpha[0]) + np.array(alpha[1])

fig = plt.figure(figsize=(10, 6))
plt.plot(SLrange, alpha[0], color='r', lw=3, alpha=0.5)
plt.plot(SLrange, alpha[1], color='b', lw=3, alpha=0.5)
plt.plot(SLrange, total, color='g', lw=3, alpha=0.5)
plt.xlabel('SL')
plt.ylabel(r'$\alpha$')
plt.xlim([400, 1800])
plt.ylim([0, 1])
plt.legend([r'$\alpha_w$', r'$\alpha_b$', r'$\alpha_w +\alpha_b$'])
plt.grid()
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 6))
plt.plot(SLrange, Tw_list, color='r', lw=3, alpha=0.5)
plt.plot(SLrange, Tb_list, color='b', lw=3, alpha=0.5)
plt.plot(SLrange, Tg_list, color='g', lw=3, alpha=0.5)
plt.plot(SLrange, T_list, color='black', lw=3, alpha=0.6)
plt.xlabel('SL')
plt.ylabel('Temp')
plt.xlim([400, 1800])
plt.legend(['Tw', 'Tb', 'Tg', 'T'])
plt.grid()
fig.tight_layout()
plt.show()


length = 100
width = 200

cell = np.zeros((length, width), int)
cell_temp = copy.deepcopy(cell)

for i in range(0, length):
    for j in range(0, width):
        cell_temp[i][j] = random.randint(-1, 1)

a = [[], []]
for t in range(50):
    # 通过读取cell矩阵，不断判断每一个单元格下一状态，将判断结果保存在cell_temp中，最终再复制给cell矩阵
    cell = copy.deepcopy(cell_temp)
    # 显示每一轮的图像
    # plt.imshow(cell)
    # plt.pause(0.1)

    # ns = [white, black, ground]
    ns = [0, 0, 0]
    for i in range(0, length):
        for j in range(0, width):
            if cell[i][j] == 1:
                ns[0] += 1
            elif cell[i][j] == -1:
                ns[1] += 1
            elif cell[i][j] == 0:
                ns[2] += 1
    alpha = np.array(ns) / (length * width)
    [alphaw, alphab, alphag] = alpha[:]
    a[0].append(alphaw)
    a[1].append(alphab)
    A = alphaw * aw + alphab * ab + alphag * ag
    T4 = SL * (1 - A) / sigma
    Tw = (q * (A - aw) + T4) ** 0.25
    Tb = (q * (A - ab) + T4) ** 0.25
    Tg = (q * (A - ag) + T4) ** 0.25
    betaw = alphag * beta(Tw)
    betab = alphag * beta(Tb)

    # 两重循环遍历整个矩阵，判断每个细胞的下一个状态
    for i in range(0, length):
        for j in range(0, width):
            neighbors = [cell[(i - 1) % length][(j - 1) % width], cell[(i - 1) % length][j],
                         cell[(i - 1) % length][(j + 1) % width],
                         cell[i][(j - 1) % width], cell[i][(j + 1) % width],
                         cell[(i + 1) % length][(j - 1) % width], cell[(i + 1) % length][j],
                         cell[(i + 1) % length][(j + 1) % width]]
            nw = 0
            nb = 0
            for neighbor in neighbors:
                if neighbor == 1:
                    nw += 1
                elif neighbor == -1:
                    nb += 1
            if nw == 0 and nb == 0:
                Pe = 1
                Pb = 0
                Pw = 0
            else:
                Pb0 = 1 - (1 - betab) ** nb
                Pw0 = 1 - (1 - betaw) ** nw
                Pe = (1 - Pb0) * (1 - Pw0)
                Pb = (1 - Pe) * Pb0 / (Pb0 + Pw0)
                Pw = (1 - Pe) * Pw0 / (Pb0 + Pw0)
            Pd = gamma

            if cell[i][j] == 0:
                r = random.random()
                if r < Pb:
                    cell_temp[i][j] = -1
                elif Pb <= r < (Pb + Pw):
                    cell_temp[i][j] = 1
                else:
                    continue
            else:
                r = random.random()
                if r < Pd:
                    cell_temp[i][j] = 0
                else:
                    continue

t2 = np.linspace(0, 50, 1000)
ans = odeint(f, [0.33, 0.33], t2)

fig = plt.figure(figsize=(10, 6))
plt.plot(a[0], color='r', lw=3, alpha=0.75)
plt.plot(a[1], color='b', lw=3, alpha=0.75)
plt.plot(t2, ans[:, 0], color='r', lw=3, alpha=0.25)
plt.plot(t2, ans[:, 1], color='b', lw=3, alpha=0.25)
plt.xlabel('t')
plt.ylabel('alpha')
plt.legend([r'$\alpha_{w} 2D$', r'$\alpha_{b} 2D$', r'$\alpha_{w} 1D$', r'$\alpha_{b} 1D$'])
plt.grid()
fig.tight_layout()
plt.show()

[a, b, c, d] = [0.2, 0.4, 0.1, 1.2]


def g(data, t):
    [x, y] = data[:]
    x1 = -a * x + b * x * y
    y1 = c * y - d * x * y
    return [x1, y1]


data0 = [0.1, 0.67]
t3 = np.linspace(0, 100, 1000)

ans = odeint(g, data0, t3)

x = ans[:, 0]
y = ans[:, 1]

fig = plt.figure(figsize=(10, 6))
plt.plot(t3, x, color='r', lw=3, alpha=0.5)
plt.plot(t3, y, color='b', lw=3, alpha=0.5)
plt.xlabel('t')
plt.ylabel('size')
plt.ylim([0, 1])
plt.legend(['predator', 'prey'])
plt.grid()
fig.tight_layout()
plt.show()

[lambda1, lambda2, lambda3] = [1, 0.5, 0.75]


def h(data, t):
    [alpha_w, alpha_b, theta] = data[:]
    alpha_g = 1 - alpha_w - alpha_b
    Albedo = alpha_w * aw + alpha_b * ab + alpha_g * ag
    T_4 = SL * (1 - Albedo) / sigma
    T_w = (q * (Albedo - aw) + T_4) ** 0.25
    T_b = (q * (Albedo - ab) + T_4) ** 0.25
    alpha_w1 = alpha_w * (alpha_g * beta(T_w) - lambda1 * theta - 0.05)
    alpha_b1 = alpha_b * (alpha_g * beta(T_b) - lambda1 * theta - 0.05)
    theta1 = -lambda2 * theta + lambda3 * (alpha_w + alpha_b) * theta
    return [alpha_w1, alpha_b1, theta1]


SL = 917
t4 = np.linspace(0, 50, 5000)

fig = plt.figure(figsize=(10, 6))

for i in range(5):
    tmp1 = random.uniform(0.1, 0.9)
    tmp2 = random.uniform(0, 1 - tmp1)
    tmp3 = random.uniform(0, 0.1)
    data0 = [tmp1, tmp2, tmp3]
    sol = odeint(h, data0, t4)
    w = sol[:, 0]
    b = sol[:, 1]
    p = sol[:, 2]
    plt.plot(t4, w, color='r', lw=3, alpha=-0.1 * i + 0.6)
    plt.plot(t4, b, color='b', lw=3, alpha=-0.1 * i + 0.6)
    plt.plot(t4, p, color='g', lw=3, alpha=-0.1 * i + 0.6)
    plt.grid()
    fig.tight_layout()

plt.xlabel('t')
plt.ylabel(r'$\alpha$')
plt.ylim([0, 1])
plt.legend([r'$\alpha_w$', r'$\alpha_b$', 'predator'])
plt.show()
