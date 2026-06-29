# 统一信号:多 σ Tweedie 去噪残差场

> 用一个**冻结的 SD2 先验**,在多个噪声级 $t$ 上测量去噪残差,得到一个有理论根基的取证场(score 场对样本的适配度剖面),而非手搓的"噪声分支"。文档涵盖:理论背景、数学推导、机理解释、三项输出、与现有工作的区别,以及完整的参数含义表。

---

## 0. 一句话主线

整套方法的核心恒等式链是:

$$
\underbrace{\hat\varepsilon_\theta(z_t,t)}_{\text{网络噪声预测}}
\;\Longleftrightarrow\;
\underbrace{s_\theta=\nabla\log p_t(z_t)}_{\text{score}}
\;\Longleftrightarrow\;
\underbrace{\hat z_0=\mathbb E[z_0\mid z_t]}_{\text{Tweedie 后验均值}}
$$

去噪器的输出**就是** score(三者是同一对象的不同写法);而我们真正拿来做取证的量,是 score 拟合的**逐样本缺口** $r_\text{eps}(t)=\mathbb E\|\varepsilon-\hat\varepsilon_\theta\|^2$ 在多个噪声级上的**剖面**。

需要先澄清一个常见混淆:**$t$、$\bar\alpha_t$、$\sigma_t$ 是同一根"加了多少噪声"轴的三种刻度**——$t$ 是离散步号,$\bar\alpha_t$ 是信号保留比例,$\sigma_t$ 是等效噪声标准差;知道任意一个就能推出另两个。

---

## 1. 坐标:把 VP 写成"加性高斯 + σ_t"

前向加噪(方差保持 / VP 形式):

$$
z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I).
$$

Tweedie 公式的经典写法要求"信号系数为 1 的加性噪声",所以先做一次重标定:

$$
x_t \equiv \frac{z_t}{\sqrt{\bar\alpha_t}}=z_0+\sigma_t\,\varepsilon,
\qquad
\boxed{\;\sigma_t^2=\frac{1-\bar\alpha_t}{\bar\alpha_t}\;},
\qquad
\text{SNR}=\frac{1}{\sigma_t^2}=\frac{\bar\alpha_t}{1-\bar\alpha_t}.
$$

在这个重标定坐标里,$x_t$ 就是干净信号叠加标准差 $\sigma_t$ 的高斯噪声,经典 Tweedie 公式直接成立。

**尺度空间视角**:边际密度 $p_{\sigma_t}$ 是先验 $p(z_0)$ 被高斯核 $\mathcal N(0,\sigma_t^2 I)$ 卷积的结果。"在多个 $t$ 上探测"等价于"用一族不同半径 $\sigma_t$ 的高斯核去看密度"——大 $\sigma$ 看粗结构(密度近似高斯、可分性低),小 $\sigma$ 看细节(细节敏感但噪声大)。可分性集中在小到中等 $\sigma$ 的窄带。

---

## 2. Tweedie 公式:去噪器 = 后验均值 = score

对边际 $p_\sigma(x)=\int p(x_0)\,\mathcal N(x;x_0,\sigma^2 I)\,dx_0$ 直接对 $x$ 求梯度:

$$
\nabla_x p_\sigma(x)
=\int p(x_0)\,\mathcal N(x;x_0,\sigma^2 I)\cdot\Big(-\frac{x-x_0}{\sigma^2}\Big)\,dx_0
=-\frac{1}{\sigma^2}\Big[x\,p_\sigma(x)-\!\int x_0\,p(x_0)\mathcal N(x;x_0,\sigma^2 I)\,dx_0\Big].
$$

两边除以 $p_\sigma(x)$,识别出后验均值 $\mathbb E[x_0\mid x]$:

$$
\nabla_x\log p_\sigma(x)=-\frac{1}{\sigma^2}\big(x-\mathbb E[x_0\mid x]\big)
\quad\Longleftrightarrow\quad
\boxed{\;\mathbb E[x_0\mid x]=x+\sigma^2\,\nabla_x\log p_\sigma(x)\;}
$$

这就是 **Tweedie / 经验贝叶斯公式**,也即原始记法 $(\hat x-x)/\sigma^2\approx\nabla\log p_\sigma(x)$(其中 $\hat x=\mathbb E[x_0\mid x]$)。它的内容是:**MMSE 去噪器的输出本身就是 score**,二者只是同一量的两种写法。

---

## 3. ε-预测与 score 的等价

条件 score 可直接算:由 $z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\varepsilon$,

$$
\nabla_{z_t}\log p(z_t\mid z_0)=-\frac{z_t-\sqrt{\bar\alpha_t}z_0}{1-\bar\alpha_t}=-\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}}.
$$

去噪 score matching 让最优网络对齐边际 score,得到标准恒等式:

$$
\boxed{\;s_\theta(z_t,t)=\nabla_{z_t}\log p_t(z_t)=-\frac{\hat\varepsilon_\theta(z_t,t)}{\sqrt{1-\bar\alpha_t}}\;}
$$

代回 Tweedie,你定义的

$$
\hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\,\hat\varepsilon_\theta}{\sqrt{\bar\alpha_t}}
$$

**恰好等于** $\mathbb E[z_0\mid z_t]$。所以真正"是 score"的对象是 $\hat\varepsilon_\theta$(或等价的 $\hat z_0$)。

> ⚠️ **措辞精确化**:残差 $r_\text{eps}(t)$ **不是** score,而是 score 拟合的逐样本缺口——它是去噪 score-matching(Vincent 2011)目标在该样本、该尺度上的取值,衡量"模型学到的 score 场对这张图适配得多好"。这恰恰是我们想要的取证量:不是 score 的大小,而是 score 场的 **goodness-of-fit**。

---

## 4. 残差到底测什么 + 一个该指出的恒等式

两条剖面分量定义为:

$$
r_\text{eps}(t)=\mathbb E_\varepsilon\big\|\varepsilon-\hat\varepsilon_\theta(z_t,t)\big\|^2,
\qquad
r_x(t)=\big\|\hat z_0-z_0\big\|^2.
$$

把 $z_t$ 代入 $\hat z_0$:

$$
\hat z_0-z_0=\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\,(\varepsilon-\hat\varepsilon_\theta)=\sigma_t\,(\varepsilon-\hat\varepsilon_\theta)
\;\Longrightarrow\;
\boxed{\;r_x(t)=\sigma_t^2\,r_\text{eps}(t)\;}
$$

**含义**:剖面向量 $[\,r_\text{eps}(t_{1..K}),\,r_x(t_{1..K})\,]$ 的两半**不是两路独立测量**,而是同一个 score-matching 残差在两种加权下的样子——$r_\text{eps}$ 是 VP/ε 权重,$r_x$ 等于乘上 $\sigma_t^2$ 后的 $x_0$ 权重。信息增益不在"双分支",而在 $\sigma_t^2$ 这个单调因子如何在噪声级间重新分配权重($r_x$ 抬高大 $t$ 端,$r_\text{eps}$ 抬高小 $t$ 端)。

> **改进建议**:若要拿到真正互补的第二路信息,应改测方向性 / 逐通道量,例如 $\langle\varepsilon,\hat\varepsilon_\theta\rangle$、残差频谱、或 $\hat z_0$ 的雅可比迹散度,而不是再算一个被 $\sigma_t^2$ 锁死的标量。

---

## 5. 为什么 score 剖面是取证场:流形—不动点机理

这是相比"手搓噪声分支"最有解释力的部分,也回答了"为什么扩散与扩散难分"。

记去噪映射 $D_t(z_t)=\hat z_0$。

- **生成图**:其 $z_0$ 是反向扩散迭代的产物,本质是 $D$ 的近似**不动点 / 吸引子**。重新加噪再去噪,网络"认出"它在流形上,$\hat\varepsilon_\theta\approx\varepsilon$,残差贴着 **in-distribution 的不可约下界(贝叶斯 DSM floor)**。
- **真实图**:带有传感器噪声、去马赛克 / JPEG 痕迹、自然图像统计——这些是冻结先验的 score 场没完全覆盖的方向。去噪会把图朝流形"拉",留下高于 floor 的缺口。

跨 $\sigma$ 看(尺度空间论证):

| 区域 | 行为 | 可分性 |
|---|---|---|
| $\sigma_t\to\infty$(大噪声) | $z_t$ 近纯噪声,$\hat\varepsilon_\theta\to 0$,$\|\varepsilon-\hat\varepsilon_\theta\|^2\to\|\varepsilon\|^2\approx d$ | 所有图趋同,**无可分性**(密度被高斯抹平) |
| $\sigma_t\to 0$(小噪声) | 细节主导;模型对 on-manifold 内容拟合好,对真实高频拟合差 | 缺口集中,**可分性最高** |

**"扩散难分扩散"的机理解释**:任何扩散生成图按构造就是去噪算子的近不动点,其剖面整体塌向 on-manifold floor,差异只剩在特定 $\sigma$ 带里的细微抬升。单尺度方法刚好把这条带平均掉了,所以失效;多 $\sigma$ 剖面把整条曲线 $t\mapsto r(t)$ 当指纹,才看得见。

---

## 6. 三项输出

### 6.1 检测 / 定位

逐位置(latent 逐空间位置)计算 $K$ 维剖面 $r_\text{eps}(t,\cdot)$,得到一个剖面**场**。

- **检测**:对整图剖面做分类 / 异常检测。
- **定位**:看剖面场的空间分区——采样区贴 floor、真实区抬升、边界处出现跳变。

真实像素与模型采样像素的 $\sigma$ 剖面系统性不同:采样区接近去噪映射的不动点、落在流形上,故贴近下界。

### 6.2 编辑算子归因(最原创点)

每种编辑在 $\sigma$ 剖面上留下可区分签名,因为签名是其**生成机制的直接投影**。以 SDEdit / img2img 为例:把引导图加噪到 $t_0$,即

$$
z_{t_0}=\sqrt{\bar\alpha_{t_0}}\,z_\text{guide}+\sqrt{1-\bar\alpha_{t_0}}\,\varepsilon,
$$

再从 $t_0$ 反向到 0。后果:$t<t_0$ 的内容是模型**合成**的(贴 floor),$t\gtrsim t_0$ 暴露被保留的引导结构(off-manifold,抬升)。剖面在 $t_0$ 附近出现**转折(kink)**,转折点横坐标即"编辑诞生尺度"。

把类型识别表述为**反估 $(t_0,c,M)$ 的逆问题**,而非"接个 softmax":

$$
\boxed{\;(\hat t_0,\hat c,\hat M)=\arg\min_{t_0,c,M}\ \mathcal D\big(r_\text{obs},\,\mathcal F(t_0,c,M)\big)+\lambda\,\mathcal R\;}
$$

即"用哪组编辑参数最能解释观测到的剖面场"。$\mathcal F$ 是把"编辑如何在剖面留痕"显式建模的前向成像模型。这比黑箱分类器强在:可加先验 / 正则、能给不确定度、对未见过的编辑类型有外推。

> **诚实告诫**:这是可能病态的逆问题。$(t_0,c)$ 之间可能简并(弱编辑+强条件 ≈ 强编辑+弱条件),掩码在低对比区不可辨识。先做合成可辨识性分析(固定真值、看后验是否单峰)再上真实数据——这正是"逆问题"框架比"softmax"诚实之处:它逼你正视可辨识性。

---

## 7. 与现有工作的区别

| 方法 | 测量 | 尺度 | 归因 | 机理解释 |
|---|---|---|---|---|
| DIRE | DDIM 重建误差 $\|z_0-\text{recon}\|$ | 单一工作点 | 无 | 无 |
| AEROBLADE | LDM 自编码重建误差 | 单一工作点 | 无 | 无 |
| **本方法** | 多 σ score-matching 残差剖面 $t\mapsto r(t)$ | 多尺度 | 算子逆估计 $(t_0,c,M)$ | 流形 / 不动点 |

单尺度方法等于只采了 score 场上的一个半径;本方法采的是整条多半径轮廓,并把编辑识别做成算子反演,且从机理上解释了单尺度为何在"扩散 vs 扩散"上失效。

---

## 8. 参数含义总表

### 8.1 前向加噪过程(扩散机制本身)

| 符号 | 含义 | 取值 / 性质 |
|---|---|---|
| $z_0$ | 干净 latent(图经 VAE 编码后的码,SD2 为 4 通道,**非像素**) | 待检测 / 归因的对象 |
| $\varepsilon$ | 注入的标准高斯噪声,与 $z_0$ 同形状 | $\varepsilon\sim\mathcal N(0,I)$ |
| $t$ | 时间步 / 噪声级编号 | 整数 $1{\dots}T$(SD2 中 $T{=}1000$),越大噪声越多 |
| $\beta_t$ | 每步方差表(noise schedule) | 由预训练固定,隐含在 $\bar\alpha_t$ 内 |
| $\bar\alpha_t$ | 累积信号保留系数 $\prod_{s\le t}(1-\beta_s)$ | 从 $\approx 1$ 单调降到 $\approx 0$ |
| $\sqrt{\bar\alpha_t}$ | 信号缩放因子 | 干净分量权重 |
| $\sqrt{1-\bar\alpha_t}$ | 噪声缩放因子 | 噪声分量权重 |
| $z_t$ | 第 $t$ 级含噪 latent | $z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\varepsilon$ |
| $\sigma_t$ | 重标定坐标下的等效噪声尺度 | $\sigma_t^2=(1-\bar\alpha_t)/\bar\alpha_t$;$\text{SNR}=1/\sigma_t^2$ |

### 8.2 网络与 Tweedie 估计(score 从哪来)

| 符号 | 含义 | 说明 |
|---|---|---|
| $\theta$ | **冻结的** SD2 UNet 权重 | 不训练,当固定先验用 |
| $\hat\varepsilon_\theta(z_t,t)$ | 网络对注入噪声的预测(ε-预测头) | 真正"承载 score"的量 |
| $s_\theta=\nabla\log p_t(z_t)$ | score(对数密度梯度)估计 | $s_\theta=-\hat\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ |
| $\hat z_0$ | 干净 latent 的 Tweedie 后验均值估计 $\mathbb E[z_0\mid z_t]$ | $\hat z_0=(z_t-\sqrt{1-\bar\alpha_t}\hat\varepsilon_\theta)/\sqrt{\bar\alpha_t}$ |
| $p_t(z_t),\,p_\sigma(x)$ | 第 $t$ 级 / 尺度 $\sigma$ 下含噪 latent 的边际密度 | 先验被高斯核卷积后的密度 |
| $d$ | latent 维数(通道×高×宽) | 决定大-$\sigma$ 端天花板 $\|\varepsilon\|^2\approx d$ |

### 8.3 残差与剖面(真正的取证量)

| 符号 | 含义 | 说明 |
|---|---|---|
| $r_\text{eps}(t)$ | 去噪 score-matching 残差 $\mathbb E_\varepsilon\|\varepsilon-\hat\varepsilon_\theta\|^2$ | score 场对该图、该尺度的适配度;期望需对多个 $\varepsilon$ 做 Monte-Carlo |
| $r_x(t)$ | 重建残差 $\|\hat z_0-z_0\|^2$ | $r_x(t)=\sigma_t^2\,r_\text{eps}(t)$,与上项确定性耦合,**非独立** |
| $K$ | 采样的噪声级数目 $t_1{\dots}t_K$ | 决定剖面分辨率 |
| $[\,r_\text{eps}(t_{1..K}),\,r_x(t_{1..K})\,]$ | 每张图的剖面特征向量 | 长度 $2K$,分类 / 异常检测 / 反演的输入 |

### 8.4 编辑算子归因(逆问题的未知量)

| 符号 | 含义 | 在剖面上的信道 |
|---|---|---|
| $t_0$ | 编辑诞生尺度(SDEdit 加噪到的级别,等价于 img2img 强度 $s$) | 剖面转折点的横坐标 |
| $c$ | 条件机制(文本 cross-attention / ControlNet 空间条件 / 无条件) | 残差的纹理与各向异性 |
| $M$ | 掩码支撑(inpainting 实际重生成的空间区域) | 剖面场的空间分区与边界签名 |
| $\mathcal F$ | 前向成像模型 $(t_0,c,M)\mapsto r(t,\cdot)$ | 把"编辑如何留痕"显式建模 |
| $\lambda,\ \mathcal R$ | 反演目标里的正则权重与正则项 | 处理 $(t_0,c)$ 的简并 / 病态 |

---

## 9. 落地时绕不开的现实约束

1. **冻结 SD2 先验与目标生成器的域差**。score 场是 SD2 学到的;若待检图来自 SDXL / Flux / 微调模型,对比可能被污染成"SD2 域内 vs 域外"。检测仍可能有效(域外都抬升),但归因前向模型 $\mathcal F$ 是 SD2 的,跨架构有系统偏差——须用**留一生成器(leave-one-generator-out)**量化。

2. **Monte-Carlo 方差**。窄可分带的信噪比取决于每个 $t$ 采多少个 $\varepsilon$。建议对偶(antithetic)采样并在 latent 空间逐通道累计,否则单次实现的剖面抖动会盖过信号。

3. **逆问题可辨识性**。$(t_0,c)$ 可能简并。先做合成可辨识性分析(固定真值、检验后验是否单峰),再上真实数据。

---

*符号约定:$z$ 系列在 latent 空间;$x$ 系列为 Tweedie 重标定坐标;$\mathbb E$ 默认对注入噪声 $\varepsilon$ 求期望,实现时以 Monte-Carlo 近似。*
