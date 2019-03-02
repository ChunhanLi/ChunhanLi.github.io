---
layout:     post
title:      概率论与数理统计
subtitle:   概率论与数理统计
date:       2019-2-23
author:     Midone
header-img: img/post-bg-re-vs-ng2.jpg
catalog: True
tags:
    - 统计基础
---


此部分是基于STA200B/C的部分笔记。主要介绍假设检验，置信区间部分。

**一个小问题： 概率论和统计推断的区别**
在概率论中， 我们处理的问题多半是知道一个随机数产生的过程，去计算某些事情发生的概率。比如，我知道每次出门遇见隔壁小姑娘的概率是0.1， 那么我出门三次至少一次遇见她的概率是多少。

而在统计推断中， 我们往往知道了某个事件的结果，而要去推断出这个随机数的分布是如何的，在分布中有些值叫做参数（paramater），我们要去对这个值去进行检验或者估计。在上述例子中，就是要去估计遇见隔壁小姑娘的概率是多少。

### 1.重要的小知识

#### 1.1 样本方差

$S^2 = \frac{1}{n-1}\sum(X_i-\bar X)^2,~E(S^2)=\sigma^2$

#### 1.2 $\chi^2$分布

- $E(\chi_k^2)=k, Var(\chi_k^2)=2k$

- k个独立的标准正态随机数的和的分布即为卡方k. 

- 对于正态分布，$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1},~~ \bar X \bot S^2$


#### 1.3 t 分布

$$U\sim N(0,1),V \sim \chi^2_{n-1}, U \bot V \text {那么 } \frac{U}{\sqrt{\frac{V}{n-1}}} \sim t_{n-1}$$

#### 1.4 F 分布

$$ \text{如果}A\sim \chi^2_p, B \sim \chi^2_q, A \bot B \text{ 那么 }\frac{A/p}{B/q} \sim F_{p,q}$$

### 2.点估计

#### 2.1 矩估计(Moment estimator)

$X_1,X_2,\dots,X_n$ 是来自参数为$\theta$分布的随机样本。$\mu_1(\theta),\mu_2(\theta),\dots,\mu_k(\theta)$ 是该分布的前k阶原点矩。

$$\mu_j(\theta)=E(X_1^j \mid \theta)$$

定义样本原点矩$m_j = \frac{1}{n}\sum_i^nX_i^j,\text{~for~j} \geq1 $

那么矩估计就是把要估计的量写成关于原点矩的函数，然后再用样本原点矩代入分布原点矩去估计想要估计的统计量。

**Example**

$$Var(X)=\sigma^2=E(X^2)-E^2(x)=\mu_2(\theta)-\mu_1^2(\theta)$$

那么矩估计：

$$\hat \sigma^2=m_2 - m_1^2=\frac{1}{n}\sum_{i=1}^nX_i^2-(\bar X)^2=\frac{1}{n}\sum_{i=1}^n(X_i-\bar X)^2$$

对于这个例子， 我们没有用到任何关于模型的性质，所以这属于一个非参数方法(nonparametric method)。

但比如对于泊松分布（均值等于方差），我们也可以用$\bar X$去矩估计方差， 此时就属于参数方法。(parametric method)

#### 2.2 最大似然估计（MLE）

最大似然估计在统计学中特别常用。

**基本概念及其可行性**、

$$L(\theta\mid x)=f_X(x \mid \theta)=\prod_{i=1}^nf(x_i \mid \theta)$$

如果X是离散的，那么$L(\theta \mid x)=P_\theta(X_n = x_n)$（这里$X_n$ = ($X_1,\dots,X_n$)）.那么我们最大似然估计的出发点就在于在已经观察到$X_n$的条件下，找出$\theta=\arg \max_\theta L$使得该事件发生的概率最大，特别make sense哈。

当然对于连续分布，这里就会出现点小问题$P_\theta(X_n = x_n)=0$,这时我们可以这么定义对于一个关于$\theta$的连续分布，$P(x-\epsilon < X < x+\epsilon) \approx 2\epsilon f(x\mid \theta) = 2\epsilon L(\theta \mid x).$所以我们仍用上述的道理去找出最大化L的$\theta$.

**一般求MLE的方法**

1.单调就直接找
2.不单调求一阶导等于0，二阶导小于0
3.一阶导等于0， 多元的 check Hessian matrix 为负定矩阵
4.没有数值解，就可以考虑牛顿迭代法


MLE， 矩估计有不变性原则，例如A的MLE是B， 那么A平方的MLE就是B的平方。

#### 2.3 贝叶斯估计（Bayes Estimate）

**贝叶斯统计是啥**

统计学上有两个分支， 一个叫做贝叶斯学派， 一个叫做频率学派。 最大的区别在于是否认为参数$\theta$是一个固定值。 贝叶斯学家认为这个参数应该是个随机变量， 而不是一个固定的值。（在前面的矩估计，最大似然中，我们都把其看做固定值）。

##### 2.3.1 先验分布

$\theta$的先验分布在实验没有进行之前（没有出样本结果前）就被选定了， 这个分布反应了实验者对其的认识。

##### 2.3.2 后验分布

在我们观察到X之后， 在给定$(x_1,\dots,x_n)的条件下， $$\theta$的分布称作其后验分布。经常表示成$\pi_{\theta \mid x}(\theta \mid x
)$

$$\pi_{\theta \mid x}(\theta \mid x)=\frac{f_{X\mid\theta}(x \mid \theta)\pi(\theta)}{\int_\Theta f_{X\mid\theta}(x \mid \theta)\pi(\theta)d\theta}$$

##### 2.3.3 贝叶斯估计

贝叶斯估计依靠于损失函数，我们目标就是为了最小化损失函数。对于平方损失，贝叶斯估计为其后验分布的均值，对于绝对损失，贝叶斯估计量为其后验分布的中位数。

### 3.置信区间

#### 3.1 定义

设$\theta$是总体的一个参数,其参数空间为$\Theta$, $x_1,\dots,x_n$是来自该总体的样本, 对给定的一个$\alpha$(0<$\alpha$<1), 假设有两个统计量$L = L(X_1,\dots,X_n)$, $U = U(X_1,\dots,X_n)$, 若对任意的$\theta \in \Theta$, 有$P_\theta(L \leq \theta \leq U) \geq 1 - \alpha$,则称[L,U]为$\theta$的置信水平为$1-\alpha$的置信区间。

**Remark:**

- 这里需要注意的是这里的L，U都是随机数，因为他们是基于随机数的统计量。$\theta$是一个未知的定值（真实值）。一旦产生了样本,我们可以将L，U对于这次样本的值代入，比如得到的0.95置信区间是[3.2,5.5]。这不是意味着$3.2\leq \theta \leq 5.5$的概率是0.95. 而是意味着我们如果能抽取100次这样的样本，那么我们得到的置信区间大致有95次捕捉到真实的$\theta$（或者每次这样的样本产生的L，U， 我们有95%把握真实$\theta$在L，U之间）.

#### 3.2 构建置信区间

**Pivot**

A random variable $Q(X,\theta)$ of $X$ and $\theta$ is called a pivot if the distribution of $Q$ does not involve $\theta$. Such functions Q are called pivot.

##### 3.2.1 常见pivots

$X_1,\dots,X_n \sim N(\mu,\sigma^2)$

1. C.I. for $\mu$. $\sigma^2$ is known.

$$\frac{\bar X - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$$

2. C.I. for $\mu$. $\sigma^2$ is unknown.

$$\frac{\bar X - \mu}{S/\sqrt{n}} \sim t_{n-1}$$

3. C.I. for $\sigma^2$. $\mu$ known.

$$\sum_i(\frac{X_i -\mu}{\sigma})^2 \sim \chi^2_n$$

4. C.I. for $\sigma^2$. $\mu$ is unknown.

$$\sum_i(\frac{X_i -\bar X}{\sigma})^2 \sim \chi^2_{n-1}$$

### 4.假设检验

假设检验通常包括以下几方面：

- 原假设$H_0$, 备择假设$H_a$.
- 拒绝域R：样本空间的一个子集，当样本落在拒绝域内，选择拒绝原假设。拒绝域的选择通常依赖于一个统计量。例如$R=\{X_1,X_2,\dots,X_n \mid \bar X >3\}$
- 显著性水平

#### 4.1 一些定义和总结

**Type I Error and Type II Error**

||$H_0$ is True|$H_0$ is False|
|:-:|:-:|:-:|
|Reject $H_0$|Type I Error|正确|
|Not Reject $H_0$|正确|Type II Error|

- 犯第一类错误的概率$\alpha = P_\theta(X \in R), \theta \in \Theta_0$
- 犯第二类错误的概率$\beta = 1- P_\theta(X \in R), \theta \in \Theta_1$
- 本文后面会讲到，我们一般控制第一类错误率在一定范围内(比如0.05)。 这是一种为了保护原假设的行为，所以我们一般会把一些我们想要去保护的结论作为原假设，比如'这个药没有效果'或者是'这个嫌疑人无罪'。因为一般来说错误的接受‘这个药有效果’或者‘嫌疑人有罪’会带来极大的损失。所以除非我们有极大的证据证明这个嫌疑人有罪，不然我们就不会给他定罪，这就是前面所说的保护原假设的一种解读。
- 另一种思想就是要把你想去证明的放在备择假设想。意思就是只有你有极大的证据证明你的结论是正确的，才会拒绝原假设。

**Power Function**

$\beta(\theta)=P_\theta(X \in R)$

最理想的power function为0， 当$\theta \in \Theta_0$, 为1,当$\theta \in \Theta_1$。 一般情况下很难达到这种情况。所以我们通常采取的策略，就是控制一类错误在一定数值下，使二类错误率达到最小。

**显著性水平**

对于任意的$\theta \in \Theta_0$, 都有 $B(\theta) \leq \alpha$则称该检验是水平为$\alpha$的检验。

在200B中定义有所不同，

- A test with power function $\beta(\theta)$ is a size $\alpha$ test if $\sup_{\theta \in \Theta_0}\beta(\theta)=\alpha$

- A test with power function $\beta(\theta)$ is a level $\alpha$ test if $\sup_{\theta \in \Theta_0}\beta(\theta)\leq \alpha$

**P-value**
![image](https://github.com/ChunhanLi/ChunhanLi.github.io/blob/master/img/p-value.png?raw=true)
 
#### 4.2 构建假设检验

##### 4.2.1 Likelihood Ratio test

$L(\theta \mid x_1,x_2,\dots,x_n) = L(\theta \mid X) = \prod_{i=1}^nf(x_i \mid \theta)$

The likelihood ratio test statistic for testing $H_0: \theta \in \Theta_0$ versus $H_1:\theta \in \Theta_0^c$ is 

$$\lambda(x)=\frac{\sup_{\Theta_0}L(\theta \mid X)}{\sup_{\Theta}L(\theta \mid X)}$$

A LRT is any test that has a rejection region of the form{x:$\lambda(x)\leq c$ },where c is any number between 0 and 1.

##### 4.2.2 Bayesian Tests

考虑其后验分布。

- 一种方法是： not reject $H_0$ if $P(\theta \in \Theta_0 \mid X) \geq P(\theta \in \Theta_0^c \mid X)$也就是前者大于0.5.
- 另一种情况是：实验设计者一般都将错误拒绝损失大的作为原假设，所以在这种情况下，这个界限0.5可能被降低。 

##### 4.2.3 Union-Intersection and  Intersection-Union Tests

这里只列举Union-Intersection方法， 后面的同理。

The null hypothesis is conveniently expressed as an intersection, say $H_0:\theta \in \bigcap_{\gamma \in \Gamma}\Theta_\gamma$. Suppose that tests are available for each of the problems of testing $H_{0\gamma}:\theta \in \Theta_\gamma$ versus $H_{1\gamma}:\theta \in \Theta_\gamma$. Then the rejection region for the union-intersection test is $\bigcup_{\gamma\in \Gamma}${x:$T_\gamma(x) \in R_\gamma$}.

##### 4.2.4 Neyman-Pearson Lemma(simple to simple)

Consider testing $H_0: \theta = \theta_0 \text{ versus } H_1: \theta = \theta_1$. A test with rejection region R is a UMP level $\alpha$ test if and only if

- $x \in R \text{ if } f(x\mid \theta_1) > kf(x \mid \theta_0)$  and  $x \in R^c \text{ if } f(x\mid \theta_1) < kf(x \mid \theta_0)$ for some k $\geq 0$
- $\alpha = P_{\theta_0}(X \in R)$  

**引理**

Suppose $T(x)$ is a sufficient statistic for $\theta$ and $g(t \mid \theta_i)$ is the pdf or pmf of T corresponding to $\theta_i$, i=0,1. Then any test based on T with rejection region S is a UMP level $\alpha$ test if it satisfies 

- $t \in S \text{ if } g(t \mid \theta_1) > kg(t \mid \theta_0)$

- $t \in S^c \text{ if } g(t \mid \theta_1) < kg(t \mid \theta_0)$
- for some k $\geq 0$, where $\alpha = P_{\theta_0}(T \in S)$

##### 4.2.5

由上述定理可以延伸到(simple to composite)

Example(UMP normal test):

Let $X_1,\dots,X_n$ be a random sample from a $n(\theta,\sigma^2)$ population, $\sigma^2$ known. The sample mean $\bar X$ is a sufficient statistic for $\theta$. Considering testing $H_0: \theta = \theta_0 \text{ versus } H_1: \theta = \theta_1 \text{ where } \theta_0 > \theta_1$.

$g(\bar x \mid \theta_1) > kg(\bar x \mid \theta_0)$, is equivalent to $\bar X < \frac{(2\sigma^2\log k)/n - \theta_0^2 + \theta_1^2}{2(\theta_1 - \theta_0)}$. It means that the test with rejection region $\bar X < c$ is the UMP level $\alpha$ test, where $\alpha = P_{\theta_0}(\bar X < c)$. Actually, $P_{\theta_0}(\bar X < c) = P_{\theta_0}(\frac{\bar X - \theta_0}{\sqrt{\sigma^2/n}} < c')$. It only depends on $\theta_0$. So we can replace $H_1$ by $H_1 :\theta \geq \theta _0$

##### 4.2.6 Asymptotic distribution of the LRT---simple $H_0$

For testing $H_0:\theta = \theta_0 \text{ versus } H_1:\theta \neq \theta_0$. Suppose $X_1,\dots, X_n$ are iid $f(x \mid \theta)$, $\hat \theta$ is the MLE of $\theta$. And $f(x \mid \theta)$ satisfies the some regularity conditions. Then under $H_0$, as $n \rightarrow \infty$,

$$-2\log \lambda(X) \rightarrow \chi_1^2 \text{ in distribution}$$

**one more**

![image](https://github.com/ChunhanLi/ChunhanLi.github.io/blob/master/img/dayangbei.png?raw=true)

Note:这里的分位数定义和我平常习惯的不一样。所以这里改成$\chi^2_{v,1-\alpha}$.

**Example: 证明筛子是均匀的**

The result of dice is distributed by Multinomial distribution. Let $\theta = (p_1,\dots,p_6)$. $P_{\theta}(X_i = j) = p_j$. Thus the pmf of $X_i$ is $f(j \mid \theta) = p_j$. And the likelihood function is $L(\theta \mid x) = \prod_{i=1}^nf(x_i \mid \theta) = p_1^{y_1}p_2^{y_2}p_3^{y_3}p_4^{y_4}p_5^{y_5}p_6^{y_6}$ where $y_j= \text{ number } of x_1,\dots,x_n \text{ equal to j}$.

Consider testing 

$$H_0: p_1 = p_2 = p_3 = p_4 = p_5 = p_6 \text{ versus }  H_1:H_0 \text{ is not true}$$

In this case, $v = 5 -0 =5$. Then, calculate $\lambda(x)$. The MLE of $p_j$ under $\Theta$ is $\hat p_j = y_j/n$. So,

$$\lambda(x) = (\frac{n}{6y_1})^{y_1}\dots(\frac{n}{6y_6})^{y_6}$$

The asymptotic size $\alpha$ test rejects $H_0$ if $-2\log \lambda(x) \geq \chi^2_{5, 0.95}$. Make sure that n is large enough.

#### 4.3 定义

##### 4.3.1 unbiased test

A test with power function $\beta(\theta)$ is unbiased if $\beta(\theta') \geq \beta(\theta'')$ for every $\theta' \in \Theta_0^c$ and $\theta'' \in \Theta_0$

##### 4.3.2 Most Powerful Tests

Let $C$ be a class of tests for testing $H_0: \theta \in \Theta_0 \text{ versus } H_1:\theta \in \Theta_0^c$. A test in class $C$, with power function $\beta(\theta)$ is a **uniformly most powerful(UMP)** class $C$ test if $\beta(\theta) \geq \beta'(\theta)$ for every $\theta \in \Theta_0^c$ and every $\beta'(\theta)$ that is a power function of a test in class $C$.

Generally, the class $C$ will be the class of all level $\alpha$ tests. It's called a UMP level $\alpha$ test.

##### monotone likelihood ratio(MLR)

A family of pdfs or pmfs {$g(t \mid \theta):\theta \in \Theta$} for a univariate random variable T with real-valued parameter $\theta$ has a monotone likelihood ratio(MLR) if , for every $\theta_2 > \theta_1$, $g(t\mid \theta_2) / g(t \mid \theta_1)$ is a monotone (nonincreasing or nondecreasing) function of t on {$t:g(t \mid \theta_1)>0 \text{ or } g(t\mid \theta_2) > 0$} 

Any regular exponential family with $g(t \mid \theta) = h(t)c(\theta)e^{w(\theta)t}$ has an MLR if $w(\theta)$ is monotone.

**Karlin-Rubin定理**

Consider testing $H_0:\theta \leq \theta_0 \text{ versus } H_1: \theta > \theta_0$. Suppose that T is a sufficient statistic for $\theta$ and the family of pdfs or pmfs {$g(t\mid\theta): \theta \in \Theta$} of T has an MLR. Then for any $t_0$, the test that rejects $H_0$ if and only if $T > t_0$ is a UMP level $\alpha$ test, where $\alpha = P_{\theta_0}(T > t_0)$.

If $H_0: \theta \geq \theta_0$ and of $H_1: \theta < \theta_0$ if and only if $T < t_0$ is a UMP level $\alpha = P_{\theta_0}(T < t_0)$ test.

### 5 充分统计量

本来不想提这个的，但是定理要用的，还是小记录一哈吧。

#### 5.1 定义

A statistic $T(X_n)$ is called a sufficient statistic for $\theta$, if the conditional distribution of $X_n=(X_1, \dots, X_n)$ given $T=t$ does not depend on $\theta$.

#### 5.2 Fisher-Neymann Factorization Theorem

A statistic $T(X_1,\dots,X_n)$ is sufficient for $\theta$ if and only if $f_n(x_n \mid \theta) = g(T(x_n) \mid \theta)h(x_n)$, where $h$ does not depend on $\theta$ and g depends on $x_n$ only through $T(x_n)$.

#### 5.3 Exponential family

An exponential family of distribution $f(x \mid \theta) = h(x)c(\theta)e^{\sum_{j=1}^kw_j(\theta)T_j(x)}$. Then $T(x_n) = (\sum_{i=1}^nT_1(X_i),\dots,\sum_{i=1}^nT_k(X_i))$


