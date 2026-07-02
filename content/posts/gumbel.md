---
title: "从离散到连续"
date: "2026-07-02T21:05:16+08:00"
lastmod: "2026-07-02T21:05:16+08:00"
description: "基于 Gumbel Softmax 的连续空间采样"
summary: "基于 Gumbel Softmax 的连续空间采样"
tags: ['math', 'idea']
categories: []
series: []
draft: False
comments: true
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowWordCount: true
cover:
  image: ""
  alt: ""
  caption: ""
  relative: false
---

这里推进一下上一篇博客提到的强化学习与连续空间采样部分的研究，参考了 [SofT-GRPO](https://arxiv.org/pdf/2511.06411)

为了计算 policy gradient，我们要解决的问题可以形式化为，假设我们有一大小为 $M$ 的向量组 $\\{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_m\\}$，定义一维 Gumbel 分布为 $G(0,1)$，给定一大小为 $M$ 的 $\mathrm{logit}$ 组 $L = \\{l_m\\}$，采样噪声 $\epsilon_i \sim G(0,1)$，计算 $\mathbf{a}= \sum_i \mathrm{softmax}(l_i+\epsilon_i)\mathbf{e}_i$，求解 $\log p(\mathbf{a}|L)$

这里如果直接算 $\log p(\mathbf{a}|L)$ 是没法算的，因为向量组可能不是线性无关组。SofT-GRPO 取了个巧，作者计算 $\log p(l_i+\epsilon_i|L)$ 作为替代，把权重视为动作，softmax加权和的过程视为一个常变换，这有好处也有坏处，好处就是这个对数概率很容易算，直接取 $\log\prod_i \exp(-\epsilon_i-\exp(-\epsilon_i))$ 即可，但坏处就是 embedding 的梯度流通不畅。我大胆假设，不妨把这个 embedding 矩阵就冻结为随机的高斯向量组，或许也能取得很好的效果？
待我之后的实验

这里其实有个问题，假设有一个场景，要求模型唯一确定地输出一个动作，gumbel-softmax 是否能满足这一需求？首先如果这个动作为 $\mathbf{e}_i$，那么很显然只要输出 $l_i$ 为 $+\infty$，其余为 $-\infty$ 即可，但如果这个动作不是向量组其中之一而是它们的线性组合呢？如果试图像离散 token 的 softmax 那样单纯地降低温度，最终并不能得到预期的效果，模型只会返回最高的 $l_i$ 对应的 $\mathbf{e}_i$，其实最根本的问题是模型需要调整 $\mathrm{logit}$ 的尺度以覆盖掉 $G(0,1)$，这一点就限制了 $\mathrm{logit}$ 之间的比例必须保持不变，我们只能整体调整他们的尺度，而这与调整温度无异，所以矛盾，现有的表达式不能满足我们这一需求。

我说其实可以改一改，我们把加权和调整成这样 $\sum_i \mathrm{softmax}(l_i+\lambda_i \cdot \epsilon_i)\mathbf{e}_i$
完全不影响 evaluation，而且模型可以自己控制探索的温度，或者干脆更暴力但省参数一点，直接学一个全局温度 $T$，然后 $\sum_i \mathrm{softmax}(l_i+ T \cdot \epsilon_i)\mathbf{e}_i$，也并不影响 $\log p$ 的求解，接下来可以试试这个