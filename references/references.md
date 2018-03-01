# Highway Networks

[@srivastava2015highway]

Standard NN layer: $y = H(x, W_H)$ where $H$ is a non-linear transformation parametrized by weights $W_H$

Highway network:

$$
y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)
$$

where T is the _transform_ gate and C is the _carry_ gate, which define the ratio in which the output is defined by transforming the input in contrast to carrying it over. For simplicity, $C = 1 - T$, producing:

$$
y = H(x, W_H) \cdot T(x, W_T) + x \cdot (1 -  T(x, W_T))
$$

**Note**: this formulation, where each layer can propagate its input $x$ further requires that all of the elements have the same dimension ($y$, $x$, $T$, $H$). An option here is to use padding to upscale $x$ or sub-sampling, in order to reduce the dimensionality. An option is also to use a regular layer (without the highway connections) to change the dimensionality, and then continue with the highway layers.

# Dropout
Introduced in: [@hinton2012improving]

# Backpropagation
Introduced in: [@rumelhart1985learning]

# Maxout networks

[@goodfellow2013maxout]

Maxout networks use the _maxout_ function as the activation. For an input $x \in \mathbb{R}^d$ the maxout is:

$$
h_i(x) = max_{j\in[1,k]} z_{ij}
$$

where $z_{ij} = x^T W_{\ldots ij} + b_{ij}$, $W \in \mathbb{R}^{d \times m \times k}$ and $b \in \mathbb{R}^{m \times k}$ are the learned model parameters.

Essentially: instead of projecting into the output dimension $m$, project into $m\times k$ and max over the $k$ additional dimensions.
Pytorch impl: [](https://github.com/pytorch/pytorch/issues/805)

# Grid LSTM
[@kalchbrenner2015grid]

Similar to Multi-dimensional Recurrent Neural Networks [@graves2009offline]

LSTM along each dimension of network (depth, T). The vertical LSTM hidden / cell states initialized by the inputs.

N-dimensional Grid LSTM accepts N hidden vectors $h_1, \ldots, h_N$ and N memory vectors $m_1, \ldots, m_N$, which are all distinct for each dimension.

All of the hidden states are then concatenated:


\begin{equation}
H = \begin{bmatrix}
  \hat{h}_ {1}\\
  \vdots \\
  \hat{h}_ {N}
\end{bmatrix}
\end{equation}

The N-dimensional block then computes N LSTM tranfsorms, one for each dimension. Each LSTM transform has its individual weight matrices. Each block accepts input hidden and memory vectors from N dimensions, and outputs them into N dimensions.

\begin{equation}
\begin{aligned}
  (\hat{h}_ 1, \hat{m}_ 1) = LSTM(H, m_1, W_1) \\
  \ldots \\
  (\hat{h}_ N, \hat{m}_ N) = LSTM(H, m_N, W_N)
\end{aligned}
\end{equation}

CONT

# References
