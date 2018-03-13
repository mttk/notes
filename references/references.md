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

# Recurrent Highway Networks
[@zilly2016recurrent]

The paper sketches out the proof of the vanishing/exploding gradient problem and isolates its relation to the largest singular value of the recurrent weight matrix:

$$
\left\vert A \right\vert \le \left\vert R^T \right\vert \left\vert diag [f'(Ry^{[t-1]})]\right\vert \le \gamma \sigma_{max}
$$

The gradient vanishes when $\gamma \sigma_{max} < 1$ and explodes when the expression is larger than one.

$\gamma$ is the maximum value the gradient of the activation function.

Geršgorin circle theorem states:

$$
spec(A) \subset \bigcup_{i \in \left\{ 1, \ldots, n\right\}} \left\{ \lambda \in \mathbb{C} \left\lVert \lambda - a_{ii} \right\rVert_ {\mathbb{C}} \le \sum_{j=1, j \neq i}^n \left\vert a_{ij} \right\vert \right\}
$$

translated, the spectrum of eigenvelues of the square matrix $A \in \mathbb{R}^{n\times n}$ lies within the union of complex circles which are centered around the **diagonal values** of the matrix $A$, with a radius equal to the sum of the absolute values of the **non-diagonal** entries of each row.

Essentially, this means that shifting the diagonal values shifts the center of the circles, and therefore the possible location of the eigenvalues. Also, increasing the values of the remaining elements increases the radius in which the eigenvalues are contained.

TODO: Add viz of theorem w/TikZ

**Initialization of recurrent weights** as mentioned in [@le2015simple], one way to circumvent this via initialization is to initialize the recurrent matrix as an identity matrix and the remainder as small random values. However, this method does nothing to mitigate the fact that the values of the matrix will change during training, resulting in the same exploding / vanishing gradient phenomenon.

Essentially, a reformulation of a RNN in the form of a vertical highway network is used (more or less equal to LSTM, where the previous cell state is propagated input).

Takeaways: 

- most of the transform-processing is done in the first layer, and then to a lesser extent (first layer contextually transforms the input features? and the remaining layers use this contextual information).
- passing the input along in a resnet-like or highway-like fashion is useful.
- Geršgorin can help limit the range of singular values of a matrix.

# Learning long term dependencies
Dataset: pixel-by-pixel MNIST image classification, introduced in [@le2015simple]

# Learning long-term dependencies in RNNs with Auxilliary Losses
[@trinh2018learning]

- Randomly sample one or multiple anchor positions
- Use an unsupervised auxilliary loss
  - Reconstruction auxilliary loss (reconstruct a subsequence given first symbol, enhances remembering)
  - Prediction auxilliary loss (predict a future token in language-model fashion)
- Trained in two phases:
  - Pure unsupervised pretraining (minimize auxilliary loss)
  - Semi-supervised learning where $min_{\theta} L_{sup}(\theta) + L_{aux}(\theta')$

Hyperparameters: 
- How frequently to sample the reconstructon segments, and how long they are

The methods help with learning long-term dependencies, even when the backprop is truncated. Essentially, signal is needed for the network to remember things. That signal can't be achieved through very long backprop due to failure of credit assignment.

Additional ablation study \& result analysis in paper.

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

# Learning to Skim Text
[@yu2017learning]

- Uses a policy gradient method to make decisions to skip a number of tokens in the input
- Hyperparam: max jump size $K$, number of tokens to read before jumping $R$
- Processing stops if
    * Jumping softmax predicts 0
    * Jump exceeds sequence length N
    * The network processed all tokens
- The last hidden state is used for prediction in downstream tasks

REINFORCE procedure (+Baselines):

Objective: 

- Minimize cross-entropy error (classification loss)
- Maximize expected reward under the current jumping policy (R = -1 for misclassification, +1 for correct)
- Baselines regularization term: minimize difference between actual reward and predicted baseline

# Annotation Artifacts in Natural Language Inference Data

Classification model on just the _hypothesis_ of NLI achieves 67% on SNLI and 53% on MultiNLI. 

"Negation and vagueness are highly correlated with ceratin inference classes. Our findings suggest that the success of natural language inference models to date has been overestimated, and that the task remains a hard open problem." -> 

- entailed hypotheses contain gender-neutral references to people
- purpose clauses are a sign of neutral hypotheses
- negation correlates with contradiction

**Annotation artifacts:** patterns in the data that occur as a result of the framing of the annotation task influencing the language used by the crowd workers.

Discussion: "Many datasets contain annotation artifacts..." -- references to other examples of this phenomenon


# References
