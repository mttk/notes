% Attention mechanisms in deep neural networks
% Martin Tutek
% \today


# Original attention mechanism and its variants

Essentially, the attention mechanism is a way to find an answer to a query. The query in this problem is a vector of real numbers, while the answer is a linear re-combination of values produced by encoding an input sequence. The attention mechanism is supposed to find the *best* linear re-combination for a given query.

- Query: $q \in \mathbb{R}^{d_q}$
- Keys: $K \in \mathbb{R}^{T \times d_k}$
- Values: $V \in \mathbb{R}^{T \times d_v}$

Where T is the time dimension (variable over different input instances), and $d_{\left\vert\cdot\right\vert}$ are the dimensions of respective embeddings. Keys and values are mapped one-to-one.

We define the _energy_ as a result of a function $f$ mapping from a key and query onto $\mathbb{R}$.

$$e_i = f(q,k_i)$$

The energy values are then normalized with a softmax function to produce a probability distribution over all keys

$$a_{i} = softmax(e_{i})$$

We then use these values as the coefficients of a linear combination over the values

$$att = \sum_i a_ih_i$$

## Usage
Initially [@bahdanau2014neural], the attention mechanism was introduced in a machine translation (sequence to sequence) problem to mitigate the problem of learning long dependencies. After a RNN encoder encoded the input sequence into a sequence of hidden states, we use the current decoder state as the query and the encoder hidden states as both the keys and the values.

Essentially, we find relevant information for the word we are currently translating in the encoded input sequence. For this to work, it is essential that the embedding spaces of the input and output vocabularies are somewhat aligned (similar words in languages should be close together).

Other examples of usage include self-attention and multi-head attention, referenced in a later chapter.

## Variants
### MLP attention 
[@bahdanau2014neural]
$f$ is *parametrized* by a feed-forward neural network (multi-layer perceptron)

$$ a_i(q, k) =  w_2^T tanh(W_1 [q;k_i])$$

- $[q;k]$ are the concatenated query and key
- $W_1$ is a linear operator
- $w_2$ is a parameter vector

### Bilinear attention
[@luong2015effective]

$f$ is parametrized by a matrix $W \in \mathbb{R}^{d_q \times d_k}$ (a bilinear operator)

$$ a(q, k) = q^T W k $$

### Dot product attention
[@luong2015effective]

$f$ is _parameter-free_, however $d_q$ **must** be equal to *d_k*.

$$ a(q, k) = q^T \cdot k $$

\pagebreak

#Extensions

## Self-attention (inter-attention)
[@cheng2016long] : The LSTMN model 
_"uses attention to induce relations between tokens"_

Idea: use attention over previous LSTM states (keys, values) with the current LSTM state as the query.

$$ a_i^{(t)} = v^T tanh(W_hh_i + W_xx^{(t)} + W_{\hat{h}} \hat{h}^{(t-1)}) $$
$$ s_i^{(t)} = softmax(a_i^{(t)})$$

Where $x^{t}$ is the current input, $\hat{h}^{(t-1)}$ the hidden state in the **previous timestep**, $v$ a parameter vector and $h_i$ the previous hidden states (i < t - 1).

**Note:** only the hidden state $h$ is used in the computation, and not the cell state $c$!

Then the state vectors ($c, h$) are updated:

\begin{equation}
\begin{bmatrix}
  \hat{h}^{t}\\
  \hat{c}^{t}
\end{bmatrix}
= 
\sum_{i=1}^{t-1} s_i^{(t)}
\begin{bmatrix}
  h_i\\
  c_i
\end{bmatrix}
\end{equation}

and then replace the un-altered state vectors in further LSTM computations.

**Attention fusion:** 
how to use self-attention in a sequence-to-sequence task where a decoder network, along with using self-attention, queries the encoder network (intra-attention).

* **Shallow attention fusion**
treats the LSTMN model as a standard LSTM and uses intra-attention on top of it.

* **Deep attention fusion**
adds an additional gating mechanism into the LSTM cell update based on intra-attention. Formula in chapter 4. of paper.


## A structured self-attentive sentence embedding
[@lin2017structured]

1. Run embedded sentence through BiLSTM
2. Self-attention over the BiLSTM hidden states
3. Use a MLP for a downstream task

**Attention**: 

(1) a MLP attention with a tanh hidden layer as in (1.2.1) 
(2) _Matrix attention_ -- the second weight of the attention MLP is a matrix instead of a vector, resulting in a matrix aggregation instead of a vector

$$
A = softmax(W_2 tanh(W_1 H^T)) 
$$

Where the softmax is applied along the second dimension of the input. The MLP has no bias! The matrix A is then multiplied with the matrix of hidden states (of dim $T\times H$), resulting in a sentence embedding matrix $M = AH$

We expect (hope) that the sentence embedding matrix will capture different aspects, however this is not necessary the case since the matrix $M$ can _"suffer from redundancy problems"_. The authors attempt to mitigate this by introducing a regularization penalty term $P$.

$$
P = \left\lVert (AA^T - I) \right\rVert_F^2
$$

----------------------------------------

**Intermezzo: Frobenius norm**: 

The Frobenius or Hilbert-Schmidt norm of the matrix A is defined as:

\begin{equation}
\begin{aligned}
\left\lVert A \right\rVert_F &= \sqrt{\sum_i^m \sum_j^n \left\vert a_{ij} \right\vert^2}\\
& = \sqrt{trace(A^TA)}\\
& = \sqrt{\sum_i^{\min\{m,n\}} \sigma_i^2(A)}\\
\end{aligned}
\end{equation}
where $\sigma_i$ is a singular value of $A$. 

-----------------------------------------

**Effect of regularization**:

The matrix A is row-normalized (each out of $r$ rows should focus on one aspect). CONT


## Attention-over-Attention
[@cui2016attention]

- Document, query $\in R^{\left\vert \mathbb{D} \right\vert \cdot 2h}, R^{\left\vert \mathbb{Q} \right\vert \cdot 2h}$
- $D, Q$ are sequence lengths of document and query respectively
- Shared embedding spaces for query and document (uses one embedding matrix for their joined vocabulary)
- two BiGRU embed query and document ($h_{doc} \in D\cdot 2h$, $h_{query} \in Q \cdot wh$)
- matrix multiplication over the shared embedding dimension $2d$ produces the `pair-wise matching score`
$$M = h_{doc}^T \cdot h_{query} \in R^{DxQ}$$
- 


# References
