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

**Note**: this formulation, where each layer can accept the previous input $x$ 