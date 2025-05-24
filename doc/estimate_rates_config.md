# Experimental setup for rate estimation and network classification

## Goal
- To estimate pairwise and higher-order infection rates from SIS dynamics on various higher-order networks 
- And to distinguish network structures based on the shapes of these estimated rate curves

## General simulation parameters
- Number of nodes: `N = 1000`
- Initial number of infected: `I0 = 50` (5% of N)
- Max simulation time: `time_max = 10.0`
- Number of Gillespie runs: `nruns = 100` (`= nsims`)
- Recovery rate `mu = 1.0`
- Target steady state: `0.75 N` within `time_max`

```python
N = 1000
I0 = 50

nsims = 100  # <- increased
time_max = 10.0

y = int(0.75 * N)
```

## Higher-order network classes
Table shows the progression from homogeneous to highly heterogeneous structures:

| Hypernetwork  | Avg. PW Degree (Var)    | Avg. HO Degree  (Var)        | $\boldsymbol{\beta_1}$ | $\boldsymbol{\beta_2}$    | PW CV $= \sigma / \mu$ | HO CV $= \sigma / \mu$ |
|---------------|-------------------------|------------------------------|------------------------|---------------------------|---------------------|---------------------------|
| Complete      | $N-1 = 999$             | $\binom{N - 1}{2} = 498,501$ | $2.4 / N = 0.0024$     | $4.4 / N^2 = 4.4 10^{-6}$ | 0                   | 0                         |
| Regular       | 9                       | 3                            | $0.18$                 | $1.33$                    | 0                   | 0                   |
| E-R           | $16.21 \, (21.66)$      | $3.10 \, (2.90)$             | $0.1$                  | $1.33$                    | 0.29                |  0.55               |
| Scale-Free    | $9.46 \, (157.25)$      | $4.87 \, (48.58)$            | $0.21$                 | $0.82$                    | 1.32                |  1.43               |

<!-- <d1> = x ; <d2> = y    <d1^2> = z ; <d2^2> = g -->
(Added a coefficient of variation (CV) column as a single measure of heterogeneity.)

Infection parameters $I_0, \beta_1, \beta_2$ and network parameters such as $d_1, d_2, m$ and $\gamma$ are chosen for each network class (or network instance in case of Scale-free) to achieve an approximate stationary state of $0.75 N$ within the simulation time $0, t_{max} = 10$.

### Complete
```python
name = "Complete"
test_name = "complete"

I0 = 50

beta1 * N = 2.4, beta2 * N^2 = 4.4
```


## Regular
```python
name = "Regular-HG"
test_name = "regular"
d1, d2 = 9, 3
n = 1000
    # Instance
	Regular-HG on 1000 nodes with 5500 edges.
    number of 2-node edges: 4500
    number of 3-node edges: 1000

lambda1 = 1.6 # <- increased
lambda2 = 4

beta1 = 0.1778, beta2 = 1.3333
```

## E-R
```python
name = "Erdos-Renyi-SC"
test_name = "random_ER"
d1, d2 = (16, 3)
    
    # Instance:
    # file_path = "../results/random_ER.pkl"
	Erdos-Renyi-SC on 1000 nodes with 9052 edges.

	Target d1: 16.00, Realized d1: 16.05
	Target d2: 3.00, Realized d2: 3.08

	Target p1:  0.01601602, Realized p1: 0.01606607
	Target p2:  0.00000602, Realized p2: 0.00000618

	Initial p_G used for G(N, p_G): 0.01008840

	Realized number of pw edges:  8025/499500
	Realized number of ho edges:  1027/166167000

	Is valid SC: True

    Some basic stats:

    Mean (SD) PW: 16.21, (4.65)
    Mean (SD) HO: 3.10, (1.70)

    CV (SD / Mean) PW: 0.29
    CV (SD / Mean) HO: 0.55
    Mean (Var) PW: 16.21 (21.66)
    Mean (Var) HO: 3.10 (2.90)

    2nd moment PW: 284.55
    2nd moment HO: 12.52

    Min, Max PW: 3, 32
    Min, Max HO: 0, 9

lambda1 = 1.6 # <- increased
lambda2 = 4

beta1 = 0.1000, beta2 = 1.3333
```


## Scale-Free
```python
name = "Scale-Free-SC"
test_name = "scale_free"
m_sc = 2
gamma_sc = 2.5
max_retries_for_stub_set = N // 100

    # Instance:
    # file_path = "../results/scale_free.pkl"
    SF-SC with 1000 nodes.
    number of 1-simplices (edges): 4732
    number of 2-simplices (triangles): 1624
        Scale-Free-SC on 1000 nodes with 6356 edges.

    PW:  Avg: 9.46, Max: 169.00, 2nd moment: 246.74
    HO:  Avg: 4.87, Max: 102.00, 2nd moment: 72.33

    # Some basic stats:
    Mean (SD) PW: 9.46, (12.54)
    Mean (SD) HO: 4.87, (6.97)

    CV (SD / Mean) PW: 1.32
    CV (SD / Mean) HO: 1.43
    Mean (Var) PW: 9.46 (157.17)
    Mean (Var) HO: 4.87 (48.60)

    2nd moment PW: 246.74
    2nd moment HO: 72.33

    Min, Max PW: 3, 169
    Min, Max HO: 2, 102


lambda2 = 4.2

beta1 = 0.2110, beta2 = 0.8160
beta1 = 0.2118, beta2 = 0.8052
beta1 = 0.2479, beta2 = 0.8929
```

## Notes: 
- 100 Gillespie runs provide $10^4$ events expected, sufficient amount data for robust rate estimation even for high-variance cases such as Scale-Free
- It might be clearer to always report $\beta_1$, $\beta_2$ rates
- Could plot diagnostics for each structure:
  * Distribution of $S_{k}^{(1)}, S_{k}^{(2)}$ for each $k$
  * Correlation between $S_{k}^{(1)}, S_{k}^{(2)}$
- Edge overlap, could calculate fraction of 3-cliques that are 2-simplices
- For each structure compute variation coeff $Var[a_K \mid |K| = k] / E[a_K \mid |K|=k]^2$