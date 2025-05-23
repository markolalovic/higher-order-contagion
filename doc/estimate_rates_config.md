# Setup
```python
N = 1000
I0 = 50
nsims = 100   # TODO: increase
time_max = 10.0
```

## Complete
```python
name = "Complete"
test_name = "complete"

I0 = 50

beta1 * N = 2.4, beta2 * N^2 = 4.4
y=int(0.75 * N)
```

## E-R
```python
name = "Erdos-Renyi-SC"
test_name = "random_ER"
d1, d2 = (16, 3)
	Erdos-Renyi-SC on 1000 nodes with 9052 edges.

	Target d1: 16.00, Realized d1: 16.05
	Target d2: 3.00, Realized d2: 3.08

	Target p1:  0.01601602, Realized p1: 0.01606607
	Target p2:  0.00000602, Realized p2: 0.00000618

	Initial p_G used for G(N, p_G): 0.01008840

	Realized number of pw edges:  8025/499500
	Realized number of ho edges:  1027/166167000

	Is valid SC: True

lambda1 = 1.6 # <- increase lambda
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

# Instance 1
    SF-SC with 1000 nodes.
    number of 1-simplices (edges): 4740
    number of 2-simplices (triangles): 1634
        ScaleFreeSC on 1000 nodes with 6374 edges.

    PW:  Avg: 9.48, Max: 193.00, 2nd moment: 254.10
    HO:  Avg: 4.90, Max: 117.00, 2nd moment: 75.94

# Instance 2
    SF-SC with 1000 nodes.
    number of 1-simplices (edges): 4722
    number of 2-simplices (triangles): 1656
        ScaleFreeSC on 1000 nodes with 6378 edges.

    PW:  Avg: 9.44, Max: 150.00, 2nd moment: 288.20
    HO:  Avg: 4.97, Max: 90.00, 2nd moment: 91.40

# Instance 3
    SF-SC with 1000 nodes.
    number of 1-simplices (edges): 4438
    number of 2-simplices (triangles): 1568
        Scale-Free-SC on 1000 nodes with 6006 edges.

    PW:  Avg: 8.88, Max: 302.00, 2nd moment: 298.74
    HO:  Avg: 4.70, Max: 219.00, 2nd moment: 116.08

lambda1 = 2.2 # <- increase lambda
lambda2 = 4.2


beta1 = 0.2110, beta2 = 0.8160
beta1 = 0.2118, beta2 = 0.8052
beta1 = 0.2479, beta2 = 0.8929
```

## Regular
```python
d1, d2 = 9, 3
n = 1000
# Instance
	Regular-HG on 1000 nodes with 5500 edges.
    number of 2-node edges: 4500
    number of 3-node edges: 1000

lambda1 = 1.6 # <- increase lambda
lambda2 = 4

beta1 = 0.1778, beta2 = 1.3333
```
