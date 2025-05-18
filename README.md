# higher-order-contagion
Code for learning pairwise and higher-order rates of complex contagion on higher-order networks using birth-and-death processes.

Higher-order networks can be represented by various structures including simplicial complexes and general hypergraphs. Here, we focus on simplicial complexes where the highest dimension of interaction is two. 

Nevertheless, the framework is extendable to higher dimensions and to general hypergraphs.

## Examples
TODO: add examples of 3 main classes of simplicial complexes, more heterogeneous E-R SC, regular, and generalized scale-free, maybe the star graph too.

### Example
Example of a higher-order network on 4 nodes with 6 edges: five 2-node edges and one 3-node hyperedge:
```python
from higher_order_structures import HigherOrderStructure
from utils import draw_hypergraph

g = HigherOrderStructure(4)
g.name = "Higher Order Example"
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1), (1, 2, 3)]
g.set_edges(edges)

positions = {0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}
file_name = "../figures/higher_order_structures/ho_example.svg"
draw_hypergraph(g, pos=positions, fname=file_name)
```
<img src="figures/higher_order_structures/ho_example.svg" alt="Higher Order Example" width="400" height="400">

### Cycle hypergraph
Example of a cycle / ring hypergraph on 10 nodes:
```python
N = 10
g = HigherOrderStructure(N)
g.name = "Cycle Hypergraph"
edges = []
for i in range(N):
    # 2-node edges form a cycle
    j = (i + 1) % N
    edges.append((i, j))
    # 3-node edges form a cycle
    k = (i + 2) % N
    edges.append((i, j, k))
g.set_edges(edges)
file_name = "../figures/higher_order_structures/ho_cycle_hypergraph.svg"
draw_hypergraph(g, fname=file_name)
```
<img src="figures/higher_order_structures/ho_cycle_hypergraph.svg" alt="Cycle Hypergraph" width="450" height="450">

## Installation
```bash
# Clone repository
git clone https://github.com/markolalovic/higher-order-contagion.git
cd higher-order-contagion

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install dependencies
python3 -m pip install -r requirements.txt
```

### Minimal dependencies
```bash
# For the main / core functionality
python3 -m pip install numpy scipy

# For plotting results and data handling
python3 -m pip install matplotlib pandas

# To run the Jupyter notebooks install:
python3 -m pip install jupyterlab notebook ipykernel

# And add the virtual environment as a kernel
ipython kernel install --user --name .venv

# For testing
python3 -m pip install pytest
```

### Optional dependencies
For running Mathematica scripts (`.wls`) and notebooks install: [Wolfram Engine kernel](https://www.wolfram.com/engine/index.php.en). 

For hypergraph drawings install: [SageMath](https://www.sagemath.org/).