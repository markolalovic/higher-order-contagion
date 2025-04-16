# higher-order-contagion
Code for learning pairwise and higher-order rates of complex contagion on higher-order networks using birth-and-death processes. We model higher-order networks using hypergraphs.

## Examples

### Simple Hypergraph
Example of a hypergraph on 4 nodes with four 2-node edges and one 3-node edge:
```python
from Hypergraphs import EmptyHypergraph
from utils import draw_hypergraph

# create a hypergraph on 4 nodes with 5 edges
g = EmptyHypergraph(4)
g.name = "Example Hypergraph"
edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 2, 3)]
g.set_edges(edges)

# set node positions and draw it
positions = {0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}
file_name = "../figures/hypergraphs/example_hypergraph.svg"
draw_hypergraph(g, pos=positions, fname=file_name)
```
<!-- ![Example Hypergraph](figures/hypergraphs/example_hypergraph.svg) -->
<img src="figures/hypergraphs/example_hypergraph.svg" alt="Example Hypergraph" width="400" height="400">

### Cycle hypergraph
Example of a cycle / ring hypergraph on 10 nodes:
```python
N = 10
g = EmptyHypergraph(N)
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
file_name = "../figures/hypergraphs/cycle_hypergraph.svg"
draw_hypergraph(g, fname=file_name)
```
![Cycle Hypergraph](figures/hypergraphs/cycle_hypergraph.svg)


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