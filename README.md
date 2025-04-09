# higher-order-contagion
This repository contains code for learning pairwise and higher-order rates of complex contagion on higher-order networks using birth-and-death processes.

## How-to
To run the code, first clone the repository:
```bash
git clone https://github.com/markolalovic/higher-order-contagion.git
cd higher-order-contagion
```

Then set up a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Alternatively, you can install dependencies individually. For the main parts you only need:
```bash
python -m pip install numpy scipy
```

For plotting results and data handling:
```bash
python -m pip install matplotlib pandas
```

To run the Jupyter notebooks, install also:
```bash
python -m pip install jupyterlab notebook ipykernel
```

and add the virtual environment as a kernel:
```bash
ipython kernel install --user --name .venv
```

Optionally, to run notebooks and scripts (`.wls`) written in Mathematica, you'll need the [Wolfram Engine kernel](https://www.wolfram.com/engine/index.php.en). 

And, for visualizing graphs and hypergraphs, you can install [SageMath](https://www.sagemath.org/).
