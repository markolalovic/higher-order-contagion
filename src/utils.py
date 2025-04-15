from sage.graphs.graph import Graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

def draw_hypergraph(g, pos=None, lab=None, fname=None, y_shift=-0.008):
    r""" Plots hypergraph g:
      * Draws 2-node edges
      * Fills 3-node edges (triangle cells) as shaded gray
      * Labels the nodes
    
    TODO: 
      * Denote infected nodes (as filled dots?) and add labels (rates as numbers?)
      * Using: g.get_node_attributes(attr) for attr = "state" or "rate"
    """
    sage_g = Graph()
    sage_g.add_vertices(range(g.N)) # add nodes first, since they can be isolated
    sage_g.add_edges(g.get_edges(order=1)) # add pairwise edges

    # node positions
    if not pos:
        # using circular layout by default
        sage_g.graphplot(save_pos=True, layout="circular")
        pos = sage_g.get_pos() # node positions dict from the chosen layout
    
    # node labels
    if not lab:
        # using labels 0, 1, ..., N by default
        lab = {node: str(node) for node in g.nodes}

    # create a figure
    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=(6, 6), frameon=False)

    # draw 3-node edges (filled in triangle cells) first
    for edge in g.get_edges(order=2):
        # get coordinates of triangle vertices
        x, y = zip(*[pos[v] for v in edge])
        
        # draw triangles as gray cells
        polygon = patches.Polygon(list(zip(x, y)), 
                                  closed=True, 
                                  color="gray", 
                                  alpha=0.5,
                                  edgecolor=None)
        ax.add_patch(polygon)

    # draw 2-node edges on top
    for edge in g.get_edges(order=1):
        # get coordinates of edge vertices
        x, y = zip(*[pos[v] for v in edge])
        
        # draw edges as black lines
        ax.plot(x, y, "k-", lw=2) 

    # node attributes and labels
    for node, (x, y) in pos.items():
        # nodes
        ax.scatter(x, y, s=300, c="white", edgecolors="black", zorder=3)
        # and labels 
        # y_shift for -0.008 to center the labels
        ax.text(x, y + y_shift, lab[node], 
                ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal") # as square

    # remove frame borders!
    # set figure background to white
    fig.patch.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if fname:
        plt.savefig(fname, format="svg", bbox_inches="tight", 
                    pad_inches=0, transparent=False, facecolor="white")
        print(f"Saved as {fname}")
        plt.show()
    else:
        plt.show()

def save_hypergraph(g, file_path):
    r"""
    Saves a hypergraph object to a file using pickle, 
    e.g. to: `../data/random_graph.pkl`
    """
    with open(file_path, "wb") as f:
        pickle.dump(g, f)

def load_hypergraph(file_path):
    r"""
    Loads a hypergraph object to a file using pickle,
    e.g. to: `../data/random_graph.pkl`
    """
    with open(file_path, "rb") as f:
        g = pickle.load(f)
    return g
