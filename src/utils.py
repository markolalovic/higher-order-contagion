from sage.graphs.graph import Graph
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pickle

def plot_hypergraph(g, file_name=None, y_shift = -0.008):
    """
    Plots a hypergraph:        
        * Draws 1-simplices (edges)
        * Fills 2-simplices (triangle cells) as shaded gray
        * Labels the nodes
    
    TODOs: 
        * Denote infected nodes (as filled dots?) and add labels (rates as numbers?)
        * Using: g.get_node_attributes(attr) for attr = "state" or "rate"
    """
    sage_g = Graph()

    sage_g.add_vertices(range(g.N)) # add nodes first, since can be isolated
    sage_g.add_edges(g.get_simplices(order=1)) # pairwise edges

    # node positions using circular layout
    sage_g.graphplot(save_pos=True, layout='circular')
    pos = sage_g.get_pos()  # node positions

    # create a figure
    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=(6, 6), frameon=False)

    # draw 2-simplices (filled in triangle cells) first
    for simplex in g.get_simplices(order=2):
        # get coordinates of triangle vertices
        x, y = zip(*[pos[v] for v in simplex])
        
        # draw triangles as gray cells
        polygon = patches.Polygon(list(zip(x, y)), closed=True, color='gray', 
                                    alpha=0.5, edgecolor=None)
        ax.add_patch(polygon)

    # draw 1-simplices (edges) second
    for edge in g.get_simplices(order=1):
        # get coordinates of edge vertices
        x, y = zip(*[pos[v] for v in edge])
        
        # draw edges as black lines
        ax.plot(x, y, 'k-', lw=2) 

    # node attributes and labels
    for node, (x, y) in pos.items():
        # nodes
        ax.scatter(x, y, s=300, c='white', edgecolors='black', zorder=3)
        # and labels 
        # TODO: How to perfectly center labels in nodes? (y_shift for -0.008 for now)
        ax.text(x, y + y_shift, str(node), ha='center', va='center', fontsize=14, 
                fontweight='bold', zorder=4)
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal') # as square

    # remove frame borders!
    # set figure background to white
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    if file_name:
        # plt.savefig(file_name, format="svg", bbox_inches='tight', 
        #             pad_inches=0, transparent=True, edgecolor='white')
        plt.savefig(file_name, format="svg", bbox_inches='tight', 
                    pad_inches=0, transparent=False, facecolor='white')
        print(f"Saved as {file_name}")
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
