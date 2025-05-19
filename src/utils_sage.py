from sage.graphs.graph import Graph
from sage.plot.polygon import polygon
from sage.plot.text import text
import pickle

def draw_hypergraph(g, pos=None, fname=None,
                     hyperedge_color="steelblue", hyperedge_alpha=0.3):    
    r"""Plots hypergraph g by
      * Drawing 2-node edges
      * Filling 3-node edges (triangle cells) with hyperedge_color
    
    TODO:
      * Label the nodes e.g. {a, b, c, d}
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
    
    # initialize the plot object
    plot_obj = sage_g.plot(pos=pos, talk=False) # for cleaner display

    # add 3-node hyperedges as filled triangles
    for edge in g.get_edges(order=2):
        # get coordinates of triangle vertices
        vertices = [pos[v] for v in edge]
        
        # draw triangles as colored cells
        tri = polygon(
            vertices,
            color=hyperedge_color, 
            alpha=hyperedge_alpha,
            fill=True,
            thickness=0)
        plot_obj += tri
    
    # set plot options
    plot_obj.axes(False) # hide axes
    
    # save it
    if fname:
        plot_obj.save(fname, figsize=[6, 6])
        print(f"Saved as {fname}")
    
    return plot_obj

