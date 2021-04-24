import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_graph_from_file(file,sep=",",header=0,len_edges=10):
    """
    Crea un objeto que representa un grafo a partir de un archivo.csv    
    
    args:
        file: nombre del archivo
        sep: separador para leer el csv
        len_edges: longitud de las aristas
    returns:
        Grafo con conexiones basadas en archivo de datos
    """
    df = pd.read_csv(file,sep=sep,header=header)
    G = nx.Graph()
    aristas = []
    G.add_nodes_from(df["cultivo"])
    # Creacion de edges
    for c in df.columns:
        if c != "cultivo":
            node_for_connections = list(df.loc[np.where(df[c]==1),"cultivo"])
            for n in node_for_connections:
                for m in node_for_connections:
                    if ((n,m) not in aristas and (m,n) not in aristas) and n!= m:
                        aristas.append((n,m))
                        G.add_edge(n,m, length=len_edges)    
    return G


#------------------------------------------------------------------------------
# Ejecucion de codigo
#G = create_graph_from_file("papa.csv")
#G = create_graph_from_file("maiz.csv")
#G = create_graph_from_file("cafe.csv")
G = create_graph_from_file("cania_panelera.csv")

sizes = []
for n in G.nodes:
    if G.degree()[n] != 0:
        sizes.append(80*G.degree()[n])
    else:
        sizes.append(40) # tamaño por defecto para nodos aislados
pos = nx.drawing.layout.spring_layout(G, k=2/np.sqrt(len(G.nodes)))
nx.draw(G, pos,node_size=sizes, with_labels = False, edgecolors = 'darkblue', node_color = 'orange')
plt.show()
