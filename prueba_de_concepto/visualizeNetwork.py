import math
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readGrapFile(filename, sep, header):
    #input: nombre del archivo csv, sep (string que designa el separador de valores en el csv)
    #header (indice en el que se espera esté el cabecero para tomarlo como header en el dataframe)

    data = pd.read_csv(filename, sep=sep, header=header)
    
    print("los datos son: ")
    print(data)
    print()

    print("calculado relaciones y seteando pesos...")
    print()
    
    graph = nx.Graph()

    #obtiene a partir del dataframe los nodos que tienen una arista entre si y su peso 
    #correspondiente, así como las componentes aisladas en caso de haberlas
    edges_and_weights, isolated_components = get_NodesAndEdges(data)

    #contruye el grafo a partir de la lista de aristas obtenida
    graph.add_weighted_edges_from(edges_and_weights)

    #en caso de haber componentes aisladas se añaden al grafo
    if(len(isolated_components) != 0):
        graph.add_nodes_from(isolated_components)


    return graph

def get_NodesAndEdges(data):
    
    nodes_weights = []      #lista para guardar las parejas de nodos y sus pesos
    isolated_nodes = []     #lista para guardar los nodos que no tienen aristas 

    for i in range(len(data)):

        edge_1_atributes = np.array(list(data.iloc[i, 1:]))

        #guarda el identificador del nodo que se va a empezar a 
        #relacionar con otros nodos
        node1 = data.iloc[i, 0]

        #verifica que si el nodo es uno de los que no tiene problemas 
        #se añada en una lista distinta para nodos aislados, o componentes aisladas
        if (euclidian_norm(edge_1_atributes) == 0):
            isolated_nodes.append(node1)
            continue
            

        for j in range(len(data)):

            if j == i: 
                continue

            edge_2_atributes = np.array(list(data.iloc[j, 1:]))
            node2 = data.iloc[j, 0]

            #se setea una linea que une a los dos nodos 
            #basándose en los problemas que tienen en común y si su
            #peso (el de los problemas en común) es distinto de cero

            edge = edge_1_atributes & edge_2_atributes

            #print("arista principal de la iteración:")
            #print(edge_1_atributes)
            #print()

            #print("arista secundaria de la iteración:")
            #print(edge_2_atributes)
            #print()
            #print(edge)
            #print()
            
            weight = euclidian_norm(edge)

            if weight != 0:
                nodes_weights.append((node1, node2, weight))

    return nodes_weights, isolated_nodes

def euclidian_norm(vector):

    norm = 0

    for i in vector:
        norm += i**2

    return math.sqrt(norm)


def get_adjacency_matrix(G):

    #crea una copia del grafo G
    H = G.copy()

    #ordena los nodos para crear una matriz de adyacencia basándose 
    #en ese ordenamiento
    nodes_list = list(H.nodes)
    nodes_list.sort()

    node_num = len(nodes_list)

    #inicializa la matriz de adyacencia
    adj_matrix = np.zeros((node_num, node_num))

    #da valores congruentes con el grafo a la matriz de adyacencia
    for v, j in zip(nodes_list, range(node_num)):
        neighbors = list(nx.classes.function.neighbors(H,v))

        for u in neighbors: 
            i = np.searchsorted(nodes_list, u)      #busca cual es el índice del vecino 'u' para poder 
                                                    #indicar en la matriz su adyacencia
            #setea los valores en la matriz de 
            #adyacencia (recuerde que para un grafo simple es simétrica)
            adj_matrix[j][i] = 1
            adj_matrix[i][j] = 1
            H.remove_edge(u,v)                      #quito un vértice para disminuir la cantidad de vecinos 
                                                    #y así eliminar la repetición de computaciones
            
    #print(adj_matrix.T == adj_matrix)              #para revisar si la matriz es simétrica y por lo tanto 
                                                    #si está bien hecha

    #el siguiente print es para verificar lo mismo
    #en caso de que la matriz sea muy grande como para realizar
    #inspección visual
    #print("la matriz es simétrica? ", pow(node_num,2) == sum(sum(adj_matrix.T == adj_matrix)) )
    #print()

    return adj_matrix, nodes_list

def floyd_warshall(H):

    adj_mat, nodes_set = get_adjacency_matrix(H)

    for i in range(len(nodes_set)):
        adj_mat = floyd_warshall_tough_work(adj_mat)

    return adj_mat
        
    
def floyd_warshall_tough_work(L_i):

    print(L_i)
    

#se obtiene el grafo generado a partir del archivo csv
graph = readGrapFile('./datos_prueba_grande.csv', ',' , 0)

#for e, datadict in graph.edges.items():
#    print(e, datadict)

floyd_warshall(graph)

print("Grafo generado: \n")

nx.draw(graph, with_labels = True, font_weight='bold')
plt.show()

#documentation used:
#1) https://networkx.org/documentation/networkx-2.5/tutorial.html#drawing-graphs
#2) https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html

#documentation about useful networkx functions:
#1) https://networkx.org/documentation/networkx-2.5/reference/functions.html

#theorical documentation: 
#1) https://en.wikipedia.org/wiki/Norm_(mathematics)
#2) https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html



