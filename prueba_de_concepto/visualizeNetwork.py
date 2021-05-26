import math
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
import os

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

    #se le pasan las opciones de maximización a la toma de entradas
    selections = take_user_input(list(data.columns)[1:])

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

            #si la selección es distinta de -1 (es decir si desea maximizar para
            #alguna o algunas características en particular, maximise esa característica)
            if (-1 not in selections):
                for select in selections:
                    edge[select] *= len(edge)
            
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
            H.remove_edge(u,v)                      #quito una arista para disminuir la cantidad de vecinos 
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

    #obtención de la matiz de adyacencia 
    adj_mat, nodes_set = get_adjacency_matrix(H)
    
    #primera iteración del algoritmo
    n_iters = len(nodes_set)
    k = 0
    result_matrix = floyd_warshall_tough_work(adj_mat, k, n_iters)

    for i in range(1, n_iters+1):
        k += 1
        result_matrix = floyd_warshall_tough_work(result_matrix, k, n_iters)

    return result_matrix, nodes_set
        
    
def floyd_warshall_tough_work(prev_mat, k, sz):

    ans = prev_mat

    if (k != 0):

        for i in range(sz):
            for j in range(sz):

                if i == j:
                    continue

                #recordar que en python se indexa desde 0 y no desde 1
                #por eso la fórmula cambia ligeramente para la escritura en python
                ans[i][j] = min([ prev_mat[i][j], prev_mat[i][k-1] + prev_mat[k-1][j] ])
    else:

        for i in range(sz):
            for j in range(sz):

                if (prev_mat[i][j] == 0) and (i != j):
                    ans[i][j] = math.inf
                else: 
                    ans[i][j] = prev_mat[i][j]

    return ans
    
def get_center(G):

    
    #obtiene las componentes del grafo para no agregar al algoritmo los nodos que no 
    #son de la componente más importante(con más nodos)
    components = list(nx.connected_component_subgraphs(G))

    max_indx = 0 #indice de la componente de components que tiene más vértices
    count = 0    #contador para asignar el max_indx
    n_G = 0      #guarda el número máximo de la cantidad de nodos de las componentes

    #selecciona la componente con el mayor número de vértices para aplicarle el 
    #floyd_warshall (esto porque en varios casos pueden haber componentes aisladas)
    for component in components:

        if len(component) > n_G:
            n_G = len(component)
            max_indx = count
        
        count += 1

    distances_mat, nodes = floyd_warshall(components[max_indx])
    n_nodes = len(nodes)

    vertex_exentricity = {}

    for vertex, col in zip( nodes, list(range(n_nodes)) ):

        #setea por defecto la excentricidad en infinito para que 
        #siempre que encuentre cualquier otro valor lo cambie 
        vertex_exentricity[vertex] = -math.inf

        #revisa las distancias entre un nodo y los demás nodos
        #en búsca de la mayor de las distancias
        for row in range(n_nodes):
            value = distances_mat[row][col]

            if  (value > vertex_exentricity[vertex]) and (value != math.inf):
                vertex_exentricity[vertex] = value

    min_excentr = min(vertex_exentricity.values())
    center = [v for v in vertex_exentricity if vertex_exentricity[v] == min_excentr]

    H = G.subgraph(center)
    return H


def take_user_input(options):
    
    msg1 = "¿Para cuales de las siguientes opciones desea optimizar el análisis de afectaciones?\n [seleccione un número o varios y sepárelos por comas]:"

    print(msg1 + "\n")

    #contador para crear la lista de opciones que se van a mostrar
    count = 0
    #lista que guarda las opciones con su forma de seleccion tal cual se debe mostrar en la interfaz de consola
    real_options = []

    for opts in options:
        real_options.append(opts + " (" + str(count) + ")")     #le da formato a cada una de las opciones
        count += 1

    msg2 = ", ".join(real_options)                              #junta todas las opciones para ser mostradas
    print(msg2 + "\n")

    opti_option = input("indique los números correspondientes a su selección [-1 si no desea maximizar una característica particular]: ")

    #borrado de posibles espacios al inicio o al final
    #borrado de salto de linea del string
    opti_option.strip(" ")
    opti_option.strip("\n")

    #separa todas las opciones escritas como número
    options = opti_option.split(",")

    #almacena las opciones convertidas a entero
    options_converted = []

    for opt in options:
        if opt != '':
            options_converted.append(int(opt))

    print("opciones seleccionadas: ", options_converted)
    return options_converted

def select_file_usr():

    files = os.listdir(os.getcwd())
    csv = [file_ for file_ in files if '.csv' in file_]
    csv.sort()

    print("***"*30)
    print("Por favor seleccione uno de los cultivos que se muestran en lista: ")

    #contador para crear la lista de opciones que se van a mostrar
    count = 0
    #lista que guarda las opciones con su forma de seleccion tal cual se debe mostrar en la interfaz de consola
    real_options = []

    for filename in csv:
        real_options.append(filename + " (" + str(count) + ")")     #le da formato a cada una de las opciones
        count += 1

    msg2 = ", ".join(real_options)                              #junta todas las opciones para ser mostradas

    print(msg2 + "\n")
    print("***"*30)
    print()

    user_select = int(input("Número del archivo escogido: "))
    print(csv[user_select])

    return csv[user_select]


def create_visualization(H, nodeColor, edgeColor, fontSize, withLabels=False):

    #obtención de los pesos del grafo para añadirlos como 
    #etiquetas
    edge_weights = {(u,v):round(d['weight'],2) for u,v,d in H.edges(data=True)}
    #print(edge_weights)

    #posicionamiento de los nodos de los vértices según la los métodos 
    #otorgados por 'spring_layout' que ve el sistema de nodos como un conjunto de vértices
    #separados por resortes
    #pos = nx.drawing.layout.spring_layout(H, k=5/math.sqrt(len(H.nodes)))
    #pos = nx.drawing.layout.circular_layout(H, scale=10)
    pos = nx.drawing.layout.fruchterman_reingold_layout(H, k=100, iterations=25000, scale=500)

    #asignación de tamaño de los vértices en la visualización
    sizes = []
    for n in H.nodes:
        if H.degree()[n] != 0:
            sizes.append(80*H.degree()[n])
        else:
            sizes.append(40) # tamaño por defecto para nodos aislados

    #dibujado de las etiquetas en el grafo a visualizar
    nx.draw_networkx_edge_labels(G=H,pos=pos, edge_labels=edge_weights, font_color='b', font_size = fontSize)

    #dibujo del grafo
    nx.draw(H, with_labels = withLabels, font_weight='bold', pos=pos, node_size = sizes, edgecolors = edgeColor,
            node_color = nodeColor)
    plt.show()

## Algoritmo de Kruskal, expansion maxima

def Kruskal(G):

    #obtiene las componentes del grafo para no agregar al algoritmo los nodos que no 
    #son de la componente más importante(con más nodos)
    components = list(nx.connected_component_subgraphs(graph))

    max_indx = 0 #indice de la componente de components que tiene más vértices
    count = 0    #contador para asignar el max_indx
    n_G = 0      #guarda el número máximo de la cantidad de nodos de las componentes

    #selecciona la componente con el mayor número de vértices para aplicarle el 
    #Kruskal (esto porque en varios casos pueden haber componentes aisladas)
    for component in components:

        if len(component) > n_G:
            n_G = len(component)
            max_indx = count
        
        count += 1

    G = components[max_indx]

    # Lista de (edge:peso)
    edge_weight = []
    for e in list(G.edges):
        edge_weight.append((e,G.get_edge_data(e[0],e[1])["weight"]))

    # Orgnizacion de pesos de mayor a menor
    edge_weight= sorted(edge_weight, key=lambda x: x[1], reverse=True)

    # Inicializacion de arbol de expansion con vertices
    T = nx.Graph()
    T.add_nodes_from(list(G.nodes)) # V(T) = V(G), E(T) = empty set
    T_copy = deepcopy(T)

    while not nx.is_connected(T) and len(T.edges) != len(G.nodes)-1:

        for i in range(len(edge_weight)):
            e = edge_weight[i]
            T_copy.add_edge(e[0][0], e[0][1], weight=e[1])

            try:

                # Verificacion si T_copy tiene un ciclo
                nx.find_cycle(T_copy)
                # Si la linea anterior funciona significa que hay ciclo
                # Se deja T_copy como estaba inicialmente
                T_copy = deepcopy(T)
            except:

                # Si no tiene ciclos agregamos a T el mismo lado
                T.add_edge(e[0][0], e[0][1], weight=e[1])
                # Eliminamos el lado de la lista de posibles lados
                edge_weight.pop(i)
                # terminamos la verificacion de aristas para este arbol
                break            
    return T

selected_file = select_file_usr()

#se obtiene el grafo generado a partir del archivo csv
graph = readGrapFile(selected_file, ',' , 0)
#graph = readGrapFile('./prueba2.csv', ',' , 0)


print("Grafo generado: \n")
create_visualization(graph, 'red', 'orange', 7, True)

print("Grafo centro:\n")
center = get_center(graph)
create_visualization(center, 'red', 'orange', 7, True)

# Generacion de arbol de expansion
T = Kruskal(graph)
print("Numero de componentes de T: ",  nx.number_connected_components(T))
print("T es conexo?: ", nx.is_connected(T))


try:
    ciclo = list(nx.find_cycle(T))
    print("ERROR: T tiene un ciclo")
    print(ciclo)
except:
    print("T no tiene ciclos")


#for e in list(T.edges):
#    print((e,T.get_edge_data(e[0],e[1])["weight"]))

create_visualization(T, 'red', 'orange', 7, True)

#for e, datadict in graph.edges.items():
#    print(e, datadict)


#documentation used:
#1) https://networkx.org/documentation/networkx-2.5/tutorial.html#drawing-graphs
#2) https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html

#documentation about useful networkx functions:
#1) https://networkx.org/documentation/networkx-2.5/reference/functions.html

#theorical documentation: 
#1) https://en.wikipedia.org/wiki/Norm_(mathematics)
#2) https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html

#add labels for network drawing
#1) https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
#2) https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edge_labels.html
