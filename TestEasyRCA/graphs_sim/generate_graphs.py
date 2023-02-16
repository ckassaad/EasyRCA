"""
Coded by Charles Assaad
"""
import networkx as nx
import string
import random
import json


def max_degree_in(dag):
    m_d_in = 0
    for d in dag.in_degree:
        m_d_in = max(m_d_in, d[1])
    return m_d_in


def max_degree_out(dag):
    m_d_out = 0
    for d in dag.out_degree:
        m_d_out = max(m_d_out, d[1])
    return m_d_out


# Generate a list DAGs (without self loops) given the number of nodes, maximal in degree and maximal out degree
def generate_graphs(nb_nodes, degree_in, degree_out, n_g):
    """
    :param nb_nodes: number of nodes in generated graph
    :param degree_in: maximal in degree in generated graph
    :param degree_out: maximal in degree in generated graph
    :param n_g: number of graphs
    :return: list of n_g graphs
    """
    alphabet_string = string.ascii_lowercase
    alphabet_string = list(alphabet_string)
    graphs = []

    while len(graphs) < n_g:
        g = nx.gnp_random_graph(nb_nodes, 0.5, directed=True)
        # g = nx.fast_gnp_random_graph(nb_nodes, 0.5, directed=True)
        dag = nx.DiGraph([(u, v) for (u, v) in g.edges() if u < v])
        mapping = dict()
        for node in dag.nodes:
            mapping[node] = alphabet_string[node]
        dag = nx.relabel_nodes(dag, mapping)

        roots = [n for n, d in dag.in_degree() if d == 0]
        if (nx.is_directed_acyclic_graph(dag)) and (max_degree_in(dag) == degree_in) and \
                (max_degree_out(dag) == degree_out) and (len(dag.nodes) == nb_nodes) and (len(roots) == 1):
            graphs.append(dag)
    return graphs


def choose_intervention_node(graph):
    roots = [n for n,d in graph.in_degree() if d==0]
    not_roots = list(set(graph.nodes) - set(roots))
    intervention_node = random.choice(not_roots)
    return intervention_node


if __name__ == "__main__":
    nb_nodes_change = False  # change nb of node and fix degree in and degree out to (2,2)
    degree_change = not nb_nodes_change  # change degree in and degree out and fix nb of node to 6
    read_or_write = "write"

    import os
    print(os.getcwd())

    if nb_nodes_change:
        degin_list = [2]
        degout_list = [2]
        nb_nodes_list = [6, 7, 8]
    elif degree_change:
        # degin_list = [2, 3, 2]
        # degout_list = [2, 2, 3]
        degin_list = [1, 1]
        degout_list = [2, 3]
        nb_nodes_list = [6]
    else:
        degin_list = 0
        degout_list = 0
        nb_nodes_list = 0
        exit(0)

    test_compatibility = len(degin_list) == len(degout_list) == len(nb_nodes_list)
    nb_iter = max(len(degin_list), len(nb_nodes_list))
    for j in range(nb_iter):
        random.seed(j)
        print("############# Iter "+str(j)+" ################")
        if nb_nodes_change:
            nb_nodes = nb_nodes_list[j]
            degin = degin_list[0]
            degout = degout_list[0]
        else:
            nb_nodes = nb_nodes_list[0]
            degin = degin_list[j]
            degout = degout_list[j]

        if read_or_write == "write":
            gs = generate_graphs(nb_nodes, degin, degout, 10)
            for i in range(10):
                print("Graph " +str(i))

                intervention = choose_intervention_node(gs[i])
                print(intervention)

                if nb_nodes_change:
                    graph_name = "graph_" + str(nb_nodes) + "_" + str(i) + ".json"
                    # data_name = "data_" + str(nb_nodes) + "_" + str(i) + ".csv"
                    intervention_name = "intervention_" + str(nb_nodes) + "_" + str(i) + ".csv"
                    path = "./graphs_sim/nb_nodes_change/"
                else:
                    graph_name = "graph_"+str(degin)+"_"+str(degout) + "_"+str(i)+".json"
                    # data_name = "data_"+str(degin)+"_"+str(degout) + "_"+str(i)+".csv"
                    intervention_name = "intervention_"+str(degin)+"_"+str(degout) + "_"+str(i)+".csv"
                    path = "./graphs_sim/degree_change/"

                dict_g = nx.to_dict_of_dicts(gs[i])
                with open(path + "graphs/" + graph_name, 'w') as dict_file:
                    dict_file.write(json.dumps(dict_g))
                # nx.write_gpickle(gs[i], path + "graphs/" + graph_name)
                file = open(path + "interventions/" + intervention_name, "w")
                file.write(str(intervention))
                file.close()

        elif read_or_write == "read":
            for i in range(10):
                if nb_nodes_change:
                    graph_name = "graph_" + str(nb_nodes) + "_" + str(i) + ".json"
                    # data_name = "data_" + str(nb_nodes) + "_" + str(i) + ".csv"
                    intervention_name = "intervention_" + str(nb_nodes) + "_" + str(i) + ".csv"
                    path = "nb_nodes_change/"
                else:
                    graph_name = "graph_"+str(degin)+"_"+str(degout) + "_"+str(i)+".json"
                    # data_name = "data_"+str(degin)+"_"+str(degout) + "_"+str(i)+".csv"
                    intervention_name = "intervention_"+str(degin)+"_"+str(degout) + "_"+str(i)+".csv"
                    path = "degree_change/"

                with open(path + "graphs/" + graph_name, 'r') as dict_file:
                    dict_str = dict_file.read()
                dict_g = json.loads(dict_str)
                g = nx.DiGraph()
                g.add_nodes_from(dict_g.keys())
                for k, v in dict_g.items():
                    g.add_edges_from(([(k, t) for t in v]))
                # g = nx.read_gpickle(path + "graphs/" + graph_name)
                file = open(path + "interventions/" + intervention_name, "r")
                intervention = file.read()
                file.close()

                print(g.edges)
                print(intervention)
