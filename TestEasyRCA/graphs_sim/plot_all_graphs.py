import json
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


if __name__ == '__main__':
    list_graphs_interventions = []
    for di in [2, 3]:
        for do in [2, 3]:
            if (di == 1) or (di == 2) or (do == 2):
                for i in range(10):
                    list_graphs_interventions.append(str(di) + "_" + str(do) + "_" + str(i))

    # list_graphs_interventions = ["3_2_9"]
    print(list_graphs_interventions)
    precision_list = []
    recall_list = []
    f1_list = []
    time_list = []
    np.random.seed(1)
    for str_i in list_graphs_interventions:
        print("####################### " + str_i + " ####################")
        with open('./graphs_sim/degree_change/graphs/graph_' + str_i + '.json', 'r') as dict_file:
            dict_str = dict_file.read()
        dict_graph = json.loads(dict_str)
        file = open('./graphs_sim/degree_change/interventions/intervention_' + str_i + '.csv', "r")
        intervention = file.read()
        file.close()

        graph = nx.DiGraph()
        graph.add_nodes_from(dict_graph.keys())
        for k, v in dict_graph.items():
            graph.add_edges_from(([(k, t) for t in v]))

