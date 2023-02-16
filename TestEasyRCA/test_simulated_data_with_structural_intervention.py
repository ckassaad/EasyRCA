import json
import time
import sys

from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from easyrca import EasyRCA
from baselines.WhyMDC import whymdc
from baselines.cloudranger import cloud_ranger, cloud_ranger0
from baselines.microcause import micro_cause, micro_cause0

from TestEasyRCA.generate_data import GenerateData, generate_data_with_structural_intervention
from TestEasyRCA.evaluation_measures import precision, recall, f1


def draw_graph(g, node_size=300):
    pos = nx.spring_layout(g, k=0.25, iterations=20)
    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=node_size)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        method = sys.argv[1]
        nb_anomalous_data = int(sys.argv[2])
        gamma_max = int(sys.argv[3])
    else:
        method = "EasyRCA"  # EasyRCA or EasyRCA* or MicroCause or CloudRanger or WhyMDC
        nb_anomalous_data = 1000
        gamma_max = 3

    nb_data = 12 * nb_anomalous_data
    # anomaly_start = int(nb_data/2)
    anomaly_start = nb_anomalous_data * 10

    list_graphs_interventions = []
    for di in [2, 3]:
        for do in [2, 3]:
            if (di == 1) or (di == 2) or (do == 2):
                for i in range(10):
                    list_graphs_interventions.append(str(di) + "_" + str(do) + "_" + str(i))

    print(list_graphs_interventions)
    precision_list = []
    recall_list = []
    f1_list = []
    time_list = []
    np.random.seed(1)
    for str_i in list_graphs_interventions:
        print("####################### " + str_i + " ####################")
        with open(str(parent) + '/graphs_sim/degree_change/graphs/graph_' + str_i + '.json', 'r') as dict_file:
            dict_str = dict_file.read()
        dict_graph = json.loads(dict_str)
        file = open(str(parent) + '/graphs_sim/degree_change/interventions/intervention_' + str_i + '.csv', "r")
        intervention = file.read()
        file.close()

        graph = nx.DiGraph()
        graph.add_nodes_from(dict_graph.keys())
        for k, v in dict_graph.items():
            graph.add_edges_from(([(k, t) for t in v]))

        anomalies_start_time = dict()
        for node in graph.nodes:
            # if node != "a":
            short_path = nx.shortest_path(graph, source="a", target=node, weight=None, method='dijkstra')
            anomalies_start_time[node] = anomaly_start + len(short_path) - 1

        if nb_anomalous_data > 100:
            eps = 100
        else:
            eps = 50
        generator = GenerateData(graph)
        data = generator.generate_data(n=nb_data, intervention=True, rootStartIntervention=anomaly_start,
                                       rootEndIntervention=anomaly_start + nb_anomalous_data + eps,
                                       secondInterventionNode=intervention,
                                       seccondStartIntervention=anomalies_start_time[intervention],
                                       secondEndIntervention=anomalies_start_time[
                                                                 intervention] + nb_anomalous_data + eps)
        # data = generate_data_with_structural_intervention(graph, nb_data, secondInterventionNode=intervention,
        #                                                 rootStartIntervention=anomaly_start,
        #                                                 rootEndIntervention=anomaly_start + nb_anomalous_data + eps)

        for node in graph.nodes:
            graph.add_edge(node, node)

        ####################
        # draw_graph(graph)
        # print(graph.edges)
        # print(intervention)
        # print(anomalies_start_time)
        # print(nb_data)
        # print(nb_anomalous_data)
        # print(data)
        ####################
        start = time.time()
        if method == "EasyRCA":
            erca = EasyRCA(graph, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                           anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=0.01)
            # erca.run_without_data()
            # print(erca.root_causes)
            # print(erca.get_recommendations)

            erca.run(data)
            print(erca.root_causes)
            # print(erca.get_recommendations)
            root_causes = []
            for subgraph_id in erca.dict_linked_anomalous_graph.keys():
                root_causes = root_causes + erca.root_causes[subgraph_id]["structure_defying"]
                # root_causes = root_causes + erca.root_causes[subgraph_id]["param_defying"]
            print(root_causes, intervention)
            # draw_graph(graph)
        elif method == "EasyRCA*":
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests import ParCorr
            from tigramite import data_processing as pp
            from easyrca import remove_self_loops

            data_normal = data.loc[:anomaly_start-10]
            dataframe = pp.DataFrame(data_normal.values,
                                     datatime=np.arange(len(data_normal)),
                                     var_names=data_normal.columns)
            parcorr = ParCorr(significance='analytic')
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

            output = pcmci.run_pcmciplus(tau_min=0, tau_max=gamma_max, pc_alpha=0.01)
            g = nx.DiGraph()
            g.add_nodes_from(data.columns)
            g = nx.DiGraph()
            g.add_nodes_from(data.columns)
            for name_y in pcmci.all_parents.keys():
                for name_x, t_xy in pcmci.all_parents[name_y]:
                    if (data.columns[name_x], data.columns[name_y]) not in g.edges:
                        if (data.columns[name_y], data.columns[name_x]) not in g.edges:
                            g.add_edges_from([(data.columns[name_x], data.columns[name_y])])
            dag = remove_self_loops(g)
            if nx.is_directed_acyclic_graph(dag):
                erca = EasyRCA(g, list(g.nodes), anomalies_start_time=anomalies_start_time,
                               anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=0.01)

                erca.run(data)
                print(erca.root_causes)
                root_causes = []
                for subgraph_id in erca.dict_linked_anomalous_graph.keys():
                    root_causes = root_causes + erca.root_causes[subgraph_id]["structure_defying"]
                    # root_causes = root_causes + erca.root_causes[subgraph_id]["param_defying"]
                print(root_causes, intervention)
            else:
                print("Cyclic!!!!!")
                root_causes = []

        elif method == "WhyMDC":
            import pandas as pd
            window_graph = nx.DiGraph()
            window_data = pd.DataFrame()
            window_anomalies_start_time = dict()
            targets = []

            temporal_node_to_node = dict()
            node_to_temporal_node = dict()
            for node in graph.nodes:
                temporal_node_to_node[node + "t"] = node
                temporal_node_to_node[node + "t-1"] = node
                # temporal_node_to_node[node + "t-2"] = node
                # temporal_node_to_node[node + "t-3"] = node
                # node_to_temporal_node[node] = [node + "t", node + "t-1", node + "t-2", node + "t-3"]
                node_to_temporal_node[node] = [node + "t", node + "t-1"]
                window_graph.add_nodes_from(node_to_temporal_node[node])
                targets.append(node_to_temporal_node[node][0])

            # prepare window graph
            for edge in graph.edges:
                cause = edge[0]
                effect = edge[1]
                window_graph.add_edge(node_to_temporal_node[cause][1], node_to_temporal_node[effect][0])
                # window_graph.add_edge(node_to_temporal_node[cause][2], node_to_temporal_node[effect][1])
                # window_graph.add_edge(node_to_temporal_node[cause][3], node_to_temporal_node[effect][2])

            # prepare window data
            for gamma in range(0, 2):
                shifteddata = data.shift(periods=-gamma)
                new_columns = []
                for node in data.columns:
                    if gamma == 0:
                        new_columns.append(node_to_temporal_node[node][0])
                    elif gamma == 1:
                        new_columns.append(node_to_temporal_node[node][1])
                    elif gamma == 2:
                        new_columns.append(node_to_temporal_node[node][2])
                    else:
                        new_columns.append(node_to_temporal_node[node][3])
                shifteddata.columns = new_columns
                window_data = pd.concat([window_data, shifteddata], axis=1, join="outer")
            window_data.dropna(axis=0, inplace=True)

            # prepare window anomalies time start
            for node in anomalies_start_time.keys():
                window_anomalies_start_time[node_to_temporal_node[node][0]] = anomalies_start_time[node] + 1
                window_anomalies_start_time[node_to_temporal_node[node][1]] = anomalies_start_time[node]
                # window_anomalies_start_time[node_to_temporal_node[node][2]] = anomalies_start_time[node] + 1
                # window_anomalies_start_time[node_to_temporal_node[node][3]] = anomalies_start_time[node]

            root_causes_temporal = whymdc(window_graph, window_data, list(window_graph.nodes), targets,
                                          anomalies_start_time=window_anomalies_start_time,
                                          anomaly_length=nb_anomalous_data, sig_threshold=0.01)
            root_causes = []
            for rct in root_causes_temporal:
                if temporal_node_to_node[rct] not in root_causes:
                    root_causes.append(temporal_node_to_node[rct])

            for node in graph.nodes:
                parents_of_node = list(graph.predecessors(node))
                if (len(parents_of_node) == 1) and (node in parents_of_node) and (node in root_causes):
                    print(node)
                    root_causes.remove(node)

            print(root_causes, intervention)
        elif method == "MicroCause":
            root_causes = micro_cause(data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                      anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=0.01)
            print(root_causes)
            for node in graph.nodes:
                parents_of_node = list(graph.predecessors(node))
                if (len(parents_of_node) == 1) and (node in parents_of_node) and (node in root_causes):
                    print(node)
                    root_causes.remove(node)
            # print(root_causes)
            print(root_causes, intervention)
        elif method == "MicroCause0":
            anomalous_graph = graph.copy()
            for edge in graph.edges:
                if edge[1] == intervention:
                    anomalous_graph.remove_edge(edge[0], edge[1])

            root_causes = micro_cause0(anomalous_graph, data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                      anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=0.01)
            print(root_causes)
            for node in graph.nodes:
                parents_of_node = list(graph.predecessors(node))
                if (len(parents_of_node) == 1) and (node in parents_of_node) and (node in root_causes):
                    print(node)
                    root_causes.remove(node)
            # print(root_causes)
            print(root_causes, intervention)
        elif method == "CloudRanger":
            root_causes = cloud_ranger(data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                       anomaly_length=nb_anomalous_data, sig_threshold=0.01)
            print(root_causes)
            for node in graph.nodes:
                parents_of_node = list(graph.predecessors(node))
                if (len(parents_of_node) == 1) and (node in parents_of_node) and (node in root_causes):
                    print(node)
                    root_causes.remove(node)
            # print(root_causes)
            print(root_causes, intervention)
        elif method == "CloudRanger0":
            anomalous_graph = graph.copy()
            for edge in graph.edges:
                if edge[1] == intervention:
                    anomalous_graph.remove_edge(edge[0], edge[1])

            root_causes = cloud_ranger0(anomalous_graph, data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                       anomaly_length=nb_anomalous_data, sig_threshold=0.01)
            print(root_causes)
            for node in graph.nodes:
                parents_of_node = list(graph.predecessors(node))
                if (len(parents_of_node) == 1) and (node in parents_of_node) and (node in root_causes):
                    print(node)
                    root_causes.remove(node)
            # print(root_causes)
            print(root_causes, intervention)
        else:
            root_causes = []
            print("Error: method does not exist")
            exit(0)
        end = time.time()

        p = precision(root_causes, [intervention])
        r = recall(root_causes, [intervention])
        f = f1(root_causes, [intervention])
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
        time_list.append(end - start)
    print("#############################################")
    print("Precision = " + str(np.mean(precision_list)) + " +- " + str(np.var(precision_list)))
    print("Recall = " + str(np.mean(recall_list)) + " +- " + str(np.var(recall_list)))
    print("F1 = " + str(np.mean(f1_list)) + " +- " + str(np.var(f1_list)))
    print("Computational time = " + str(np.mean(time_list)) + " +- " + str(np.var(time_list)))
