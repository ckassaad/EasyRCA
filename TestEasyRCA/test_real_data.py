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

import pandas as pd
import numpy as np
import networkx as nx

from easyrca import EasyRCA
from baselines.cloudranger import cloud_ranger
from baselines.microcause import micro_cause


if __name__ == '__main__':
    if len(sys.argv) > 1:
        method = sys.argv[1]
        gamma_max = int(sys.argv[2])
    else:
        method = "EasyRCA"  # EasyRCA or EasyRCAPCMCI or MicroCause or CloudRanger
        gamma_max = 3

    list_id_metric = [
        "1aa1ea46-a30d-4c83-a6ef-25ab6e1e1912",
        "2a75e210-5626-43c4-b478-725ee57fea3c",
        "2e3c7d7e-fd17-4c70-adab-db88c81a09da",
        "14fbfb3c-b94c-42dc-80a8-87e043d31b51",
        "90a75961-332c-48e3-a48e-0332d9a9b592",
        "bdbca314-bf54-4404-92d4-d250cce9e6d1",
        "ebe01255-9c2e-4d2f-8c4d-6d3a93504d4b",
        "f1168af3-4d5b-4b6d-8314-ad93e14f3633"
    ]
    map_id_to_name = {
        "1aa1ea46-a30d-4c83-a6ef-25ab6e1e1912": "capacity_last_metric_bolt",
        "2a75e210-5626-43c4-b478-725ee57fea3c": "capacity_elastic_search_bolt",
        "2e3c7d7e-fd17-4c70-adab-db88c81a09da": "pre_Message_dispatcher_bolt",
        "14fbfb3c-b94c-42dc-80a8-87e043d31b51": "check_message_bolt",
        "90a75961-332c-48e3-a48e-0332d9a9b592": "message_dispatcher_bolt",
        "bdbca314-bf54-4404-92d4-d250cce9e6d1": "metric_bolt",
        "ebe01255-9c2e-4d2f-8c4d-6d3a93504d4b": "group_status_information_bolt",
        "f1168af3-4d5b-4b6d-8314-ad93e14f3633": "Real_time_merger_bolt"
    }
    anomaly_start = 46683
    anomaly_end = 46783
    nb_anomalous_data = anomaly_end - anomaly_start + 1
    print(nb_anomalous_data)

    dataFrame = pd.DataFrame()
    pathNameData = str(parent) + "/monitoring_data/real_data_for_causality_relabel_v2/"

    list_nodes = []
    for id_metric in list_id_metric:
        fileName = pathNameData + id_metric + '.json'
        fileNameCsv = pathNameData + id_metric + '.csv'
        df = pd.read_csv(fileNameCsv, sep=',', header=0)
        del df['timestamp']
        nodeName = map_id_to_name[id_metric]
        list_nodes.append(nodeName)
        df = df.rename(columns={'value': nodeName})
        dataFrame = dataFrame.join(df[nodeName], how='outer')

    data = dataFrame.loc[45683:50000]
    print(data)

    # import graph
    graph = nx.DiGraph()
    graph.add_nodes_from(list_nodes)
    graph.add_edges_from([("pre_Message_dispatcher_bolt", "message_dispatcher_bolt"),
                          ("message_dispatcher_bolt", "check_message_bolt"),
                          ("message_dispatcher_bolt", "Real_time_merger_bolt"),
                          ("check_message_bolt", "Real_time_merger_bolt"),
                          ("check_message_bolt", "metric_bolt"),
                          ("metric_bolt", "capacity_last_metric_bolt"),
                          ("Real_time_merger_bolt", "group_status_information_bolt"),
                          ("Real_time_merger_bolt", "capacity_elastic_search_bolt"),
                          ("group_status_information_bolt", "capacity_elastic_search_bolt")])

    for node in graph.nodes:
        graph.add_edge(node, node)

    anomalies_start_time = dict()
    for node in graph.nodes:
        anomalies_start_time[node] = anomaly_start

    start = time.time()
    if method == "EasyRCA":
        erca = EasyRCA(graph, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                       anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=0.01)

        erca.run(data)
        print(erca.root_causes)
        # root_causes = []
        # for subgraph_id in erca.dict_linked_anomalous_graph.keys():
        #     root_causes = root_causes + erca.root_causes[subgraph_id]["structure_defying"]
        #     root_causes = root_causes + erca.root_causes[subgraph_id]["structure_defying"]
        #     root_causes = root_causes + erca.root_causes[subgraph_id]["param_defying"]
        # print(root_causes)
        # draw_graph(graph)
    elif method == "EasyRCAPCMCI":
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests import ParCorr
        from tigramite import data_processing as pp
        from easyrca import remove_self_loops

        data_normal = data.loc[:anomaly_start - 10]
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
        else:
            print("Cyclic!!!!!")
            root_causes = []
    elif method == "MicroCause":
        root_causes = micro_cause(data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                   anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=0.01)
        print(root_causes)
    elif method == "CloudRanger":
        root_causes = cloud_ranger(data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                   anomaly_length=nb_anomalous_data, sig_threshold=0.01)
        print(root_causes)
    else:
        root_causes = []
        print("Error: method does not exist")
        exit(0)
    end = time.time()


