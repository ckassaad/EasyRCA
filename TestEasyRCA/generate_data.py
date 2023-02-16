"""
Coded by Lei Zan
"""

import numpy as np
import pandas as pd
import networkx as nx
import copy


class GenerateData:
    """
    Sampling mixed data from DAG.

    :param numNodeDisc: the number of discrete nodes in the DAG
    :param scalingValue: the maximum scaling value used to transfer quantitative values to qualitative values
    """
    def __init__(self, DAG, numNodeDisc=0, scalingValue=3):
        self.DAG = DAG
        self.listNode = self.DAG.nodes
        self.root = []
        for node in self.listNode:
            if len(nx.ancestors(self.DAG, node)) == 0:
                self.root.append(node)
        self.alphaList = ['A', 'B', 'C']
        self.scalingValue = scalingValue

        if numNodeDisc > len(self.listNode):
            raise ValueError('Out of list length')

        # self.discNode = np.random.choice(self.listNode, size=numNodeDisc, replace=False)
        orderingNodes = list(nx.topological_sort(self.DAG))
        if numNodeDisc == 0:
            self.discNode = []
        else:
            self.discNode = orderingNodes[-numNodeDisc:]

        self.lag = 1

    def generate_node_data(self, n, nodeName, listParent, searchDict, firstGeneration):
        """
        Sampling data for each node according to its type and types of its parents.

        :param n: the number of sampling points
        :param nodeName: the name of the node
        :param listParent: a list of its parents (a list of list, if not empty)
        :return: the sampling data of the node
        """
        uniqueAlpha = []
        # generating mapping function
        if firstGeneration is True:
            mappingAlphaNum = {}
            mappingNumAlpha = {}
            coeff = []
            noise = []
        else:
            mappingAlphaNum = searchDict[nodeName]['mappingAlphaNum']
            mappingNumAlpha = searchDict[nodeName]['mappingNumAlpha']
            coeff = searchDict[nodeName]['coeff']
            noise = searchDict[nodeName]['noise']
        if len(listParent) != 0:
            if firstGeneration is True:
                for parent in listParent:
                    if isinstance(parent[0], str):
                        uniqueAlpha += parent
                if len(uniqueAlpha) != 0:
                    uniqueAlpha = sorted(set(uniqueAlpha))
                    indexList = np.random.choice(range(3*len(uniqueAlpha)), size=len(uniqueAlpha), replace=False)
                    for (alpha, number) in zip(uniqueAlpha, indexList):
                        mappingAlphaNum[alpha] = number
        data = []
        if nodeName in self.discNode:
            if len(listParent) == 0:
                if firstGeneration is True:
                    noise = list(np.random.choice(['0', '1'], size=n, replace=True, p=[0.5, 0.5]))
                    data = noise
                else:
                    data = noise
            else:
                if firstGeneration is True:
                    alphaList = ['A', 'B', 'C']
                    np.random.shuffle(alphaList)
                    for (number, alpha) in zip(range(3), alphaList):
                        mappingNumAlpha[str(number)] = alpha
                # Store scaling ratios for each continuous parent
                ratioList = []
                for m in range(len(listParent)):
                    if isinstance(listParent[m][0], str):
                        ratioList.append((None, None))
                    else:
                        mediaList = copy.deepcopy(listParent[m])
                        minValue = min(mediaList)
                        mediaList = np.array(mediaList) - minValue
                        ratioList.append((self.scalingValue/max(mediaList), minValue))
                for i in range(n):
                    if i == 0:
                        data.append(None)
                        if firstGeneration is True:
                            noise.append('1')
                    else:
                        res = ''
                        for m in range(len(listParent)):
                            if isinstance(listParent[m][0], str):
                                res += listParent[m][i-self.lag]
                            else:
                                res += mappingNumAlpha[str(int(np.mod((listParent[m][i-self.lag]-ratioList[m][1])*ratioList[m][0], 3)))]
                        if firstGeneration is True:
                            noise.append(np.random.choice(['0', '1'], size=1, p=[0.5, 0.5])[0])
                        res += noise[i]
                        data.append(res)
                data[0] = data[n-1]
        else:
            if firstGeneration is True:
                coeff.append(np.random.uniform(low=0.1, high=1.0, size=1)[0])
            if len(listParent) == 0:
                for i in range(n):
                    if i == 0:
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0])
                        data.append(noise[i])
                    else:
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0])
                        data.append(coeff[0]*data[i-1]+noise[i])
            else:
                if firstGeneration is True:
                    coeffList = np.random.uniform(low=0.1, high=1.0, size=len(listParent))
                    for i in coeffList:
                        coeff.append(i)
                    numParentStr = 0
                    for m in range(len(listParent)):
                            if isinstance(listParent[m][0], str):
                                numParentStr+=1
                for i in range(n):
                    if i == 0:
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0])
                        data.append(noise[i])
                    else:
                        res = 0
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0]+np.sum(np.random.normal(size=numParentStr)))
                        res += coeff[0]*data[i-1]+noise[i]
                        for m in range(len(listParent)):
                            if isinstance(listParent[m][0], str):
                                res += coeff[m+1]*mappingAlphaNum[listParent[m][i-self.lag]]
                            else:
                                res += coeff[m+1]*listParent[m][i-self.lag]
                        data.append(res)
        return {'data':data, 'mappingAlphaNum':mappingAlphaNum, 'mappingNumAlpha':mappingNumAlpha, 'coeff':coeff, 'noise':noise}

    def update_node(self, n, node, searchDict, firstGeneration):
        """
        Update the node according to predecessors.

        :param n: the number of data
        :param node: the chosen node
        :param searchDict: current data dictionary
        :return: updated node
        """
        dataParents = []
        listParents = list(self.DAG.predecessors(node))
        if len(listParents) != 0:
            for parent in listParents:
                dataParents.append(searchDict[parent]['data'])
        return self.generate_node_data(n=n, nodeName=node, listParent=dataParents, searchDict=searchDict, firstGeneration=firstGeneration)


    def generate_data(self, n, intervention, rootStartIntervention, rootEndIntervention, secondInterventionNode, seccondStartIntervention, secondEndIntervention):
        """
        Sampling data for the DAG.

        :param n: the number of sampling points
        :param intervention: whether there are interventions or not
        :param rootStartIntervention: index of start point of root intervention
        :param rootEndIntervention: index of end point of root intervention
        :param seccondStartIntervention: index of start point of second intervention
        :param secondEndIntervention: index of end point of second intervention
        :return: a dataframe which contains sampling points for each node
        """
        firstGeneration = True
        orderingNodes = list(nx.topological_sort(self.DAG))
        searchDict = {}
        for node in orderingNodes:
            dataParents = []
            listParents = list(self.DAG.predecessors(node))
            if len(listParents) != 0:
                for parent in listParents:
                    dataParents.append(searchDict[parent]['data'])
            searchDict[node] = self.generate_node_data(n=n, nodeName=node, listParent=dataParents, searchDict=searchDict, firstGeneration=firstGeneration)

        firstGeneration = False
        if intervention is True:
            for root in self.root:
                if root in self.discNode:
                    searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.choice(list(sorted(set(searchDict[root]['data']))), size=(rootEndIntervention-rootStartIntervention), replace=True))
                else:
                    # searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.exponential(scale=2, size=(rootEndIntervention-rootStartIntervention)))
                    searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.normal(1, 1, size=(rootEndIntervention-rootStartIntervention)))

            for node in orderingNodes:
                if node in self.root:
                    continue
                else:
                    searchDict[node] = self.update_node(n=n, node=node, searchDict=searchDict, firstGeneration=firstGeneration)

            childNodeList = list(nx.descendants(self.DAG, secondInterventionNode))
            if secondInterventionNode in self.discNode:
                searchDict[secondInterventionNode]['data'][seccondStartIntervention:secondEndIntervention] = list(np.random.choice(list(sorted(set(searchDict[secondInterventionNode]['data']))), size=(secondEndIntervention-seccondStartIntervention), replace=True))
            else:
                # searchDict[secondInterventionNode]['data'][seccondStartIntervention:secondEndIntervention] = list(np.random.exponential(scale=2, size=(secondEndIntervention-seccondStartIntervention)))
                searchDict[secondInterventionNode]['data'][seccondStartIntervention:secondEndIntervention] = list(np.random.normal(1, 1, size=(secondEndIntervention-seccondStartIntervention)))

            if len(childNodeList) != 0:
                for node in orderingNodes:
                    if node in childNodeList:
                        searchDict[node] = self.update_node(n=n, node=node, searchDict=searchDict, firstGeneration=firstGeneration)
        resData = {}
        for key in searchDict.keys():
            resData[key] = searchDict[key]['data']
        resData = pd.DataFrame(resData)
        return resData


class GenerateDataSoft:
    """
    Sampling mixed data from DAG.

    :param numNodeDisc: the number of discrete nodes in the DAG
    :param scalingValue: the maximum scaling value used to transfer quantitative values to qualitative values
    """
    def __init__(self, DAG, numNodeDisc=0, scalingValue=3):
        self.DAG = DAG
        self.listNode = self.DAG.nodes
        self.root = []
        for node in self.listNode:
            if len(nx.ancestors(self.DAG, node)) == 0:
                self.root.append(node)
        self.alphaList = ['A', 'B', 'C']
        self.scalingValue = scalingValue

        if numNodeDisc > len(self.listNode):
            raise ValueError('Out of list length')

        # self.discNode = np.random.choice(self.listNode, size=numNodeDisc, replace=False)
        orderingNodes = list(nx.topological_sort(self.DAG))
        if numNodeDisc == 0:
            self.discNode = []
        else:
            self.discNode = orderingNodes[-numNodeDisc:]

        self.lag = 1

    def generate_node_data(self, n, nodeName, listParent, searchDict, firstGeneration):
        """
        Sampling data for each node according to its type and types of its parents.

        :param n: the number of sampling points
        :param nodeName: the name of the node
        :param listParent: a list of its parents (a list of list, if not empty)
        :return: the sampling data of the node
        """
        uniqueAlpha = []
        # generating mapping function
        if firstGeneration is True:
            mappingAlphaNum = {}
            mappingNumAlpha = {}
            coeff = []
            noise = []
        else:
            mappingAlphaNum = searchDict[nodeName]['mappingAlphaNum']
            mappingNumAlpha = searchDict[nodeName]['mappingNumAlpha']
            coeff = searchDict[nodeName]['coeff']
            noise = searchDict[nodeName]['noise']
        if len(listParent) != 0:
            if firstGeneration is True:
                for parent in listParent:
                    if isinstance(parent[0], str):
                        uniqueAlpha += parent
                if len(uniqueAlpha) != 0:
                    uniqueAlpha = sorted(set(uniqueAlpha))
                    indexList = np.random.choice(range(3*len(uniqueAlpha)), size=len(uniqueAlpha), replace=False)
                    for (alpha, number) in zip(uniqueAlpha, indexList):
                        mappingAlphaNum[alpha] = number
        data = []
        if nodeName in self.discNode:
            if len(listParent) == 0:
                if firstGeneration is True:
                    noise = list(np.random.choice(['0', '1'], size=n, replace=True, p=[0.5, 0.5]))
                    data = noise
                else:
                    data = noise
            else:
                if firstGeneration is True:
                    alphaList = ['A', 'B', 'C']
                    np.random.shuffle(alphaList)
                    for (number, alpha) in zip(range(3), alphaList):
                        mappingNumAlpha[str(number)] = alpha
                # Store scaling ratios for each continuous parent
                ratioList = []
                for m in range(len(listParent)):
                    if isinstance(listParent[m][0], str):
                        ratioList.append((None, None))
                    else:
                        mediaList = copy.deepcopy(listParent[m])
                        minValue = min(mediaList)
                        mediaList = np.array(mediaList) - minValue
                        ratioList.append((self.scalingValue/max(mediaList), minValue))
                for i in range(n):
                    if i == 0:
                        data.append(None)
                        if firstGeneration is True:
                            noise.append('1')
                    else:
                        res = ''
                        for m in range(len(listParent)):
                            if isinstance(listParent[m][0], str):
                                res += listParent[m][i-self.lag]
                            else:
                                res += mappingNumAlpha[str(int(np.mod((listParent[m][i-self.lag]-ratioList[m][1])*ratioList[m][0], 3)))]
                        if firstGeneration is True:
                            noise.append(np.random.choice(['0', '1'], size=1, p=[0.5, 0.5])[0])
                        res += noise[i]
                        data.append(res)
                data[0] = data[n-1]
        else:
            if firstGeneration is True:
                coeff.append(np.random.uniform(low=0.1, high=1.0, size=1)[0])
            if len(listParent) == 0:
                for i in range(n):
                    if i == 0:
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0])
                        data.append(noise[i])
                    else:
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0])
                        data.append(coeff[0]*data[i-1]+noise[i])
            else:
                if firstGeneration is True:
                    coeffList = np.random.uniform(low=1, high=2, size=len(listParent))
                    for i in coeffList:
                        coeff.append(i)
                    numParentStr = 0
                    for m in range(len(listParent)):
                            if isinstance(listParent[m][0], str):
                                numParentStr+=1
                for i in range(n):
                    if i == 0:
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0])
                        data.append(noise[i])
                    else:
                        res = 0
                        if firstGeneration is True:
                            noise.append(0.1*np.random.normal(size=1)[0]+np.sum(np.random.normal(size=numParentStr)))
                        res += coeff[0]*data[i-1]+noise[i]
                        for m in range(len(listParent)):
                            if isinstance(listParent[m][0], str):
                                res += coeff[m+1]*mappingAlphaNum[listParent[m][i-self.lag]]
                            else:
                                res += coeff[m+1]*listParent[m][i-self.lag]
                        data.append(res)
        return {'data':data, 'mappingAlphaNum':mappingAlphaNum, 'mappingNumAlpha':mappingNumAlpha, 'coeff':coeff, 'noise':noise}

    def update_node(self, n, node, searchDict, firstGeneration):
        """
        Update the node according to predecessors.

        :param n: the number of data
        :param node: the chosen node
        :param searchDict: current data dictionary
        :return: updated node
        """
        dataParents = []
        listParents = list(self.DAG.predecessors(node))
        if len(listParents) != 0:
            for parent in listParents:
                dataParents.append(searchDict[parent]['data'])
        return self.generate_node_data(n=n, nodeName=node, listParent=dataParents, searchDict=searchDict, firstGeneration=firstGeneration)


    def generate_data(self, n, intervention, secondInterventionNode, rootStartIntervention=0, rootEndIntervention=0, seccondStartIntervention=0, secondEndIntervention=0):
        """
        Sampling data for the DAG.

        :param n: the number of sampling points
        :param intervention: "structure", "parameter" or NULL, to decide whether it is a structure intervention or a parameter intervention
        :param rootStartIntervention: index of start point of root intervention
        :param rootEndIntervention: index of end point of root intervention
        :param secondInterventionNode: the second intervention node
        :param seccondStartIntervention: index of start point of second intervention
        :param secondEndIntervention: index of end point of second intervention
        :param parameter_intervention: whether there are parameter changings or not
        :return: a dataframe which contains sampling points for each node
        """
        firstGeneration = True
        orderingNodes = list(nx.topological_sort(self.DAG))
        searchDict = {}
        for node in orderingNodes:
            dataParents = []
            listParents = list(self.DAG.predecessors(node))
            if len(listParents) != 0:
                for parent in listParents:
                    dataParents.append(searchDict[parent]['data'])
            searchDict[node] = self.generate_node_data(n=n, nodeName=node, listParent=dataParents, searchDict=searchDict, firstGeneration=firstGeneration)

        firstGeneration = False
        if intervention == "structure":
            # print("Structure changing !!! ")
            for root in self.root:
                if root in self.discNode:
                    searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.choice(list(sorted(set(searchDict[root]['data']))), size=(rootEndIntervention-rootStartIntervention), replace=True))
                else:
                    searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.exponential(scale=2, size=(rootEndIntervention-rootStartIntervention)))

            for node in orderingNodes:
                if node in self.root:
                    continue
                else:
                    searchDict[node] = self.update_node(n=n, node=node, searchDict=searchDict, firstGeneration=firstGeneration)

            childNodeList = list(nx.descendants(self.DAG, secondInterventionNode))
            if secondInterventionNode in self.discNode:
                searchDict[secondInterventionNode]['data'][seccondStartIntervention:secondEndIntervention] = list(np.random.choice(list(sorted(set(searchDict[secondInterventionNode]['data']))), size=(secondEndIntervention-seccondStartIntervention), replace=True))
            else:
                searchDict[secondInterventionNode]['data'][seccondStartIntervention:secondEndIntervention] = list(np.random.exponential(scale=2, size=(secondEndIntervention-seccondStartIntervention)))

            if len(childNodeList) != 0:
                for node in orderingNodes:
                    if node in childNodeList:
                        searchDict[node] = self.update_node(n=n, node=node, searchDict=searchDict, firstGeneration=firstGeneration)

        #parameter changing
        if intervention == "parameter":
            # print("Parameter changing !!! ")
            for root in self.root:
                if root in self.discNode:
                    searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.choice(list(sorted(set(searchDict[root]['data']))), size=(rootEndIntervention-rootStartIntervention), replace=True))
                else:
                    # searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.exponential(scale=2, size=(rootEndIntervention-rootStartIntervention)))
                    searchDict[root]['data'][rootStartIntervention:rootEndIntervention] = list(np.random.normal(1, 1, size=(rootEndIntervention-rootStartIntervention)))

            for node in orderingNodes:
                if node in self.root:
                    continue
                else:
                    searchDict[node] = self.update_node(n=n, node=node, searchDict=searchDict, firstGeneration=firstGeneration)
            # searchDict[secondInterventionNode]['coeff'][:-1] = list(np.random.uniform(low=0.1, high=1.0, size=(len(searchDict[secondInterventionNode]['coeff'])-1)))
            # searchDict[secondInterventionNode]['coeff'][1:] = list(np.random.uniform(low=0.01, high=0.5, size=(len(searchDict[secondInterventionNode]['coeff'])-1)))
            searchDict[secondInterventionNode]['coeff'][1:] = [5] *(len(searchDict[secondInterventionNode]['coeff'])-1)
            searchDict[secondInterventionNode]['data'][rootStartIntervention:rootEndIntervention] = self.update_node(n=n, node=secondInterventionNode, searchDict=searchDict, firstGeneration=firstGeneration)['data'][rootStartIntervention:rootEndIntervention]
            childNodeList = list(nx.descendants(self.DAG, secondInterventionNode))
            if len(childNodeList) != 0:
                for node in orderingNodes:
                    if node in childNodeList:
                        searchDict[node] = self.update_node(n=n, node=node, searchDict=searchDict, firstGeneration=firstGeneration)
        resData = {}
        for key in searchDict.keys():
            resData[key] = searchDict[key]['data']
        resData = pd.DataFrame(resData)
        return resData


def generate_data_with_parametric_intervention(DAG,n, secondInterventionNode, rootStartIntervention=0, rootEndIntervention=0):
    orderingNodes = list(nx.topological_sort(DAG))
    noise = pd.DataFrame(np.zeros([n, len(DAG.nodes)]), columns=orderingNodes)
    data = pd.DataFrame(np.zeros([n, len(DAG.nodes)]), columns=orderingNodes)

    anomaly_size = rootEndIntervention - rootStartIntervention + 1

    coef_dict = dict()
    for edge in DAG.edges:
        coef_dict[edge] = np.random.uniform(low=0.1, high=1.0, size=1)[0]
    for node in DAG.nodes:
        coef_dict[(node, node)] = np.random.uniform(low=0.1, high=1.0, size=1)[0]

    for node in orderingNodes:
        data[node] = 0.1 * np.random.normal(size=n)
        noise[node] = 0.1 * np.random.normal(size=n)
    # self cause on root
    data[orderingNodes[0]] = noise[orderingNodes[0]] + coef_dict[(orderingNodes[0], orderingNodes[0])] *data[orderingNodes[0]].shift(periods=-1)

    for i in range(1, n):
        for node in orderingNodes[1:]:
            for par in DAG.predecessors(node):
                # data[node] = data[node] + data[par].shift(periods=1)
                data[node].loc[i] = noise[node].loc[i] + coef_dict[(node, node)] * data[node].loc[i-1] + coef_dict[(par, node)] * data[par].loc[i-1]

    # intervention on root
    data[orderingNodes[0]].loc[rootStartIntervention: rootEndIntervention] = \
        np.random.normal(1, 1, size=anomaly_size) + \
        0.1 * data[orderingNodes[0]].loc[rootStartIntervention: rootEndIntervention].shift(periods=-1)

    # generate new intervention coeff
    intervention_coef = np.random.uniform(low=1, high=2, size=1)[0]
    # treating other nodes
    for i in range(rootStartIntervention, rootEndIntervention):
        for node in orderingNodes[1:]:
            for par in DAG.predecessors(node):
                # nodeStartIntervention = rootStartIntervention
                # nodeEndIntervention = rootEndIntervention
                # intervention on second node
                if node == secondInterventionNode:
                    # data[node].loc[nodeStartIntervention: nodeEndIntervention] = noise[node].loc[rootStartIntervention: rootEndIntervention] + 5 * data[par].shift(periods=1)
                    data[node].loc[i] = noise[node].loc[i] + coef_dict[(node, node)]*data[node].loc[i-1] + intervention_coef * data[par].loc[i-1]
                # propagate interventions
                else:
                    # data[node].loc[nodeStartIntervention: nodeEndIntervention] = noise[node].loc[rootStartIntervention: rootEndIntervention] + data[par].shift(periods=1)
                    data[node].loc[i] = noise[node].loc[i] + coef_dict[(node, node)]*data[node].loc[i-1] + coef_dict[(par, node)] * data[par].loc[i-1]

    data.dropna(axis=0, inplace=True)

    return data


def generate_data_with_structural_intervention(DAG,n, secondInterventionNode, rootStartIntervention=0, rootEndIntervention=0):
    orderingNodes = list(nx.topological_sort(DAG))
    noise = pd.DataFrame(np.zeros([n, len(DAG.nodes)]), columns=orderingNodes)
    data = pd.DataFrame(np.zeros([n, len(DAG.nodes)]), columns=orderingNodes)

    anomaly_size = rootEndIntervention - rootStartIntervention + 1

    coef_dict = dict()
    for edge in DAG.edges:
        coef_dict[edge] = np.random.uniform(low=0.1, high=1.0, size=1)[0]
    for node in DAG.nodes:
        coef_dict[(node, node)] = np.random.uniform(low=0.1, high=1.0, size=1)[0]

    for node in orderingNodes:
        data[node] = 0.1 * np.random.normal(size=n)
        noise[node] = 0.1 * np.random.normal(size=n)
    # self cause on root
    data[orderingNodes[0]] = noise[orderingNodes[0]] + coef_dict[(orderingNodes[0], orderingNodes[0])] * data[orderingNodes[0]].shift(periods=-1)

    for i in range(1, n):
        for node in orderingNodes[1:]:
            for par in DAG.predecessors(node):
                # data[node] = data[node] + data[par].shift(periods=1)
                data[node].loc[i] = noise[node].loc[i] + coef_dict[(node, node)] * data[node].loc[i-1] + coef_dict[(par, node)] * data[par].loc[i-1]

    # intervention on root
    data[orderingNodes[0]].loc[rootStartIntervention: rootEndIntervention] = \
        np.random.normal(1, 1, size=anomaly_size) + \
        0.1 * data[orderingNodes[0]].loc[rootStartIntervention: rootEndIntervention].shift(periods=-1)

    # generate new intervention coeff
    intervention_coef = 0
    # treating other nodes
    for i in range(rootStartIntervention, rootEndIntervention):
        for node in orderingNodes[1:]:
            for par in DAG.predecessors(node):
                # nodeStartIntervention = rootStartIntervention
                # nodeEndIntervention = rootEndIntervention
                # intervention on second node
                if node == secondInterventionNode:
                    # data[node].loc[nodeStartIntervention: nodeEndIntervention] = noise[node].loc[rootStartIntervention: rootEndIntervention] + 5 * data[par].shift(periods=1)
                    data[node].loc[i] = noise[node].loc[i] + coef_dict[(node, node)]*data[node].loc[i-1] + intervention_coef * data[par].loc[i-1]
                # propagate interventions
                else:
                    # data[node].loc[nodeStartIntervention: nodeEndIntervention] = noise[node].loc[rootStartIntervention: rootEndIntervention] + data[par].shift(periods=1)
                    data[node].loc[i] = noise[node].loc[i] + coef_dict[(node, node)]*data[node].loc[i-1] + coef_dict[(par, node)] * data[par].loc[i-1]

    data.dropna(axis=0, inplace=True)

    return data


if __name__ == "__main__":
    graph = nx.DiGraph()
    # graph.add_edges_from([("1", "2"), ("1", "3"), ("3", "4"), ("4", "5"), ("4", "6")])
    graph.add_edges_from([("1", "2"), ("1", "3"), ("3", "4"), ("1", "4"), ("4", "5"), ("4", "6")])
    # graph.add_edges_from([("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6")])

    anomalous = ["1", "2", "3", "4", "5", "6"]
    # anomalies_start = dict()
    # for node in anomalous:
    #     anomalies_start[node] = 1000

    rootStartIntervention = 9000
    rootEndIntervention = 9999
    n = 10000
    anomaly_size = rootEndIntervention - rootStartIntervention + 1

    print(n, rootEndIntervention, rootStartIntervention)
    res_df = generate_data_with_parametric_intervention(graph, n, secondInterventionNode="4",
                                                        rootStartIntervention=rootStartIntervention,
                                                        rootEndIntervention=rootEndIntervention)
    for node in graph.nodes:
        graph.add_edge(node, node)

    # print(res_df)
    # np.random.seed(0)
    # one = np.random.normal(size=2000)
    # two = 2 * one + 0.1 * np.random.normal(size=2000)
    # three = 5 * one + 0.2 * np.random.normal(size=2000)
    # four = 5 * three + 0.2 * np.random.normal(size=2000)
    # five = 5 * four + 0.2 * np.random.normal(size=2000)
    # six = 5 * four + 0.2 * np.random.normal(size=2000)
    # data = pd.DataFrame(np.array([one, two, three, four, five, six]).T, columns=["1", "2", "3", "4", "5", "6"])
    #
    # data["2"].loc[1000: 1599] = np.array(0.2 * np.random.normal(size=600))
    # data["4"].loc[1000: 1599] = np.array(0.2 * np.random.normal(size=600))
    # data["5"].loc[1000: 1599] = 5 * data["4"].loc[1000: 1599].values + 0.2 * np.random.normal(size=600)
    # data["6"].loc[1000: 1599] = np.array(10.05 * np.random.uniform(0, 1, size=600))
    #

    from easyrca import EasyRCA
    # anomalies_start = dict()
    # for node in anomalous:
    #     anomalies_start[node] = rootStartIntervention

    anomalies_start_time = dict()
    for node in graph.nodes:
        short_path = nx.shortest_path(graph, source="1", target=node, weight=None, method='dijkstra')
        anomalies_start_time[node] = rootStartIntervention + len(short_path) - 1

    AG = EasyRCA(graph, anomalous, anomalies_start_time=anomalies_start_time, anomaly_length=anomaly_size)
    # res = AG.run_without_data()
    res = AG.run(res_df)
    print(AG.root_causes)

