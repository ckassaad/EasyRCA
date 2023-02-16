"""
Coded by Charles Assaad, Imad Ez-Zejjari and Lei Zan
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import scipy as scp
from sklearn.linear_model import LinearRegression as lr
from sklearn.feature_selection import f_regression as fr


# from scipy import special
# from scipy.spatial import cKDTree
# import itertools
# from joblib import Parallel, delayed, cpu_count
# import statsmodels.api as sm


class CiTests:
    def __init__(self, x, y, cond_list=None):
        super(CiTests, self).__init__()
        self.x = x
        self.y = y
        if cond_list is None:
            self.cond_list = []
        else:
            self.cond_list = cond_list

    def get_dependence(self, df):
        print("To be implemented")

    def get_pvalue(self, df):
        print("To be implemented")


class FisherZ(CiTests):
    def __init__(self, x, y, cond_list=None):
        CiTests.__init__(self, x, y, cond_list)

    def get_dependence(self, df):
        list_nodes = [self.x, self.y] + self.cond_list
        df = df[list_nodes]
        a = df.values.T

        if len(self.cond_list) > 0:
            cond_list_int = [i + 2 for i in range(len(self.cond_list))]
        else:
            cond_list_int = []

        correlation_matrix = np.corrcoef(a)
        var = list((0, 1) + tuple(cond_list_int))
        sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
        if np.linalg.det(sub_corr_matrix) == 0:
            r = 1
        else:
            inv = np.linalg.inv(sub_corr_matrix)
            r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
        return r

    def get_pvalue(self, df):
        r = self.get_dependence(df)
        if r == 1:
            r = r - 0.0000000001
        z = 0.5 * np.log((1 + r) / (1 - r))
        pval = np.sqrt(df.shape[0] - len(self.cond_list) - 3) * abs(z)
        pval = 2 * (1 - norm.cdf(abs(pval)))

        return pval, r

    def diff_two_fisherz(self, df1, df2):
        r1 = self.get_dependence(df1)
        if r1 == 1:
            r1 = r1 - 0.0000000001
        r2 = self.get_dependence(df2)
        if r2 == 1:
            r2 = r2 - 0.0000000001
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        n1 = df1.shape[0]
        n2 = df2.shape[0]
        sezdiff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        ztest = (z1 - z2) / sezdiff
        pval = 2 * (1 - norm.cdf(abs(ztest), 0, 1))
        return pval


class LinearRegression:
    def __init__(self, x, y, cond_list=[]):
        self.x = x
        self.y = y
        self.list_nodes = [x] + cond_list

    def get_coeff(self, df):
        X_data = df[self.list_nodes].values
        Y_data = df[self.y].values
        reg = lr().fit(X_data, Y_data)

        return reg.coef_[0]

    def test_zeo_coef(self, df):
        X_data = df[self.list_nodes].values
        Y_data = df[self.y].values
        pval = fr(X_data, Y_data)[1][0]

        # X_data = df[self.list_nodes].values
        # Y_data = df[self.y].values
        # reg1 = lr()
        # reg1.fit(X_data, Y_data)
        # ssef = sum(Y_data - reg1.predict(X_data))**2
        #
        # reduce_list_nodes = self.list_nodes.copy()
        # reduce_list_nodes.remove(self.x)
        # X_data = df[reduce_list_nodes].values
        # Y_data = df[self.y].values
        # reg2 = lr()
        # reg2.fit(X_data, Y_data)
        # sser = sum(Y_data - reg2.predict(X_data))**2
        #
        # dof_r = df.shape[0] - 1
        # dof_f = df.shape[0] - 2
        # dof_factors = len(self.list_nodes)
        #
        # F = ((sser - ssef)/(dof_r - dof_f)) / (ssef/dof_f)
        # pval = (1 - norm.cdf(F, dof_f, dof_factors))
        return pval


def grubb_test(quatification_list, confidence_level=0.05):
    n = len(quatification_list)

    stat_value = max(abs(np.array(quatification_list) - np.mean(quatification_list)))/ np.std(quatification_list)

    t_dist = scp.stats.t.ppf(1 - confidence_level / (2 * n), n - 2)
    numerator = (n - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(n) * np.sqrt(n - 2 + np.square(t_dist))
    critical_value = numerator / denominator

    quatification_list_sorted_idx = sorted(range(len(quatification_list)), key=quatification_list.__getitem__)
    quatification_list_sorted = sorted(quatification_list)
    # locate potential outlier
    l1 = quatification_list_sorted[1] - quatification_list_sorted[0]
    l2 = quatification_list_sorted[-1] - quatification_list_sorted[-2]

    anomaly_position = None
    if stat_value > critical_value:
        if l1 > l2:
            anomaly_position = quatification_list_sorted_idx[0]
        else:
            anomaly_position = quatification_list_sorted_idx[-1]

    return {"anomaly_position": anomaly_position, "stat_value": stat_value, "critical_value": critical_value}


###############################################################################################################
# # The input data is purely continuous and well tailored, the output is the relative distance array
# def getDistArray(data):
#     """
#         calculate the distance from each element in data to another element
#     :param data: list of list , each list inside the list is a column
#     :return:list of list of list (list of matrices) containing the distance between one point and another point
#     """
#     N = data.shape[0]
#     nDim = data.shape[1]
#     # inds = sorted(list(data.columns))
#     inds = list(data.columns)
#     disArray = []
#     for m in inds:
#         dataDim = data[m]
#
#         # calculate the distance of Manhattan, return list
#         # dataDim est un dataframe, convert to list
#         dataDim = list(map(float, list(dataDim)))
#         listForDistance = list(np.abs(np.subtract.outer(list(dataDim), list(dataDim))))
#         disArray.append([list(element) for element in listForDistance])
#     return disArray
#
# def getEpsilonDistance(k, disArray):
#     """
#         Get epsilon used in Knn for each point
#     :param k: the parameter k for k nearest neighbors
#     :param disArray: the distance btw points
#     :return: the maximum distance for each index of matrices comparing the different dimensions inside disArray
#     """
#     # takes the most time
#     epsilonDisArray = []
#     N = len(disArray)
#     lengthDisArray0 = len(disArray[0])
#     lengthDisArray00 = len(disArray[0][0])
#     for elementIndex in range(lengthDisArray0):
#         listTemp = []
#         for element2Index in range(lengthDisArray00):
#             listTemp.append(max([disArray[m][elementIndex][element2Index] for m in range(N)]))
#         # sort the list to take the k nearest neighbor
#         listTempSorted = sorted(listTemp)
#         # take the k éme point
#         epsilonDisArray.append(2*sorted(listTemp)[k])
#
#     return epsilonDisArray
# def parralelGetEpsilon(k, lenRaws, lenColumns, elementIndex, data):
#     listTemp = []
#     for element2Index in range(lenRaws):
#         maxVal = -float('inf')
#         for numColumn in range(lenColumns):
#             val = np.abs(data.iloc[elementIndex, numColumn] - data.iloc[element2Index, numColumn])
#             if maxVal < val:
#                 maxVal = val
#         listTemp.append(maxVal)
#     return 2*sorted(listTemp)[k]
# def getEpsDistOptimizedParallel(k, data):
#     lenColumns = data.shape[1]
#     lenRaws = data.shape[0]
#     resultParallel = Parallel(n_jobs=cpu_count()-1)(delayed(parralelGetEpsilon)(
#         k, lenRaws, lenColumns, elementIndex, data) for elementIndex in range(lenRaws))
#     return resultParallel
# def getEpsilonDistanceOptimized(k, data):
#     # not used
#     lenColumns = data.shape[1]
#     lenRaws = data.shape[0]
#     epsilonDisArray = []
#     for elementIndex in range(lenRaws):
#         listTemp = []
#         for element2Index in range(lenRaws):
#             maxVal = -float('inf')
#             for numColumn in range(lenColumns):
#                 val = np.abs(data.iloc[elementIndex, numColumn] - data.iloc[element2Index, numColumn])
#                 if maxVal < val:
#                     maxVal = val
#             listTemp.append(maxVal)
#         epsilonDisArray.append(2*sorted(listTemp)[k])
#     return epsilonDisArray
# def getEpsilonDistanceFast(k, data):
#     tree_xyz = cKDTree(data)
#     epsarray = tree_xyz.query(data, k=[k + 1], p=np.inf,
#                               eps=0., workers=-1)[0][:, 0].astype(np.float64)
#     # distArray = getDistArray(data2)
#     # epsilonDis = getEpsilonDistance(k, distArray)
#     # print("epsarray", 2*epsarray)
#     # print("epsilonDis", epsilonDis)
#
#     return 2*epsarray
# def condEntropyEstimator(data, k, dN):
#     """
#         calculate the conditional entropy estimator
#     :param data: the continuous data: data frame
#     :param k: the parameter k for k nearest neighbors
#     :param dN: number of continuous dimensions
#     :return:
#     """
#     N = data.shape[0]
#     if N == 1:
#         return 0
#     # distArray = getDistArrayParallel(data)
#     # distArray2 = getDistArray2(data)
#
#     # distArray = getDistArray(data)
#     # epsilonDis = getEpsilonDistance(k, distArray)
#
#     # epsilonDis = getEpsilonDistanceOptimized(k, data)
#     # epsilonDis = getEpsDistOptimizedParallel(k, data)
#     # cython
#     # distArray = getEpsDistance.getDistArray(data)
#     # epsilonDis = getEpsDistance.getEpsilonDistance(k, distArray)
#     # epsilonDis = getEpsDistance.getEpsilonDistanceOptimized(k, data.to_numpy())
#     # print("epsilonDis",  epsilonDis)
#     epsilonDis = getEpsilonDistanceFast(k, data.to_numpy())
#     if 0 in epsilonDis:
#         # delete all null values
#         epsilonDis = list(filter(lambda value: value != 0, epsilonDis))
#         N = len(epsilonDis)
#         if N == 0:
#             return 0
#     # calculate the entropy using the famous equation
#     # if list(epsilonDis)[0] == 0:
#     # epsilonDis = list(map(float, epsilonDis))
#     entropy = -special.digamma(k) + special.digamma(N) + (dN*sum(np.log(epsilonDis)))/N
#     return entropy
#
#
# def calcDfComb(data, dimDis, allCombinations):
#     """
#         Calculate the data frames for all combination in combinations
#     :param data: the data frame source
#     :param dimDis: The index for discrete columns
#     :param allCombinations: All possible unique combinations
#     :return: list of all combinations
#     """
#     result = []
#     for element in allCombinations:
#         dataComb = data
#         indiceElement = 0
#         for i in dimDis:
#             # select all element in the data frame having a specific value
#             dataComb = dataComb.loc[dataComb[i] == element[indiceElement]]
#             indiceElement += 1
#
#         # To investigate if len(dataComb.index) == 0:
#         result.append(dataComb)
#
#     return result
#
#
# def mixedEntroEstimator(data, dimCon, dimDis):
#     """
#         Calculate the information entropy for mixed data: the principal function
#     :param data: the data frame which contains all the information
#     :param dimCon: the indexes in the data frame - continuous ones
#     :param dimDis: the indexes in the data frame - discrete ones
#     :return: the value of the entropy
#     """
#     # Input data should be as matrix
#     dN = len(dimCon)
#     if data is not None:
#         N = data.shape[0]
#     # dataDis: data discrete taken from data
#     # dataCon: data continuous taken from data
#     dataDis = []
#     dataCon = []
#     estimatorCont = 0
#     estimatorDisc = 0
#     if len(dimCon) != 0:
#         # Select only continuous dimensions
#         dataCon = data[dimCon]
#     if len(dimDis) != 0:
#         # Select only discrete dimensions
#         dataDis = data[dimDis]
#
#     # if len(dimCon) == 0 and len(dimDis) == 0:
#     #     print("The data is Null")
#
#     # If the data is purely continuous!
#     if len(dimDis) == 0 and len(dimCon) != 0:
#         estimatorCont = condEntropyEstimator(dataCon, max(1, int(0.1*N)), dN)
#
#     # This list takes all unique combinations in the list
#     classByDimList = []
#     # data frame takes all the combinations of points
#     listDfComb = []
#     # Calculate the probability of different bins
#     probBins = []
#
#     if len(dimDis) != 0:
#         for i in dimDis:
#             if len(dataDis) > 0:
#                 classByDimList.append(list(np.unique(dataDis[i])))
#
#         # search all combination of the list
#         allCombinations = list(itertools.product(*classByDimList))
#         listDfComb = calcDfComb(data, dimDis, allCombinations)
#
#         for element in listDfComb:
#             probBins.append(len(element.index)/N)
#
#         for proba in probBins:
#             # To delete the possibility having log(0)!
#             if proba != 0:
#                 estimatorDisc -= proba*np.log(proba)
#
#     if len(dimDis) != 0 and len(dimCon) != 0:
#         # if the data is mixed
#         for i in range(len(probBins)):
#             proba = probBins[i]
#             if proba != 0 and listDfComb != 0:
#                 # define k to delete problems
#                 k = max(1, int(0.1*len(listDfComb[i].index)))
#                 estimatorCont += proba*condEntropyEstimator(dataCon.iloc[list(listDfComb[i].index), :], k, dN)
#
#     finalResult = estimatorDisc + estimatorCont
#     return finalResult
#
#
# def mixedEstimator(data, xind, yind, zind, isCat):
#     """
#         This method estimate the conditional mutual information
#         between X, Y / Z. Here we calculate all the elements
#     :param data: is a dataframe which contain many columns including those of X,Y,Z
#     :param xind: Are the indexes of X, list
#     :param yind: Are the indexes of Y, list
#     :param zinds: Are the indexes of Z, list
#     :param isCat: Are the indexes of discrete variables, list
#     :return:
#     """
#     # to delete the possibility to have double values, we use the set the transform it to a list
#     xDimCon = list(set(xind).difference(isCat))
#     xDimDis = list(set(xind).difference(xDimCon))
#
#     yDimCon = list(set(yind).difference(isCat))
#     yDimDis = list(set(yind).difference(yDimCon))
#
#     zDimCon = list(set(zind).difference(isCat))
#     zDimDis = list(set(zind).difference(zDimCon))
#
#     conXYZ = list(set(xDimCon + yDimCon + zDimCon))
#     disXYZ = list(set(xDimDis + yDimDis + zDimDis))
#     # print("here", conXYZ, disXYZ)
#     hXYZ = mixedEntroEstimator(data, conXYZ, disXYZ)
#
#     conXZ = list(set(xDimCon + zDimCon))
#     disXZ = list(set(xDimDis + zDimDis))
#     hXZ = mixedEntroEstimator(data, conXZ, disXZ)
#
#     conYZ = list(set(yDimCon + zDimCon))
#     disYZ = list(set(yDimDis + zDimDis))
#     hYZ = mixedEntroEstimator(data, conYZ, disYZ)
#
#     conZ = list(set(zDimCon))
#     disZ = list(set(zDimDis))
#     # calculate hZ, in case of Z null this will give 0 value
#     hZ = mixedEntroEstimator(data, conZ, disZ)
#     # to parallelize the jobs
#     # listeParam = [(conXYZ, disXYZ), (conXZ, disXZ), (conYZ, disYZ), (conZ, disZ)]
#     # resultParallel = Parallel(n_jobs=5)(delayed(mixedEntroEstimator)(element[0], element[1], element[2]) for element in listeParam)
#     # resultParallel = Parallel(n_jobs=4, prefer="threads")(delayed(mixedEntroEstimator)(data, element[0], element[1]) for element in listeParam)
#
#     # cmi = resultParallel[1] + resultParallel[2] - resultParallel[0] - resultParallel[3]
#     cmi = hXZ + hYZ - hXYZ - hZ
#     return cmi
#
#
# class OutlierTest:
#     def __init__(self, x, y, cond_list=None):
#         super(OutlierTest, self).__init__()
#         self.x = x
#         self.y = y
#         if cond_list is None:
#             self.cond_list = []
#         else:
#             self.cond_list = cond_list
#
#
#
#
# def dixon_test(quatification_list, confidence_level):
#     dict_critical_value = dict()
#     dict_critical_value[0.1] = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
#            0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
#            0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
#            0.277, 0.273, 0.269, 0.266, 0.263, 0.26
#            ]
#
#     dict_critical_value[0.05] = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
#            0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
#            0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
#            0.308, 0.305, 0.301, 0.29
#            ]
#
#     dict_critical_value[0.01] = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
#            0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
#            0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
#            0.384, 0.38, 0.376, 0.372
#            ]
#
#     n = len(quatification_list)
#
#     critical_value = dict_critical_value[confidence_level][n-3]
#
#
#     quatification_list_sorted_idx = sorted(range(len(quatification_list)), key=quatification_list.__getitem__)
#     quatification_list_sorted = sorted(quatification_list)
#     # locate potential outlier
#     l1 = quatification_list_sorted[1] - quatification_list_sorted[0]
#     l2 = quatification_list_sorted[-1] - quatification_list_sorted[-2]
#     print(l1, l2, quatification_list_sorted[-1] - quatification_list_sorted[0])
#
#     anomaly_position = None
#     if l1 > l2:
#         stat_value = l1/(quatification_list_sorted[-1] - quatification_list_sorted[0])
#         if stat_value > critical_value:
#             anomaly_position = quatification_list_sorted_idx[0]
#     else:
#         stat_value = l2 / (quatification_list_sorted[-1] - quatification_list_sorted[0])
#         if stat_value > critical_value:
#             anomaly_position = quatification_list_sorted_idx[-1]
#
#     return {"anomaly_position": anomaly_position, "stat_value": stat_value, "critical_value": critical_value}
#
#
# def boxplot_test(quatification_list):
#     Q1, Q3 = np.percentile(quatification_list, [25, 75])
#     IQR = Q3 - Q1
#     ul = Q3 + 1.5 * IQR
#     ll = Q1 - 1.5 * IQR
#     outliers = quatification_list[(quatification_list > ul) | (quatification_list < ll)]
#     return outliers

if __name__ == '__main__':
    x = np.random.normal(size=100)
    y = 2*x + 0.1 * np.random.normal(size=100)
    z = 5*x + 0.2* np.random.normal(size=100)
    data = pd.DataFrame(np.array([x, y, z]).T, columns=["x", "y", "z"])

    ci = FisherZ("z", "y", ["x"])
    p, d = ci.get_pvalue(data)
    print(p, d)

    ci = FisherZ("z", "y")
    p, d = ci.get_pvalue(data)
    print(p, d)
