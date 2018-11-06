__author__ = 'Hanmant Lokare'

"""

Implementation Agglomerative hierarchical clustering algorithm
and also calculates cross-corelation co-efficient.

"""

import pandas as pd
import math
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

class cluster:
    """
        This class creates a cluster.
        :param: id - list of ID of cluster
        :param: df - Dataframe consisting of whole dataset
        :return: None
        """

    def __init__(self,id,df):
        self.id = id
        self.df = df
        self.values = []
        self.centroid_list = []
        self.distance = 0
        list =  df.values.tolist()
        if len(self.id) == 1:
            for i in range(len(list[self.id[0]])):
                self.values.append(list[self.id[0]][i])
            self.centroid_list = self.values
        else:
            single_list = [item for sublist in df.values.tolist() for item in sublist]
            self.values = single_list

class program:

    """
        This class consist of actual clustering algorithm and its helping function.

        """
    def __init__(self):
        self.merge_list = []


    def centroid(self,cluster1,cluster2,df):
        """
        This function calculates the centroid between 2 clusters
        :param cluster1: First cluster
        :param cluster2: Second Cluster
        :param df: Dataframe consisting of whole dataset
        :return: Centroid list - consisting of all the centroid values of 2 clusters
        """

        centroid_list = []
        id_list = cluster1.id + cluster2.id

        if len(cluster1.id) <len(cluster2.id):
            self.merge_list.append(len(cluster1.id))
        else:
            self.merge_list.append(len(cluster2.id))

        length = len(cluster1.id) + len(cluster2.id)
        for i in range(len(df.columns)):
           sum = 0
           for j in range(length):
               sum+=df.iat[id_list[j],i]
           centroid_list.append(sum/length)
        return centroid_list

    def distance(self,cluster1,cluster2):
        """
        This calculates the Eculidian distance between two clusters
        :param cluster1: First cluster
        :param cluster2: Second Cluster
        :return: Distance between two clusters
        """
        add =0
        list1 = cluster1.values
        list2 = cluster2.values

        for i in range(len(list1)):
            diff = list2[i] - list1[i]
            add = add + (diff*diff)

        return math.sqrt(add)

    def agglomerative(self,df):
        """
        The actual Agglomerative hierarchical clustering algorithm.
        :param df: DataFrame
        :return: None
        """
        cluster_list = []
        cluster1 = None
        cluster2 = None
        distance = []
        for i in range(len(df)):
            list  = []
            list.append(i)
            cluster_list.append(cluster(list,df))

        while(len(cluster_list)!=1):
            min = 100000
            for i in range(len(cluster_list)):
                for j in range(len(cluster_list)):
                    if i>j:
                        temp = self.distance(cluster_list[i],cluster_list[j])
                        if temp < min:
                            min = temp
                            cluster1 = cluster_list[i]
                            cluster2 = cluster_list[j]
            distance.append(min)
            clus = self.centroid(cluster1,cluster2,df)
            temp_id = cluster1.id + cluster2.id
            clus_df = pd.DataFrame(clus)
            new_cluster = cluster(temp_id,clus_df)
            remove = []
            remove.append(cluster1.id)
            remove.append(cluster2.id)
            new_cluster_list = []
            for i in range(len(cluster_list)):
                if cluster_list[i].id not in remove:
                    new_cluster_list.append(cluster_list[i])
            new_cluster_list.append(new_cluster)
            cluster_list = new_cluster_list

    def positive_find_max(self,list):
        """
        Helper function to positive best Cross- correlation co-efficient
        :param list: Cross-relation of that attribute
        :return: dict
        """
        max = 0
        dict = {}
        count = 0
        for i in range(len(list)):
            if list[i]>max and list[i]<1:
                max = list[i]
                count = i
        dict[max] = count
        return dict

    def negative_find_max(self, list):
        """
        Helper function to negative best Cross- correlation co-efficient
        :param list: Cross- relation of that attribute
        :return: dict
        """
        max = 0
        dict = {}
        count = 0
        for i in range(len(list)):
            if list[i] < max and list[i] >= -1:
                max = list[i]
                count = i
        dict[max] = count
        return dict


    def main(self):
        """
        This function read the input data and calculates cross-corealtion co-efficient
        :return:
        """
        # Reading the dataset from the .csv file
        df = pd.read_csv("HW_AG_SHOPPING_CART_v805.csv")
        

        df = df.drop(columns='ID')
        # df = df.drop(columns='  Eggs')
        # df = df.drop(columns='  Meat')

        corr = df.corr(method='pearson')
        print(corr.to_string())
        list = corr.values
        pos_max_list = []
        neg_max_list = []
        for i in list:
            post_max = self.positive_find_max(i)
            neg_max = self.negative_find_max(i)
            pos_max_list.append(post_max)
            neg_max_list.append(neg_max)

        self.agglomerative(df)
        link = hierarchy.linkage(df.values, 'centroid')
        plt.figure()
        plt.ylabel('Eucledian Distance')
        plt.xlabel('Clusters')
        hierarchy.dendrogram(link)
        plt.show()

p = program()
p.main()
