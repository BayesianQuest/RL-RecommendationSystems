'''

THis is the script for the rfm process using lifetime library
'''

import sys
sys.path.append('/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/RL_Recomendations/rlreco')
import pandas as pd
import lifetimes
from sklearn.cluster import KMeans
from utils import helperFunctions



class rfmMaker:
    def __init__(self,custDetails,conf):
        self.custDetails = custDetails
        self.conf = conf

    def rfmMatrix(self):
        # Converting data to RFM format
        RfmAgeTrain = lifetimes.utils.summary_data_from_transaction_data(self.custDetails, self.conf['customer_id'], 'Parse_date','grossValue')
        # Reset the index
        RfmAgeTrain = RfmAgeTrain.reset_index()
        return RfmAgeTrain

    # Function for ordering cluster numbers

    def order_cluster(self,cluster_field_name, target_field_name, data, ascending):
        # Group the data on the clusters and summarise the target field(recency/frequency/monetary) based on the mean value
        data_new = data.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        # Sort the data based on the values of the target field
        data_new = data_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
        # Create a new column called index for storing the sorted index values
        data_new['index'] = data_new.index
        # Merge the summarised data onto the original data set so that the index is mapped to the cluster
        data_final = pd.merge(data, data_new[[cluster_field_name, 'index']], on=cluster_field_name)
        # From the final data drop the cluster name as the index is the new cluster
        data_final = data_final.drop([cluster_field_name], axis=1)
        # Rename the index column to cluster name
        data_final = data_final.rename(columns={'index': cluster_field_name})
        return data_final

    # Function to do the cluster ordering for each cluster
    #

    def clusterSorter(self,target_field_name,RfmAgeTrain, ascending):
        # Defining the number of clusters
        nclust = self.conf['nclust']
        # Make the subset data frame using the required feature
        user_variable = RfmAgeTrain[['CustomerID', target_field_name]]
        # let us take four clusters indicating 4 quadrants
        kmeans = KMeans(n_clusters=nclust)
        kmeans.fit(user_variable[[target_field_name]])
        # Create the cluster field name from the target field name
        cluster_field_name = target_field_name + 'Cluster'
        # Create the clusters
        user_variable[cluster_field_name] = kmeans.predict(user_variable[[target_field_name]])
        # Sort and reset index
        user_variable.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
        # Sort the data frame according to cluster values
        user_variable = self.order_cluster(cluster_field_name, target_field_name, user_variable, ascending)
        return user_variable


    def clusterCreator(self):
        
        #data : THis is the dataframe for which we want to create the clsuters
        #clustName : This is the variable name
        #nclust ; Numvber of clusters to be created
        
        # Get the RFM data Frame
        RfmAgeTrain = self.rfmMatrix()
        # Implementing for user recency
        user_recency = self.clusterSorter('recency', RfmAgeTrain,False)
        #print('recency grouping',user_recency.groupby('recencyCluster')['recency'].mean().reset_index())
        # Implementing for user frequency
        user_freqency = self.clusterSorter('frequency', RfmAgeTrain, True)
        #print('frequency grouping',user_freqency.groupby('frequencyCluster')['frequency'].mean().reset_index())
        # Implementing for monetary values
        user_monetary = self.clusterSorter('monetary_value', RfmAgeTrain, True)
        #print('monetary grouping',user_monetary.groupby('monetary_valueCluster')['monetary_value'].mean().reset_index())

        # Merging the individual data frames with the main data frame
        RfmAgeTrain = pd.merge(RfmAgeTrain, user_monetary[["CustomerID", 'monetary_valueCluster']], on='CustomerID')
        RfmAgeTrain = pd.merge(RfmAgeTrain, user_freqency[["CustomerID", 'frequencyCluster']], on='CustomerID')
        RfmAgeTrain = pd.merge(RfmAgeTrain, user_recency[["CustomerID", 'recencyCluster']], on='CustomerID')
        # Calculate the overall score
        RfmAgeTrain['OverallScore'] = RfmAgeTrain['recencyCluster'] + RfmAgeTrain['frequencyCluster'] + RfmAgeTrain['monetary_valueCluster']
        return RfmAgeTrain

    def segmenter(self):
        
        #This is the script to create segments after the RFM analysis
        
        # Get the RFM data Frame
        RfmAgeTrain = self.clusterCreator()
        # Segment data
        RfmAgeTrain['Segment'] = 'Q1'
        RfmAgeTrain.loc[(RfmAgeTrain.OverallScore == 0), 'Segment'] = 'Q2'
        RfmAgeTrain.loc[(RfmAgeTrain.OverallScore == 1), 'Segment'] = 'Q2'
        RfmAgeTrain.loc[(RfmAgeTrain.OverallScore == 2), 'Segment'] = 'Q3'
        RfmAgeTrain.loc[(RfmAgeTrain.OverallScore == 4), 'Segment'] = 'Q4'
        RfmAgeTrain.loc[(RfmAgeTrain.OverallScore == 5), 'Segment'] = 'Q4'
        RfmAgeTrain.loc[(RfmAgeTrain.OverallScore == 6), 'Segment'] = 'Q4'

        # Merging the customer details with the segment
        custDetails = pd.merge(self.custDetails, RfmAgeTrain, on=['CustomerID'], how='left')
        # Saving the details as a pickle file
        helperFunctions.save_clean_data(custDetails,self.conf["custDetails"])
        print("[INFO] Saved customer details ")

        return custDetails


