'''
This is the script for Self Learn process

'''

import pandas as pd
from numpy.random import normal as GaussianDistribution
from collections import OrderedDict
from collections import Counter
import operator
from random import sample
import numpy as np
from pymongo import MongoClient
client = MongoClient(port=27017)
db = client.rlRecomendation



class rlLearn:
    def __init__(self,custDetails,conf):
        # Get the date  as a seperate column
        custDetails['Date'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%d"))
        # Converting date to float for easy comparison
        custDetails['Date'] = custDetails['Date'].astype('float64')
        # Get the period of month column
        custDetails['monthPeriod'] = custDetails['Date'].apply(lambda x: int(x > conf['monthPer']))
        # Aggregate the custDetails to get a distribution of rewards
        rewardFull = custDetails.groupby(['Segment', 'Month', 'monthPeriod', 'Day', conf['product_id']])[conf['prod_qnty']].agg(
            'sum').reset_index()
        # Get these data frames for all methods
        self.custDetails = custDetails
        self.conf = conf
        self.rewardFull = rewardFull
        # Defining some dictionaries for storing the values
        self.countDic = {}  # Dictionary to store the count of products
        self.polDic = {}  # Dictionary to store the value distribution
        self.rewDic = {}  # Dictionary to store the reward distribution
        self.recoCountdic = {}  # Dictionary to store the recommendation counts

    # Method to find unique values of each of the variables
    def uniqeVars(self):
        # Finding unique value for each of the variables
        segments = list(self.rewardFull.Segment.unique())
        months = list(self.rewardFull.Month.unique())
        monthPeriod = list(self.rewardFull.monthPeriod.unique())
        days = list(self.rewardFull.Day.unique())
        return segments,months,monthPeriod,days

    # Method to consolidate all products
    def prodConsolidator(self):
        # Get all the unique values of the variables
        segments, months, monthPeriod, days = self.uniqeVars()
        # Creating the consolidated dictionary
        for seg in segments:
            for mon in months:
                for period in monthPeriod:
                    for day in days:
                        # Get the subset of the data
                        subset1 = self.rewardFull[(self.rewardFull['Segment'] == seg) & (self.rewardFull['Month'] == mon) & (
                                self.rewardFull['monthPeriod'] == period) & (self.rewardFull['Day'] == day)]
                        # INitializing a temporary dictionary to storing in mongodb
                        tempDic = {}
                        # Check if the subset is valid
                        if len(subset1) > 0:
                            # Iterate through each of the subset and get the products and its quantities
                            stateId = str(seg) + '_' + mon + '_' + str(period) + '_' + day
                            # Define a dictionary for the state ID
                            self.countDic[stateId] = {}
                            tempDic[stateId] = {}
                            for i in range(len(subset1.StockCode)):
                                # Store in the Count dictionary
                                self.countDic[stateId][subset1.iloc[i]['StockCode']] = int(subset1.iloc[i]['Quantity'])
                                tempDic[stateId][subset1.iloc[i]['StockCode']] = int(subset1.iloc[i]['Quantity'])
                            # Dumping each record into mongo db
                            db.rlQuantdic.insert(tempDic)

        # Consolidate the rewards and value functions based on the quantities
        for key in self.countDic.keys():
            # Creating two temporary dictionaries for loading in Mongodb
            tempDicpol = {}
            tempDicrew = {}
            # First get the dictionary of products for a state
            prodCounts = self.countDic[key]
            self.polDic[key] = {}
            self.rewDic[key] = {}
            # Initializing temporary dictionaries also
            tempDicpol[key] = {}
            tempDicrew[key] = {}
            # Update the policy values
            for pkey in prodCounts.keys():
                # Creating the value dictionary using a Gaussian process
                self.polDic[key][pkey] = GaussianDistribution(loc=prodCounts[pkey], scale=1, size=1)[0].round(2)
                tempDicpol[key][pkey] = self.polDic[key][pkey]
                # Creating a reward dictionary using a Gaussian process
                self.rewDic[key][pkey] = GaussianDistribution(loc=prodCounts[pkey], scale=1, size=1)[0].round(2)
                tempDicrew[key][pkey] = self.rewDic[key][pkey]
            # Dumping each of these in mongo db
            db.rlRewarddic.insert(tempDicrew)
            db.rlValuedic.insert(tempDicpol)
        print('[INFO] Dumped the quantity dictionary,policy and rewards in MongoDB')

class rlRecomend:
    def __init__(self, custDetails, conf):
        # Get the date  as a seperate column
        custDetails['Date'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%d"))
        # Converting date to float for easy comparison
        custDetails['Date'] = custDetails['Date'].astype('float64')
        # Get the period of month column
        custDetails['monthPeriod'] = custDetails['Date'].apply(lambda x: int(x > conf['monthPer']))
        # Aggregate the custDetails to get a distribution of rewards
        rewardFull = custDetails.groupby(['Segment', 'Month', 'monthPeriod', 'Day', conf['product_id']])[
            conf['prod_qnty']].agg(
            'sum').reset_index()
        # Get these data frames for all methods
        self.custDetails = custDetails
        self.conf = conf
        self.rewardFull = rewardFull

    # Method to find unique values of each of the variables
    def uniqeVars(self):
        # Finding unique value for each of the variables
        segments = list(self.rewardFull.Segment.unique())
        months = list(self.rewardFull.Month.unique())
        monthPeriod = list(self.rewardFull.monthPeriod.unique())
        days = list(self.rewardFull.Day.unique())
        return segments, months, monthPeriod, days

    # Method to sample a state
    def stateSample(self):
        # Get the unique state elements
        segments, months, monthPeriod, days = self.uniqeVars()
        # Get the context of the customer. For the time being let us randomly select all the states
        seg = sample(segments, 1)[0]  # Sample the segment
        mon = sample(months, 1)[0]  # Sample the month
        monthPer = sample([0, 1], 1)[0]  # sample the month period
        day = sample(days, 1)[0]  # Sample the day
        # Get the state id by combining all these samples
        stateId = str(seg) + '_' + mon + '_' + str(monthPer) + '_' + day
        self.seg = seg
        return stateId

    # Method to update a dictionary in case a state Id is not available
    def collfinder(self,stateId,countDic,polDic,rewDic,recoCountdic):
        # Defining some dictionaries for storing the values
        self.countDic = countDic  # Dictionary to store the count of products
        self.polDic = polDic  # Dictionary to store the value distribution
        self.rewDic = rewDic  # Dictionary to store the reward distribution
        self.recoCountdic = recoCountdic  # Dictionary to store the recommendatio
        self.stateId = stateId
        print("[INFO] THe current state is :", stateId)
        if self.countDic is None:
            print("[INFO] State ID do not exist")
            self.countDic = {}
            self.countDic[stateId] = {}
            self.polDic = {}
            self.polDic[stateId] = {}
            self.rewDic = {}
            self.rewDic[stateId] = {}
        if self.recoCountdic is None:
            self.recoCountdic = {}
            self.recoCountdic[stateId] = {}
        else:
            self.recoCountdic[stateId] = {}

    # Method to update the recommendation dictionary
    def recoCollChecker(self):
        recoCol = db.rlRecotrack.find_one({self.stateId: {'$exists': True}})
        if recoCol is None:
            print("[INFO] Inserting the record in the recommendation collection")
            db.rlRecotrack.insert_one({self.stateId: {}})
        return recoCol

    # Create a function to get a list of products for a certain segment
    def segProduct(self,seg, nproducts):
        # Get the list of unique products for each segment
        seg_products = list(self.rewardFull[self.rewardFull['Segment'] == seg]['StockCode'].unique())
        seg_products = sample(seg_products, nproducts)
        return seg_products

    # This is the function to get the top n products based on value
    def sortlist(self,nproducts,seg):
        # Get the top products based on the values and sort them from product with largest value to least
        topProducts = sorted(self.polDic[self.stateId].keys(), key=lambda kv: self.polDic[self.stateId][kv])[-nproducts:][::-1]
        # If the topProducts is less than the required number of products nproducts, sample the delta
        while len(topProducts) < nproducts:
            print("[INFO] top products less than required number of products")
            segProducts = self.segProduct(seg, (nproducts - len(topProducts)))
            newList = topProducts + segProducts
            # Finding unique products
            topProducts = list(OrderedDict.fromkeys(newList))
        return topProducts

    # This is the function to create the number of products based on exploration and exploitation
    def sampProduct(self,seg, nproducts,epsilon):
        # Initialise an empty list for storing the recommended products
        seg_products = []
        # Get the list of unique products for each segment
        Segment_products = list(self.rewardFull[self.rewardFull['Segment'] == seg]['StockCode'].unique())
        # Get the list of top n products based on value
        topProducts = self.sortlist(nproducts,seg)
        # Start a loop to get the required number of products
        while len(seg_products) < nproducts:
            # First find a probability
            probability = np.random.rand()
            if probability >= epsilon:
                # print(topProducts)
                # The top product would be first product in the list
                prod = topProducts[0]
                # Append the selected product to the list
                seg_products.append(prod)
                # Remove the top product once appended
                topProducts.pop(0)
                # Ensure that seg_products is unique
                seg_products = list(OrderedDict.fromkeys(seg_products))
            else:
                # If the probability is less than epsilon value randomly sample one product
                prod = sample(Segment_products, 1)[0]
                seg_products.append(prod)
                # Ensure that seg_products is unique
                seg_products = list(OrderedDict.fromkeys(seg_products))
        return seg_products

    # This is the method for updating the dictionaries after recommendation
    def dicUpdater(self,prodList, stateId):
        for prod in prodList:
            # Check if the product is in the dictionary
            if prod in list(self.countDic[stateId].keys()):
                # Update the count by 1
                self.countDic[stateId][prod] += 1
            else:
                self.countDic[stateId][prod] = 1
            if prod in list(self.recoCountdic[stateId].keys()):
                # Update the recommended products with 1
                self.recoCountdic[stateId][prod] += 1
            else:
                # Initialise the recommended products as 1
                self.recoCountdic[stateId][prod] = 1
            if prod not in list(self.polDic[stateId].keys()):
                # Initialise the value as 0
                self.polDic[stateId][prod] = 0
            if prod not in list(self.rewDic[stateId].keys()):
                # Initialise the reward dictionary as 0
                self.rewDic[stateId][prod] = GaussianDistribution(loc=0, scale=1, size=1)[0].round(2)

        print("[INFO] Completed the initial dictionary updates")

    def dicAdder(self,prodList, stateId):
        #self.countDic[stateId] = {}
        #self.polDic[stateId] = {}
        #self.recoCountdic[stateId] = {}
        #self.rewDic[stateId] = {}
        # Loop through the product list
        for prod in prodList:
            # Initialise the count as 1
            self.countDic[stateId][prod] = 1
            # Initialise the value as 0
            self.polDic[stateId][prod] = 0
            # Initialise the recommended products as 1
            self.recoCountdic[stateId][prod] = 1
            # Initialise the reward dictionary as 0
            self.rewDic[stateId][prod] = GaussianDistribution(loc=0, scale=1, size=1)[0].round(2)
        print("[INFO] Completed the dictionary initialization")
        # Next update the collections with the respective updates
        print('[INFO] Updating all the collections')
        # Updating the quantity collection
        db.rlQuantdic.insert_one({stateId: self.countDic[stateId]})
        # Updating the recommendation tracking collection
        db.rlRecotrack.insert_one({stateId: self.recoCount[stateId]})
        # Updating the value function collection for the products
        db.rlValuedic.insert_one({stateId: self.polDic[stateId]})
        # Updating the rewards collection
        db.rlRewarddic.insert_one({stateId: self.rewDic[stateId]})
        print('[INFO] Completed updating all the collections')


    # Method to sample a stateID and then initialize the dictionaries
    def rlRecommender(self):
        # First sample a stateID
        stateId = self.stateId
        # Initialise the dictionaries based on stateId
        #self.collfinder(stateId)
        # Start the recommendation process
        if len(self.polDic[stateId]) > 0:
            print("The context exists")
            # Implement the sampling of products based on exploration and exploitation
            seg_products = self.sampProduct(self.seg, self.conf["nProducts"],self.conf["epsilon"])
            # Check if the recommendation count collection exist
            recoCol = self.recoCollChecker()
            print('Recommendation collection existing :',recoCol)
            # Update the dictionaries of values and rewards
            self.dicUpdater(seg_products, stateId)
        else:
            print("The context dosent exist")
            # Get the list of relavant products
            seg_products = self.segProduct(self.seg, conf["nProducts"])
            # Add products to the value dictionary and rewards dictionary
            self.dicAdder(seg_products, stateId)
        print("[INFO] Completed the recommendation process")

        return seg_products

    # Function to initiate customer action
    def custAction(self,segproducts):
        print('[INFO] getting the customer action')
        # Sample a value to get how many products will be clicked
        click_number = np.random.choice(np.arange(0, 10),
                                        p=[0.50, 0.35, 0.10, 0.025, 0.015, 0.0055, 0.002, 0.00125, 0.00124, 0.00001])
        # Sample products which will be clicked based on click number
        click_list = sample(segproducts, click_number)

        # Sample for buy values
        buy_number = np.random.choice(np.arange(0, 10),
                                      p=[0.70, 0.15, 0.10, 0.025, 0.015, 0.0055, 0.002, 0.00125, 0.00124, 0.00001])
        # Sample products which will be bought based on buy number
        buy_list = sample(segproducts, buy_number)

        return click_list, buy_list

    def getReward(self,loc):
        rew = GaussianDistribution(loc=loc, scale=1, size=1)[0].round(2)
        return rew

    def saPolicy(self,rew, prod):
        # This function gets the relavant algorithm for the policy update
        # Get the current value of the state
        vcur = self.polDic[self.stateId][prod]
        # Get the counts of the current product
        n = self.recoCountdic[self.stateId][prod]
        # Calculate the new value
        Incvcur = (1 / n) * (rew - vcur)
        return Incvcur

    def valueUpdater(self,seg_products, loc, custList, remove=True):
        for prod in custList:
            # Get the reward for the bought product. The reward will be centered around the defined reward for each action
            rew = self.getReward(loc)
            # Update the reward in the reward dictionary
            self.rewDic[self.stateId][prod] += rew
            # Update the policy based on the reward
            Incvcur = self.saPolicy(rew, prod)
            self.polDic[self.stateId][prod] += Incvcur
            # Remove the bought product from the product list
            if remove:
                seg_products.remove(prod)
        return seg_products

    # Function to update the reward dictionary and the value dictionary based on customer action
    def rewardUpdater(self, seg_products,custBuy=[], custClick=[]):
        # Check if there are any customer purchases
        if len(custBuy) > 0:
            seg_products = self.valueUpdater(seg_products, self.conf['buyReward'], custBuy)
            # Repeat the same process for customer click
        if len(custClick) > 0:
            seg_products = self.valueUpdater(seg_products, self.conf['clickReward'], custClick)
            # For those products not clicked or bought, give a penalty
        if len(seg_products) > 0:
            custList = seg_products.copy()
            seg_products = self.valueUpdater(seg_products, -2, custList,False)
        # Next update the collections with the respective updates
        print('[INFO] Updating all the collections')
        # Updating the quantity collection
        db.rlQuantdic.replace_one({self.stateId: {'$exists': True}}, {self.stateId: self.countDic[self.stateId]})
        # Updating the recommendation tracking collection
        db.rlRecotrack.replace_one({self.stateId: {'$exists': True}}, {self.stateId: self.recoCountdic[self.stateId]})
        # Updating the value function collection for the products
        db.rlValuedic.replace_one({self.stateId: {'$exists': True}}, {self.stateId: self.polDic[self.stateId]})
        # Updating the rewards collection
        db.rlRewarddic.replace_one({self.stateId: {'$exists': True}}, {self.stateId: self.rewDic[self.stateId]})
        print('[INFO] Completed updating all the collections')


