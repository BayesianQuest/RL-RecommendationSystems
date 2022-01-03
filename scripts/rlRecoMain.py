'''

This is the main driver script for the application

# Implement this script using the below command line
python rlRecoMain.py --conf config/custprof.json
'''

import argparse
import pandas as pd
from utils import Conf,helperFunctions
from Data import DataProcessor
from processes import rfmMaker,rlLearn,rlRecomend
import os.path
from pymongo import MongoClient

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c','--conf',required=True,help='Path to the configuration file')
args = vars(ap.parse_args())

# Load the configuration file
conf = Conf(args['conf'])

print("[INFO] loading the raw files")
dl = DataProcessor(conf)

# Check if custDetails already exists. If not create it
if os.path.exists(conf["custDetails"]):
    print("[INFO] Loading customer details from pickle file")
    # Load the data from the pickle file
    custDetails = helperFunctions.load_files(conf["custDetails"])
else:
    print("[INFO] Creating customer details from csv file")
    # Let us load the customer Details
    custDetails = dl.gvCreator()
    # Starting the RFM segmentation process
    rfm = rfmMaker(custDetails,conf)
    custDetails = rfm.segmenter()
    # Save the custDetails file as a pickle file
    #helperFunctions.save_clean_data(custDetails,conf["custDetails"])

# Starting the self learning Recommendation system

# Check if the collections exist in Mongo DB
client = MongoClient(port=27017)
db = client.rlRecomendation

# Get all the collections from MongoDB
countCol = db["rlQuantdic"]
polCol = db["rlValuedic"]
rewCol = db["rlRewarddic"]
recoCountCol = db['rlRecotrack']

print(countCol.estimated_document_count())

# If Collections do not exist then create the collections in MongoDB
if countCol.estimated_document_count() == 0:
    print("[INFO] Main dictionaries empty")
    rll = rlLearn(custDetails, conf)
    # Consolidate all the products
    rll.prodConsolidator()
    print("[INFO] completed the product consolidation phase")
    # Get all the collections from MongoDB
    countCol = db["rlQuantdic"]
    polCol = db["rlValuedic"]
    rewCol = db["rlRewarddic"]

# start the recommendation phase
rlr = rlRecomend(custDetails,conf)
# Sample a state since the state is not available
stateId = rlr.stateSample()

print(stateId)


# Get the respective dictionaries from the collections

countDic = countCol.find_one({stateId: {'$exists': True}})
polDic = polCol.find_one({stateId: {'$exists': True}})
rewDic = rewCol.find_one({stateId: {'$exists': True}})

# The count dictionaries can exist but still recommendation dictionary can not exist. So we need to take this seperately

if recoCountCol.estimated_document_count() == 0:
    print("[INFO] Recommendation tracking dictionary empty")
    recoCountdic = {}
else:
    # Get the dictionary from the collection
    recoCountdic = recoCountCol.find_one({stateId: {'$exists': True}})


print('recommendation count dic', recoCountdic)


# Initialise the Collection checker method
rlr.collfinder(stateId,countDic,polDic,rewDic,recoCountdic)
# Get the list of recommended products
seg_products = rlr.rlRecommender()
print('List of recommended products',seg_products)
# Initiate customer actions

click_list,buy_list = rlr.custAction(seg_products)
print('click_list',click_list)
print('buy_list',buy_list)

# Get the reward functions for the customer action
rlr.rewardUpdater(seg_products,buy_list ,click_list)


