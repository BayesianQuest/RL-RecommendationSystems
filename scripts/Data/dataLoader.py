'''
This is the scrip for loading data for RL Recommendation system

'''

import os
import pandas as pd
import pickle
import numpy as np
import random
from utils import helperFunctions
from datetime import datetime, timedelta,date
from dateutil.parser import parse

class DataProcessor:
    def __init__(self,configfile):
        # This is the first method in the DataProcessor class
        self.config = configfile

    def dataLoader(self):
        # THis is the method to load data from the input files
        inputPath = self.config["inputData"]
        dataFrame = pd.read_csv(inputPath,encoding = "ISO-8859-1")
        return dataFrame
    # This is the process for parsing dates
    def dateParser(self):
        custDetails = self.dataLoader()
        #Parsing  the date
        custDetails['Parse_date'] = custDetails[self.config["order_date"]].apply(lambda x: parse(x))
        # Parsing the weekdaty
        custDetails['Weekday'] = custDetails['Parse_date'].apply(lambda x: x.weekday())
        # Parsing the Day
        custDetails['Day'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%A"))
        # Parsing the Month
        custDetails['Month'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%B"))
        # Getting the year
        custDetails['Year'] = custDetails['Parse_date'].apply(lambda x: x.strftime("%Y"))
        # Getting year and month together as one feature
        custDetails['year_month'] = custDetails['Year'] + "_" +custDetails['Month']

        return custDetails

    def gvCreator(self):
        custDetails = self.dateParser()
        # Creating gross value column
        custDetails['grossValue'] = custDetails[self.config["prod_qnty"]] * custDetails[self.config["unit_price"]]

        return custDetails






