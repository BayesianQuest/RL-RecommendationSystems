'''
This is the method for reading the configuration files in json format
'''

from json_minify import json_minify
import json

class Conf:

    def __init__(self,confPath):
        # Read the json file and load it into a dictionary
        conf = json.loads(json_minify(open(confPath).read()))
        self.__dict__.update(conf)
    def __getitem__(self, k):
        return self.__dict__.get(k,None)
