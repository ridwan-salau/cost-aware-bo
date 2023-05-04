import json

def read_json(filename):
    file = open(filename + '.json')
    params = json.load(file)
    return params
