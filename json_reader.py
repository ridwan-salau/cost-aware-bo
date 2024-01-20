import json


def read_json(filename):
    file = open(f"{filename}.json")
    params = json.load(file)
    return params
