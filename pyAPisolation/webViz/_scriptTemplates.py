#template out some scripts for the webViz

import os
import json


onload = """ window.onload = function() {%s}; """

umap_call = """generate_umap($datatb, $umapkeys); """

paracoords_call = """generate_paracoords($datatb, $paracoordskeys); """

def generate_umap(data, keys):
    return umap_call.replace("$datatb", data).replace("$umapkeys", str(keys))

def generate_paracoords(data, keys):
    return paracoords_call.replace("$datatb", data).replace("$paracoordskeys", str(keys))

def generate_onload(script):
    return onload % script