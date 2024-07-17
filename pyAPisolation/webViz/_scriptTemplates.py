#template out some scripts for the webViz

import os
import json


onload = """ window.onload = function() {%s}; """

umap_call = """generate_umap($datatb, $umapkeys); """

paracoords_call = """generate_paracoords($datatb, $paracoordskeys); """

colors = """var embed_colors = {'': ['#0000FF', '#A5E41F', '#FF24FF', '#B8B2B2', '#fc0303']"""
colors_end = """};"""

def generate_umap(data, keys):
    return umap_call.replace("$datatb", data).replace("$umapkeys", str(keys))

def generate_paracoords(data, keys):
    return paracoords_call.replace("$datatb", data).replace("$paracoordskeys", str(keys))

def generate_onload(script):
    return onload % script