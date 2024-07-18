#template out some scripts for the webViz

import os
import json


onload = """ window.onload = function() {%s}; """

umap_call = """generate_umap($datatb, $umapkeys); """

paracoords_call = """generate_paracoords($datatb, $paracoordskeys, '$default_color'); """

colors = """var embed_colors = """


def generate_umap(data, keys):
    return umap_call.replace("$datatb", data).replace("$umapkeys", str(keys))

def generate_paracoords(data, keys, colors):
    return paracoords_call.replace("$datatb", data).replace("$paracoordskeys", str(keys)).replace("$default_color", colors)

def generate_colors(colors_in=None):
    if colors is None:
        return str({'': [0, 0, 0]})
    else:
        return colors + str(colors_in)
def generate_onload(script):
    return onload % script