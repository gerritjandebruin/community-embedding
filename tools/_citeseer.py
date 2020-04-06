import csv

import networkx as nx
import pandas as pd

from ._network import gc


def get_data(giant_component=True) -> (pd.Series, nx.Graph):
    """Return ground_truth and graph."""
    with open("citeseer/citeseer.content") as f:
        ground_truth = {l[0]: l[-1] for l in csv.reader(f, delimiter="\t")}
    graph = nx.read_edgelist("citeseer/citeseer.cites")
    if giant_component:
        graph = gc(graph)
    return pd.Series(ground_truth), graph