from pathlib import Path

D_DATA = Path(".") / "raw"
F_NODE_TYPES = "dcbn-node_types.npy"
F_EDGES = 'dcbn-t0001_edges.npy'

NODE_TYPE_NORMAL = 0
NODE_TYPE_BOUNDARY = 1
NODE_TYPE_INFLOW = 2
NODE_TYPE_OUTFLOW = 3
