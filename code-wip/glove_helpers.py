import struct
from scipy.sparse import coo_matrix

def load_cooccurrences(path, *args, **kwargs):
  """ Usage: load_cooccurrences("cooccurrence.bin") """
  with open(path, "rb") as f:
    raw = f.read()
  i, j, data = zip(*struct.iter_unpack("@iid", raw))
  return coo_matrix((data, (i, j)), *args, **kwargs)
