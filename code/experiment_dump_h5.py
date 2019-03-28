import os
import sys
adirCode=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,adirCode)

import h5py
import numpy as np
import json

def dump_h5_tree_impl(h5, path):
    if isinstance(h5, h5py.Group):
        yield {'type':'Group', 'path':path}
        for k in h5.keys():
            pathB = path + [k]
            h5B = h5[k]
            yield from dump_h5_tree_impl(h5B, pathB)
    elif isinstance(h5, h5py.Dataset):
        name = h5.name # our "path" variable concatenated with strings is what h5 calls the "name"
        shape = ",".join(list([str(d) for d in h5.shape]))
        yield {'type':'Group', 'path':path, 'name':name, 'shape':shape}
    else:
        print("class={}".format(h5.__class__))

def dump_h5_tree(h5):
    for dict in dump_h5_tree_impl(h5, []):
        print("found {}".format(json.dumps(dict, sort_keys=True)))

def load_and_dump_h5_tree(rfile):
#    print({
#        'rfile': rfile,
#        'adirCode': adirCode
#    })
    afileData = os.path.join(adirCode, rfile)
    f = h5py.File(afileData, "r")
    dump_h5_tree(f)
    f.close()

def usage():
    print("usage:\n\tTODO")

if(len(sys.argv)<=1):
    usage()
elif(len(sys.argv[1])<=0):
    usage()
else:
    rfile = sys.argv[1]
    load_and_dump_h5_tree(rfile)
    # ok: checkpoints/small_model_epoch_5.h5
    # fail: alphago_sl_policy.h5
