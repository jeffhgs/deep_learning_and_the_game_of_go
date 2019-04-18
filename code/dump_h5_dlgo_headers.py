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

def get(h5, fd):
    return (None if h5 is None else
            None if not fd in h5 else
            h5[fd])

def dump_h5_tree(rfile, h5file):
    try:
        encoder=get(h5file, 'encoder')
        encoder_name=None
        board_width=None
        board_height=None
        if encoder is not None:
            encoder_name = None if encoder is None else encoder.attrs['name']
            if encoder_name is not None and not isinstance(encoder_name, str):
                encoder_name = encoder_name.decode('ascii')
            board_width = h5file['encoder'].attrs['board_width']
            board_height = h5file['encoder'].attrs['board_height']
        numNodes = sum(1 for i in dump_h5_tree_impl(h5file, ['model']))
        print("file={} encoder_name={} board={}x{} numNodes={}".format(
            rfile, encoder_name, board_width, board_height, numNodes))
    except Exception as ex:
        print("file={} exception={}".format(rfile, ex))


def load_and_dump_h5_tree(rfile):
    f = h5py.File(rfile, "r")
    dump_h5_tree(rfile, f)
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
