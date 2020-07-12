import os
import sys

dir_name = sys.argv[1]
for fname in os.listdir(dir_name):
    with open(os.path.join(dir_name, fname)) as f:
        if '-' in f.read():
            print(fname)
