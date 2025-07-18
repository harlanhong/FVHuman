import glob
import os
import json
import numpy as np
vid_meta = json.load(open('/projects/p32296/fating/src/novel-view-human-master/data/dyhuman_meta.json', "r"))
    # self.vid_meta = vid_meta
vid_meta = [item for item in vid_meta if item.get("mode") == 'test']
root = '/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/'
for meta in vid_meta:
    id = meta['id']
    