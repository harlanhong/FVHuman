import json
import pdb
import random
import decord
import os
vid_meta = json.load(open('/sensei-fs/users/fhong/src/pose2video/data/final_ted_meta.json', "r"))
    # all_test_cases = vid_meta["mode"] == "test"
all_test_cases = [item for item in vid_meta if item.get("mode") == "test"]
total_test_sample = []
for i in range(len(all_test_cases)):
    id_sample = all_test_cases[i]
    for i in range(3):
        selected_items = random.choices(id_sample['data'], k=5)
        vr = decord.VideoReader(os.path.join('/mnt/localssd/final_ted_dataset',selected_items[0]['video_path'])) #source len
        num_frames = len(vr)
        selected_ref1_index = random.sample(list(range(num_frames)), 3)
        
        vr = decord.VideoReader(os.path.join('/mnt/localssd/final_ted_dataset',selected_items[1]['video_path'])) #source len
        num_frames = len(vr)
        selected_ref2_index = random.sample(list(range(num_frames)), 3)
        
        vr = decord.VideoReader(os.path.join('/mnt/localssd/final_ted_dataset',selected_items[2]['video_path'])) #source len
        num_frames = len(vr)
        selected_ref3_index = random.sample(list(range(num_frames)), 3)
        
        vr = decord.VideoReader(os.path.join('/mnt/localssd/final_ted_dataset',selected_items[3]['video_path'])) #source len
        num_frames = len(vr)
        selected_ref4_index = random.sample(list(range(num_frames)), 3)
        
        vr = decord.VideoReader(os.path.join('/mnt/localssd/final_ted_dataset',selected_items[4]['video_path'])) #source len
        num_frames = len(vr)
        selected_tgt_index = random.sample(list(range(num_frames)), 3)
        
        test_sample = {}
        test_sample['ref'] = selected_items[:-1]
        test_sample['target'] = selected_items[-1]
        test_sample['ref_idx'] = [selected_ref1_index,selected_ref2_index,selected_ref3_index,selected_ref4_index]
        test_sample['tgt_idx'] = selected_tgt_index
        total_test_sample.append(test_sample)
        
        vr = decord.VideoReader(os.path.join('/mnt/localssd/final_ted_dataset',selected_items[0]['video_path'])) #source len
        num_frames = len(vr)
        selected_ref1_index = random.sample(list(range(num_frames)), 3)
        selected_ref2_index = random.sample(list(range(num_frames)), 3)
        selected_ref3_index = random.sample(list(range(num_frames)), 3)
        selected_ref4_index = random.sample(list(range(num_frames)), 3)
        selected_tgt_index = random.sample(list(range(num_frames)), 3)
        test_sample = []
        test_sample = {}
        test_sample['ref'] = [selected_items[0]]*4
        test_sample['target'] = selected_items[0]
        test_sample['ref_idx'] = [selected_ref1_index,selected_ref2_index,selected_ref3_index,selected_ref4_index]
        test_sample['tgt_idx'] = selected_tgt_index
        total_test_sample.append(test_sample)
        
json.dump(total_test_sample, open(f"./data/stage1_ms_test_metav2.json", "w"))

print('aa')