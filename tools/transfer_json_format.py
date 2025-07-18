import json

# 转换为目标格式
def convert_json(original):
    new_data = []
    for entry in original:
        target = entry['target']
        target['idx'] = entry['tgt_idx']
        ref_list = []
        
        for ref, idx in zip(entry['ref'], entry['ref_idx']):
            ref['idx'] = idx
            ref_list.append(ref)
        
        new_data.append({
            "target": target,
            "ref": ref_list
        })
    return new_data

# 进行转换
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

original_data = read_json_file('/projects/p32296/fating/src/novel-view-human-master/data/stage1_ms_test_metav2_bk.json')
converted_data = convert_json(original_data)
write_json_file('/projects/p32296/fating/src/novel-view-human-master/data/stage1_ms_test_metav2.json', converted_data)