import os
from pathlib import Path
import json, ftfy
from datasets import load_dataset

data = []
for path in Path('./gen_data').rglob('*.json'):
    print(path)
    with open(path) as f:
        d = json.load(f)
        data += d

out_file_name_txt = "evol_instruct.json" 
out = open(out_file_name_txt, "w")
out.write(json.dumps(data, indent=2, ensure_ascii=False))
out.close()

ds = load_dataset("json", data_files=out_file_name_txt)
print(ds)
