import requests
from tqdm import tqdm
import re
import torch

def use_pretrained(model,
                   folder='weights/',
                   file_name="swint_victim_pretrained.pth",
                   download=False,
                   url=None, ):
    if download:
        response = requests.get(url, stream=True)
        t = int(response.headers.get('content-length', 0))  # total file size
        block_size = 1024 ** 2  # 1 Mbit
        progress_bar = tqdm(total=t, unit='iB', unit_scale=True)
        with open(f"weights/{file_name}", 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if (t != 0) and (progress_bar.n != t):
            print("ERROR downloading weights!")
            return -1
        print(f"Weights downloaded in {folder} directory!")
    model.load_state_dict(torch.load(os.path.join(folder, file_name)))
    return model


def fetch_subpart(data, subpart_key, transform_keys=None):
    if subpart_key == "full": return data
    keys = subpart_key.split(".")
    for key in keys:
        data = data[key]
    if transform_keys is not None:
        pattern, replacement = re.match(r"s/(.*)/(.*)/", transform_keys).groups()
        for key in list(data.keys()):
            data[re.sub(pattern, replacement, key)] = data.pop(key)
    return data