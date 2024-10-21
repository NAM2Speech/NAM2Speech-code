import json 
import numpy as np
from pathlib import Path
import torch
import os
from torch.utils.data import Dataset

def convert_arr_to_tensor(arr, dtype):
    return torch.tensor(arr, dtype=dtype)

def get_mask_from_batch(batch, pad_idx):
    return (batch != pad_idx)

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("Input length exceeds max_len")

        s = np.shape(x)[1]  # Get the second dimension size
        x_padded = np.pad(
            x, ((0, max_len - np.shape(x)[0]), (0, 0)), mode="constant", constant_values=PAD
        )
        return x_padded

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

class ParrotDataset(Dataset):
    def __init__(self, split, data_config):
        self.root_dir = Path(data_config["path"]["root_path"])
        self.data_file = self.root_dir / f"{split}.txt"

        # path of NAM and simulated ground-truth speech
        self.nam_path = data_config["path"]["nam_path"]
        self.gt_path = data_config["path"]["gt_path"]

        # path of simulated LJNAM and LJSpeech speech
        self.LJNAM_path = data_config["path"]["LJNAM_path"]
        self.LJSpeech_path = data_config["path"]["LJSpeech_path"]
        
        self.data_list = []
        
        with open(self.data_file) as f:
            data_lines = f.readlines()
            for l in data_lines:
                self.data_list.append(l)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        basename = self.data_list[idx].strip()

        if basename+'.npy' in os.listdir(self.gt_path):
            gt_path = os.path.join(self.gt_path, f"{basename}.npy")  # Read output Groundtruth hubert
            nam_path = os.path.join(self.nam_path, f"{basename}.npy") # Read input NAMs hubert
        else:
            gt_path = os.path.join(self.LJSpeech_path, f"{basename}.npy")  # Read output Groundtruth hubert
            nam_path = os.path.join(self.LJNAM_path, f"{basename}.npy") # Read input NAMs hubert
        
        gt_code = np.load(gt_path) 
        nam_code = np.load(nam_path)

        # Read input CTC tokens
        ctc_path = os.path.join(os.path.join("runs/TTE/ASR_tokens_character"), "{}.npy".format(basename),)
        ctc_label = np.load(ctc_path)

        # Ensure that both have the same length by truncating
        # print(f"nam_code: {len(nam_code)} and gt_code: {len(gt_code)}")
        if len(nam_code) != len(gt_code):
            min_len = min(len(nam_code), len(gt_code))
            nam_code = nam_code[:min_len]
            gt_code = gt_code[:min_len]

        return {
            'id': basename,
            'nam_code': nam_code,
            'gt_code': gt_code,
            "ctc_label": ctc_label
        }

    def collate_fn(self, data_list):
        ids = [d['id'] for d in data_list]
        nam_codes = [d['nam_code'] for d in data_list]
        gt_codes = [d['gt_code'] for d in data_list]
        ctc_labels = [d["ctc_label"] for d in data_list]

        # Pad the 2D sequences using pad_2D
        max_len = max(len(nam_code) for nam_code in nam_codes)
        nam_codes_padded = convert_arr_to_tensor(pad_2D(nam_codes, max_len), dtype=torch.float32)
        gt_codes_padded = convert_arr_to_tensor(pad_2D(gt_codes, max_len), dtype=torch.float32)

        ctc_labels = pad_1D(ctc_labels)
        src_lens = np.array([nam_code.shape[0] for nam_code in nam_codes])

        # Use a fixed padding index of 0
        src_mask = get_mask_from_batch(nam_codes_padded, pad_idx=0)
        src_mask = src_mask.any(dim=-1)

        tgt_mask = get_mask_from_batch(gt_codes_padded, pad_idx=0)
        tgt_mask = tgt_mask.any(dim=-1)

        data = {
            'ids': ids,
            'nam_codes': nam_codes_padded,
            'src_lens': src_lens,
            'gt_codes': gt_codes_padded,
            'ctc_labels': ctc_labels,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask
        }

        return data