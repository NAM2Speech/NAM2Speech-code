# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import glob
import json
import os
import shutil
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from dataset import CodeDataset, parse_manifest, mel_spectrogram, \
    MAX_WAV_VALUE
from utils import AttrDict
from models import CodeGenerator

h = None
device = None


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, code):
    start = time.time()
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf


def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global f0_stats
    global spkrs_emb
    global dataset
    global spkr_dataset
    global idx
    global device
    global a
    global h
    global spkrs

    a = arguments
    idx = queue.get()
    device = idx
    
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(idx)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])

    file_list = parse_manifest(a.input_code_file)
    dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                            fmax_loss=h.fmax_for_loss, device=device,
                            multispkr=h.get('multispkr', None),
                            pad=a.pad)
    if os.path.exists(a.output_dir):
        shutil.rmtree(a.output_dir)
    os.makedirs(a.output_dir)

    if h.get('multispkr', None):
        spkrs = random.sample(range(len(dataset.id_to_spkr)), k=min(5, len(dataset.id_to_spkr)))

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def inference(item_index):
    code, gt_audio, filename, _ = dataset[item_index]
    code = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in code.items()}

    if a.parts:
        parts = Path(filename).parts
        fname_out_name = '_'.join(parts[-3:])[:-4]
    else:
        fname_out_name = Path(filename).stem

    new_code = dict(code)
    audio, rtf = generate(h, generator, new_code)
    output_file = os.path.join(a.output_dir, fname_out_name + '.wav')
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(output_file, h.sampling_rate, audio)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_code_file', default='runs/vocoder/LJSpeech/val.txt')
    parser.add_argument('--config', default='utils/vocoder/LJSpeech_config.json')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parts', action='store_true')
    parser.add_argument('-n', type=int, default=10)
    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    file_list = parse_manifest(a.input_code_file)
    dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                            multispkr=h.get('multispkr', None),
                            pad=a.pad)

    if a.debug:
        ids = list(range(1))
        import queue
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, a)

        for i in range(0, len(dataset)):
            inference(i)
            bar = progbar(i, len(dataset))
            message = f'{bar} {i}/{len(dataset)} '
            stream(message)
            if a.n != -1 and i > a.n:
                break
    else:
        idx = list(range(len(dataset)))
        random.shuffle(idx)
        with Pool(8, init_worker, (idQueue, a)) as pool:
            for i, _ in enumerate(pool.imap(inference, idx), 1):
                bar = progbar(i, len(idx))
                message = f'{bar} {i}/{len(idx)} '
                stream(message)
                if a.n != -1 and i > a.n:
                    break


if __name__ == '__main__':
    main()
