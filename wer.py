# CREATE GROUND TRUTH TEXTS FOR WER CALCULATION

import os
speaker = "Stetho"

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o","--out_path", type=str, required=True, help="path to output wavs")
parser.add_argument("-t","--text_path", type=str, required=True, help="path to ground truth text")
parser.add_argument("-s","--store_path", type=str, required=True, help="path to store hypothesis text")
args = parser.parse_args()

out_path = args.out_path
textpath = args.text_path
storefile = args.store_path

valfile = "runs/data/metadata_forwer.csv"
if os.path.exists(valfile):
    os.remove(valfile)

for i in os.listdir(textpath):
    name = i.split('.')[0]
    with open(os.path.join(textpath,i),'r') as f:
        text = f.readlines()[0].strip('\n')
    final = name + '|' + text + '|' + text
    with open(valfile,'a') as fp:
        fp.write(final+'\n')
        
import os
import numpy as np
import torch
import pandas as pd
import whisper
from tqdm import tqdm
import glob
import jiwer
from whisper.normalizers import EnglishTextNormalizer

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

class Bliz(torch.utils.data.Dataset):

    def __init__(self, device=DEVICE):

        self.device = device
        self.wavpaths = glob.glob(out_path+'/*.wav')
        # self.wavpaths = self.wavpaths[:2]
        
        print('num files',len(self.wavpaths))
        self.gts = {}
        with open(valfile,'r') as vf:
            trans = vf.readlines()
            for t in trans:
                self.gts[t.split('|')[0]] = t.split('|')[-1]

    def __len__(self):
        return len(self.wavpaths)

    def __getitem__(self, idx):

        wav_path = self.wavpaths[idx]
        gt = self.gts[wav_path.split('/')[-1][:-4]]
        basename = os.path.basename(wav_path)
        audio = whisper.load_audio(wav_path)
        duration = len(audio)/16000
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        return gt,mel,duration,basename

dataset = Bliz()
loader = torch.utils.data.DataLoader(dataset, batch_size=4)
model = whisper.load_model("medium",device=DEVICE)
model = model.to(DEVICE)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)
hypotheses = []
references = []
durations = []
names = []

for texts, mels, duration, name in tqdm(loader):

    _, probs = model.detect_language(mels)
    probs = probs[0]
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)
    durations.extend(duration)
    names.extend(name)

gts = [ r.strip('\n').split('|')[-1] for r in references ]
data = pd.DataFrame(dict(hypothesis=hypotheses, reference=gts, duration=durations, name=names)) 
normalizer = EnglishTextNormalizer()

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
data["duration"] = [text for text in data["duration"]]
data["name"] = [text for text in data["name"]]


my_dictionary = {}
dur_lst = [round(float(tensor.numpy()), 2) for tensor in data["duration"]]

with open(storefile, "w") as f:
    for i in range(0, len(data["reference_clean"])):
        cer = jiwer.cer(list(data["reference_clean"])[i], list(data["hypothesis_clean"])[i])
        wer = jiwer.wer(list(data["reference_clean"])[i], list(data["hypothesis_clean"])[i])

        my_dictionary[data["name"][i]] = (round(wer,2), dur_lst[i])
        
        f.write(f"NAME: " + str(data["name"][i]) + "\n")
        f.write(f"GT: " + str(data["reference_clean"][i]) + "\n")
        f.write(f"HYP: " + str(data["hypothesis_clean"][i]) + "\n")
        f.write(f"WER: " + str(wer) + "\n") 
        f.write("\n")
    
    # compute wer for all files
    cer = jiwer.cer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

    print(f"CER: {cer * 100:.2f} %")
    print(f"WER: {wer * 100:.2f} %")
    f.write(f"\nCER: {cer}\nWER: {wer}\n")