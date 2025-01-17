from glob import glob
import os
from pathlib import Path
import joblib
import yaml
import argparse
from hubert_api import HubertFeatureReader
import soundfile as sf
import numpy as np

class HubertInference:
    def __init__(self, ckpt, km):
        self.reader = HubertFeatureReader(checkpoint_path=ckpt, layer=11, use_cuda=False)
        self.km_model = joblib.load(open(km, 'rb'))        
        
    def get_codes_from_path(self, wav_path):
        feats = self.reader.get_feats(wav_path)
        out = self.km_model.predict(feats.cpu().numpy())
        return out
    
    def get_codes(self, wav):
        feats = self.reader.get_feats_from_wav(wav)
        out = self.km_model.predict(feats.cpu().numpy())
        return out  
    
    def get_features(self, wav):
        feats = self.reader.get_feats(wav)
        return feats          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="utils/hubert_extraction/config.yaml", help="Path to DFA data.yaml")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found.")
        exit(1)
    
    hubert_base_ckpt = config["path"]["hubert_ckpt"]
    hubert_base_kmeans = config["path"]["km_path"]
    dataset_dir = Path(config["path"]["dataset_dir"])
    root_dir = Path(config["path"]["root_dir"])
    speaker = config["path"]["dataset_dir"].split("/")[-1]
    save_feature_only = config["path"]["save_feature_only"]

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    hubert = HubertInference(hubert_base_ckpt, hubert_base_kmeans)
        
    processed_lines = list()
    
    def get_audio_duration(audio_file_path):
        with sf.SoundFile(audio_file_path) as f:
            duration = len(f) / f.samplerate
        return duration
    
    print(f"Processing speaker {dataset_dir}...")
    
    wav_files = glob(os.path.join(dataset_dir, "*.wav"))
    
    for awav in wav_files:
        print(f"Processing audio {awav}...")
        
        if not save_feature_only:
            data_dict = {}
            
            # store audio path
            data_dict['audio'] = awav
            
            # get codes and store it
            codes = hubert.get_codes_from_path(awav)
            data_dict['hubert'] = ' '.join(map(str, codes))
            
            # store duration
            duration = get_audio_duration(awav)
            data_dict['duration'] = duration

            processed_lines.append(data_dict)
            
        else:
            if not os.path.exists(root_dir / f"{speaker}"):
                os.makedirs(root_dir / f"{speaker}")
            basename = awav.split('/')[-1].split('.')[0]
            feats = hubert.get_features(awav)
            np.save(root_dir / f"{speaker}" / f"{basename}.npy", feats)
    
    if not save_feature_only:
        with open(root_dir / f"{speaker}.txt", 'w') as f:
            for line in processed_lines:
                f.write(str(line) + "\n")