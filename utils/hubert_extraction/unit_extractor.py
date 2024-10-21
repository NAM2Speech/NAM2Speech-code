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

    def get_units_from_features(self, feats):
        out = self.km_model.predict(feats)
        return out            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="utils/hubert_extraction/config.yaml", help="Path to DFA data.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    hubert_base_ckpt = config["path"]["hubert_ckpt"]
    hubert_base_kmeans = config["path"]["km_path"]
    predicted_feat_dir = Path(config["path"]["predicted_feat_dir"])
    speaker = config["path"]["predicted_feat_dir"].split("/")[-1]
    root_dir = Path(config["path"]["root_dir"])

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    hubert = HubertInference(hubert_base_ckpt, hubert_base_kmeans)
        
    processed_lines = list()
    
    def get_audio_duration(audio_file_path):
        with sf.SoundFile(audio_file_path) as f:
            duration = len(f) / f.samplerate
        return duration
        
    hubert_files = glob(os.path.join(predicted_feat_dir, "*.npy"))
    
    for afile in hubert_files:
        print(f"Processing file {afile}...")
        basename = afile.split('/')[-1].split('.')[0]
        data_dict = {}
        awav = os.path.join(config["path"]["audio_dir"],basename+'.wav')
        
        feat = np.load(afile)
        codes = hubert.get_units_from_features(feat)

        data_dict['audio'] = awav

        data_dict['hubert'] = ' '.join(map(str, codes))
        
        # store duration
        duration = get_audio_duration(awav)
        data_dict['duration'] = duration      

        processed_lines.append(data_dict)  

    with open(root_dir / f"{speaker}.txt", 'w') as f:
        for line in processed_lines:
            f.write(str(line) + "\n")
    print("Saved units at:",root_dir / f"{speaker}.txt")