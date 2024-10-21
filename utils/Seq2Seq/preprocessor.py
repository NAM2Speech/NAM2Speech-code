from pathlib import Path
import json
import os
import argparse
import yaml
import numpy as np
from transformers import Wav2Vec2CTCTokenizer
import re
import shutil
import random

chars_to_ignore_regex = r'[\$\&\,\?\.\!\-\;\:\"\(\)\[\]\’\'\“\”]'
def remove_special_characters(batch):
    lst = []
    for i in batch:
        lst.append(re.sub(chars_to_ignore_regex, '', i).lower())
    return lst

def extract_all_chars(batch):
    all_text = ''
    for i in batch:
        all_text += " ".join(batch)
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

def replace_characters(sentences, replacements):
    modified_sentences = []
    for sentence in sentences:
        modified_sentence = sentence
        for target, replacement in replacements.items():
            modified_sentence = modified_sentence.replace(target, replacement)
        modified_sentences.append(modified_sentence)
    return modified_sentences

# Define the characters and their replacements
character_replacements = {
    'é': 'e',
    'à': 'a',
    'â': 'a',
    'é': 'e',
    'è': 'e',
    'ê': 'e',
    'ü': 'u',
    # Add more character replacements as needed
}

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config["path"]["root_path"])
        self.nam_path = config["path"]["nam_path"]
        self.ljnam_path = config["path"]["LJNAM_path"]
        self.val_size = config["preprocess"]["val_size"]
        
    def build_from_path(self):
        print("Processing data...")
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir)
        
        nam_files = os.listdir(self.nam_path)
        ljnam_files = os.listdir(self.ljnam_path)
        files = nam_files+ljnam_files
        random.shuffle(files)
        processed_lines = list()
        texts_list = []
        names_list = []
        
        for l in files: 
            basename = l.split('.')[0]
            processed_lines.append(basename)

        # get groundtruth CTC tokens
        text_path = "runs/data/text"
        for text in os.listdir(text_path):
            with open(os.path.join(text_path,text),'r') as f:
                texts_list.extend(file.split('\n')[0] for file in f.readlines())
                names_list.append(text.split('.')[0])
            f.close()
        rem_char = remove_special_characters(texts_list)

        modified_sentences = replace_characters(rem_char, character_replacements)
        vocab_train = extract_all_chars(modified_sentences)
        vocab_list = list(set(vocab_train["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open(os.path.join(self.root_dir,'vocab_character.json'), 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        ctc_label_lst = []

        tokenizer = Wav2Vec2CTCTokenizer(os.path.join(self.root_dir,'vocab_character.json'), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

        for atext in modified_sentences:
            ctc_label_lst.append(tokenizer.encode(atext))

        asr_tokens_save_path = os.path.join(self.root_dir,'ASR_tokens_character')
        os.makedirs(asr_tokens_save_path,exist_ok=True)
        for i in range(len(names_list)):
            with open(os.path.join(asr_tokens_save_path,names_list[i]+'.npy'), 'wb') as f:
                np.save(f, ctc_label_lst[i])

        cnt = 0
        for line in processed_lines:
            # if line+'.npy' in os.listdir(self.nam_path):
            if line+'.npy' in os.listdir(self.nam_path) and cnt!=self.val_size:
                with open(self.root_dir / "val.txt", 'a') as f:
                    f.write(str(line) + "\n")
                f.close()
                cnt += 1
            else:
                with open(self.root_dir / "train.txt", 'a') as f:
                    f.write(str(line) + "\n")
                f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="utils/TTE/TTE_config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    Prep = Preprocessor(config)
    Prep.build_from_path()