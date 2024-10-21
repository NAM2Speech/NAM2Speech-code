# NAM2Speech-code
Official code for Interspeech 2024 paper on "Towards Improving NAM-to-Speech Synthesis Intelligibility using Self-Supervised Speech Models". We propose a novel approach to significantly improve the intelligibility in the Non-Audible Murmur (NAM)-to-speech conversion task, leveraging self-supervision and sequence-to-sequence (Seq2Seq) learning techniques.

## Table of Contents
- [Libraries Installation](#libraries-installation)
- [Training NAM2Speech on CSTR NAM TIMIT Plus Corpus](#training-nam2speech-on-cstr-nam-timit-plus-corpus)
- [Things-to-do](#things-to-do)
- [Cite](#cite)

## Libraries Installation

- Note: You may need to clone and install fairseq to extract HuBERT features and units.

1. **Create and activate a new Conda environment:**
    ```bash
    conda create --name nam2speech python=3.8.19
    conda activate nam2speech
    ```

2. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu125
    ```

## Training NAM2Speech on CSTR NAM TIMIT Plus corpus

To train NAM2Speech, follow these steps:

### Step 1: Download CSTR NAM TIMIT Plus corpus's `NAM` and `whisper` files and `LJSpeech` corpus. The zip folder and its content will be stored at `runs/data`
`   ``bash
    python download.py
    ```

### Step 2: Extract HuBERT Units for speech samples from `LJSpeech` corpus

- Download the HuBERT checkpoint and quantizer from [this link](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) and [this link](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin) and store them in `utils/hubert_extraction`. Once downloaded, the following command can be run. Note: You may need to clone and install fairseq to run this step. Change the path where whisper samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to False.
- Run the following command to extract HuBERT units for audio files. The units will be stored at `runs/hubert_extraction/LJSpeech.txt`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```
- Note: HuBERT units have already been extracted for the corpus and are available at [this Google Drive link](https://drive.google.com/file/d/1h93GVPA4R1XZ53IphE3snZ3e5cDQ1mL1/view?usp=sharing). Download and save it at `runs/hubert_extraction/LJSpeech.txt`.

### Step 3: Create Training and Validation Files for Vocoder training on `LJSpeech` data

- Generate training and validation files for the vocoder:
    ```bash
    python utils/vocoder/preprocessor.py --input_file runs/hubert_extraction/LJSpeech.txt --root_path runs/vocoder/LJSpeech
    ```

### Step 4: Train HiFi-GAN Vocoder on `LJSpeech` data

- Set the number of GPUs in the `nproc_per_node` variable and run the following command:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 utils/vocoder/train.py --checkpoint_path runs/vocoder/LJSpeech/checkpoints --config utils/vocoder/LJSpeech_config.json
    ```
- Note: The LJSPeech vocoder pretrained model is available at [this Google Drive link](https://drive.google.com/file/d/10qDW4SjcUflC58u7E4uijZvgmFxVqV5K/view?usp=sharing). Download and save it at `runs/vocoder/LJSpeech/checkpoints/g_00200000`.

### Step 5: Extract HuBERT Units for `Whisper` samples from CSTR NAM TIMIT Plus corpus

- Run the following command to extract HuBERT units for `whisper` audio files. Change the path where `whisper` samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to False. The units will be stored at `runs/hubert_extraction/whisper.txt`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```
- Note: HuBERT units have already been extracted for the corpus and are available at [this Google Drive link](https://drive.google.com/file/d/1zY9oT221wcvMvyY5iTJ4_4EsSrLdEwfs/view?usp=sharing). Download and save it at `runs/hubert_extraction/whisper.txt`.

### Step 5: Infer using LJSpeech trained vocoder on HuBERT units extracted from `whisper` samples.

- Infer the LJSpeech trained vocoder. This creates speech using `whisper` content in `LJSpeech` voice. The generated files are available at `runs/data/simulated_gt_from_whisper_content_inLJSpeechvoice`:
    ```bash
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/LJSpeech/checkpoints -n 500 --input_code_file runs/hubert_extraction/whisper.txt --output_dir runs/data/simulated_gt_from_whisper_content_inLJSpeechvoice --config utils/vocoder/LJSpeech_config.json
    ```

### Step 6: Extract HuBERT Units for NAM samples from CSTR NAM TIMIT Plus corpus

- Run the following command to extract HuBERT units for `NAM` files. Change the path where whisper samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to False. The units will be stored at `runs/hubert_extraction/nam.txt`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```
- Note: HuBERT units have already been extracted for the corpus and are available at [this Google Drive link](https://drive.google.com/file/d/1K4jLmWfrvsb1Cq7Ghz7Ft0IfAqAbrunq/view?usp=sharing). Download and save it at `runs/hubert_extraction/nam.txt`.

### Step 7: Create Training and Validation Files for Vocoder training on `NAM` data

- Generate training and validation files for the vocoder:
    ```bash
    python utils/vocoder/preprocessor.py --input_file runs/hubert_extraction/nam.txt --root_path runs/vocoder/NAM
    ```

### Step 8: Train HiFi-GAN Vocoder on `NAM` samples from CSTR NAM TIMIT Plus corpus

- Set the number of GPUs in the `nproc_per_node` variable and run the following command:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 utils/vocoder/train.py --checkpoint_path runs/vocoder/NAM/checkpoints --config utils/vocoder/NAM_config.json
    ```

### Step 9: Infer using NAM trained vocoder on HuBERT units extracted from `LJSpeech` samples.

- Infer the NAM trained vocoder. This creates speech using `LJSpeech` content in `NAM` voice. The generated files are available at `runs/data/LJNAM`:
    ```bash
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/NAM/checkpoints -n 15000 --input_code_file runs/hubert_extraction/LJSpeech.txt --output_dir runs/data/LJNAM --config utils/vocoder/NAM_config.json
    ```

### Step 10: Extract HuBERT features for `LJSpeech`

- Run the following command to extract HuBERT features for `LJSpeech` files. Change the path where `LJSpeech` samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to True. The features will be stored at `runs/hubert_extraction/LJSpeech/`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```

### Step 11: Extract HuBERT features for `LJNAM`

- Run the following command to extract HuBERT features for `LJNAM` files. Change the path where `LJNAM` samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to True. The features will be stored at `runs/hubert_extraction/LJNAM/`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```

### Step 12: Extract HuBERT features for `NAM`

- Run the following command to extract HuBERT features for `LJSpeech` files. Change the path where `LJSpeech` samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to True. The features will be stored at `runs/hubert_extraction/nam/`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```

### Step 13: Extract HuBERT features for `simulated_gt_from_whisper_content_inLJSpeechvoice`

- Run the following command to extract HuBERT features for `simulated_gt_from_whisper_content_inLJSpeechvoice` files. Change the path where `simulated_gt_from_whisper_content_inLJSpeechvoice` samples are available in `dataset_dir` in `utils/hubert_extraction/hubert_config.yaml`. Also set `save_feature_only` to True. The features will be stored at `runs/hubert_extraction/simulated_gt_from_whisper_content_inLJSpeechvoice/`: 
    ```bash
    python utils/hubert_extraction/extractor.py utils/hubert_extraction/hubert_config.yaml
    ```

### Step 14: Create Files for Sequence-to-Sequence (Seq2Seq) Training

- Prepare the necessary files for training the Seq2Seq module:
    ```bash
    python utils/Seq2Seq/preprocessor.py utils/Seq2Seq/Seq2Seq_config.yaml
    ```

### Step 15: Train the Seq2Seq Module

- Train the Seq2Seq module using the following command:
    ```bash
    python train.py --config utils/Seq2Seq/Seq2Seq_config.yaml --num_gpus 1
    ```

### Step 16: Infer HuBERT embeddings from trained Seq2Seq module 

- Run inference to predict HuBERT from the trained Seq2Seq module:
    ```bash
    python inference.py --config utils/Seq2Seq/Seq2Seq_config.yaml --checkpoint_pth runs/Seq2Seq/ckpt/Seq2Seq_model-step=30000-val_total_loss_step=0.00.ckpt --device cuda:2
    ```

### Step 17: Derive HuBERT units from inferred HuBERT embeddings

- Run the following command to extract HuBERT units for `NAM` files. Change the path where `NAM` samples are available in `audio_dir` in `utils/hubert_extraction/unit_config.yaml`. Also provide path of predicted features from above step in `predicted_feat_dir` in the same config file: 
    ```bash
    python utils/hubert_extraction/unit_extractor.py utils/hubert_extraction/unit_config.yaml
    ```

### Step 18: Infer using `LJSpeech` trained vocoder on the predicted HuBERT units.

- Infer the LJSpeech trained vocoder. The generated files are available at `runs/data/predicted_features`.:
    ```bash
    python utils/vocoder/inference.py --checkpoint_file runs/vocoder/LJSpeech/checkpoints -n 500 --input_code_file runs/hubert_extraction/predicted_features.txt --output_dir runs/data/predicted_speech --config utils/vocoder/LJSpeech_config.json
    ```

### Step 19: Calculate Word Error Rate (WER) and Character Error Rate (CER)

- Calculater wer and cer on validation set of CSTR NAM TIMIT Plus corpus:
    ```bash
    python wer.py -o runs/data/predicted_speech/ -t runs/data/text -s runs/data/calculated_wer.txt
    ```

## Things-to-do

- The initial and trailing silence can be removed from the samples of the CSTR NAM TIMIT Plus corpus. However, this may misalign the data with the simulated ground truth. To address this, a DTW algorithm can be employed to temporally align the input NAM with the simulated targets for Seq2Seq training as shown in our paper, potentially enhancing performance.

## Cite
If you use this code, please cite the following paper:

Shah, N., Karande, S., Gandhi, V. (2024) Towards Improving NAM-to-Speech Synthesis Intelligibility using Self-Supervised Speech Models. Proc. Interspeech 2024, 2470-2474, doi: 10.21437/Interspeech.2024-672