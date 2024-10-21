import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from modules.fft import FFTBlock, SinusoidalPosEmb
from modules.loss import ModelLoss

class Parrot(nn.Module):
    def __init__(self, data_config):
        super().__init__()
        self.max_len = data_config["transformer"]["max_len"]
        self.d_model = data_config["transformer"]["d_model"]
        transformer_config = data_config["transformer"]

        # map input 768 dim HuBERT to 256 dim
        self.enclin = nn.Linear(768,data_config["transformer"]["d_model"])

        # map encoder output 256 dim to 768 dim CTC tokens
        self.ctc_fc = nn.Linear(data_config["transformer"]["d_model"], 768)
        self.dropout = nn.Dropout(data_config["transformer"]["decoder"]["dropout_p"])
        with open(os.path.join(data_config["path"]["root_path"],"vocab_character.json"), 'r') as json_file:
            data = json.load(json_file)
        length = len(data)
        self.lm_head = nn.Linear(768, length) 

        self.pos_emb = SinusoidalPosEmb(self.max_len, self.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    transformer_config["encoder"]["n_head"],
                    transformer_config["conv_n_filter"],
                    transformer_config["conv_kernel_sizes"],
                    transformer_config["encoder"]["dropout_p"]
                ) for n in range(transformer_config["encoder"]["n_layer"])
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model,
                    transformer_config["decoder"]["n_head"],
                    transformer_config["conv_n_filter"],
                    transformer_config["conv_kernel_sizes"],
                    transformer_config["decoder"]["dropout_p"]
                ) for n in range(transformer_config["decoder"]["n_layer"])
            ]
        )

        self.head = nn.Linear(self.d_model, data_config["preprocess"]["hubert_codes"])
    
    def forward_encoder(self, x, mask):
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, key_padding_mask=mask)
        return x

    def forward_decoder(self, x, mask):
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, key_padding_mask=mask)
        return x


    def forward(self, batch, inference=False):

        out = batch["nam_codes"]
        
        # pass through linear layer that brings 768 dim to 256 dimension
        out = self.enclin(out) 

        # Encoder block
        out = self.pos_emb(out)
        out = self.forward_encoder(out, ~batch['src_mask'])

        # CTC block
        output_forctc = self.ctc_fc(out)      
        if inference == False:
            
            ctc_labels = batch["ctc_labels"]
            ctc_labels = torch.from_numpy(ctc_labels)

            hidden_states = output_forctc  #[0]
            hidden_states = self.dropout(hidden_states)

            # each element is True if the corresponding element in ctc_labels is greater than zero
            labels_mask = ctc_labels > 0  #>= 0
            target_lengths = labels_mask.sum(-1)

            # masked_select selects the elements from ctc_labels where labels_mask is True
            # a 1-dimensional tensor containing the non-padding target labels.
            flattened_targets = ctc_labels.masked_select(labels_mask)
            
            # map the intermediate hidden states to CTC logits, which will be used to calculate the CTC loss.
            ctc_logits = self.lm_head(hidden_states)

            # converts the logits into log probabilities.
            log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32)

        # Decoder block
        out = self.pos_emb(out)    
        out = self.forward_decoder(out, ~batch['tgt_mask'])

        # Linear head after Decoder to map to 768 dim output 
        out = self.head(out)

        if inference == False:
            src_lens = torch.from_numpy(batch["src_lens"])
            return (out, batch['src_mask'], ['tgt_mask'], src_lens, target_lengths, flattened_targets, log_probs)
        else:
            return (out, batch['src_mask'], ['tgt_mask'])

    def infer(self, batch):
        assert self.training == False
        out, _, tgt_mask = self.forward(batch, inference=True)
        return out