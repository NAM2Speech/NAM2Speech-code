import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLoss():
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, src_lens, target_lengths):
        # Temporarily disable deterministic algorithms
        torch.use_deterministic_algorithms(False)

        _loss = nn.functional.ctc_loss(
            input,
            target,
            src_lens,
            target_lengths,
            reduction='mean',  # Adjust reduction as necessary
            blank=0  # Specify your blank index if needed
        )

        # Optionally, you can set it back to true afterward if required
        # torch.use_deterministic_algorithms(True)

        return _loss

    def __call__(self, input: torch.Tensor, target: torch.Tensor, src_lens, target_lengths) -> torch.Tensor:
        return self.forward(input, target, src_lens, target_lengths)

class ModelLoss(nn.Module):
    def __init__(self, data_config):
        super().__init__()
        self.ctc_loss = CTCLoss()
        self.num_codes = data_config["preprocess"]["hubert_codes"]
        self.code_loss = nn.MSELoss()

    def forward(self, out, batch, src_lens, target_lengths, flattened_targets, log_probs):
        code_loss = self.code_loss(out, batch['gt_codes'])

        # CTC loss calculation
        ctcloss = self.ctc_loss(log_probs.transpose(0, 1), flattened_targets, src_lens, target_lengths)

        loss = code_loss + 0.0001 * ctcloss
        return loss, code_loss, ctcloss