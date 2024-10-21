import torch
from torch.utils.data import DataLoader
from modules import ParrotDataset, Parrot
import lightning as L
import yaml
import argparse
import os
import numpy as np
import shutil

class LitParrot(L.LightningModule):
    
    # define model architecture
    def __init__(
        self, data_config
    ):
        super().__init__()
        self.save_hyperparameters()
        self.parrot = Parrot(data_config)
    
    def infer(self, batch):
        self.eval()
        res = self.parrot.infer(batch)
        return res
    
def main(args):
    data_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    # setup datasets
    val_dataset = ParrotDataset("val", data_config=data_config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.collate_fn,
        num_workers=4,
    )
    
    # load checkpoint
    checkpoint = args.checkpoint_pth
    
    # init the model
    model = LitParrot.load_from_checkpoint(checkpoint,strict=False)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    
    # Move model to the correct device
    model = model.to(device)

    if os.path.exists(data_config["path"]["store_predicted_feats"]):
        shutil.rmtree(data_config["path"]["store_predicted_feats"])
    os.makedirs(data_config["path"]["store_predicted_feats"])

    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in val_loader:
                    
            # Move batch to the same device as the model
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            basename = batch["ids"][0]
            codes = model.infer(batch)[0].to('cpu').numpy()

            # Store the codes as a numpy file using the basename
            output_path = os.path.join(data_config["path"]["store_predicted_feats"], f"{basename}.npy")
            np.save(output_path, codes)
            print(f"Saved generated codes to: {output_path}")
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint_pth", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference")

    args = parser.parse_args()
    L.seed_everything(42, workers=True)

    main(args)