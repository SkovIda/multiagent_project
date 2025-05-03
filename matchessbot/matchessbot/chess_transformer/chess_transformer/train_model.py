import torch

from config import import_config
from dataset import ChessDataset
from model import ChessTransformerEncoder


if __name__ == '__main__':
    config = import_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ChessTransformerEncoder(config).to()
    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters()])
    )