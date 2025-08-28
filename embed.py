import pickle
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_model(model_path: str, data_path: Path, output_path: Path):
    """
    Run ESM model to extract embeddings for sequences in the given data path.

    :param model_name: Name of the ESM model to use.
    :param num_layers: Number of layers to extract embeddings from.
    :param data_path: Path to the CSV file containing sequences.
    :param output_path: Path to save the extracted embeddings.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path).to(DEVICE).eval()

    for idx, seq in tqdm(data[["ID", "sequence"]].values):
        inputs = tokenizer(seq[:1022])
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(inputs["input_ids"]).reshape(1, -1).to(DEVICE),
                attention_mask=torch.tensor(inputs["attention_mask"]).reshape(1, -1).to(DEVICE),
                output_attentions=False,
            )
            del inputs
            with open(output_path / f"{idx}.pkl", "wb") as f:
                pickle.dump(outputs.last_hidden_state[0, 1: -1].cpu().numpy().mean(axis=0), f)
        del outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ESM model to extract embeddings.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the ESM model directory.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the CSV file containing sequences.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the extracted embeddings.")
    
    args = parser.parse_args()
    
    run_model(args.model_path, args.data_path, args.output_path)
