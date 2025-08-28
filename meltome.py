import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'valid': 'data/valid-00000-of-00001.parquet'}
train = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["train"])
train["split"] = "train"
test = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["test"])
test["split"] = "test"
valid = pd.read_parquet("hf://datasets/cradle-bio/meltome_cluster_split/" + splits["valid"])
valid["split"] = "valid"

df = pd.concat([train, test, valid])
df["ID"] = [f"P{i:05d}" for i in range(len(df))]
df.rename(columns={"target": "label"}, inplace=True)
df[["ID", "sequence", "label", "split"]].to_csv("/wibicomfs/STBS/roman/meltome_atlas.csv", index=False)
