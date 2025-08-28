from pathlib import Path

import numpy as np

BLUB = {
    0.1: "uniref50_0",
    0.2: "uniref50_10",
    0.3: "uniref50_20",
    0.4: "uniref50_30",
    0.5: "uniref50_40",
    0.6: "uniref50_50",
    0.7: "uniref50_60",
    0.8: "uniref50_70",
    0.9: "uniref50_80",
    1.0: "uniref50_90",
    1.1: "uniref50_100",
}


def parse_fasta(path):
    """
    Parse a FASTA file and do some validity checks if requested.

    Args:
        path: Path to the FASTA file

    Returns:
        Dictionary mapping sequences IDs to amino acid sequences
    """
    print(f"Parsing {path}...")
    seq_map = {}

    with open(path, "r") as fasta:
        for line in fasta.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '>':
                entry_id = line[1:]
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map


def collect_seqs(max_sim):
    seqs = dict()
    for s in BLUB:
        if s <= max_sim:
            seqs.update(parse_fasta(f"/wibicomfs/STBS/roman/{BLUB[s]}.fasta"))
    return seqs


def sample(max_sim, num_seqs, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    seqs = list(collect_seqs(max_sim).items())

    with open(output_file, "w") as f:
        for pos in np.random.choice(len(seqs), num_seqs, replace=False):
            f.write(f"{seqs[pos][1][:1022]}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or (len(sys.argv[1]) >= 2 and sys.argv[1] in {"-h", "--help"}):
        print("Usage: python sample.py <max_sim> <num_seqs> <output_file>")
        sys.exit(0)

    max_sim = float(sys.argv[1])
    num_seqs = int(sys.argv[2])
    output_file = Path(sys.argv[3])

    sample(max_sim, num_seqs, output_file)
    print(f"Sampled {num_seqs} sequences with maximum similarity {max_sim} to {output_file}.")
