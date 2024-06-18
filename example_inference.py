POET_PATH = '/share/apps/genomics/PoET'
import argparse
from pathlib import Path
import string
import numpy as np
import torch
import sys
sys.path.append(f"{POET_PATH}/poet")
sys.path.append(POET_PATH)
from poet.alphabets import Uniprot21

from poet.models.poet import PoET
from scripts.score import (
    append_startstop,
    get_logps_tiered_fast,
    get_seqs_from_fastalike,
    jit_warmup,
)

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()

def get_encoded_msa_from_a3m_seqs(msa_sequences: list[bytes], alphabet: Uniprot21) -> np.ndarray:
    max_length = max(len(seq) for seq in msa_sequences)
    encoded_seqs = []
    for seq in msa_sequences:
        encoded_seq = alphabet.encode(seq.translate(None, delete=ASCII_LOWERCASE_BYTES))
        padded_seq = np.pad(encoded_seq, (0, max_length - len(encoded_seq)), constant_values=alphabet.gap_token)
        encoded_seqs.append(padded_seq)
    return np.stack(encoded_seqs)

def load_poet_model():
    # Load model checkpoint
    ckpt = torch.load(args.ckpt_path)
    model = PoET(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict({k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()})
    model = model.cuda().half().eval()
    alphabet = Uniprot21(include_gap=True, include_startstop=True, distinct_startstop=True)
    jit_warmup(model, alphabet)
    return model, alphabet

@torch.inference_mode()
def main(args):
    all_sequences = get_seqs_from_fastalike(args.fasta_path)
    prompt_seqs = all_sequences[:7]
    eval_seqs = all_sequences[7:10]
    model, alphabet = load_poet_model()
    prompt_encoded = get_encoded_msa_from_a3m_seqs(prompt_seqs, alphabet)
    logps = get_logps_tiered_fast(
        msa_sequences=prompt_encoded,
        variants=eval_seqs,
        model=model,
        batch_size=args.batch_size,
        alphabet=alphabet,
    )
    print(f"log likelihoods for {len(eval_seqs)} sequences:")
    print(logps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",
                        type=Path,
                        default="data/poet.ckpt"
                        )
    parser.add_argument("--fasta_path",
                        type=Path,
                        default="example_data/expasy_ec/1_1_1_3.fasta",
                        required=True
                        )
    args = parser.parse_args()
    main(args)