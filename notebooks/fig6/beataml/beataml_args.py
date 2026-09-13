import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune on BeatAML')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--split_method", type=str, default="patient", choices=["drug", "patient"])
    parser.add_argument("--freeze_level", type=str, default="madrigal", choices=["madrigal", "encoder_only", "none"])
    parser.add_argument("--drug_combo_dim", type=int, default=128)
    parser.add_argument("--scorer_hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--scorer_dropout", type=float, default=0.2)
    parser.add_argument("--use_rnaseq_profile", action="store_true")
    parser.add_argument("--use_mut_profile", action="store_true")
    parser.add_argument("--use_clin_feature", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    return args
    
