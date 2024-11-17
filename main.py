import argparse
import os
from src.utils.config import load_config
from src.train import train_ast, train_sed
from src.inference import inference_ast, inference_sed


def parse_args():
    parser = argparse.ArgumentParser(description='ASVSpoof AIRI Competition')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                        help='Run mode: train or inference')
    parser.add_argument('--model_type', type=str, default='ast',
                        help='A type of a model: ["ast", "sed"]')
    parser.add_argument('--config', type=str, default='configs/ast_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for inference')
    parser.add_argument('--wandb-key', type=str, default=None,
                        help='WandB API key')
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config, args.model_config)

    if args.wandb_key:
        os.environ['WANDB_API_KEY'] = args.wandb_key

    if args.model_type == 'ast':
        if args.mode == 'train':
            train_ast(config)
        else:
            if args.checkpoint is None:
                raise ValueError('Checkpoint path is required for inference mode')
            inference_ast(config, args.checkpoint)
    elif args.model_type == 'sed':
        if args.mode == 'train':
            train_sed(config)
        else:
            if args.checkpoint is None:
                raise ValueError('Checkpoint path is required for inference mode')
            inference_sed(config, args.checkpoint)
    else:
        raise ValueError('A type of a model should be selected from ["ast", "sed"]')


if __name__ == '__main__':
    main()
