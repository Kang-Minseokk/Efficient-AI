import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='binary', choices=['binary', 'trick_ternary', 'qat_ternary', 'fp16', 'mlp_mixer'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='fashionmnist', choices=['fashionmnist', 'cifar10'])
    
    parser.add_argument('--quantize_type', type=str, default='off', choices=['real', 'binary', 'qat_ternary', 'trick_ternary'])
    return parser.parse_args()
