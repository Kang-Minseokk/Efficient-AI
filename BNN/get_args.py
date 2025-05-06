import argparse


class ParamProcessor(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace, self.dest, [])
        if param_dict is None:
            param_dict = {}

        for value in values.split(","):
            k, v = value.split("=")
            param_dict[k] = v
        setattr(namespace, self.dest, param_dict)


def str2bool(s):
    return s.lower() == "true"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="MNIST", choices=["Digit", "MNIST", "FashionMNIST","CIFAR10", "CIFAR100", "TinyIN"])
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)


    parser.add_argument("--model", type=str, default="MLP", choices=["MLP", "VGG"])
    parser.add_argument("--vgg_mode", type=str, default="A")
    parser.add_argument("--b_act", type=str2bool, default=True)
    parser.add_argument("--b_weight", type=str2bool, default=True)

    parser.add_argument("--batch_norm", type=str2bool, default=True)
    parser.add_argument("--batch_norm_out", type=str2bool, default=False)
    parser.add_argument("--affine", type=str2bool, default=False)
    parser.add_argument("--bias", type=str2bool, default=False)

    parser.add_argument("--training_space", type=str, default="real", choices=["binary", "real"])
    parser.add_argument("--mask_type", type=str, default="EMP", choices=["EMP", "MMP", "RAND"])
    parser.add_argument("--scheduler_type", type=str, default="gaussian", choices=["const", "cosine", "gaussian"])

    # wandb
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--agent", action="store_true")

    args = parser.parse_args()
    return args
