import re
import matplotlib.pyplot as plt

def main():
    log_path = "../result/AdanBNN_BinSFO_400.out"
    with open(log_path, "r") as f:
        content = f.read()

    epochs, train_accs, valid_accs = [], [], []

    for block in re.findall(r"Epoch: \d+.*?(?=Epoch: \d+|$)", content, flags=re.S):
        e = re.search(r"Epoch: (\d+)", block)
        tr = re.search(r"train accuracy \(top-1\)\s*:\s*([\d.]+)", block)
        va = re.search(r"valid accuracy \(top-1\)\s*:\s*([\d.]+)", block)
        if e and tr and va:
            epochs.append(int(e.group(1)))
            train_accs.append(float(tr.group(1)))
            valid_accs.append(float(va.group(1)))

    plt.plot(epochs, train_accs, label="Train Acc", marker='o')
    plt.plot(epochs, valid_accs, label="Valid Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.title("AdamBNN BinSFO w Ternary Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("AdamBNN_BinSFO_400_ternary.png")
    print("그래프 저장 완료: AdamBNN_BinSFO_400_ternary.png")

if __name__ == "__main__":
    main()
