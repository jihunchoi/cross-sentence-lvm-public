import argparse
import random
from pathlib import Path


random.seed(2019)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--num-labeled-data', required=True, type=int)
    parser.add_argument('--save', required=True)
    args = parser.parse_args()

    lines = []
    with open(args.data, 'r') as f:
        for line in f:
            lines.append(line)
    random.shuffle(lines)

    labeled_data = lines[:args.num_labeled_data]
    unlabeled_data = lines[args.num_labeled_data:]

    save_dir = Path(args.save)
    save_dir.mkdir(exist_ok=True, parents=True)
    labeled_path = save_dir / 'labeled.txt'
    unlabeled_path = save_dir / 'unlabeled.txt'

    for data, path in zip([labeled_data, unlabeled_data],
                          [labeled_path, unlabeled_path]):
        with open(path, 'w') as f:
            for d in data:
                f.write(d)


if __name__ == '__main__':
    main()
