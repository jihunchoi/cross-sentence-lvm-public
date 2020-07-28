import argparse
import random
from pathlib import Path


random.seed(2019)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--save', required=True)
    args = parser.parse_args()

    samples = []
    with open(args.data, 'r') as f:
        for line in f:
            label, s1, s2, pair_id = line.split('\t')
            samples.append((label.strip(), s1.strip(), s2.strip(),
                            pair_id.strip()))
            samples.append((label.strip(), s2.strip(), s1.strip(),
                            f'{pair_id.strip()}_D'))

    save_path = Path(args.save)
    save_dir = save_path.parent
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(args.save, 'w') as f:
        for d in samples:
            f.write(f'{d[0]}\t{d[1]}\t{d[2]}\t{d[3]}\n')


if __name__ == '__main__':
    main()
