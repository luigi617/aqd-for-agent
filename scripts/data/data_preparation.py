import json
from datasets import Dataset
import argparse
import os


def main(args):
    raw_data = json.load(open(args.input_dataset_path))

    dataset = Dataset.from_list(raw_data)
    split_dataset = dataset.train_test_split(test_size=0.3, shuffle=True, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    test_split_dataset = test_dataset.train_test_split(test_size=0.5)
    eval_dataset = test_split_dataset['train']
    test_dataset = test_split_dataset['test']

    # save the json files
    json.dump(train_dataset.to_dict(), open(os.path.join(args.output_dataset_dir, "train_dataset.json"), 'w'), indent=4)
    json.dump(eval_dataset.to_dict(), open(os.path.join(args.output_dataset_dir, "eval_dataset.json"), 'w'), indent=4)
    json.dump(test_dataset.to_dict(), open(os.path.join(args.output_dataset_dir, "test_dataset.json"), 'w'), indent=4)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dataset_path", type=str, default="/local/data/shared/aqd/lucy/cleaned_data.json", help="path to the input dataset")
    args.add_argument("--output_dataset_dir", type=str, default="/home/yunanlu/code/aqd-for-agent/data", help="path to the output dataset directory")
    args = args.parse_args()

    if not os.path.exists(args.output_dataset_dir):
        os.makedirs(args.output_dataset_dir)

    main(args)

# python -m scripts.data.data_preparation --input_dataset_path /local/data/shared/aqd/lucy/cleaned_data.json --output_dataset_dir /home/yunanlu/code/aqd-for-agent/data
