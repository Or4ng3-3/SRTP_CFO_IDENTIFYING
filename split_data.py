import json
import random


def split_dataset(input_file, train_file, dev_file, test_file):
    # 1. 读取原始大文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]  # 假设你的源文件已经是 JSONL 格式

    total_len = len(data)
    print(f"loaded {total_len} lines。")

    # 2. 设定随机种子，保证每次打乱的结果一模一样！
    random.seed(42)
    random.shuffle(data)

    # 3. 计算切分点 (8:1:1)
    train_end = int(total_len * 0.8)  # 0 到 1600
    dev_end = train_end + int(total_len * 0.1)  # 1600 到 1800

    # 4. 执行切片
    train_data = data[:train_end]
    dev_data = data[train_end:dev_end]
    test_data = data[dev_end:]

    # 5. 保存为三个独立的文件
    def save_jsonl(subset, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_data, train_file)
    save_jsonl(dev_data, dev_file)
    save_jsonl(test_data, test_file)

    print(f"split completed")
    print(f" - Train: {len(train_data)} lines -> {train_file}")
    print(f" - Dev: {len(dev_data)} lines -> {dev_file}")
    print(f" - Test: {len(test_data)} lines -> {test_file}")


if __name__ == "__main__":
    # 假设你所有的 2000 条数据都在 all_data.json 里
    split_dataset("data.json", "train.json", "dev.json", "test.json")
