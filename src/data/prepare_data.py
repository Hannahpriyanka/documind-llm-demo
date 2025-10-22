# src/data/prepare_data.py
import json
import argparse
import os

def prepare_data(input_file, output_file):
    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    data = []
    with open(input_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "text": obj.get("text", ""),
                "metadata": obj.get("metadata", {})
            })

    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Processed {len(data)} records into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--out', required=True, help="Output JSONL file")
    args = parser.parse_args()
    prepare_data(args.input, args.out)
