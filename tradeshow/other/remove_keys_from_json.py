import json
import argparse


def remove_keys_from_json(data, keys_to_remove_lower):
    if isinstance(data, dict):
        return {
            k: remove_keys_from_json(v, keys_to_remove_lower)
            for k, v in data.items()
            if k.lower() not in keys_to_remove_lower
        }
    elif isinstance(data, list):
        return [remove_keys_from_json(item, keys_to_remove_lower) for item in data]
    else:
        return data


def process_json(input_json_path, output_json_path, keys_txt_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    with open(keys_txt_path, "r") as f:
        keys_to_remove = set(line.strip().lower() for line in f if line.strip())

    cleaned_data = remove_keys_from_json(data, keys_to_remove)

    with open(output_json_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Remove specified keys from a JSON file (case-insensitive matching)."
    )
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("output_json", help="Path to the output JSON file")
    parser.add_argument(
        "keys_txt", help="Path to the text file containing keys to remove"
    )

    args = parser.parse_args()
    process_json(args.input_json, args.output_json, args.keys_txt)


if __name__ == "__main__":
    main()
