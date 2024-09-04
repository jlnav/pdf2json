import argparse
from text_processing.json_structure import restructure_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure JSON file based on sub-sections.")
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output JSON file')
    
    args = parser.parse_args()
    restructure_json(args.input_file, args.output_file)
