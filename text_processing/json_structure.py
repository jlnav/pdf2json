import json
import re

def restructure_json(input_file, output_file):
    # Load the JSON data from the file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # New structure to hold the modified JSON
    new_data = []
    current_section = ""

    # Process each page in the original JSON
    for entry in data['instances']:
        text = entry['text']
        # Split the text into sub-sections based on '###' and '\n##'
        sub_sections = re.split(r'###|\n##', text)

        # Add first part to the current_section if it's a continuation
        if current_section:
            current_section += " " + sub_sections[0].strip()
        else:
            current_section = sub_sections[0].strip()

        # Add full entries for all full sub-sections
        for sub_section in sub_sections[1:]:
            if current_section:
                new_data.append({"text": current_section})
            current_section = sub_section.strip()

    # Add the last section if it's non-empty
    if current_section:
        new_data.append({"text": current_section})

    # Save the new structured data to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)