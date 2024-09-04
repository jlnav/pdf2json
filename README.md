
# Text Processing Package

This package provides utilities to extract text from PDFs and restructure JSON files based on sections. It is designed to handle academic papers and other structured PDFs, allowing the user to process and analyze large documents efficiently.

## Features

- **PDF to Text Extraction**: Convert PDFs into structured text data using the `pdf2text` function. The package uses OCR techniques with models from Hugging Face to extract text from each page of the PDF.
- **Text Restructuring**: The package allows you to restructure the extracted text into sections and save them in a cleaner JSON format, making it easier to work with structured documents.

## Installation

To install the package, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/nesar/pdf2json.git
    ```

2. Navigate to the root of the project:

    ```bash
    cd pdf2json
    ```

3. (Optional) Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

4. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Install the package in editable mode:

    ```bash
    pip install -e .
    ```

## Usage

### Extract Text from PDF

You can extract text from a PDF using the \`run_pdf_to_text.py\` script. The output will be stored as JSON files. Example:

```bash
python scripts/run_pdf_to_text.py
```

This will extract text from the specified PDF file and save the results to a JSON file.

### Restructure JSON Based on Sections

To restructure the JSON files based on section markers, use the \`run_json_restructure.py\` script. Example:

```bash
python scripts/run_json_restructure.py input.json output.json
```

This will restructure the contents of \`input.json\` into cleaner sections and save the result in \`output.json\`.

## Project Structure

```
text_processing_package/
│
├── text_processing/         # Main package directory
│   ├── __init__.py
│   ├── pdf_to_text.py       # Module for PDF to text extraction
│   ├── json_structure.py    # Module for restructuring JSON files
│
├── tests/                   # Unit tests for the package
│   ├── test_pdf_to_text.py
│   ├── test_json_structure.py
│
├── scripts/                 # Example scripts for using the package
│   ├── run_pdf_to_text.py   # Script to extract text from PDFs
│   ├── run_json_restructure.py  # Script to restructure JSON files
│
├── data/                    # Sample data for testing
│   ├── sample.pdf
│   ├── sample_output.json
│
├── README.md                # Documentation
├── setup.py                 # Package setup script
├── requirements.txt         # Package dependencies
└── .gitignore               # Ignore files for Git
```

## Dependencies

- `transformers`
- `torch`
- `Pillow`
- `PyMuPDF (fitz)`

These can be installed automatically by running:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.
