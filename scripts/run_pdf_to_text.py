from text_processing.pdf_to_text import pdf2text, text2json

if __name__ == "__main__":
    filepath = 'data/sample.pdf'
    text_samples = pdf2text(filepath)
    fileOut = 'data/sample_output'
    text2json(text_samples, fileOut)
