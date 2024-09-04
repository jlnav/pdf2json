from setuptools import setup, find_packages

setup(
    name='text_processing_package',
    version='0.1',
    description='A Python package for extracting text from PDFs and restructuring JSON files by sections.',
    author='Nesar Ramachandra',
    author_email='nesar1202@gmail.com',
    packages=find_packages(),  # Automatically find all sub-packages
    install_requires=[
        'transformers',
        'torch',
        'Pillow',
        'fitz',
    ],
    entry_points={
        'console_scripts': [
            'pdf_to_text=scripts.run_pdf_to_text:main',
            'restructure_json=scripts.run_json_restructure:main',
        ],
    },
)
