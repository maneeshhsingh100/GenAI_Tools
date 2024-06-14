import os

def read_text_from_file(filepath):
    """Read text from a file and return it as a string."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as file:
        text = file.read()
    return text
