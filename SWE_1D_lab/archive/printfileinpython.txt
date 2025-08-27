import os

def print_file(head: str, foot: str, filename: str, format_str: str, data, path: str):
    """
    Writes formatted data to a text file with optional header and footer.

    Parameters:
        head (str): Header string to write at the top of the file.
        foot (str): Footer string to write at the bottom of the file.
        filename (str): Name of the output file.
        format_str (str): Format string for each data row (e.g., '%.4f\\n').
        data (array-like): Data to write to the file.
        path (str): Directory where the file will be saved.
    """
    print(f"Writing file {filename} ...")

    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)

    with open(full_path, 'w') as f:
        f.write(head)
        for row in data:
            f.write(format_str % row)
        f.write(foot)

    print("File written.")