import os


def get_files(file_path) -> list:
    files = []
    if os.path.isdir(file_path):
        for entry in os.listdir(file_path):
            full_path = os.path.join(file_path, entry)
            if os.path.isfile(full_path):
                files.append(full_path)
    elif os.path.isfile(file_path):
        files.append(file_path)
    return files
