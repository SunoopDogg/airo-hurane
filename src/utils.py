import os
from pathlib import Path
from typing import List, Union


def get_files(file_path) -> list:
    """
    Get list of files from a directory or single file path.

    Args:
        file_path: Path to directory or file

    Returns:
        List of file paths
    """
    files = []
    if os.path.isdir(file_path):
        for entry in os.listdir(file_path):
            full_path = os.path.join(file_path, entry)
            if os.path.isfile(full_path):
                files.append(full_path)
    elif os.path.isfile(file_path):
        files.append(file_path)
    return files


def get_video_files(file_path: Union[str, Path], extensions: List[str] = None) -> List[str]:
    """
    Get list of video files from a directory or validate single video file.

    Args:
        file_path: Path to directory or video file
        extensions: List of valid video extensions (default: common video formats)

    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    file_path = Path(file_path)
    video_files = []

    if file_path.is_dir():
        for ext in extensions:
            video_files.extend([str(f) for f in file_path.glob(f'*{ext}')])
            video_files.extend([str(f) for f in file_path.glob(f'*{ext.upper()}')])
    elif file_path.is_file():
        if file_path.suffix.lower() in extensions:
            video_files.append(str(file_path))

    return sorted(video_files)


def validate_video_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a valid video file.

    Args:
        file_path: Path to video file

    Returns:
        True if valid video file, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    if not file_path.is_file():
        return False

    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    if file_path.suffix.lower() not in valid_extensions:
        return False

    return True


def get_output_path(input_path: Union[str, Path], output_dir: Union[str, Path], prefix: str = "annotated_") -> str:
    """
    Generate output path for processed files.

    Args:
        input_path: Input file path
        output_dir: Output directory path
        prefix: Prefix to add to output filename

    Returns:
        Output file path as string
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{prefix}{input_path.name}"
    output_path = output_dir / output_filename

    return str(output_path)
