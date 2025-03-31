#!/usr/bin/env python3
import os
import sys

def delete_png_files(root_dir):
    # Walk through all subdirectories of the given root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file ends with '.png' (case-insensitive)
            if filename.lower().endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":    
    directory = "logs"
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    delete_png_files(directory)
