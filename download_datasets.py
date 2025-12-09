import kagglehub
import os

try:
    path1 = kagglehub.dataset_download("mexwell/football-players-stats-2024-2025")
    print(f"Dataset 1 downloaded to: {path1}")
    for file in os.listdir(path1):
        file_path = os.path.join(path1, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({file_size:.2f} MB)")
except Exception as e:
    print(f"ERROR downloading Dataset 1: {e}")
    path1 = None

try:
    path2 = kagglehub.dataset_download("davidcariboo/player-scores")
    print(f"Dataset 2 downloaded to: {path2}")
    for file in os.listdir(path2):
        file_path = os.path.join(path2, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({file_size:.2f} MB)")
except Exception as e:
    print(f"ERROR downloading Dataset 2: {e}")
    path2 = None

if path1 and path2:
    print(f"Dataset 1 Location: {path1}")
    print(f"Dataset 2 Location: {path2}")
    with open("dataset_paths.txt", "w") as f:
        f.write(f"DATASET1_PATH={path1}\n")
        f.write(f"DATASET2_PATH={path2}\n")
    print("Paths saved to dataset_paths.txt")
else:
    print("Download failed")
