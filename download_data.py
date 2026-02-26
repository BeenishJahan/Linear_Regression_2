import kagglehub


path = kagglehub.dataset_download("wordsforthewise/lending-club", output_dir="./data")

print("Path to dataset files:", path)
