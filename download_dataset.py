import gdown
import zipfile
import os

file_id = "1vIeDYgb1zQcnnWIWUU0Jz4SsQsaet6xT"
output = "dataset.zip"

gdown.download(id=file_id, output=output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(".")

os.remove(output)

print("✅ Dataset downloaded and extracted successfully.")
