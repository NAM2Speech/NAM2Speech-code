import gdown
import os
import zipfile

# URLs and File IDs for the files to download
file_id_whisper_zip = "12fFd4yThDWTpzEUsm9_pRKA0Rn4U7hMJ"  # ZIP file
file_id_nam_zip = "1zUyx2mlGftI-2jutnhgzdnBUjr12gCE-"
file_id_ljspeech_zip = "1X1twVbL6avBGHiMcardfRH6hvSZKutiG"
file_id_txt = "1DBxoApvpp_XeHdHpO_hvaag9YBMkGI8Y"  # Text file

# Download URLs
url_whisper_zip = f"https://drive.google.com/uc?id={file_id_whisper_zip}&export=download"
url_nam_zip = f"https://drive.google.com/uc?id={file_id_nam_zip}&export=download"
url_ljspeech_zip = f"https://drive.google.com/uc?id={file_id_ljspeech_zip}&export=download"
url_txt = f"https://drive.google.com/uc?id={file_id_txt}&export=download"

# Paths to save files
whisper_path = "runs/data/whisper.zip"  # Path where the zip file will be saved
nam_path = "runs/data/nam.zip"  # Path where the zip file will be saved
ljspeech_path = "runs/data/ljspeech.zip"  # Path where the zip file will be saved
text_path = "runs/data/text.zip"          # Path where the text file will be saved
extract_dir = "runs/data"    # Directory where contents will be extracted

# Ensure the directories exist
os.makedirs(os.path.dirname(whisper_path), exist_ok=True)

# Download the ZIP file
gdown.download(url_whisper_zip, whisper_path, quiet=False)
print(f"Downloaded whisper ZIP file and saved to {whisper_path}")

gdown.download(url_nam_zip, nam_path, quiet=False)
print(f"Downloaded NAM ZIP file and saved to {nam_path}")

gdown.download(url_ljspeech_zip, ljspeech_path, quiet=False)
print(f"Downloaded NAM ZIP file and saved to {ljspeech_path}")

# Download the text file
gdown.download(url_txt, text_path, quiet=False)
print(f"Downloaded text file and saved to {text_path}")

# Unzip the ZIP file
with zipfile.ZipFile(ljspeech_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted LJSpeech data to {ljspeech_path}")
os.rename(extract_dir+'/wavs', extract_dir+'/LJSpeech')

# Unzip the ZIP file
with zipfile.ZipFile(whisper_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted whisper data to {extract_dir}")

# Unzip the ZIP file
with zipfile.ZipFile(nam_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted whisper data to {nam_path}")

# Unzip the text file if it's a zip file
with zipfile.ZipFile(text_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted text data to {extract_dir}")

# Remove the downloaded ZIP files
os.remove(whisper_path)
os.remove(nam_path)
os.remove(ljspeech_path)
os.remove(text_path)
print(f"Removed temporary files: {whisper_path}, {nam_path}, {ljspeech_path} and {text_path}")