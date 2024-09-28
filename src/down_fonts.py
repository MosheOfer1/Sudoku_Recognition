import os
import random
import requests
from dotenv import load_dotenv
from fontTools.ttLib import TTFont
import json

load_dotenv()
FONTS_KEY = os.getenv("FONTS_KEY")


def download_and_save_fonts(num_fonts=150):
    font_dir = "downloaded_fonts"
    os.makedirs(font_dir, exist_ok=True)

    fonts_info_file = os.path.join(font_dir, "fonts_info.json")

    if os.path.exists(fonts_info_file):
        with open(fonts_info_file, 'r') as f:
            fonts_info = json.load(f)
        print(f"Found {len(fonts_info)} previously downloaded fonts.")
        return [font['path'] for font in fonts_info]

    google_fonts_api_url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={FONTS_KEY}"
    response = requests.get(google_fonts_api_url)
    fonts_data = response.json()

    downloaded_fonts = []
    fonts_info = []
    for font in random.sample(fonts_data['items'], num_fonts):
        font_files = font['files']
        font_url = None

        # Try to get the 'regular' version first, then fall back to any other available version
        if 'regular' in font_files:
            font_url = font_files['regular']
        elif font_files:
            # Get the first available font file
            font_url = next(iter(font_files.values()))

        if not font_url:
            print(f"No suitable font file found for {font['family']}. Skipping.")
            continue

        font_name = font['family'].replace(' ', '_') + '.ttf'
        font_path = os.path.join(font_dir, font_name)

        if not os.path.exists(font_path):
            response = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(response.content)

            try:
                TTFont(font_path)
                print(f"Downloaded: {font_name}")
            except:
                os.remove(font_path)
                print(f"Failed to download or invalid font: {font_name}")
                continue
        else:
            print(f"Font already exists: {font_name}")

        downloaded_fonts.append(font_path)
        fonts_info.append({
            "name": font['family'],
            "path": font_path
        })

    with open(fonts_info_file, 'w') as f:
        json.dump(fonts_info, f)

    return downloaded_fonts


# Main execution
if __name__ == "__main__":
    # Download fonts and save them locally
    downloaded_fonts = download_and_save_fonts(50)
