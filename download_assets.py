import os
import requests
import re
from urllib.parse import urljoin

STATIC_DIR = os.path.join("k:", os.sep, "kisan_django", "static")
FONTS_DIR = os.path.join(STATIC_DIR, "fonts")
FA_DIR = os.path.join(STATIC_DIR, "vendor", "fontawesome")
FA_CSS_DIR = os.path.join(FA_DIR, "css")
FA_WEBFONTS_DIR = os.path.join(FA_DIR, "webfonts")

os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(FA_CSS_DIR, exist_ok=True)
os.makedirs(FA_WEBFONTS_DIR, exist_ok=True)

# 1. Download FontAwesome
print("Downloading FontAwesome...")
fa_css_url = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
fa_css = requests.get(fa_css_url).text
with open(os.path.join(FA_CSS_DIR, "all.min.css"), "w", encoding="utf-8") as f:
    f.write(fa_css)

# Extract webfonts from FontAwesome CSS
print("Downloading FontAwesome Webfonts...")
webfont_urls = re.findall(r'url\(([^)]+\.(?:woff2|woff|ttf))\)', fa_css)
for url in set(webfont_urls):
    url = url.strip("'\"")
    # URL might be relative like '../webfonts/fa-solid-900.woff2'
    full_url = urljoin(fa_css_url, url)
    filename = url.split('/')[-1].split('?')[0].split('#')[0]
    print(f"  Downloading {filename}...")
    try:
        content = requests.get(full_url).content
        with open(os.path.join(FA_WEBFONTS_DIR, filename), "wb") as f:
            f.write(content)
    except Exception as e:
        print(f"Failed to download {full_url}: {e}")

# 2. Download Google Fonts
print("Downloading Google Fonts...")
gfonts_urls = [
    "https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
]

# We must send a User-Agent that gets woff2
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

all_css = []
font_counter = 1

for gurl in gfonts_urls:
    css = requests.get(gurl, headers=headers).text
    # find all woff2 urls
    urls = re.findall(r'url\((https://[^)]+\.woff2)\)', css)
    for u in set(urls):
        filename = f"font_{font_counter}.woff2"
        font_counter += 1
        print(f"  Downloading {filename} from {u}...")
        try:
            content = requests.get(u).content
            with open(os.path.join(FONTS_DIR, filename), "wb") as f:
                f.write(content)
            # replace url in css with local url
            css = css.replace(u, f"./{filename}")
        except Exception as e:
            print(f"Failed to download {u}: {e}")
    all_css.append(css)

with open(os.path.join(FONTS_DIR, "fonts.css"), "w", encoding="utf-8") as f:
    f.write("\n".join(all_css))

print("Assets downloaded successfully.")
