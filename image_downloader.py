# -----------------------------
# pip install icrawler
# -----------------------------
from icrawler.builtin import GoogleImageCrawler
import os

# -----------------------------
# Config
# -----------------------------
breeds = [
    'Vechur', 'Mehsana', 'Hallikar', 'Amritmahal', 'Kankrej', 'Sahiwal', 'Surti',
    'Pulikulam', 'Nagpuri', 'Nagori', 'Malnad_gidda', 'Dangi', 'Murrah', 'Jaffrabadi',
    'Krishna_Valley', 'Kherigarh', 'Rathi', 'Khillari', 'Bargur', 'Banni', 'Toda',
    'Deoni', 'Kangayam', 'Kenkatha', 'Kasargod', 'Nimari', 'Tharparkar', 'Bhadawari',
    'Ongole', 'Red_Sindhi', 'Hariana', 'Umblachery', 'Gir', 'Nili_Ravi'
]

root_dir = "new/cow_images"
download_limit = 600  # Number of images per breed

os.makedirs(root_dir, exist_ok=True)

# -----------------------------
# Download function
# -----------------------------
def download_breed(breed):
    keyword = f"{breed} cow"  # add 'cow' to each breed
    dest = os.path.join(root_dir, breed.replace(" ", "_"))
    os.makedirs(dest, exist_ok=True)
    
    print(f"🔍 Downloading images for: {keyword}")
    crawler = GoogleImageCrawler(storage={'root_dir': dest})
    crawler.crawl(
        keyword=keyword,
        max_num=download_limit,
        min_size=(200, 200),
        overwrite=False
    )
    print(f"✅ Completed: {breed}")

# -----------------------------
# Run for all breeds
# -----------------------------
for breed in breeds:
    download_breed(breed)

print("\n🎉 All breed images downloaded successfully!")
