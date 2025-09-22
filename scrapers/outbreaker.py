import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

PAGE_URL = "https://idsp.mohfw.gov.in/index4.php?lang=1&level=0&linkid=406&lid=3689"

SAVE_DIR = "latest_weekly_outbreak"
os.makedirs(SAVE_DIR, exist_ok=True)

resp = requests.get(PAGE_URL)
soup = BeautifulSoup(resp.text, "html.parser")

pdf_links = [(a.text.strip(), urljoin(PAGE_URL, a["href"])) 
             for a in soup.find_all("a", href=True) if a["href"].endswith(".pdf")]

if not pdf_links:
    print("No PDF links found.")
    exit()

target_week = "31"
week_pdf = [(week, link) for week, link in pdf_links if target_week in week]

if not week_pdf:
    print(f"No PDF found for week {target_week}")
    exit()

latest_week, latest_pdf_url = week_pdf[0]

file_name = os.path.join(SAVE_DIR, f"{latest_week}_weekly_outbreak.outpdf")
print(f"Found PDF for week {latest_week}: {latest_pdf_url}")

pdf_resp = requests.get(latest_pdf_url)
with open(file_name, "wb") as f:
    f.write(pdf_resp.content)

print(f"Saved as {file_name}")
