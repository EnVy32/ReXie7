import requests
from bs4 import BeautifulSoup
import re

def fetch_page(url):
    """
    Downloads the raw HTML content of a website.
    Uses a fake User-Agent to avoid detection/blocking.
    """
    print(f"--- FETCHING: {url} ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_search_results(html_text):
    """
    Parses HTML based on the 'info-area__main-info' structure.
    Target: div.info-area__main-info -> Price and Mileage.
    Context: Year is in a preceding span.title__year.
    """
    print("--- PARSING HTML (Targeted Strategy) ---")
    soup = BeautifulSoup(html_text, 'html.parser')
    
    cars_data = []
    
    # 1. Find all main info containers (The double underscore is key!)
    info_containers = soup.find_all('div', class_='info-area__main-info')
    print(f"--> Found {len(info_containers)} info blocks.")
    
    for container in info_containers:
        try:
            # --- EXTRACT PRICE ---
            # We look for the box that says "FOB Price"
            price = 0
            # Find all small boxes inside this container
            boxes = container.find_all('div', class_='main-info__box')
            
            for box in boxes:
                title = box.find('div', class_='box__title')
                body = box.find('div', class_='box__body')
                
                if title and body:
                    title_text = title.get_text(strip=True)
                    body_text = body.get_text(strip=True)
                    
                    if "FOB Price" in title_text:
                        # Clean string: "US$1,595" -> 1595
                        price_str = re.sub(r'[^\d]', '', body_text)
                        price = int(price_str) if price_str else 0
                        
                        # CONVERT USD to JPY (Approximate Exchange Rate: 1 USD = 150 JPY)
                        # Our model uses '000s JPY (e.g. 200 = 200,000)
                        # So: 1595 USD * 150 = 239,250 JPY -> ~239 in our scale
                        price = int((price * 150) / 1000)

            # --- EXTRACT MILEAGE ---
            # Mileage is usually in another box in the same container, or we text-mine the container
            mileage = 0
            text_content = container.get_text(separator=' ', strip=True)
            mileage_match = re.search(r'(\d{1,3}(,\d{3})*)\s*km', text_content, re.IGNORECASE)
            if mileage_match:
                mileage = int(mileage_match.group(1).replace(',', ''))

            # --- EXTRACT YEAR ---
            # The Year is NOT in 'info-area__main-info'. It is in the parent/sibling structure.
            # Strategy: Look at the HTML context. The container is usually inside a 'car-item'.
            # We will try to find the closest 'title__year' preceding this info block.
            
            year = None
            # Traverse up to finding the car card, then search down for year
            # This part depends heavily on structure, but let's try finding the preceding 'title__year'
            
            # Alternative: Search specifically for title__year globally and match by index? 
            # Risk: Mismatch if counts differ.
            
            # Best Local Strategy: Go to parent's parent and find 'title__year'
            # (Based on your snippet: ...</a></div></div><div class="info-area__main-info">)
            try:
                # Go up 2 levels (div -> div -> maybe parent)
                parent = container.find_parent()
                if parent:
                    grandparent = parent.find_parent()
                    if grandparent:
                        year_tag = grandparent.find('span', class_='title__year')
                        if year_tag:
                            year = int(year_tag.get_text(strip=True))
            except:
                pass

            # --- DEFAULT ENGINE & SPECS ---
            # If we can't scrape them easily, we use defaults for Honda Fit
            engine = 1300
            transmission = 'at' # Most JP imports are AT
            drive = '2wd'

            # Only add valid data
            if year and price > 0:
                cars_data.append({
                    'price': price,        # Now in '000 JPY
                    'year': year,
                    'mileage': mileage if mileage else 50000, # Default if missing
                    'engine_capacity': engine,
                    'transmission': transmission,
                    'drive': drive,
                    'mark': 'honda',
                    'model': 'fit'
                })
                
        except Exception as e:
            # print(f"Debug Error: {e}") # Uncomment to see details
            continue
            
    return cars_data