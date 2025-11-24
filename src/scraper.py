import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
import time

async def get_usd_jpy_rate_async(session):
    """
    Fetches live exchange rate asynchronously.
    """
    print("--- [FINANCE] Fetching Live Exchange Rate (Async) ---")
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                rate = data['rates']['JPY']
                print(f"--> Live Rate: 1 USD = {rate} JPY")
                return rate
    except Exception as e:
        print(f"--> Warning: API failed ({e}). Using fallback rate.")
    
    return 150.0

async def fetch_page_async(session, url, semaphore):
    """
    Fetches a single page respecting the semaphore limit.
    """
    async with semaphore:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/'
        }
        try:
            # Random sleep to act human even while async
            await asyncio.sleep(0.5) 
            
            async with session.get(url, headers=headers, timeout=20) as response:
                if response.status == 200:
                    html = await response.text()
                    return html
                else:
                    print(f"--> Error: Status {response.status} for {url}")
                    return None
        except Exception as e:
            print(f"--> Network Error on {url}: {e}")
            return None

def extract_price(container, text_content, exchange_rate):
    price_usd = 0
    price_tag = container.find(['p', 'span', 'div'], class_=re.compile(r'(price|fob)', re.IGNORECASE))
    if price_tag:
        raw_price = price_tag.get_text(strip=True)
        digits = re.sub(r'[^\d]', '', raw_price)
        if digits:
            price_usd = int(digits)

    if price_usd == 0:
        match = re.search(r'US\$\s*([\d,]+)', text_content)
        if match:
            price_clean = match.group(1).replace(',', '')
            price_usd = int(price_clean)

    if price_usd > 0:
        return int((price_usd * exchange_rate) / 1000)
    return 0

def parse_search_results(html_text, exchange_rate=150.0):
    if not html_text: return []
    
    soup = BeautifulSoup(html_text, 'html.parser')
    cars_data = []
    
    containers = soup.find_all(['li', 'div'], class_=re.compile(r'car-item'))
    if not containers:
        containers = soup.find_all('div', class_=re.compile(r'(product|item|listing)'))

    for container in containers:
        try:
            full_text = container.get_text(separator=' ', strip=True)
            
            link_tag = container.find('a', href=True)
            car_link = None
            if link_tag:
                href = link_tag['href']
                car_link = href if href.startswith('http') else f"https://www.tc-v.com{href}"

            price = extract_price(container, full_text, exchange_rate)
            
            grade = "Unknown"
            grade_tag = container.find('p', class_=re.compile(r'grade-wrap__grade'))
            if grade_tag:
                grade = grade_tag.get_text(strip=True)

            year = None
            year_match = re.search(r'\b(199\d|200\d|201\d|202\d)\b', full_text)
            if year_match:
                year = int(year_match.group(0))

            mileage = 0
            mile_match = re.search(r'([\d,]+)\s*km', full_text, re.IGNORECASE)
            if mile_match:
                mileage = int(mile_match.group(1).replace(',', ''))
            
            engine = 1300
            eng_match = re.search(r'([\d,]+)\s*cc', full_text, re.IGNORECASE)
            if eng_match:
                engine = int(eng_match.group(1).replace(',', ''))

            is_mt = bool(re.search(r'\b(MT|Manual|F5|F6)\b', full_text, re.IGNORECASE))
            is_4wd = bool(re.search(r'\b(4WD|4x4|AWD)\b', full_text, re.IGNORECASE))
            
            if price > 0 and year:
                cars_data.append({
                    'price': price,
                    'year': year,
                    'mileage': mileage,
                    'engine_capacity': engine,
                    'transmission': 'mt' if is_mt else 'at',
                    'drive': '4wd' if is_4wd else '2wd',
                    'grade': grade,
                    'mark': 'honda',
                    'model': 'fit',
                    'link': car_link
                })
        except Exception:
            continue

    return cars_data

async def scrape_listings_async_runner(base_url, max_pages, progress_callback):
    all_cars = []
    sem = asyncio.Semaphore(5)
    
    async with aiohttp.ClientSession() as session:
        rate = await get_usd_jpy_rate_async(session)
        
        # Progress Tracking Wrapper
        completed_tasks = 0
        
        async def monitored_fetch(url):
            nonlocal completed_tasks
            html = await fetch_page_async(session, url, sem)
            completed_tasks += 1
            
            if progress_callback:
                # Send COMPLETED count, not Page Number
                progress_callback(completed_tasks, max_pages)
            
            return html

        tasks = []
        for i in range(1, max_pages + 1):
            url = f"{base_url}?pn={i}"
            tasks.append(monitored_fetch(url))
        
        print(f"--- STARTING ASYNC SCRAPE ({max_pages} pages) ---")
        html_pages = await asyncio.gather(*tasks)
        
        for html in html_pages:
            if html:
                cars = parse_search_results(html, rate)
                all_cars.extend(cars)
                
    print(f"--- SCRAPING COMPLETE. Total Cars: {len(all_cars)} ---")
    return all_cars

def scrape_listings(base_url, max_pages=100, progress_callback=None):
    """
    Synchronous wrapper for the async engine.
    """
    return asyncio.run(scrape_listings_async_runner(base_url, max_pages, progress_callback))