from src.scraper import fetch_page, parse_search_results

#Target
target_url = "https://www.tc-v.com/used_car/honda/fit/"

html = fetch_page(target_url)

if html:
    data= parse_search_results(html)

    print(f"\nExtracted Data (first 3 cars):")
    for car in data[:3]:
        print(car)
    #Print the first 500 characters to prove
    print(html[:500])
else:
    print("Failed to fetch HTML")