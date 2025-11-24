from src.scraper import fetch_page
import re

#Target URL
url = "https://www.tc-v.com/used_car/honda/fit/"

print("---DIAGNOSTIC RUN---")

#Fetch the raw HTML
html = fetch_page(url)

if html:

    #Save it to a file so we can inspect it manually if needed
    with open("bot_view.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("--> HTML saved to 'bot_view.html'.")

    #Check if we are blocked
    #Automated investigation

    print("\n---CONTENT ANALYSIS---")

    if "Fit" in html:
        print("✅ Word 'Fit' found. We are on the right page.")
    else:
        print("❌ Word 'Fit' NOT found. We might be on a Captcha/Block page.")
    
    #Check for any price-like patterns
    price_pattern = re.compile(r'US\$\s*[\d,]+')
    matches = price_pattern.findall(html)

    if matches:
        print(f"✅ Found {len(matches)} price tags (e.g., {matches[0]}).")

        #KEY STEP: find the container of the first price
        #Split the HTML to find the context of the first price
        part = html.split(matches[0])[0]
        #Print the last 200 characters before the price to see the tags
        print("\n---HTML CONTEXT (Tags around the price)---")
        print(part[-300:] + "[[[PRICE HERE]]]")
    else:
        print("❌ No prices found. The page might be rendering via JavaScript.")

else:
    print("Failed to fetch page.")

    