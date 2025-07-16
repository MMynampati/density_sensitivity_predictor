import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from datetime import datetime
from scraper import scrape_table, clean_gmtkn_data
from helpers import get_seed_urls, get_seed_urls_test

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; GMTKN55Crawler/1.0)'
}


#seed_urls = ['http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/W4-11.html']
seed_urls = get_seed_urls()

attempted_page_count = 0
actual_page_count = 0 


def crawl(seed_urls):
    '''
    Goes through each subset URL, extracts links to the associated functional result pages, 
    scrapes tables from each, and returns a single merged dataframe.
    '''
    global attempted_page_count, actual_page_count
    master_df = pd.DataFrame()

    for subset_url in seed_urls:
        # Get subset name from URL (e.g., "W4-11" from ".../W4-11.html")
        subset_name = subset_url.split('/')[-1].replace('.html', '')

        # Fetch subset page and parse it
        response = requests.get(subset_url, HEADERS)
        if response.status_code != 200:
            print(f"Failed to access subset page: {subset_url}")
            continue

        print(f"Now checking subset_url: {subset_url}")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links to functional result pages
        links = soup.find_all('a', href=True)
        functional_urls = [link['href'] for link in links if 'result.html' in link['href']]

        # Ensure full URL paths
        base_url = '/'.join(subset_url.split('/')[:-1])
        full_functional_urls = [f"{base_url}/{relative}" for relative in functional_urls]

        for functional_url in full_functional_urls:
            functional_name = functional_url.split('/')[-2]  # e.g., "PBE" from ".../PBE/result.html"
            attempted_page_count += 1
            print(f"--Checking functional url #{attempted_page_count}: {functional_url}--")

            try:
                df = scrape_table(functional_url)
                df['functional'] = functional_name
                df['subset'] = subset_name
            
                # Optional: clean/normalize the scraped data
                if clean_gmtkn_data:
                    df = clean_gmtkn_data(df)

                # Append to master dataframe
                master_df = pd.concat([master_df, df], ignore_index=True)
                
                actual_page_count += 1

                #Politeness delay
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"Error scraping {functional_url}: {e}")
                continue

    return master_df


if __name__ == "__main__":
    df = crawl(seed_urls)
    print(df)
    print(f'attempted page count: {attempted_page_count}, actual_page_count: {actual_page_count}')
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"scraped_gmtkn_data_{timestamp}.csv"

    df.to_csv(filename, index=False)

