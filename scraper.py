import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive"
}

def scrape_table(url, table_id):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "lxml")
    table = soup.find("table", {"id": table_id})
    if table is None:
        print(f"  Table '{table_id}' not found at {url}")
        return None
    df = pd.read_html(io.StringIO(str(table)))[0]
    df = df[df.iloc[:, 0] != df.columns[0]]
    df = df.dropna(how="all")
    return df

def clean_salary(val):
    try:
        return int(str(val).replace("$", "").replace(",", ""))
    except:
        return None

if __name__ == "__main__":
    print("Scraping per game stats...")
    per_game = scrape_table(
        "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html",
        "per_game_stats"
    )
    if per_game is not None:
        print(f"  Got {len(per_game)} records")
        per_game.to_csv("data/per_game_stats.csv", index=False)
    time.sleep(5)

    print("Scraping advanced stats...")
    advanced = scrape_table(
        "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html",
        "advanced_stats"
    )
    if advanced is not None:
        print(f"  Got {len(advanced)} records")
        advanced.to_csv("data/advanced_stats.csv", index=False)
    time.sleep(5)

    print("Scraping salary data...")
    salaries = scrape_table(
        "https://www.basketball-reference.com/contracts/players.html",
        "player-contracts"
    )
    if salaries is not None:
        print(f"  Got {len(salaries)} records")
        salaries.to_csv("data/salaries.csv", index=False)

    print("Done. Check the data/ folder for output files.")