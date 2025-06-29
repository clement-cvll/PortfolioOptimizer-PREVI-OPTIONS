import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from bs4 import BeautifulSoup


# Function to fetch unit data from the official website
def fetch_unit_data_from_official_website() -> pd.DataFrame:
    url = "https://www.previ-direct.com/web/eclient-suravenir/perf-uc-previ-options"

    # Fetch the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    data = []

    # Scrap name and ISIN of each unit
    rows = soup.find_all("tr", class_="portlet-section-alternate results-row")
    for row in rows:
        tds = row.find_all("td")
        if len(tds) >= 2:
            a1 = tds[0].find("a")
            a2 = tds[1].find("a")
            unit_isin = a1.text.strip() if a1 else None
            unit_name = a2.text.strip() if a2 else None
            if unit_isin and unit_name:
                data.append({"unit_isin": unit_isin, "unit_name": unit_name})

    # Get all unique tickers from the scraped data
    info = []
    for unit in tqdm(data, desc="Fetching tickers"):
        info.extend(yf.Lookup(unit["unit_name"]).all.index)
        info.extend(yf.Lookup(unit["unit_isin"]).all.index)
    info = list(set(info))
    print("Number of unique tickers:", len(info))

    # Create a DataFrame with the unique tickers
    pre_df = {}
    for ticker in tqdm(info, desc="Fetching ticker details"):
        pre_df[ticker] = {
            "region": "",
            "currency": "",
            "name": "",
            "size_quotes_10y": 0
        }
        detail = yf.Ticker(ticker).info
        pre_df[ticker]["region"] = detail.get("region", "")
        pre_df[ticker]["currency"] = detail.get("currency", "")
        pre_df[ticker]["name"] = detail.get("shortName", "").replace('"', "")
        pre_df[ticker]["long_name"] = detail.get("longName", "").replace('"', "")
        pre_df[ticker]["size_quotes_10y"] = len(yf.Ticker(ticker).history(period="10y")) if yf.Ticker(ticker).history(period="10y") is not None else 0
    df = pd.DataFrame.from_dict(pre_df, orient="index")
    df = df.loc[df.groupby("long_name")["size_quotes_10y"].idxmax()]
    df.drop(df[df["currency"] != "EUR"].index, inplace=True)
    df.drop(df[df["size_quotes_10y"] < 252].index, inplace=True)
    df.drop_duplicates(subset=["name"], inplace=True)
    df.drop(columns=["currency", "region", "name", "size_quotes_10y"], inplace=True)
    df.rename(columns={"long_name": "name"}, inplace=True)
    return df


if __name__ == "__main__":
    # Example usage
    df = fetch_unit_data_from_official_website()
    df.to_csv("tickers.csv", index=True, index_label="ticker")
    print("Fetched Tickers:", df)
