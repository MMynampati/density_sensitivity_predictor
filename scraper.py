import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; GMTKN55Crawler/1.0'
}

def scrape_table(url):
    '''
    Scrapes table from URL. 
    '''

    # Fetch the webpage content
    response = requests.get(url, HEADERS)
    if response.status_code != 200:
        return f"Failed to retrieve page: Status code {response.status_code}"
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table
    table = soup.find('table')
    
    # Extract headers (combining colspan cells)
    headers = []
    header_row = table.find('tr')
    for th in header_row.find_all('th'):
        header_text = th.text.strip()
        colspan = int(th.get('colspan', 1))
        if colspan > 1 and header_text != "Systems" and header_text != "Stoichiometry":
            # For other headers with colspan, duplicate the header
            headers.extend([header_text] * colspan)
        elif header_text == "Systems" or header_text == "Stoichiometry":
            # For "Systems" and "Stoichiometry", create numbered columns
            headers.extend([f"{header_text}_{i+1}" for i in range(colspan)])
        else:
            headers.append(header_text)
    
    # Extract rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip the header row
        row_data = []
        for td in tr.find_all('td'):
            row_data.append(td.text.strip())
        rows.append(row_data)

    # Filter out summary rows 'MD', 'MAD', 'RMSD'
    filtered_rows = []
    for row in rows:
        if (row[0] not in ["MD", "MAD", "RMSD"]):
            filtered_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(filtered_rows, columns=headers)
    
    # Clean up and convert numeric columns
    numeric_cols = ['Ref.', 'without', 'D3(0)', 'D3(BJ)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: float(x.strip()) if x.strip() else None)
    
    return df

def clean_gmtkn_data(df):
    '''
    Finds all the stoichiometry and systems columns, combines them all into a single stoichiometry and systems column each, and 
    then drops the old intermediate columns.
    
    For both stoichiometry and systems, it takes all non-empty entries in a row and combines them into a single entry, 
    where each entry has a list of string values.
    
    Args:
        df: Extracted dataframe from webpage

    Returns:
        df: Reformatted dataframe with new columns 'Systems' and 'Stoichiometry'
    '''
    # Dynamically find all Systems columns
    systems_cols = [col for col in df.columns if col.startswith('Systems_')]
    # Get the maximum index number
    max_systems_index = max([int(col.split('_')[1]) for col in systems_cols]) if systems_cols else 0

    # Dynamically find all Stoichiometry columns
    stoich_cols = [col for col in df.columns if col.startswith('Stoichiometry_')]
    # Get the maximum index number
    max_stoich_index = max([int(col.split('_')[1]) for col in stoich_cols]) if stoich_cols else 0

    # Verify they have the same number of columns (as expected)
    if max_systems_index != max_stoich_index:
        print(f"Warning: Number of Systems columns ({max_systems_index}) differs from Stoichiometry columns ({max_stoich_index})")

    # Combine system columns into a list of non-empty strings for each row
    df['Systems'] = df.apply(
        lambda row: [str(row[f'Systems_{i}']) for i in range(1, max_systems_index + 1)
                    if f'Systems_{i}' in df.columns and row[f'Systems_{i}'] and str(row[f'Systems_{i}']).strip()],
        axis=1
    )

    # Combine stoichiometry columns into a list of non-empty strings for each row
    df['Stoichiometry'] = df.apply(
        lambda row: [str(row[f'Stoichiometry_{i}']) for i in range(1, max_stoich_index + 1)
                    if f'Stoichiometry_{i}' in df.columns and row[f'Stoichiometry_{i}'] and str(row[f'Stoichiometry_{i}']).strip()],
        axis=1
    )

    # Create new DataFrame without the temporary columns
    clean_df = df.drop(columns=systems_cols + stoich_cols)

    return clean_df

def string_processing(df: pd.DataFrame):
    '''
    Creates string representations of table info. 

    Args:
        df (pd.DataFrame): Extracted table from webpage

    Returns: 
        list: list of strings
        
    '''


if __name__ == '__main__':
    url = "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/results/W4-11/PBE/result.html"
    url2 = "http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/results/G21EA/PBE/result.html"
    df = scrape_table(url2)
    print(df)
    clean_df = clean_gmtkn_data(df)
    print(clean_df)
