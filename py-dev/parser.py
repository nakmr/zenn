from bs4 import BeautifulSoup

# Load the HTML file
file_path = '/Users/nakimura/Projects/zenn/py-dev/Sustainability_ADAv5_Reference Models.html'
output_file_path = '/Users/nakimura/Projects/zenn/py-dev/extracted_data.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

results = []

# Find all relevant elements in the order they appear in the HTML
elements = soup.find_all([
    'div'
])

for element in elements:
    # Check for <div class="treemaptooltip w-full">
    # if 'treemaptooltip w-full' in element.get('class', []):
    #     h_tags = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
    #     for h_tag in h_tags:
    #         h_text = h_tag.get_text(strip=True)
    #         results.append(h_text)

    # Check for <div class="treemapcollapse">
    if 'treemapcollapse' in element.get('class', []):
        headers = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5'], class_='collapseheader')
        for header in headers:
            header_text = header.get_text(strip=True)
            collapsesection = element.find('div', class_='collapsesection')
            if collapsesection:
                p_tags = collapsesection.find_all('p', class_='text-xs')
                for p_tag in p_tags:
                    p_text = p_tag.get_text(strip=True)
                    results.append(f"{header_text}: {p_text}")
            else:
                results.append(header_text)

    # Check for <div class="w-full flex flex-row">
    elif 'w-full flex flex-row' in element.get('class', []):
        h_tags = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
        for h_tag in h_tags:
            h_text = h_tag.get_text(strip=True)
            results.append(h_text)

# Write results to an external file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write('\n'.join(results))

print(f"Extraction complete. Results saved to {output_file_path}")
