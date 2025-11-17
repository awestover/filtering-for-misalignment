import json
import glob

# Find all search_terms JSON files
json_files = glob.glob("search_terms*.json")

all_search_terms = []

# Extract search terms from each file
for file_path in sorted(json_files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'search_terms' in data:
                all_search_terms.extend(data['search_terms'])
                print(f"Extracted {len(data['search_terms'])} terms from {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Write to markdown file
with open('search_terms.md', 'w') as f:
    for term in all_search_terms:
        f.write(f"{term}\n")

print(f"\nTotal search terms extracted: {len(all_search_terms)}")
print(f"Written to search_terms.md")
