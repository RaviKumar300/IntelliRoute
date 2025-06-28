import os
import pandas as pd

# Set parent directory path
parent_folder = r"D:\1_RAVI_STARTS\ML\intelliroute\bbc news summaries\BBC News Summary\News Articles"  # replace with actual path

entries = []

# Traverse all subfolders and read .txt files
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    entries.append({"query": text, "class": "summary"})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Create DataFrame
df_summary = pd.DataFrame(entries)

# Save to CSV
df_summary.to_csv("summary.csv", index=False)

print(f"Saved {len(df_summary)} entries to summary.csv")
