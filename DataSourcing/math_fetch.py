import os
import json
import pandas as pd

# Set your parent directory path here
parent_folder = r"D:\1_RAVI_STARTS\ML\intelliroute\archive\MATH\train"  # replace with actual path

problems = []

# Traverse all subfolders and files
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if "problem" in data:
                        problems.append({"problem": data["problem"]})
            except json.JSONDecodeError as e:
                print(f"Skipped {file_path}: {e}")

# Create DataFrame
df_problems = pd.DataFrame(problems)
df_problems["class"] = "math"
df_problems.columns = ["query", "class"]
# Show preview
print(f"Total problems loaded: {len(df_problems)}")
# df_problems.head()

df_problems.to_csv("maths.csv", index=False)
