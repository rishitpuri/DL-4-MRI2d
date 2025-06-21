import os
import json
import pandas as pd

folders = ["/N/project/SingleCell_Image/5kADprocessed", "/N/project/SingleCell_Image/5kCNprocessed", "/N/project/SingleCell_Image/5kMCIprocessed"]

# Initialize an empty list to store the data
data = []
map_dict = {}

# Traverse through each folder
for i,folder in enumerate(folders):
	map_dict[i] = folder
	for filename in os.listdir(folder):
		if filename.endswith(".jpeg"):
			# Extract the patient ID from the filename
			patient_id = "_".join(filename.split("_")[2:5])
			# Get the full path to the file
			full_path = os.path.join(folder, filename)
			# Append the data to the list
			data.append([patient_id, full_path, i])

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=["IID", "Image Path", "Label"])

# Display or save the DataFrame
print(len(df))
print(df.head(5))
print(map_dict)

# Save map_dict
with open('map.json', 'w') as jsonFile:
	json.dump(map_dict, jsonFile, indent=4)

# Optionally, you can save the DataFrame to a CSV file
df.to_csv("patient_data.csv", index=False)

print("Dataframe saved as patient_data.csv")
print("Map dict saved as map.json")

