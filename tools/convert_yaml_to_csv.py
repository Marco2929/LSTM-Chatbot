import yaml
import csv

# Specify the input YAML file and output CSV file
input_yaml_file = 'data/botprofile.yml'
output_csv_file = 'data/botprofile.csv'

# Load the YAML content from the file
with open(input_yaml_file, 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# Open the CSV file for writing
with open(output_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header
    csv_writer.writerow(['Question', 'Answer', 'Class'])

    # Write the conversations
    for conversation in data['conversations']:
        question = conversation[0]
        answer = conversation[1]
        csv_writer.writerow([question, answer, data['categories'][0]])

print(f"CSV file '{output_csv_file}' created successfully.")

