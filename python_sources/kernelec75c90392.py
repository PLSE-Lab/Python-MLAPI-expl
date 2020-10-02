import csv # Use to .csv files

List = []

for i in range(0, 101): # Genarate id that 0 to 100
    Dict = {}
    Dict["id"] = str(i)
    Dict["new_number"] = str(i + 2)
    List.append(Dict)
    
with open("Solution.csv", "w") as output_file:
    file_writer = csv.DictWriter(output_file, fieldnames=List[0].keys()) # Write Solution.csv file using dictionary
    file_writer.writeheader() # Write the header into Solution.csv file
    file_writer.writerows(List) # Write the id and new_number into Solution.csv file