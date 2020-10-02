
data_dir = "../input"

valid_id = dict()

for category in ("beauty", "fashion", "mobile"):
    with open("%s/%s_data_info_val_competition.csv" % (data_dir, category), "r") as infile:
        next(infile)
        for line in infile:
            curr_id = line.strip().split(',')[0]
            valid_id[curr_id] = True

# This is the new output submission file containing 977987 rows
with open("submission-out.csv", "w") as outfile:
    outfile.write("id,tagging\n")
    
    # Please change the file below to your current submission filename containing 1174802 rows
    # with open("submission-in.csv", "r")  as infile:
    with open("%s/data_info_val_sample_submission.csv" % data_dir, "r") as infile:
        next(infile)
        for line in infile:
            curr_id = line.strip().split('_')[0]
            if curr_id in valid_id:
                outfile.write(line.strip() + '\n')
