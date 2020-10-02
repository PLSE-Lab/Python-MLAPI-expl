from subprocess import check_output
print(check_output(["free"]).decode("utf8"))