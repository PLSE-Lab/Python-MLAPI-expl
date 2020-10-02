import kagglegym
import numpy as np

text_file = open("output.txt", "w")
text_file.write("You should never see this text file if not an admin!!!")
text_file.close()