# %% [code]
import requests 
import os
from zipfile import ZipFile
url=["http://www.ktbs.kar.nic.in/New/website%20textbooks/class2/2nd-english-maths.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class2/2nd-marati-maths.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class2/2nd-english-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class2/2nd-marati-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class3/3rd-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class3/3rd-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class3/3rd-english-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class3/3rd-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class3/3rd-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class3/3rd-marati-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class4/4th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class4/4th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class4/4th-english-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class4/4th-marathi-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class4/4th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class4/4th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class5/5th-english-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class5/5th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class5/5th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class5/5th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class5/5th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class5/5th-marathi-evs.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-english-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-english-science.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-marathi-science.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class6/6th-marathi-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-english-science-01.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-english-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-marathi-science-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class7/7th-marathi-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-english-science-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-english-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-marathi-science-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class8/8th-marathi-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-english-science.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-english-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-marathi-science-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-marathi-science-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class9/9th%20standard/9th-marathi-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-english-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-english-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-english-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-english-science-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-english-science-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-marathi-phy-edu.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-marathi-maths-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-marathi-maths-2.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-marathi-science-1.pdf",
     "http://www.ktbs.kar.nic.in/New/website%20textbooks/class10/10th%20standard/10th-marathi-science-2.pdf"     
     ]

os.chdir('/kaggle/output/working')

listdir_2=os.listdir()
fles= list(filter(lambda x: x[-4:] == '.pdf', listdir_2))
zipObj = ZipFile('Marati_files.zip', 'w')
import re
for i in url:
  image_url = i
  s=re.split("/",image_url)
  s1=s[len(s)-1]
  if s1 not in fles:
    print(1+2)
    r = requests.get(image_url)
    fles.append(s1)
    # send a HTTP request to the server and save 
    # the HTTP response in a response object called r 
    with open(s1,'wb') as f: 
      
        # Saving received content as a png file in 
        # binary format 
      
        # write the contents of the response (r.content) 
        # to a new file in binary mode. 
        f.write(r.content) 
        zipObj.write(s1)
  else:
      print(s1)
  

zipObj.close()
#from google.colab import files
#files.download('Marati_files.zip') 