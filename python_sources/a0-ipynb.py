{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"A0.ipynb","provenance":[],"collapsed_sections":[]},"kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":[{"cell_type":"markdown","metadata":{"id":"bmL4kM-WY5vB","colab_type":"text"},"source":"\n#A0: Getting started\nThis assignment consists of seven steps. We would like you to complete this assignment by the second class session, and that is the due date. If you have barriers in your critical path that are keeping you from doing part or all of this assignment please contact the instructors.\n\n##Step 1\n\nThis file, with its .ipynb extension is a Jupyter notebook, which you can run in Google CoLab. You could also run the Python code locally on your own computer, but for this assignment, run it in your CoLab notebook. Run the code in the **Code Cell** below by hovering your mouse over the [ ] brackets and pressing the run button that will appear."},{"cell_type":"code","metadata":{"id":"4BdSDTmmZWte","colab_type":"code","colab":{}},"source":"def hello():\n    print(\"--------------------------------\")\n    print(\"Welcome to HCDE 530.\")\n    print(\"\")\n    print(\"We are glad you are here\")\n    print(\"and we hope you enjoy the class.\")\n    print(\"--------------------------------\")\n\n\nhello()\nprint(\"... Let's say that again... \\n\")\nhello()","execution_count":0,"outputs":[]},{"cell_type":"markdown","metadata":{"id":"mvA8QAC7GQ6v","colab_type":"text"},"source":"Note that the output is displayed beneath the Code Cell. You can clear the output by pressing the Clear Output button (the box with an arrow going out of it)."},{"cell_type":"markdown","metadata":{"id":"Jn7MIe9BZfNz","colab_type":"text"},"source":"#Step 2\nNow try deleting the second `hello()` in the **Code Cell** below.  Run the program again to see the results."},{"cell_type":"code","metadata":{"id":"mu-4ZSrQaUCb","colab_type":"code","colab":{}},"source":"def hello():\n    print(\"--------------------------------\")\n    print(\"Welcome to HCDE 530.\")\n    print(\"\")\n    print(\"We are glad you are here\")\n    print(\"and we hope you enjoy the class.\")\n    print(\"--------------------------------\")\n\n\nhello()\nprint(\"... Let's say that again... \\n\")\nhello()\n","execution_count":0,"outputs":[]},{"cell_type":"markdown","metadata":{"colab_type":"text","id":"DdcfV1VVcZ0C"},"source":"#Step 3\nNow insert the second `hello()` back at the end of the code in **Code Cell** below. Try using the auto-complete feature. After you type \"hel\", possible completions should appear. Use the arrow keys or the mouse to select, and hit enter. Then run the code and examine the output."},{"cell_type":"code","metadata":{"id":"Z4WTw373d3jR","colab_type":"code","colab":{}},"source":"hello()\nprint(\"... Let's say that again... \\n\")","execution_count":0,"outputs":[]},{"cell_type":"markdown","metadata":{"id":"90y14IhwePt1","colab_type":"text"},"source":"# Step 4\n\nNow, try a Python program that uses variables. Uncomment the lines below, by removing the # at the start of each line. Fill in the values for length, width, height, and your name. Then run the program."},{"cell_type":"code","metadata":{"id":"ewohiWW9eaZw","colab_type":"code","colab":{}},"source":"# length = 0\n# width = 0\n# height = 0\n#\n# me = \"<your name here>\"\n# print(\"Volume =\", width * length * height)\n# print(\"My name is\", me)","execution_count":0,"outputs":[]},{"cell_type":"markdown","metadata":{"id":"Q9eKjfi0edF5","colab_type":"text"},"source":"## Step 5 Running python written in CoLab notebooks locally\nYou can run Python programs written in CoLab locally on your laptop, but you need to download them as native Python files first.\n\nTo try this, download this notebook as a native Python file by selecting **File -> Download .py** in the CoLab window in your browser. After you have downloaded the file, you can run it from the command line terminal by typing python followed by the name of the file:\n\n`python filename.py`\n\n(Note, there are platforms such as *Anaconda* that allow you to run native .ipynb notebooks locally, but you don't need them for this class.)\n\nYou can also run Python programs on your local computer within a program called IDLE, which is an interactive python interpreter. You can open this file in IDLE (double-clicking on the file should do it). After you have opened the file you can run its contents within IDLE by selecting **Run -> Run Module**.\n"},{"cell_type":"markdown","metadata":{"id":"D0Jtd3NUHY4V","colab_type":"text"},"source":"If you didnt already to it, download this file (a0.ipynb) as a native Python file (.py). How is the downloaded .py file different from your Colab notebook? \n###Write your answer in the space below:\n-"},{"cell_type":"markdown","metadata":{"id":"H_pklc9HCVki","colab_type":"text"},"source":"#Step 5 Design a program\nThe purpose of this exercise is to get you familiar with issues in data manipulation and to get you thinking about how a computer might do it. For this exercise, write an English description (not a computer program!) of instructions for counting the number of words in a document (we've provided hw0.txt as a starter). Try to be as precise as possible and remove all ambiguity. It may help you to eliminate ambiguity if you anticipate that the person to whom you are giving the instructions will purposely try to misinterpret your instructions so as to get the wrong result."},{"cell_type":"markdown","metadata":{"id":"5gdvZHeXCgpv","colab_type":"text"},"source":"###Write your \"program\" in the space below:\n-"},{"cell_type":"markdown","metadata":{"id":"Ulm7GO43Cj8f","colab_type":"text"},"source":"#Step 6\nWas the previous exercise easy or hard for you? What was satisfying or unsatisfying about trying to describe the instructions in English? \n\n###Write a few remarks in the space below:\n-"},{"cell_type":"markdown","metadata":{"id":"9dHomE2rDKyD","colab_type":"text"},"source":"#Step 7\nWhen you have finished, submit this homework by *sharing* your CoLab notebook. Select the Share icon in the upper right of your screen. Make sure that Anyone at UW with the link can view. Select **Get shareable link** and copy this URL. <br><br>To submit, select Submit Assignment on Canvas and paste this link into the URL submission field."}]}