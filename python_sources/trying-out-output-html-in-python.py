html = '''
<html>
<p>Hi!</p>
<body>
<ul>
<li>a</li>
<li>b</li>
<li>c</li>
</ul>
</body>
</html>
'''

with open("output.html", "w") as output:
    output.write(html)