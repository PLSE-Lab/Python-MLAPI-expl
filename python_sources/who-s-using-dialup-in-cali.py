import shapefile
import pandas as pd
import sqlite3
def rgb_hex(rgb):
    return '#%02x%02x%02x' % rgb
housing_a = pd.read_csv("../input/pums/ss13husa.csv")
#population_a = pd.read_csv("../input/pums/ss13pusa.csv")
housing_a = housing_a[housing_a.ST==6]
c=sqlite3.connect(':memory:')
housing_a.to_sql('housing', c)
housing_a = pd.read_sql_query("SELECT PUMA, SUM(DIALUP) AS S FROM housing GROUP BY PUMA;", c)
housing_a.to_sql('s', c)
r = shapefile.Reader("../input/shapefiles/pums/tl_2013_06_puma10")
ll=[42.5,32,-114,-125] #latitude & longitude binding
colors=[]
for i in range(11):
    colors.append(rgb_hex((255,255-(i*20),255-(i*20))))
h=[]
k=[]
h.append("<!DOCTYPE html><html><body><svg width='850' height='850'><script type='text/javascript'>function info(idx) {alert(idx);}</script>")
for shape in r.shapeRecords():
    svg_ploygon=""
    for i in shape.shape.points[:]:
        y = (((i[1]-ll[0])/(ll[1] - ll[0]))*850)
        x = abs(850-(((i[0]-ll[2])/(ll[3] - ll[2]))*850))
        svg_ploygon += str(int(x))+","+str(int(y))+" "
    k.append([int(shape.record[1]),shape.record[3]])
    t = int(shape.record[1])
    t_pd = pd.read_sql_query("SELECT S FROM s WHERE PUMA=?;", c, params=[t])
    t=t_pd.S[0]
    t = ((t - housing_a.S.min())/(housing_a.S.max()-housing_a.S.min()))*10
    h.append("<polygon id='" + str(shape.record[1]) + "' points='" + svg_ploygon + "' style='fill:" + colors[int(t)] +  ";stroke:black;stroke-width:1;' onclick='info(this.id)' />")
h.append("</svg>")
housing_c=pd.DataFrame(k,columns=['PUMA_I', 'PUMA_N'])
housing_c=housing_c.to_sql('CT', c)
housing_c= pd.read_sql_query("SELECT PUMA, PUMA_N, SUM(DIALUP), SUM(BROADBND), SUM(DSL), SUM(FIBEROP) FROM housing JOIN CT on housing.PUMA=CT.PUMA_I GROUP BY PUMA_N ORDER BY PUMA_N;", c)
h.append(housing_c.to_html())
h.append("</body></html>")
with open("output.html","w",encoding="ascii", errors="surrogateescape") as f:
    for i in range(len(h)):
        f.write(h[i]+"\n")
f.close()