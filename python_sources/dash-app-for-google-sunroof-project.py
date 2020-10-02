import dash
from dash.dependencies import Input, Output, State, #Event
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import plotly as plotly
from textwrap import dedent
from plotly import tools
import base64
from textwrap import dedent


# Pre-Startup -------------------------------------------------------------------------------------------
print("Initializing Dash and Loading Dataset...")

external_stylesheets = ['/home/bhavya/Desktop/DV/DASH/assets/main_style_sheet.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

state = pd.read_csv('/home/bhavya/Desktop/DV/DASH/state2.csv')
zipcode = pd.read_csv('/home/bhavya/Desktop/DV/DASH/zipcode2.csv')
city = pd.read_csv('/home/bhavya/Desktop/DV/DASH/city2.csv')
county = pd.read_csv('/home/bhavya/Desktop/DV/DASH/county2.csv')
state_cols = pd.read_csv('/home/bhavya/Desktop/DV/DASH/state_cols.csv')
image = '/home/bhavya/Desktop/DV/DASH/projectsunroof.png'
encoded_image = base64.b64encode(open(image, 'rb').read())
tree = '/home/bhavya/Desktop/DV/DASH/tree.JPEG'
encoded_tree = base64.b64encode(open(tree, 'rb').read())
mapbox_access_token = 'pk.eyJ1IjoiYmhhdnlhcmFtZ2lyaSIsImEiOiJjanR6d3lqdDgycng5M3lxdW5tcWN3bW5hIn0.15q7MVN8v1rkemwfomu-gA'



# Layout style -------------------------------------------------------------------------------------------
print("layouts initializations..............")
layout_table = dict(
    autosize=True,
    height=500,
    width=1222,
    font=dict(color='#FFFFF0'),
    titlefont=dict(color='#FFFFF0', size='8'),
    margin=dict(
        l=75,
        r=75,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    legend=dict(font=dict(size=10), orientation='v'),
)
layout_table['font-size'] = '11'
layout_table['margin-top'] = '20'



layout_map = dict(
    autosize=True,
    height=500,
    font=dict(color='#FFFFF0'),
    titlefont=dict(color='#FFFFF0', size='20'),
    margin=dict(
        l=75,
        r=75,
        b=35,
        t=45    
    ),
    hovermode="closest",
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    legend=dict(font=dict(size=15), orientation='h'),
    title='Potential Annual Solar Energy Generation',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="dark",
        bearing=0,
        center=dict(
            lon=-94,
            lat=38
        ),
        pitch=0,
        zoom=3,
    )
)

# function for generating map ---------------------------------------------------------------------------------
print('functions to create map ............................')
def gen_map_zipcode(zipcode):
    zipcodez = zipcode.sort_values(by='yearly_sunlight_kwh_total', ascending=False)
    nrows = round(len(zipcodez)/3)
    df_1 = zipcodez[0:nrows]
    df_2 = zipcodez[nrows : nrows*2]
    df_3 = zipcodez[nrows*2 : nrows*3]
    site_lat1 = df_1['lat_avg']
    site_lon1 = df_1['lng_avg']
    site_lat2 = df_2['lat_avg']
    site_lon2 = df_2['lng_avg']
    site_lat3 = df_3['lat_avg']
    site_lon3 = df_3['lng_avg']
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(site_lat1),
                "lon": list(site_lon1),
                "hoverinfo": "text",
                "hovertext": [["zipcode: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_1['region_name'], df_1['state_name'],df_1['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} - {:,.2f} million kwh".format(zipcodez[0:nrows].yearly_sunlight_kwh_total.max(),zipcodez[0:nrows].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": 'orangered',
                    "size": 15,
                    "opacity": 0.3 }},
                {
                "type": "scattermapbox",
                "lat": list(site_lat2),
                "lon": list(site_lon2),
                "hoverinfo": "text",
                "hovertext": [["zipcode: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_2['region_name'], df_2['state_name'],df_2['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} - {:,.2f} million kwh".format(zipcodez[nrows: nrows*2].yearly_sunlight_kwh_total.max(),zipcodez[nrows : nrows*2].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": 'yellow',
                    "size": 10,
                    "opacity": 0.4 }},
               {
                "type": "scattermapbox",
                "lat": list(site_lat3),
                "lon": list(site_lon3),
                "hoverinfo": "text",
                "hovertext": [["zipcode: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_3['region_name'], df_3['state_name'],df_3['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} million - {:,.2f} thousand kwh".format(zipcodez[nrows*2 : nrows*3].yearly_sunlight_kwh_total.max(),zipcodez[nrows*2 : nrows*3].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": 'lightgreen',
                    "size": 4,
                    "opacity": 0.5 }}],
        "layout": layout_map 
    }

def gen_map_city(city):
    cityz = city.sort_values(by='yearly_sunlight_kwh_total', ascending=False)
    nrows = round(len(cityz)/3)
    df_1 = cityz[0 : nrows]
    df_2 = cityz[nrows : nrows*2]
    df_3 = cityz[nrows*2 : nrows*3]
    site_lat1 = df_1['lat_avg']
    site_lon1 = df_1['lng_avg']
    site_lat2 = df_2['lat_avg']
    site_lon2 = df_2['lng_avg']
    site_lat3 = df_3['lat_avg']
    site_lon3 = df_3['lng_avg']
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(site_lat1),
                "lon": list(site_lon1),
                "hoverinfo": "text",
                "hovertext": [["city: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_1['region_name'], df_1['state_name'],df_1['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} billion - {:,.2f} million kwh".format(cityz[0:nrows].yearly_sunlight_kwh_total.max(),cityz[0:nrows].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": '#00ffff',
                    "size": 15,
                    "opacity": 0.3 }},
                {
                "type": "scattermapbox",
                "lat": list(site_lat2),
                "lon": list(site_lon2),
                "hoverinfo": "text",
                "hovertext": [["city: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_2['region_name'], df_2['state_name'],df_2['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} - {:,.2f} million kwh".format(cityz[nrows: nrows*2].yearly_sunlight_kwh_total.max(),cityz[nrows : nrows*2].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": '#ffff00',
                    "size": 10,
                    "opacity": 0.4 }},
               {
                "type": "scattermapbox",
                "lat": list(site_lat3),
                "lon": list(site_lon3),
                "hoverinfo": "text",
                "hovertext": [["city: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_3['region_name'], df_3['state_name'],df_3['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} million - {:,.2f} thousand kwh".format(cityz[nrows*2 : nrows*3].yearly_sunlight_kwh_total.max(),cityz[nrows*2 : nrows*3].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": '#cc3385',
                    "size": 4,
                    "opacity": 0.5 }}],
        "layout": layout_map 
    }

def gen_map_county(county):
    countyz = county.sort_values(by='yearly_sunlight_kwh_total', ascending=False)
    nrows = round(len(countyz)/3)
    df_1 = countyz[0 : nrows]
    df_2 = countyz[nrows : nrows*2]
    df_3 = countyz[nrows*2 : nrows*3]
    site_lat1 = df_1['lat_avg']
    site_lon1 = df_1['lng_avg']
    site_lat2 = df_2['lat_avg']
    site_lon2 = df_2['lng_avg']
    site_lat3 = df_3['lat_avg']
    site_lon3 = df_3['lng_avg']
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(site_lat1),
                "lon": list(site_lon1),
                "hoverinfo": "text",
                "hovertext": [["county: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_1['region_name'], df_1['state_name'],df_1['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} billion - {:,.2f} million kwh".format(countyz[0:nrows].yearly_sunlight_kwh_total.max(),countyz[0:nrows].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": '#66ff33',
                    "size": 15,
                    "opacity": 0.3 }},
                {
                "type": "scattermapbox",
                "lat": list(site_lat2),
                "lon": list(site_lon2),
                "hoverinfo": "text",
                "hovertext": [["county: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_2['region_name'], df_2['state_name'],df_2['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} - {:,.2f} million kwh".format(countyz[nrows: nrows*2].yearly_sunlight_kwh_total.max(),countyz[nrows : nrows*2].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": '#cf006e',
                    "size": 10,
                    "opacity": 0.4 }},
               {
                "type": "scattermapbox",
                "lat": list(site_lat3),
                "lon": list(site_lon3),
                "hoverinfo": "text",
                "hovertext": [["county: {} <br>state: {} <br>solar energy in kwh: {}".format(i,j,k)]
                                for i,j,k in zip(df_3['region_name'], df_3['state_name'],df_3['yearly_sunlight_kwh_total'])],
                "mode": "markers",
                "name": "{:,.2f} million - {:,.2f} thousand kwh".format(countyz[nrows*2 : nrows*3].yearly_sunlight_kwh_total.max(),countyz[nrows*2 : nrows*3].yearly_sunlight_kwh_total.min()),
                "marker": {
                    "color": '#ffff00',
                    "size": 4,
                    "opacity": 0.7 }}],
        "layout": layout_map 
    }

def gen_bar(state_cols):
    data = [
        go.Bar(
            x = ['No. of buildings qualified<br>for solar panel installation', 'No. of buildings with<br>installed solar panels'],
            y = [(state_cols.count_qualified).sum(), (state_cols.existing_installs_count).sum()],
            marker = dict(color=['#ffcc00','#ff4700']),
            opacity = 1,)
        ]
    layout = go.Layout(
        xaxis = dict(automargin= True),
        yaxis = dict(automargin= True),
        title = 'Number of Buildings - Solar PV Qualified and Installed',
        plot_bgcolor = '#000000',
        paper_bgcolor="#000000",
        font=dict(color='#FFFFF0'),
    )    
    return go.Figure(data=data, layout=layout)    


def gen_pie(state_cols):
    north = (state_cols.yearly_sunlight_kwh_n).sum()
    south = (state_cols.yearly_sunlight_kwh_s).sum()
    east = (state_cols.yearly_sunlight_kwh_e).sum()
    west = (state_cols.yearly_sunlight_kwh_w).sum()
    flat = (state_cols.yearly_sunlight_kwh_f).sum()
    data_pie= [
        go.Pie(
            values = [north, south, east, west, flat],
            labels = ["North-facing","South-facing","East-facing","West-facing","Flat-roof"],
            domain = dict(column = 0),
            name = "Roof<br>Orientation",
            hoverinfo = "label+percent+name",
            hole = .4,
            marker = dict(
                colors = ["#94f08f","#ff5973","#ffcc00","#ff4700","#33c9cf"],
                line = dict(color='#000000', width=5)),
            opacity = 0.8,
        )]
    layout_pie = go.Layout(
        title = "Annual Solar Energy Generation Potential<br>for various Roof Orientations",
        plot_bgcolor = '#000000',
        paper_bgcolor="#000000",
        font=dict(color='#FFFFF0'),
        annotations =[
            dict(
                font = dict( size = 20),
                text = "Roof<br>Orientation",
                x = 0.5,
                y = 0.5,
                showarrow = False,
            )
        ]    
    )
    return go.Figure(data=data_pie, layout=layout_pie)

# app layout ---------------------------------------------------------------------------------------------------------------
print('app layout..............................................')
app.layout = html.Div(style={"backgroundColor":"#000000"},children=[
                 html.Div(style={"backgroundColor":"#000000"}, children=[
                     html.H3(children='Solar energy generation potential estimated by selected States for USA.',
                     style={'color': '#FFFFF0','padding-top': 15, 'padding-left':30 },
                     className='six columns'),
                     html.Img(
                         src='data:image/JPG;base64,{}'.format(encoded_image.decode()),
                         style={
                             'height': '10%',
                             'width': '15%',
                             'float': 'right',
                             'position': 'relative',
                             'padding-top': 45,
                             'padding-right': 55,
                          },
                          className='six columns',
                      ),
                      html.P('Application summary:',style={'color': '#FFFFF0','padding-left':30 },className='twelve columns'),
                      html.P('This application allows the user to organize and view the data for solar energy potential in the USA by State and display it on a map. Further it allows the user to summarize the solar energy potential estimates at Zip Code, City or County levels. User is able to select as many States as desired. If no States are selected then the data for the entire Country is displayed. CO2 offset potential and the acres of trees required to sequester that CO2 are also displayed for the selections made. The application displays two charts for the total number of buildings qualified for solar installation, number of buildings with installed solar systems and a comparison of the roof orientations for the selections made. ',style={'color': '#FFFFF0','padding-left':30,'padding-right':55 },className='twelve columns'),
                      html.H6(children='''
                              In the table below, please check the boxes to select the States of your choice.
                              ''',
                              style={'color': '#FFFFF0','padding-left':30}, 
                              className='nine columns'
                      )
                 ], className="row"),     
        # table + dropdown + map-----------------------------
        html.Div(style={'backgroundColor': "#000000"}, children=[
                html.Div(
                    [
                        dt.DataTable(
                            id='datatable',
                            data=state.to_dict("rows"),
                            columns=[{"name": i, "id": i} for i in state.columns],
                            style_header={'backgroundColor': "#000000",
                                           'fontWeight': 'bold'},
                            style_cell={
                                'backgroundColor': "#000000",
                                'color': '#FFFFF0'},
                            row_selectable='multi',
                            #filtering=True,
                            sorting=True,
                            selected_rows=[],
                            #n_fixed_rows=1,
                            style_table={
                                'maxHeight': '500',
                                'overflowY': 'scroll',
                                'backgroundColor': "#000000",
                                'color': '#FFFFF0',
                                'padding-left': 10,
                                'padding-right': 5
                            },
                            #pagination_mode="fe",
                                #pagination_settings={
                                    #"displayed_pages": 1,
                                    #"current_page": 0,
                                    #"page_size": 14,
                                #},
                                #navigation="page",
                        ),
                    ],
                    style = layout_table,
                    className="twelve columns"
                ),
                html.Div(
                    [
                        html.H6('Please pick the level at which results need to be displayed (County/City/Zipcode) in the map.',
                                style={'backgroundColor': "#000000",'color': '#FFFFF0','padding-top':20, 'padding-left':20},
                              ),
                        dcc.Dropdown(
                            id='dropdown',
                            options=[
                                {'label': 'zipcode', 'value': 'zipcode'},
                                {'label': 'county', 'value': 'county'},
                                {'label': 'city', 'value': 'city'},
                            ],
                            #style={'backgroundColor': "#000000",'color': '#FFFFF0'},
                            value='zipcode',
                        )    
                    ], 
                    className = "twelve columns"
                ),              
                html.Div(
                    [
                        dcc.Graph(id='map-graph',
                                  animate=True,
                                  style={'margin-top': '20'},
                                  config = { 'scrollZoom': True })
                    ], className = "twelve columns"
                ),
            ], className="row"
        ),
#carbon offset info ----------------------------------------------------------------------        
        html.Div(style={'backgroundColor': "#000000"}, children=[
                html.Div(
                    [
                        html.H6(id='carbon',
                               style={'color': '#FFFFF0','padding-top':20, 'padding-left':40}),
                    ],
                    className = "five columns"),
                html.Img(
                    src='data:tree/JPEG;base64,{}'.format(encoded_tree.decode()),
                    style={
                        'height': '2%',
                        'width': '10%',
                        #'float': 'right',
                        #'position': 'relative',
                        #'padding-top': 12,
                        #'padding-left': 50,
                    },
                    className='two columns',
                ),
                html.Div(
                    [
                        html.H6(id='tree',
                               style={'color': '#FFFFF0','padding-top':20,'padding-left':100}),
                    ], 
                    className = "five columns"),
            ], className="row"          
        ),
  #bar+pie+name---------------------------------      
        html.Div(style={'backgroundColor': "#000000"}, children=[
                html.Div(
                    [
                        dcc.Graph(id='bar-graph'),
                    ],
                    className = "six columns"),
                html.Div(
                    [
                        dcc.Graph(id='pie-graph')
                    ],
                    className = "six columns"),
                html.Div(
                    [
                        html.P('Developed by Bhavya.Ramgiri', style = {'display': 'inline','color': '#FFFFF0'}),
                    ], 
                    className = "four columns",
                    style = {'fontSize': 18, 'padding-top': 20,'color': '#FFFFF0'}),
                html.Div(
                    [
                        dcc.Markdown(dedent('''
                        This app is based on Google's [Sunroof Project](https://www.google.com/get/sunroof/data-explorer/) data from Kaggle.'''
                        ),containerProps=dict(style={'color': '#FFFFF0'})),
                    ],
                    className = "eight columns"),
            ],
            className='row'
        ),
    ])
        #className='twelve columns offset-by-one-third'

#callback------------------------------------------------------------------------------------------------
print('callback................')
@app.callback(
    Output('map-graph', 'figure'),
    [Input('dropdown', 'value'),
     Input('datatable', 'data'),
     Input('datatable', 'selected_rows')])
def map_selection(value, data, selected_rows):
    aux = pd.DataFrame(data)
    temp_df = pd.DataFrame()
    for i in selected_rows:
        dftemp = aux.iloc[i]
        temp_df = temp_df.append(dftemp)
    if len(selected_rows) == 0:
        if value == 'city':
            df1 = pd.DataFrame()
            for i in aux.state_name:
                data1 = city[city.state_name == i]
                df1 = df1.append(data1)
            return gen_map_city(df1)
        elif value == 'county':
            df1 = pd.DataFrame()
            for i in aux.state_name:
                data1 = county[county.state_name == i]
                df1 = df1.append(data1)
            return gen_map_county(df1)
        else:
            df1 = pd.DataFrame()
            for i in aux.state_name:
                data1 = zipcode[zipcode.state_name == i]
                df1 = df1.append(data1)
            return gen_map_zipcode(df1)
    if len(selected_rows) != 0:
        if value == 'city':
            df2 = pd.DataFrame()
            for i in temp_df.state_name:
                data1 = city[city.state_name == i]
                df2 = df2.append(data1)
            return gen_map_city(df2)
        elif value == 'county':
            df2 = pd.DataFrame()
            for i in temp_df.state_name:
                data1 = county[county.state_name == i]
                df2 = df2.append(data1)
            return gen_map_county(df2)
        else:
            df2 = pd.DataFrame()
            for i in temp_df.state_name:
                data1 = zipcode[zipcode.state_name == i]
                df2 = df2.append(data1)
            return gen_map_zipcode(df2)

@app.callback(
    Output('bar-graph', 'figure'),
    [Input('datatable', 'data'),
     Input('datatable', 'selected_rows')])
def plots(data, selected_rows):
    aux = pd.DataFrame(data)
    temp_df = pd.DataFrame()
    for i in selected_rows:
        dftemp = aux.iloc[i]
        temp_df = temp_df.append(dftemp)
    if len(selected_rows) == 0:
        df3 = pd.DataFrame()
        for i in aux.state_name:
                data1 = state_cols[state_cols.state_name == i]
                df3 = df3.append(data1)
        return gen_bar(df3)  
    df4 = pd.DataFrame()
    for i in temp_df.state_name:
        data1 = state_cols[state_cols.state_name == i]
        df4 = df4.append(data1)
    return gen_bar(df4)
#carbon offset text=============================================================================================
@app.callback(
    Output('carbon', 'children'),
    [Input('datatable', 'data'),
     Input('datatable', 'selected_rows')])
def carbon(data, selected_rows):
    aux = pd.DataFrame(data)
    temp_df = pd.DataFrame()
    for i in selected_rows:
        dftemp = aux.iloc[i]
        temp_df = temp_df.append(dftemp)    
    if len(selected_rows) == 0:
        return "If the entire solar PV potential in the selection made above is installed, it will reduce CO2 emissions by {:,.2f} metric tons".format((aux.carbon_offset_metric_tons).sum()) 
    return "If the entire solar PV potential in the selection made above is installed, it will reduce CO2 emissions by {:,.2f}  metric tons".format((temp_df.carbon_offset_metric_tons).sum()) 

@app.callback(
    Output('tree', 'children'),
    [Input('datatable', 'data'),
     Input('datatable', 'selected_rows')])
def carbon(data, selected_rows):
    aux = pd.DataFrame(data)
    temp_df = pd.DataFrame()
    for i in selected_rows:
        dftemp = aux.iloc[i]
        temp_df = temp_df.append(dftemp)
    if len(selected_rows) == 0:
        return "{:,.2f} acres of trees will be needed to sequester this amount of CO2 in one year.".format(((aux.carbon_offset_metric_tons).sum())*1.2)
    return "{:,.2f} acres of trees will be needed to sequester this amount of CO2 in one year.".format(((temp_df.carbon_offset_metric_tons).sum())*1.2)

#=============================================================================================

@app.callback(
    Output('pie-graph', 'figure'),
    [Input('datatable', 'data'),
     Input('datatable', 'selected_rows')])
def plots(data, selected_rows):
    aux = pd.DataFrame(data)
    temp_df = pd.DataFrame()
    for i in selected_rows:
        dftemp = aux.iloc[i]
        temp_df = temp_df.append(dftemp)
    if len(selected_rows) == 0:
        df5 = pd.DataFrame()
        for i in aux.state_name:
                data1 = state_cols[state_cols.state_name == i]
                df5 = df5.append(data1)
        return gen_pie(df5) 
    df6 = pd.DataFrame()
    for i in temp_df.state_name:
        data1 = state_cols[state_cols.state_name == i]
        df6 = df6.append(data1)
    return gen_pie(df6)



if __name__ == '__main__':
    app.run_server(debug=True)












