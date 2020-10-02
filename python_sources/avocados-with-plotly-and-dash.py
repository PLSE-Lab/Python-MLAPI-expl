#!/usr/bin/env python
# coding: utf-8

# #A Live Avocado Analysis Page
# I've started working with Plotly and Dash, what better data set to test out?
# 
# https://mattocado.herokuapp.com/
# 
# I'm hoping this page will be useful as a quick tool for analyzing the data. 
# MOST of the combinations don't really make sense, but I left them all in anyway; I didn't want to make any pre-judgements.
# Also it's pretty SLOOOW so be patient.  I have a version with a smaller data sample, so I'll upload that too.  
# But for now...
# 
# Also here's the DASH/Plotly code.  Work in progress...
# 
# ```
# import dash
# import dash_auth
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.graph_objs as go
# import pandas as pd
# 
# app = dash.Dash()
# server = app.server
# 
# df = pd.read_csv('avocados.csv', index_col=0)
# 
# features = df.columns
# featuresZAxis = df.drop(columns=['Date', 'type', 'year', 'region']).columns
# graphStyles = ['Bar', 'Markers', 'Lines', 'Lines+Markers']
# app.layout = html.Div([
#     html.H1('AVOCADOS!'),
#         html.Div([
#             html.H3('X Axis:'),
#             dcc.Dropdown(
#                 id='xaxis',
#                 options=[{'label': i.title(), 'value': i} for i in features],
#                 value='Large Bags'
#             )
#         ],
#         style={'width': '20%', 'display': 'inline-block'}),
#         html.Div([
#             html.H3('Y Axis:'),
#             dcc.Dropdown(
#                 id='yaxis',
#                 options=[{'label': i.title(), 'value': i} for i in features],
#                 value='Total Bags'
#             )
#         ],style={'width': '20%', 'display': 'inline-block'}),
#         html.Div([
#             html.H3('Z Axis (Color):'),
#             dcc.Dropdown(
#                 id='zaxis',
#                 options=[{'label': i.title(), 'value': i} for i in featuresZAxis],
#                 value='AveragePrice'
#             )
#         ],style={'width': '20%', 'display': 'inline-block'}),
# 
#         html.Div([
#             html.H3('Plot Type:'),
#             dcc.Dropdown(
#                 id='style',
#                 options=[{'label': i.title(), 'value': i} for i in graphStyles],
#                 value='Markers'
#             )
#         ],style={'width': '20%', 'display': 'inline-block'}),
# 
#     dcc.Graph(id='feature-graphic')
# ], style={'padding':10, 'className':'dark-table'})
# 
# @app.callback(
#     Output('feature-graphic', 'figure'),
#     [Input('xaxis', 'value'),
#      Input('yaxis', 'value'),
#      Input('zaxis', 'value'),
#      Input('style', 'value')])
# def update_graph(xaxis_name, yaxis_name, zaxis_name, style_name):
#     if (style_name=="Markers"):
#         return {
#             'data': [go.Scatter(
#                 x=df[xaxis_name],
#                 y=df[yaxis_name],
#                 text=df['Date'],
#                 mode='markers',
#                 marker={
#                     'size': 15,
#                     'color': df[zaxis_name],
#                     'opacity': 0.5,
#                     'line': {'width': 2, 'color': 'black'},
#                     'colorscale': 'Viridis',
#                     'showscale': True
#                 }
#             )],
#             'layout': go.Layout(
#                 xaxis={'title': xaxis_name.title()},
#                 yaxis={'title': yaxis_name.title()},
#                 # margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
#                 hovermode='closest'
#             )
#         }
#     elif (style_name=="Bar"):
#         return {
#             'data': [go.Bar(
#                 x=df[xaxis_name],
#                 y=df[yaxis_name],
#                 text=df['Date'],
#                 marker=dict(
#                     color=df[zaxis_name],
#                     line=dict(
#                     width=1.5),
#                 ),
#                 opacity=0.6
#             )],
#             'layout': go.Layout(
#                 barmode='stack'
#             )
#         }
#     elif (style_name=="Lines"):
#         return {
#             'data': [go.Scatter(
#                 x=df[xaxis_name],
#                 y=df[yaxis_name],
#                 text=df['Date'],
#                 mode='lines',
#                 marker={
#                     'size': 15,
#                     'color': df[zaxis_name],
#                     'opacity': 0.5,
#                     'line': {'width': 2, 'color': 'black'}
#                 }
#             )],
#             'layout': go.Layout(
#                 xaxis={'title': xaxis_name.title()},
#                 yaxis={'title': yaxis_name.title()},
#                 hovermode='closest'
#             )
#         }
#     elif (style_name=="Lines+Markers"):
#             return {
#                 'data': [go.Scatter(
#                     x=df[xaxis_name],
#                     y=df[yaxis_name],
#                     text=df['Date'],
#                     mode='lines+markers',
#                     marker={
#                         'size': 15,
#                         'color': df[zaxis_name],
#                         'opacity': 0.5,
#                         'line': {'width': 2, 'color': 'black'}
#                     }
#                 )],
#                 'layout': go.Layout(
#                     xaxis={'title': xaxis_name.title()},
#                     yaxis={'title': yaxis_name.title()},
#                     hovermode='closest'
#                 )
#             }
# 
# if __name__ == '__main__':
#     app.run_server()
# ```
# 
# 
# 
# 
# 

# In[ ]:




