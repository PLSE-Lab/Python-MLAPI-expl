"""
Access the visualization by clicking on the HTML file in the Files tab.

Bokeh is a fantastic interactive visualization library. Check it out here:
http://bokeh.pydata.org/en/latest/

I'm happy to answer any questions you may have, just leave a comment. Feel free to share any ideas or thoughts you have
as well. I'd especially love to see what other kinds of visualizations you guys can come up with using Bokeh!
"""

import pandas as pd
from bokeh.models import CustomJS, ColumnDataSource, Paragraph, Select, HoverTool, BoxZoomTool, ResetTool,\
    DatetimeTickFormatter, HBox, VBox
from bokeh.plotting import Figure, output_file, save


def create_cds_key(s):
    """ColumnDataSource keys can't have special chars in them"""
    s = s.replace(' ', '_')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace("'", '')
    s = s.replace('.', '')
    s = s.lower()
    return s

# mode=inline bundles the bokeh js and css in the html rather than accessing the cdn
# this is handy since kaggle scripts can't access internet resources
output_file('visualize.html', mode='inline')

# read the data
data = pd.read_csv('../input/GlobalLandTemperaturesByState.csv', parse_dates=['dt'])

# date range which will be used to give all dataframes a common index
dr = pd.date_range(start='1743-01-01', end='2013-12-01', freq='MS')

# dict to hold bokeh model objects which can be passed to CustomJS
plot_sources = dict()

#
countries = list(data['Country'].unique())
for country in countries:
    country_data = data.loc[data['Country'] == country]
    country_states = list(country_data['State'].unique())
    state_select = Select(value=country_states[0], title='State', options=country_states)

    country_key = create_cds_key(country)
    plot_sources[country_key] = state_select

    # create a ColumnDataSource for each state in country
    for state in country_states:
        state_data = country_data.loc[country_data['State'] == state]
        state_data = state_data.drop(['Country', 'AverageTemperatureUncertainty', 'State'], axis=1)
        state_data = state_data.set_index('dt')
        state_data = state_data.reindex(dr)
        state_data.index.name = 'dt'

        state_key = create_cds_key(state)
        plot_sources[state_key] = ColumnDataSource(state_data)

# create a ColumnDataSource to use for the actual plot, default on Oregon, United States
plot_data = data.loc[(data['Country'] == 'United States') & (data['State'] == 'Oregon')]
plot_data = plot_data.drop(['Country', 'AverageTemperatureUncertainty', 'State'], axis=1)
plot_data['dt_formatted'] = plot_data['dt'].apply(lambda x: x.strftime('%b %Y'))
plot_data = plot_data.set_index('dt')
plot_data = plot_data.reindex(dr)
plot_data.index.name = 'dt'
plot_sources['plot_source'] = ColumnDataSource(plot_data)

# configure HoverTool
hover = HoverTool(
        tooltips=[
            ("Date", "@dt_formatted"),
            ("Temperature", "@AverageTemperature"),
        ]
    )

# setup some basic tools for the plot interactions
TOOLS = [BoxZoomTool(), hover, ResetTool()]

# define our plot and set various plot components
plot = Figure(plot_width=1000, x_axis_type='datetime', title='Oregon, United States', tools=TOOLS)
plot.line('dt', 'AverageTemperature', source=plot_sources['plot_source'], line_width=3, line_alpha=0.6)
plot.yaxis.axis_label = "Temperature (C)"
plot.axis.axis_label_text_font_size = "12pt"
plot.axis.axis_label_text_font_style = "bold"
plot.xaxis[0].formatter = DatetimeTickFormatter(formats=dict(months=["%b %Y"], years=["%Y"]))

# add the plot and yaxis to our sources dict so we can manipulate various properties via javascript
plot_sources['plot'] = plot
plot_sources['yaxis_label'] = plot.yaxis[0]

# callback when the units are changed
units_callback = CustomJS(args=plot_sources, code="""
        var unit = cb_obj.get('value');
        var plot_data = plot_source.get('data');

        if (unit == 'Fahrenheit') {
            yaxis_label.set('axis_label', 'Temperature (F)')
            for (i = 0; i < plot_data['AverageTemperature'].length; i++) {
                plot_data['AverageTemperature'][i] = plot_data['AverageTemperature'][i] * 1.8 + 32.0
            }
        } else if (unit == 'Celcius') {
            yaxis_label.set('axis_label', 'Temperature (C)')
            for (i = 0; i < plot_data['AverageTemperature'].length; i++) {
                plot_data['AverageTemperature'][i] = (plot_data['AverageTemperature'][i] - 32.0) * 0.5556
            }
        }

        plot_source.trigger('change');
    """)
unit_select = Select(value='Celcius', title='Units', options=['Celcius', 'Fahrenheit'], callback=units_callback)
plot_sources['unit_select'] = unit_select

# callback when a new state is selected
states_callback = CustomJS(args=plot_sources, code="""
        var state = cb_obj.get('value');
        var countries = {'Brazil': eval('brazil'),
                         'Russia': eval('russia'),
                         'United States': eval('united_states'),
                         'Canada': eval('canada'),
                         'India': eval('india'),
                         'China': eval('china'),
                         'Australia': eval('australia')};

        var country_title = '';
        Object.keys(countries).forEach( function (country) {
            if (countries[country].get('options').indexOf(state) >= 0) {
                country_title = country;
            }
        });

        plot.set('title', state + ', ' + country_title);

        var plot_data = plot_source.get('data');

        state = state.replace(/\s+/g, '_').toLowerCase();
        var eval_str = state + ".get('data')"
        var new_data = eval(eval_str);

        plot_data['dt'] = []
        plot_data['AverageTemperature'] = []

        var units = unit_select.get('value');
        if (units == 'Celcius') {
            for (i = 0; i < new_data['dt'].length; i++) {
                plot_data['dt'].push(new_data['dt'][i])
                plot_data['AverageTemperature'].push(new_data['AverageTemperature'][i])
            }
        } else if (units == 'Fahrenheit') {
            for (i = 0; i < new_data['dt'].length; i++) {
                plot_data['dt'].push(new_data['dt'][i])
                plot_data['AverageTemperature'].push(new_data['AverageTemperature'][i] * 1.8 + 32.0)
            }
        }

        plot_source.trigger('change');
    """)


# widgets for more interactions
state_select = Select(value='Oregon', title='State', options=plot_sources['united_states'].options,
                      callback=states_callback)
plot_sources['state_select'] = state_select

# callback when a new country is selected
countries_callback = CustomJS(args=plot_sources, code="""
        var country = cb_obj.get('value');
        var country_title = cb_obj.get('value');
        country = country.replace(/\s+/g, '_').toLowerCase();

        var eval_str = country + ".get('options')"
        var states = eval(eval_str);
        var state = states[0];

        plot.set('title', state + ', ' + country_title);
        state_select.set('value', state)
        state_select.set('options', states)

        var plot_data = plot_source.get('data');

        state = state.replace(/\s+/g, '_').toLowerCase();
        var eval_str = state + ".get('data')"
        var new_data = eval(eval_str);

        plot_data['dt'] = []
        plot_data['AverageTemperature'] = []

        var units = unit_select.get('value');
        if (units == 'Celcius') {
            for (i = 0; i < new_data['dt'].length; i++) {
                plot_data['dt'].push(new_data['dt'][i])
                plot_data['AverageTemperature'].push(new_data['AverageTemperature'][i])
            }
        } else if (units == 'Fahrenheit') {
            for (i = 0; i < new_data['dt'].length; i++) {
                plot_data['dt'].push(new_data['dt'][i])
                plot_data['AverageTemperature'].push(new_data['AverageTemperature'][i] * 1.8 + 32.0)
            }
        }

        plot_source.trigger('change');
    """)
country_select = Select(value='United States', title='Country', options=countries, callback=countries_callback)

# paragraph widgets to add some text
p0 = Paragraph(text="")
p1 = Paragraph(text="Update data in graph with options above.")
p2 = Paragraph(text="Click and drag on graph to zoom in with box-zoom tool (reset zoom on toolbar).")
p3 = Paragraph(text="Hover over data for values.")
p4 = Paragraph(text="Try selecting a tropical location to see an obvious warming trend. For example: Amazonas, Brazil")

# set the page layout
layout = HBox(VBox(country_select, state_select, unit_select, p0, p1, p2, p3, p4, width=200), plot, width=1250)
save(layout)
