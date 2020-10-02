#!/usr/bin/env python
# coding: utf-8

# # Rendering Altair Plots in Kaggle
# 
# Background: [Altair](http://altair-viz.github.io) is a visualization library in Python... kaggle supports it, but the plots do not render in static views.
# 
# There is a popular workaround published here: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey
# 
# It would be better if there were a built-in rendering solution that didn't involve all that copying and pasting.

# ## Simple Chart definition
# 
# Let's create a dataframe and visualize it with Altair:

# In[ ]:


import altair as alt
import numpy as np
import pandas as pd


# In[ ]:


rand = np.random.RandomState(578493)
data = pd.DataFrame({
    'x': pd.date_range('2012-01-01', freq='D', periods=365),
    'y1': rand.randn(365).cumsum(),
    'y2': rand.randn(365).cumsum(),
    'y3': rand.randn(365).cumsum()
})

data = data.melt('x')
data.head()


# In[ ]:


chart = alt.Chart(data).mark_line().encode(
    x='x:T',
    y='value:Q',
    color='variable:N'
).interactive(bind_y=False)


# ## Notebook renderer
# 
# Altair ships with a notebook renderer that uses the [ipyvega](http://github.com/vega/ipyvega) package and jupyter extension to render plots.
# 
# This works in Kaggle if you are using a live kernel, but the plot does not show up when looking at the static view. I suspect this is because it accesses JS resources from the Jupyter kernel extension, which is not available in the static view:

# In[ ]:


alt.renderers.enable('notebook')
chart


# ## Kaggle Renderer 
# 
# A workaround has been published by a kaggle user at https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey/. This defines a ``render()`` function that loads resources from an external CDN that is available in both the executed and in the static view (note that it is the frontend accessing the resources rather than the backend, so it's fine even if the internet is blocked).
# 
# I've adapted the approach there to define and register an altair renderer, so that users just have to run ``alt.renderers.enable('kaggle')``. Source is here:

# In[ ]:


# Define and register a kaggle renderer for Altair

import altair as alt
import json
from IPython.display import HTML

KAGGLE_HTML_TEMPLATE = """
<style>
.vega-actions a {{
    margin-right: 12px;
    color: #757575;
    font-weight: normal;
    font-size: 13px;
}}
.error {{
    color: red;
}}
</style>
<div id="{output_div}"></div>
<script>
requirejs.config({{
    "paths": {{
        "vega": "{base_url}/vega@{vega_version}?noext",
        "vega-lib": "{base_url}/vega-lib?noext",
        "vega-lite": "{base_url}/vega-lite@{vegalite_version}?noext",
        "vega-embed": "{base_url}/vega-embed@{vegaembed_version}?noext",
    }}
}});
function showError(el, error){{
    el.innerHTML = ('<div class="error">'
                    + '<p>JavaScript Error: ' + error.message + '</p>'
                    + "<p>This usually means there's a typo in your chart specification. "
                    + "See the javascript console for the full traceback.</p>"
                    + '</div>');
    throw error;
}}
require(["vega-embed"], function(vegaEmbed) {{
    const spec = {spec};
    const embed_opt = {embed_opt};
    const el = document.getElementById('{output_div}');
    vegaEmbed("#{output_div}", spec, embed_opt)
      .catch(error => showError(el, error));
}});
</script>
"""

class KaggleHtml(object):
    def __init__(self, base_url='https://cdn.jsdelivr.net/npm'):
        self.chart_count = 0
        self.base_url = base_url
        
    @property
    def output_div(self):
        return "vega-chart-{}".format(self.chart_count)
        
    def __call__(self, spec, embed_options=None, json_kwds=None):
        # we need to increment the div, because all charts live in the same document
        self.chart_count += 1
        embed_options = embed_options or {}
        json_kwds = json_kwds or {}
        html = KAGGLE_HTML_TEMPLATE.format(
            spec=json.dumps(spec, **json_kwds),
            embed_opt=json.dumps(embed_options),
            output_div=self.output_div,
            base_url=self.base_url,
            vega_version=alt.VEGA_VERSION,
            vegalite_version=alt.VEGALITE_VERSION,
            vegaembed_version=alt.VEGAEMBED_VERSION
        )
        return {"text/html": html}
    
alt.renderers.register('kaggle', KaggleHtml())
print("Define and register the kaggle renderer. Enable with\n\n"
      "    alt.renderers.enable('kaggle')")


# In[ ]:


alt.renderers.enable('kaggle')
chart


# Multiple renderings in one notebook work:

# In[ ]:


chart.mark_circle()


# We should think about where to host this so that users can use Altair in kaggle without any copying and pasting required.
