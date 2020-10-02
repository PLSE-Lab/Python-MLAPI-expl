#!/usr/bin/env python
# coding: utf-8

# # Add Table of Contents to Notebooks

# ## TLDR;
# Add the cell below to your notebook and run it.

# In[ ]:


get_ipython().run_cell_magic('javascript', '', '!function(){const e=IPython.page.site_div_element.context,t=e.querySelector("#site");if(t.style.paddingRight="300px",e.querySelector("#toc"))return;const i=({parent:t=e.body,type:i="div",id:o,cssText:n,value:r=""})=>{let d=e.createElement(i);return o&&(d.id=o),n&&(d.style.cssText=n),d.innerHTML=r,t.appendChild(d),d};i({id:"toc",cssText:"position:fixed; top:0; right:0; width:300px; height:100%; border-left:1px solid #ccc; padding:20px; overflow-y:auto;"}),i({id:"toc-close",value:"Close",type:"button",cssText:"position: fixed; top: 0; right: 0;"}).addEventListener("click",()=>{t.style.paddingRight="0px",e.body.removeChild(e.querySelector("#toc"))},!1);let o="";const n=()=>{const i=e.querySelector("#toc");if(!i)return;const r=[...t.querySelectorAll("h1, h2, h3, h4, h5, h6")],d=r.map(e=>e.nodeName+e.id).join(",");d!=o&&(o=d,i.innerHTML=r.map(e=>`<a href="#${e.id}" style="${(e=>{return`font-weight:bold; display:block; padding-left:${8*(parseInt(e.nodeName.replace("H",""))-1)}px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;`})(e)}">${e.innerText.trim()}</a>`).join("")),window.requestAnimationFrame(n)};n()}();')


# ## Features
# 
# 1. Auto-refresh - updating cells updates the TOC.
# 2. Cell Jumping - Clicking TOC links jump to the cell (yep it's a TOC)
# 3. Cell Nesting - H1 > H6
# 4. Toggle-able - Able to close (clicking button) and open (by rerunning the cell).
# 5. Minimal overhead

# ## Source
# If interested in how it works, review the following:

# In[ ]:


get_ipython().run_cell_magic('javascript', '', '(function () {\n    const doc = IPython.page.site_div_element.context;\n    // Add some space.\n    const site = doc.querySelector("#site");\n    site.style.paddingRight = "300px";\n    // Toc already exists.\n    if (doc.querySelector("#toc")) return;\n    // Helpers.\n    const $ = ({parent=doc.body, type="div", id, cssText, value=""}) => {\n        let el = doc.createElement(type);\n        if (id) el.id = id;\n        if (cssText) el.style.cssText = cssText;\n        el.innerHTML = value;\n        parent.appendChild(el);\n        return el;\n    };\n    const aStyle = h => {\n        const indent = (parseInt(h.nodeName.replace("H", "")) - 1) * 8;\n        return `font-weight:bold; display:block; padding-left:${indent}px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;`;\n    };\n    // Initialize TOC.\n    $({\n        id: "toc",\n        cssText: "position:fixed; top:0; right:0; width:300px; height:100%; border-left:1px solid #ccc; padding:20px; overflow-y:auto;",\n    });\n    $({\n        id: "toc-close",\n        value: "Close",\n        type: "button",\n        cssText: "position: fixed; top: 0; right: 0;" \n    }).addEventListener("click", () => {\n        site.style.paddingRight = "0px";\n        doc.body.removeChild(doc.querySelector("#toc"));\n    }, false);\n    \n    let key = "";\n    const buildTree = () => {\n        const toc = doc.querySelector("#toc");\n        if (!toc) return;\n        const headers = [...site.querySelectorAll("h1, h2, h3, h4, h5, h6")];\n        // Prevent unnecessary renders.\n        const newKey = headers.map(h => h.nodeName + h.id).join(",");\n        if (newKey != key) {\n            key = newKey;\n            toc.innerHTML = headers.map(h => `<a href="#${h.id}" style="${aStyle(h)}">${h.innerText.trim()}</a>`).join("");\n        }\n        window.requestAnimationFrame(buildTree);\n    };\n    buildTree();\n})();')


# # H1 Example

# ## H2 Example

# ### H3 Example

# #### H4 Example

# ##### H5 Example

# ###### H6 Example
