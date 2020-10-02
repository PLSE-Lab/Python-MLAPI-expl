#!/usr/bin/env python
# coding: utf-8

# # Welcome to Advanced Pandas
# 
# Welcome to the Advanced Pandas tutorial!
# 
# The `pandas` library is arguably the most important library in the Python data science stack. It is the library of choice for working with small to medium sized data (anything that isn't quote-unqutoe "big data"). Much to most of the data manipulation that's done in the Python world is done using `pandas` memory structures and tools, making it one of the most important libraries in the ecosystem.
# 
# Understanding the tools that `pandas` provides is foundational to much of the rest of what you will do. Hence this tutorial track is targetted at users completely new to data science and machine learning. Users wishing to refresh their data munging skills will also find this material helpful.
# 
# Our goal is to get you started off performing data munging operations as quickly as possible. As such, this tutorial is highly opinionated. We will not attempt to be comprehensive, but will instead cover a sequence of topics which you will need most of all in your own work.
# 
# By the end of this tutorial you will have a strong foundation in `pandas` operations and in working with data in Python. You will understand the core tenets of working with data in the Python ecosystem. You will be ready to start doing your own exploratory data analysis, and to confidently move on to any of the other Learn tutorials.
# 
# ## Contents
# 
# This tutorial consists of seven sections. Each section is divided into a workbook with exercises and a reference guide with explanations.
# 
# If you are completely new to Python, we recommend reading the references, then doing the exercises in each section. 
# 
# If you have _some_ prior experience, you may prefer to just do the exercises, and only consult the complimentary references if you absolutely need to.
# 
# <table style="width:800px">
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/creating-reading-and-writing-workbook">Creating, reading, writing workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/creating-reading-and-writing-reference">Creating, reading, writing reference</a></td>
# </tr>
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/indexing-selecting-assigning-workbook">Indexing, selecting, assigning workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/indexing-selecting-assigning-reference">Indexing, selecting, assigning reference</a></td>
# </tr>
# 
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/summary-functions-and-maps-workbook">Summary functions and maps workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/summary-functions-and-maps-reference">Summary functions and maps reference</a></td>
# </tr>
# 
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/grouping-and-sorting-workbook">Grouping and sorting workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/grouping-and-sorting-reference">Grouping and sorting reference</a></td>
# </tr>
# 
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/data-types-and-missing-data-workbook">Data types and missing values workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/data-types-and-missing-data-reference">Data types and missing values reference</a></td>
# </tr>
# 
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/renaming-and-combining-workbook">Renaming and combining workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/renaming-and-combining-reference">Renaming and combining reference</a></td>
# </tr>
# 
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/method-chaining-workbook">Method chaining workbook</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/method-chaining-reference">Method chaining reference</a></td>
# </tr>
# </table>
# 
# Ready? [To start the tutorial, proceed to the next section, "Creating, reading, writing"](https://www.kaggle.com/residentmario/creating-reading-and-writing-workbook).
