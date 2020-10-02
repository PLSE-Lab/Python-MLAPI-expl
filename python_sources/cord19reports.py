"""
CORD-19 reporting support. This script contains methods to support reporting notebooks.
"""

import os
import re
import shutil

import pandas as pd
import yaml

from IPython import get_ipython
from IPython.display import display, Markdown

def install():
    """
    Installs the cord19q project and supporting files.
    """

    # Install cord19q project
    get_ipython().magic("%pip install git+https://github.com/neuml/cord19q")

    # Download models to prevent download progressbars
    from paperai.pipeline import Pipeline
    Pipeline("NeuML/bert-small-cord19qa", False)

    # Copy vectors locally for predictable performance
    shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")

def run(task):
    """
    Main entry point to build a report.
    
    Args:
        task: YAML configuration string
    """

    # Load task YAML
    config = yaml.safe_load(task)

    # Task name
    tid = config["id"]    
    name = config["name"]

    # Parse out files based on query tasks
    files = ["%s.csv" % key for key in config if key not in ("id", "name", "fields")]

    # Build base reports
    report(task)

    # Build consolidated csv with all results
    concat(name)

    # Merge results into existing files
    merge("%s_%s" % (tid, name), files)

    # Render Markdown
    render(name)

# Builds Markdown and CSV reports
def report(task, model="../input/cord-19-analysis-with-sentence-embeddings/cord19q"):
    """
    Builds Markdown and CSV reports for the input task.
    
    Args:
        task: YAML configuration string
        model: path to embeddings model, if not provided will use default model
    """

    from paperai.report.execute import Execute

    Execute.run(task, 50, "md", model)
    Execute.run(task, 50, "csv", model)

def concat(name):
    """
    Builds a concatenated csv file.
    
    Args:
        name: task name
    """

    output = "%s.csv" % name

    # Delete prior run if it exists
    if os.path.exists(output):
        os.remove(output)

    df = None

    # Process each csv
    for f in sorted(os.listdir(".")):
        if f.endswith(".csv"):
            # Read CSV
            csv = pd.read_csv(f)
            csv.insert(0, "Query Name", queryname(f))

            # Combine into single dataframe
            df = csv if df is None else pd.concat((df, csv))

    # Write combined csv
    df.to_csv(output, index=False)

def queryname(name):
    """
    Formats a query name string to use with a concat csv report.
    
    Args:
        name: input query name
    
    Returns:
        formatted query name
    """

    # Get base name
    name = os.path.splitext(name)[0]
    name = name.replace("_", " ").replace(".", " ")

    # Capital Case String
    return name.capitalize()

def merge(source, files):
    """
    Merges a list of existing reviewed files with newly generated files.

    Args:
        source: data source
        files: list of files to merge
    """

    # Directories
    kpath = os.path.join("../input/CORD-19-research-challenge/Kaggle/target_tables/", source)
    path = "./"

    # Additional fields to add to help with analysis
    additional = ("Sample Text", "Matches", "Entry")

    # Combine files
    for f in files:
        combine(os.path.join(kpath, f), os.path.join(path, f), additional)

def combine(review, generate, additional):
    """
    Combines an existing CSV file with newly generated rows, deduplicating new rows on the study name.

    Args:
        review: existing review csv
        generate: newly generated csv
        additional: additional fields to add to help with analysis
    """

    # Output file path
    output = os.path.basename(review)

    # Read CSVs
    review = pd.read_csv(review)
    generate = pd.read_csv(generate)

    # New rows
    rows = []

    # Key field
    key = "Study"

    # Process dataframe
    for index, row in generate.iterrows():
        if not key in review or not review[key].str.match(re.escape(row[key])).any():
            # Merge review and additional columns from row into new entry
            entry = [find(row, generate.columns, column) for column in review.columns]
            rows.append(entry + [row[column] for column in additional])

    # Merge, name index column, and write output
    review = pd.concat([pd.DataFrame(rows, columns=list(review.columns) + list(additional)), review], sort=False).reset_index(drop=True)
    review.columns.values[0] = ""
    review.to_csv(output, index=False)

def find(row, columns, column):
    """
    Searches row (case-insensitive, strips whitespace) for column. If found, column value is returned. None returned otherwise.

    Args:
        row: input row
        columns: input columns
        column: search column name

    Returns:
        row[column] if found (case-insensitive, strips whitespace search), false otherwise
    """

    for x in columns:
        if x.strip().lower() == column.strip().lower():
            return row[x]
    return None

def render(task):
    """
    Renders a markdown report.
    
    Args:
        task: task name with Markdown content to render
    """

    display(Markdown(filename="%s.md" % task))
