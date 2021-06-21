import ast
from pprint import pprint
import os
import pathlib
import json


def main():
    totalErrors = 0
    res = {"np": [], "pd": []}
    with open("results_modules.txt", "w") as f:
        f.write("values")
        directory = os.fsencode('C:\\Users\\Kate\\Documents\\ECU\\Thesis\\python_sources\\')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".py"):
                with open(pathlib.Path(directory.decode("utf-8"), filename), "r") as source:
                    try:
                        tree = ast.parse(source.read())
                        analyzer = Analyzer(res)
                        analyzer.visit(tree)
                        for item in analyzer.report():
                            f.write(",\n" + item)

                    except Exception as e:
                        totalErrors = totalErrors + 1
    pprint(totalErrors)


class Analyzer(ast.NodeVisitor):
    def __init__(self, results: dict):
        self.stats = []
        self.module = ["np", "pd"]

    def visit_Import(self, node):
        for alias in node.names:
            self.stats.append(alias.name.split(".", 1)[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            module_name = node.module.split(".", 1)[0]
            self.stats.append(module_name)
        self.generic_visit(node)

    def report(self):
        return self.stats


if __name__ == "__main__":
    main()
