import ast
from pprint import pprint
import os
import pathlib

# Path to the source files extracted from Kaggle
sourceDataFolder = '~/Documents/ECU/Thesis/python_sources//'


def main():
    totalerrors = 0
    res = {}
    with open("results.txt", "w") as f:
        directory = os.fsencode(os.path.expanduser(sourceDataFolder))
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".py"):
                with open(pathlib.Path(directory.decode("utf-8"), filename), "r") as source:
                    try:
                        tree = ast.parse(source.read())
                        analyzer = Analyzer(res)
                        analyzer.visit(tree)
                        res = analyzer.report()
                    except Exception:
                        totalerrors = totalerrors + 1
                        # pprint(e)
        f.write(str(res))
    pprint(totalerrors)


class Analyzer(ast.NodeVisitor):
    def __init__(self, results: dict):
        self.stats = []
        self.results = results

    # def visit_Name(self, node:ast.Name):
    #     pprint(node.id)
    def visit_Attribute(self, node: ast.Attribute):
        if node.value is not None and isinstance(node.value, ast.Attribute):
            if node.value.attr in self.module:
                self.results[node.value.attr].append(node.attr)
            else:
                if node.value.value is not None and isinstance(node.value.value, ast.Name):
                    if node.value.value.id in self.module:
                        self.results[node.value.value.id].append(node.attr + "." + node.value.attr)
        elif node.value is not None and isinstance(node.value, ast.Name):
            if node.value.id in self.module:
                self.results[node.value.id].append(node.attr)
        # pprint(node.attr)

    # def visit_Call(self, node: ast.Call):
    #     pprint(node.func)
    # if (isinstance(node.ctx, ast.Load)):
    #     node.ctx.value
    def generic_visit(self, node):
        # pprint(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    # def visit_Import(self, node):
    #     for alias in node.names:
    #         self.stats.append(alias.name)
    #     self.generic_visit(node)
    #
    # def visit_ImportFrom(self, node):
    #     for alias in node.names:
    #         self.stats.append(alias.name)
    #     self.generic_visit(node)

    def report(self):
        return self.results


if __name__ == "__main__":
    main()
