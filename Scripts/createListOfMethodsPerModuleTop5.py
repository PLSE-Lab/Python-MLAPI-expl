import ast
from pprint import pprint
import os
import pathlib

sourceDataFolder = '~/Documents/ECU/Thesis/python_sources//'


def main():
    totalErrors = 0  # count of errors
    # Define global dictionary with libraries to store methods from all files
    res = {"numpy": [], "pandas": [], "sklearn": [], "keras": [], "matplotlib": []}
    with open("results_methods.txt", "w") as f:
        directory = os.fsencode(os.path.expanduser(sourceDataFolder))
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".py"):
                with open(pathlib.Path(directory.decode("utf-8"), filename), "r") as source:
                    try:
                        tree = ast.parse(source.read())
                        if res == '':
                            res = {"np": [], "pd": [], "sklearn": [], "keras": [], "plt": []}
                        analyzer = Analyzer(res, filename)
                        analyzer.visit(tree)
                        res = analyzer.report()
                    except Exception:
                        totalErrors = totalErrors + 1
                        # pprint(e)
        f.write(str(res))
    pprint(totalErrors)


class Analyzer(ast.NodeVisitor):
    def __init__(self, results: dict, filename: str):
        self.stats = []
        self.filename = filename
        self.module = ["numpy", "pandas", "sklearn", "keras", "matplotlib"]
        self.results = results
        # Dictionaries for storing import from and import as values
        self.imports = {"numpy": [], "pandas": [], "sklearn": [], "keras": [], "matplotlib": []}
        self.importsAs = {"numpy": [], "pandas": [], "sklearn": [], "keras": [], "matplotlib": []}
        self.importsCalled = []

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
            Assign sub-libraries to the main libraries
            E.g.: from sklearn.svm import SVC, LinearSVC
            All calls related to SVC and LinearSVC will be count to sklearn
        """
        if node.module in self.module or node.module.split(".", 1)[0] in self.module:
            for importNode in node.names:
                # use split to support from *.* expressions
                self.imports[node.module.split(".", 1)[0]].append(importNode.name)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Import(self, node):
        """
            Assign aliases to the main libraries
            E.g.: import pandas as pd
            All calls with pd will be count to pandas
        """
        if len(node.names) == 1 and node.names[0].asname:
            if node.names[0].name.split(".", 1)[0] in self.importsAs.keys():
                self.importsAs[node.names[0].name.split(".", 1)[0]].append(node.names[0].asname)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if None != node.value and isinstance(node.value, ast.Attribute):
            if node.value.attr in self.module:
                self.results[node.value.attr].append(node.attr)
            else:
                if None != node.value.value and isinstance(node.value.value, ast.Name):
                    if node.value.value.id in self.module:
                        self.results[node.value.value.id].append(node.attr + "." + node.value.attr)
        elif node.value is not None and isinstance(node.value, ast.Name):
            if node.value.id in self.module:
                self.results[node.value.id].append(node.attr)
            else:
                for keyAs, valueAs in self.importsAs.items():
                    if node.value.id in valueAs:
                        self.results[keyAs].append(node.attr)
        elif node.value is not None and isinstance(node.value, ast.Call):
            if node.value.func.id in self.module:
                self.results[node.value.func.id].append(node.attr)
            else:
                for keyAs, valueAs in self.importsAs.items():
                    if node.value.func.id in valueAs:
                        self.results[keyAs].append(node.attr)
                for keyAs, valueAs in self.imports.items():
                    if node.value.func.id in valueAs:
                        self.results[keyAs].append(node.attr)

    def visit_Call(self, node: ast.Call):
        # pprint(self._name)
        if isinstance(node.func, ast.Name):
            for key, value in self.imports.items():
                if value and node.func.id in value:
                    self.importsCalled.append(node.func.id)
                    self.results[key].append(node.func.id)
            for key, value in self.importsAs.items():
                if value and node.func.id in value:
                    self.importsCalled.append(node.func.id)
                    self.results[key].append(node.func.id)
        if isinstance(node.func, ast.Attribute):
            for key, value in self.imports.items():
                if isinstance(node.func.value, ast.Name) and node.func.value.id in value:
                    self.importsCalled.append(node.func.value.id)
                    self.results[key].append(node.func.attr)
            for key, value in self.importsAs.items():
                if isinstance(node.func.value, ast.Name) and node.func.value.id in value:
                    self.importsCalled.append(node.func.value.id)
                    self.results[key].append(node.func.attr)
        ast.NodeVisitor.generic_visit(self, node)

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def report(self):
        with open("resultsPerFile.txt", "a") as f:
            f.write(self.filename)
            for library_list in self.results:
                if len(self.results[library_list]) == 0:
                    f.write(',0')
                else:
                    f.write(',1')
            f.write('\n')
        with open("resultsPerFileImports.txt", "a") as f:
            f.write(self.filename + ',')
            excessiveImport = 0
            modules = ''
            for module in self.module:
                checkValue = excessiveImport
                for imports in self.imports[module]:
                    if imports not in self.importsCalled:
                        excessiveImport += 1
                for imports in self.importsAs[module]:
                    if imports not in self.importsCalled:
                        excessiveImport += 1
                if checkValue < excessiveImport:
                    modules = modules + ',1'
                else:
                    modules = modules + ',0'
            f.write(str(excessiveImport) + modules)
            f.write('\n')
        return self.results


if __name__ == "__main__":
    main()
