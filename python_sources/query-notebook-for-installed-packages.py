import pandas as pd
import pkg_resources

# reference: https://github.com/pypa/pip/issues/5243#issuecomment-381513000
packages = sorted(
    (package for package in pkg_resources.working_set),
    key=lambda x: x.project_name
)

rows = [(package.project_name, package.version) for package in packages]

packages = pd.DataFrame(rows, columns=['package_name', 'version'])
packages.set_index('package_name', inplace=True)
packages.to_csv('kaggle-notebook-packages.csv')
