# from: https://www.kaggle.com/yagays/two-sigma-financial-modeling/python-package-list

import pkg_resources
for dist in pkg_resources.working_set:
    print(dist.project_name, dist.version)