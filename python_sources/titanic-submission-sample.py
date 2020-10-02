import base64
import gzip
import os
from pathlib import Path
from typing import Dict

# this is base64 encoded source code
file_data: Dict = {
    'titanic_sample/dataset.py': 'H4sIAHQXAl4C/6VVUW/aMBB+R+I/WLzE0VCA7Q0pD6grVbVp7QZvCEUmOahVx7FsB8Gm/fednYQktKs2Na3ks+/7Pp/Pd2avi5woZp8E3xGeq0Jb8ojT4WA4qKeyzNWZMEOkuqwpJjNcwX+VDQd7J2KeBTAto7zIQCQGBKSWF7IRtZpxmVgwNjFKcOs28DzLLZM8TQzLlYCotFyYhvTwLfmyuLv7euvQnxfrRfLj4WFNYh8iDaJowqUq7aTWCAjftxwCwgAJookGU5Q6BROEw8ENs3AoNE+ZSJbAbIleFNwEt/mO6WfIgjEJHlPBjHHWCk7Bdjioond5QmwbyISMvCdKzXGEKHe6V0HoqDGm3OX8ddQBZAY68QhjMHc1xf1lsCdKg9IFnsPQjFkWzocDgp+zN1WgESIES4FugpwJcAfYg7e2Y7KZjskMR0yZw8RrXULYlbhkYBvtuRCS0WAV/BMeT/JCImeK/kKBOcF9gxscZzh+x/Hj7zBixp4V0ErOfVzanviSaWgDkSrKgUna9YVvhLY4dMjdpRwyjjpvcZcs5+K84j+hc7BHptMnnH9o0s13K+Xnsy733ixEISvitHVEokg3r6nHPildVi2nAStTenJbAKJgWVUelja3XwNVhnePXqwY2imx8Irs9AxYWvUhBhFPo0+NlC9lDKGr1RZ+nSPHvMY0Ze938yCmD2CTtBAIHa1KfeRHyEZtQiqFtJAps3TjNxl7aaxPg50fLxk2b9gnXJV/7cSHxoLbqsxl1cnfWO5L/xF7GOQB9L1v6jVPn8E664btuHRtfbmfTBeK9qXGhJ24iWcvKqWfLH+rcwGySlW47aXJe1vnfNvQz0kj4MdNm7Fa4NQHVBG2qCa68IKuNvQvzd+xPe0xGkcmeDZuwnGGX2m27TzY9NRiKqNt3v//2gK8WO/S0/iDVOSJsfi8x9N3SRk8neX7c1yfM+w1Gn2ZvCr5+KTQ6zQi9Q+pvtnNYQcAAA==',
    'titanic_sample/predict.py': 'H4sIAHQXAl4C/31QwWrEIBC9B/IPspcYCNJLL4UU2kPPhR5LGcw6SYUYg06W3b+vRpPdbqEi6Pjec948bWbriEk3zNJ5LIuy0Olp1MM3DZ1h0rNx6CLSO2sYaZKTPoKXZh5RKEnSI7FNZaWC/Nakyi+d0RT1ZaGwZwMShH6LwYk8r5/KgoW1tnes3a2Il8x5XxFe3/KEVGr/hB+MVTjCLOn78B/NLjQvlA39ogeOD72zaj2izm9dHdLippV2HcRIPe3+8w93wyWMQ8PCPgOhp7pJdR3Yt2lt5OBuQ5LRDViHjNDQiVdrPYVQ0uC9HrGNBsQ1iCy6wOxQBdWKiFjoI/Fs5Y7D8+2ZPYjHWkhPlxm5nuhq7bP6WNxJn1BVX0GRBDsqyMLRn/jq5W/aDdOTwnP7JkePdQpS9wxgkgYBWNuyCiDGClDlXFPIZfEDh06KTakCAAA=',
    'titanic_sample/train.py': 'H4sIAHQXAl4C/5VUTY+bMBC9R8p/sHKBSCxN1dtKPqy0Pe+q3dtqZQ0wsG4xINuJklb97x1/AEmWRiocsPzezLzxPCPV0GvLQDcDaIPr1Xolw1Yrm3fbFIqBYW1TOKTWvWJWWuhkKQyoocW8AgsGLYtRjw8vD+Lb09NLxtoeKhFhF201yE4MYN8Zn3nsE9t4JC/NYUMsNHaZREDkuLfCmjVoBSnfK+ysSbf36xWjxzeiKXxsKn+InGePpNtzXg5VNSVJN6qvsPX1N7dod3dep5G/cJMxexqQ19SvzRjpgn1r+S7/EhNQlCE1MY//uExm1KHR7nXnaXNr/kjSo/DfjJ3GxVEcoJWV24mLEiw2vZYltKJGoFxoxqOguYVAqk/r/DFM42PeQP/Xs1CCL5adqyKp+1D0WrrGGjV2JfJJ6G0l/yFkGh4od/q/57xJX/zA0soDJvcsKWQH+pQE+M8Y520Q9YdRzPEhZ8aWNPvGBHVr+OuMTyfydk5FXfRkBbfPP+/OkG6vRNH35C/d77uKwN05jKDbkzC2HwbZNYFjiBQYl6byfcyuUq6T0RzRl1eXKGCTu8M1Mvm0EfDZQdNUaYc424ylp9ldHtu6gzz7GaRTslhtaYQU85p8VQXon1glGUueyxaMcavveEzeLud0cV34tb35LZvHBvi1Pflo06WgJcW3roTXmRs4oPDL1B/q/LPZhhnJmgnRgUIhGOcsEcJNTIgkjizMb736C/g+V6a2BQAA',
    'titanic_sample/utils.py': 'H4sIAHQXAl4C/yXNsQqDMBCA4T2Qdwg4OOnVLqVCB6FFRFFw6SimBD2a5II5C337tnT8v+VHF2hjRVEKKRK1ModYAizI667zBzmwFPYVPTznZbEmQ2c4Ox6KM2hLGtwc2WzwU9gZbczDO+mKkxRDP7VVXXe3Umkiqy4q/fd0H8a26evp2oypQv+d58a/cCMvxQet+8gdkAAAAA==',
    'titanic_sample/__init__.py': 'H4sIAHQXAl4C/wMAAAAAAAAAAAA=',
    'setup.py': 'H4sIAHQXAl4C/0srys9VKE4tKS0oyc/PKVbIzC3ILyqBiPBy8XKBGRq8XApAkJeYm2qrXpJZkpiXmRxfnJhbkJOqrgORK0hMzk5MTy22jUZXEAtUocnLBQBmZqg3aQAAAA=='}

for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
run('python titanic_sample/train.py model.text --test_size 0.3')
run('python titanic_sample/predict.py model.text submission.csv')
