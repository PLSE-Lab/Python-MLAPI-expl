# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 11617,
      "digest": "sha256:0473857677cc00277a750af3733caed6ccf78c01335648a5bdf4c0d835b92da7"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 45339314,
         "digest": "sha256:c5e155d5a1d130a7f8a3e24cee0d9e1349bff13f90ec6a941478e558fde53c14"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95104141,
         "digest": "sha256:86534c0d13b7196a49d52a65548f524b744d48ccaf89454659637bee4811d312"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1571501372,
         "digest": "sha256:5764e90b1fae3f6050c1b56958da5e94c0d0c2a5211955f579958fcbe6a679fd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1083072,
         "digest": "sha256:ba67f7304613606a1d577e2fc5b1e6bb14b764bcc8d07021779173bcc6a8d4b6"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 13108480,
         "digest": "sha256:9c191e3783f6796bf8de77c8cb91adbc3bfe7567ffca3400e4222e98c3cbcb3a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 263,
         "digest": "sha256:458c86006e8da01fd9b1f48aadc7ed9e66513cbea0cf56e7c16e6709922f00f8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 275,
         "digest": "sha256:721486bee6b8ca9ab02ac828dae260382e86cb8e1c968bc41c0e4ff527803de2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 12906,
         "digest": "sha256:c7f15835ad50f08da5af7764b3748b33f89a36de9e5696a56fc30f62984b69b9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1981681751,
         "digest": "sha256:e59fa20112206d71fb2fab7b9f70fcd8a9b31656f457c34b05031cb4dfb68d18"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 869464956,
         "digest": "sha256:7c67555fe1015616e561fe70f4c016dbc12270f5f6670d0841dabcd4d8096e79"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 210087845,
         "digest": "sha256:c40ff4a9b122bed63cd1e4219271e043f472f81666bf68319e1e37737ac0668d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 556091563,
         "digest": "sha256:8e023c35e0a320d29d5088dd392a5377aca59c8d019a5c46aec328555e79fa9e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 623319097,
         "digest": "sha256:ed83aff7922691f88d76664533309aa3817d6ae48e3b7ef9123dfae1418e38b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 236993773,
         "digest": "sha256:5be0c2fce6efd43d016cee2ae8d8fc1c47305802622a71bf3dc347a15f80bbbf"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 12552,
         "digest": "sha256:ed268e613527f4f5379c5e839c328cdce02c023feecf9dbd0199ecbb16a7fe24"
      }
   ]
}
