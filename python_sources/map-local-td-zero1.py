# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas as pd

{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 27512,
      "digest": "sha256:8ef19b5397d8b13638e69746589be8265f4e9f565ee5af4d64f43f5d14a68a64"
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
         "size": 526,
         "digest": "sha256:36c8cee5dcabe015f8e5b00d9e5f26f3dc43c685616a9ff657aeac32dcb0dec7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 456,
         "digest": "sha256:fbde6884bcec90a734814ab616cc8abcf34cde78a99498df8da757431c6c28fd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 13117845,
         "digest": "sha256:4aceba2705e51efc04a48b7883d097f3c89d00a2f96b2fb16b54a7d5fc410e53"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 558997840,
         "digest": "sha256:690778d6efe115dbba1239a78693944fe179985f5a5d31078d376731eb900635"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 718986208,
         "digest": "sha256:cfc8fe521bf9c7e028edea60d6f3cbd2a50f56751c0e8d7415d6d364453b41d0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 33502401,
         "digest": "sha256:5a2d591ac4f68ab561f030733f354b722051f02fb7114a632a980d4095e9f6a5"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95925401,
         "digest": "sha256:b720a0e96c3024ee325ea8e1874a33d66d097c990ac50e8229b1c76076ae869a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 142490052,
         "digest": "sha256:a661d11e464bd9abfabe3ec4b4b4e22b01c228481ba20d5dd6c066ab512e26c2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1038014421,
         "digest": "sha256:555cc8cba1c97f86ef332cee16e11b952f18352f55b915df4a9a776d81edd234"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 75007872,
         "digest": "sha256:7ed8a9307f830dad6f7b8b273c80b0a820bdf9f9db7ad1c762282ef8b63e4122"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 80474425,
         "digest": "sha256:29eb9237adacfa8ff7974c2ce5e9f1ffc5047e625347bbb03b5a170d397153fb"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 89451858,
         "digest": "sha256:6b054a59f9fd46636ecb9f0c31a837127ae856fd44e5b998286f5f1111bf1d30"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 415384338,
         "digest": "sha256:19c088e1afee706e063eaff6a2d259efb55b962f4da47927f9461a83d904c8a1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 44168262,
         "digest": "sha256:838b7e776f75e4fdce36596b5ee8e250ebb50cdc2717033290df2bff0e70a7dc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 107758705,
         "digest": "sha256:1ad83d09763421093849a7abef397f8610f79e07767f51e3248ab9ef52679705"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 958342282,
         "digest": "sha256:569c6fda9d84413ea844c2f25799b7449b2fd6ac486bcbb8be2eb1ca65b6c51e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 588654554,
         "digest": "sha256:d8ad2accaf088624da0281224e625ff49da8212cf9e21423898f50f648542d40"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21654895,
         "digest": "sha256:1754ab792f2a4062623d4f461f9196ed41ec1bf9eb81b45ea05fe0fe6a4df3c0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 227370423,
         "digest": "sha256:153e0f49aeb357a372ddccfffe487d5b431fc81143728d1a65805a38454c477a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 20307,
         "digest": "sha256:6f8920d2e4f3aa5274096bbbf084a9587d1ff438026fac3ee7931e3f75008de3"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 85317368,
         "digest": "sha256:5857d6464a2ed88c711915621eb005d178603b9daabbf65ccfe2fd2e72d7be36"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3264,
         "digest": "sha256:5347c992f3b56e47242dd8a5694638c839cad94e9635876f2bfe9e8dd36dd62c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2163,
         "digest": "sha256:dd6f840a7b975737ae3f11a10036c7501bd6796ca86befd2596712365a9fd073"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1269,
         "digest": "sha256:a12c0432261d580586748b11db6bbfe798f5957a9ad57a71230c0f9986826114"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 641,
         "digest": "sha256:112b56a741fa6492ba1a4f9eda937bcb52f02f7c31265e142a592824bf830c36"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2052,
         "digest": "sha256:bcd81def64e80646bbebb0cd99ecfe423c0ec3df21c607fceb2f9c3a2b782e1e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 876,
         "digest": "sha256:daf7bad905212cda27468f9f136e888189f0cde90182e6eb488937740a70ac38"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 210,
         "digest": "sha256:37f94f1dfe09302f5ab426ed04a71a4bad5cc9585d65611518efb8ebc1ea5ba5"
      }
   ]
}


# Any results you write to the current directory are saved as output.