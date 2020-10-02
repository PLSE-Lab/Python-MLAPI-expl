# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 8427,
      "digest": "sha256:32a1a53fdb28035db0bdb568c202ca7f13195427d01962a8f62e25bc9d6622eb"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 51081712,
         "digest": "sha256:ab85f11dd933cfda19130a6b6986c591e8e70d8a2ac8a73dddd8722c7ec8eab1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 825,
         "digest": "sha256:2922171b2513070bd982f9030107e0a9b22fa141ce22fe185cea0b22dc34dc32"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 442,
         "digest": "sha256:03dd2bb85c8ab012f80afe6dee177f3c05e20bdfc059beeb2c03ad6ca14189f2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 676,
         "digest": "sha256:4a9039a61c36f62d62cb71a8087c71fc6046b0918617a8ae13f20fa48060983f"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 163,
         "digest": "sha256:82e5f1ef816256c961c51a8deae0ff5c038e7a4e92f0b52bf1d60340f21d9502"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 833526336,
         "digest": "sha256:67eed9560bab0b1bf847ffbc73f384a2841d3680a37a3681c9ac8cd89e38e46d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 568,
         "digest": "sha256:ce8abd2c7b1cd94ca3fc88a34ea7c6d451b197b28bc02c5f6cf9714b95996ecd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 343945402,
         "digest": "sha256:11bc6f91b846cb9c1efddd0e132773099bc29e577eca085742566e5c258e8321"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3029994,
         "digest": "sha256:91433bc796c21fe3d7b513c5aab421b10c784bb22712ec1b68698b0910b8116e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 57935,
         "digest": "sha256:1e5c91351872e093c596d34ad80f63f8a539a57b18fba87a7ab708aa79983ed1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 81865183,
         "digest": "sha256:fb8c8966667d0589428e3f40598742586a49b4e484339cdac3196c2c22317394"
      }
   ]
}

{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 31231,
      "digest": "sha256:79f52292b1d0c079b84bc79d002947be409a09b15a1320355a4de834f57b2ee8"
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
         "digest": "sha256:19abed793cf0a9952e1a08188dbe2627ed25836757d0e0e3150d5c8328562b4e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 458,
         "digest": "sha256:df204f1f292ae58e4c4141a950fad3aa190d87ed9cc3d364ca6aa1e7e0b73e45"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 13119161,
         "digest": "sha256:1f7809135d9076fb9ed8ee186e93e3352c861489e0e80804f79b2b5634b456dd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 555884253,
         "digest": "sha256:03a365d6218dbe33f5b17d305f5e25e412f7b83b38394c5818bde053a542f11b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 102870915,
         "digest": "sha256:00e3d0b7af78551716541d2076836df5594948d5d98f04f382158ef26eb7c907"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95925388,
         "digest": "sha256:59782fefadba835c1e83cecdd73dc8e81121eae05ba58d3628a44a1c607feb6e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 142481172,
         "digest": "sha256:f81b01cf2c3f02e153a71704cc5ffe6102757fb7c2fcafc107a64581b0f6dc10"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1128076783,
         "digest": "sha256:f08bbb5c2bce948f0d12eea15c88aad45cdd5b804b71bee5a2cfdbf53c7ec254"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 444800302,
         "digest": "sha256:b831800c60a36c21033cb6e85f0bd3a5f5c9d96b2fa2797d0e8d4c50598180b8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 157365696,
         "digest": "sha256:6d354ec67fa4ccf30460efadef27d48edf9599348cbab789c388f1d3a7fee232"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 63237273,
         "digest": "sha256:464f9b4eca5cdf36756cf0bef8c76b23344d0e975667becb743e8d6b9019c3cd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 427820542,
         "digest": "sha256:6c1f6bcbc63b982a86dc94301c2996505cec571938c60a434b3de196498c7b89"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 44581800,
         "digest": "sha256:c0a8110c6fede3cf54fa00a1b4e2fcb136d00b3cf490f658ec6d596e313c986e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 127637178,
         "digest": "sha256:c25df885c8dea40780f5b479bb6c7be924763399a21fa46c204d5bfac45056bd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 956429221,
         "digest": "sha256:7c1d98590e22f78a1b820f89b6ce245321437639957264e628b4abf4862e1223"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 586276809,
         "digest": "sha256:aab720d802b7d006b679ac10df4a259b3556812dea4dfc52d6111db47fc41e62"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21717560,
         "digest": "sha256:5ee4a4cda8613a3fb172a827143aadacb98128479a22a280495604f989bf4483"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 93512644,
         "digest": "sha256:c4699852e987bc3fe9adde2544ffa690ad52ebec229c20f7e4153b015ac238ff"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 19141,
         "digest": "sha256:8d93692c8dcecacb8aca746a868f53d0b0cf1207e08ced8ffb2134bb01c1f871"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 84125618,
         "digest": "sha256:57c74d175611802a57531be97d19f586dc9cd810a5490eab04fd40b648312ead"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3261,
         "digest": "sha256:1ac7a265bf03308e06e9cad7e77d12b22ca8bc6b7791d46398d95977e0042574"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2162,
         "digest": "sha256:1b4a5be69a4439f3de72877b7d408400e3aa0b4c37e9c70c4490b480bce682c0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1270,
         "digest": "sha256:648046d6f6c28a42a39c9e68a9da90ccdabbd1ecfd0be77941114df4eb2406a4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 644,
         "digest": "sha256:19a794f6956d460edfe74d5562d44366a7cf8bd46d83f408f1bf3c46e7282464"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2052,
         "digest": "sha256:880f92e310c2e03c28c5db85b342069b1a56cd13de7998ae52f699829774f075"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 875,
         "digest": "sha256:cad389727d6cd1696ed7e91b70eedd4c86fd30babb648e7be6cc1639582b0928"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 214,
         "digest": "sha256:c873da9a657a590abeae80bd3c0d0d87a6bfdfaf1d3873a0f210760a4050d6db"
      }
   ]
}
