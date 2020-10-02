# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

{
     "id": community id,
     "obj_type": "CommunityMap",
     "map_version": map version,
     "entity_version": entity version,
     "properties": property map // see below
     "languages": languages array
     "default_lang": lang,
     "location": lat lon geojson point for community
     "drawings":[drawing list...] //see "Drawing JSON
}

{
     "id": drawing id,
     "obj_type": "Drawing",
     "levels": level list, // see "Level JSON"
     "properties": property map, //
     "object_groups": map type list, // see below
     "ref_frame":{
             "transform": transform from local to lat lon coordinates
             "height": height in local coordinates
             "width": width in local coordinates
             "angle_deg": default angle,
             "local2m": scale factor to convert local coords to meters
             }   
}



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 33518,
      "digest": "sha256:5b0fd34e48683888500052a7b246e6e2eb4962406c17baae9637c3ffc5e77222"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 26692096,
         "digest": "sha256:423ae2b273f4c17ceee9e8482fa8d071d90c7d052ae208e1fe4963fceb3d6954"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 35365,
         "digest": "sha256:de83a2304fa1f7c4a13708a0d15b9704f5945c2be5cbb2b3ed9b2ccb718d0b3d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 852,
         "digest": "sha256:f9a83bce3af0648efaa60b9bb28225b09136d2d35d0bed25ac764297076dec1b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 163,
         "digest": "sha256:b6b53be908de2c0c78070fff0a9f04835211b3156c4e73785747af365e71a0d7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 403170736,
         "digest": "sha256:5650063cfbfb957d6cfca383efa7ad6618337abcd6d99b247d546f94e2ffb7a9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 81117097,
         "digest": "sha256:89142850430d0d812f21f8bfef65dcfb42efe2cd2f265b46b73f41fa65bef2fe"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 6868,
         "digest": "sha256:498b10157bcd37c3d4d641c370263e7cf0face8df82130ac1185ef6b2f532470"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 144376365,
         "digest": "sha256:a77a3b1caf74cc7c9fb700cab353313f1b95db5299642f82e56597accb419d7c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1551901872,
         "digest": "sha256:0603289dda032b5119a43618c40948658a13e954f7fd7839c42f78fd0a2b9e44"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 467065,
         "digest": "sha256:c3ae245b40c1493b89caa2f5e444da5c0b6f225753c09ddc092252bf58e84264"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 324,
         "digest": "sha256:67e85692af8b802b6110c0a039f582f07db8ac6efc23227e54481f690f1afaae"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 450,
         "digest": "sha256:ea72ab3b716788097885d2d537d1d17c9dc6d9911e01699389fa8c9aa6cac861"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 197,
         "digest": "sha256:b02850f0d90ca01b50bbfb779bcf368507c266fc10cc1feeac87c926e9dda2c1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 198,
         "digest": "sha256:4295de6959cedecdd0ba31406e15c19e38c13c0ebc38f3d6385725501063ef46"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:d651a7c122d62d2869af2a5330c756f2f4b35a8e44902174be5c8ce1ad105edd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:69e0b993e5f56695ee76b3776275dac236d38d32ba1f380fd78b900232e006ec"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 526,
         "digest": "sha256:1dc42bc46e4704e68d73eba9f14668f44ff306304b7148b9cb94911c3ea67ac1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 457,
         "digest": "sha256:2f9fa19e23808a073614dac66f46a89a4a61975a39bcc78b5be0522cdabdd69d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21088835,
         "digest": "sha256:edefd1a34c9ba833ae00c0f23e7c550cf016b23ddc63062c0fa8655bce43f6c5"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 109595241,
         "digest": "sha256:f93b619c001cdf51be82f32e7c67a582ab2ed6bab2a076ecb18aad2125098563"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 278220707,
         "digest": "sha256:8140fe8e7b974809954776ac5cf8c9699c14382653b84adb09be9d70fcd7d8f7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 207936701,
         "digest": "sha256:5f6466a5f48e7fe6f34b059ac6da15315a7f68d5b73a5765788eec6215f587ea"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 96474599,
         "digest": "sha256:1418d017696767955fad4bad41af034184107bfc3def4e48e939e843777f7109"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 108017897,
         "digest": "sha256:4aeb9e2fb029bf2f0483eb5a300a845011b538954041647090fe635ed9790690"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1026766656,
         "digest": "sha256:f571cf35405bc544211adcae3ff2a187383507ef5dd8e289bb195e32d031d714"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 74960227,
         "digest": "sha256:ac74d2d1559ca10ab9af19736b2a622bcbef5d4922bdd027a934fd0e7c469700"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46046810,
         "digest": "sha256:1d2b51f6632b960987440a64ff32ec2a189a8a98dfb7316f412028023cbb6d6d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 59099647,
         "digest": "sha256:dc7712be29c06e83f55a0d6a7adcc53a7017a1d5159f978a3f2e44755b5b5db9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 394268094,
         "digest": "sha256:6a8a6bf4138f9896a1de7b8f8d1c92c90e20a5ed41e9dd05a83a3751f7aa1771"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46394381,
         "digest": "sha256:93fd52d4e7d2ff6e6e53c1bae96dcb2401a3ffb6567d56cf3a0a163798d44796"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 105256399,
         "digest": "sha256:b9c3bcfb4b8048d8fe79946b0a16fa62f3d794774c43d8f626b0c4da73b805e9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 890007036,
         "digest": "sha256:63248a2cdbafbef594eb5779c8faeb4023c04ff1d0d2a8c05b26c3537ac1fb53"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 179782435,
         "digest": "sha256:421454e42880b4068973c167b95dac12bd6a85bbebb71fb655232d5acf1be25e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 10624138,
         "digest": "sha256:de47f097b69bd70ca58ac379cb992e562c7a64fef0a9af44a0a088620cfb0290"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1845155,
         "digest": "sha256:77a9afd7d9ee91207123b10e4eba4f956a7862c47fafcd19aa3218729ca74269"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 79288429,
         "digest": "sha256:a6b3dc3a871f299c8b7d7b4e48501d2f5c2f3c323d42b74e0c16acd7fdb77021"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3266,
         "digest": "sha256:473d50efc90a18a0593f653fb5717e4001b705a3b34a9cdc8e15344edc4a0ccd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2167,
         "digest": "sha256:d6057bf7d941280131654230fabb903d30b3af5dad1fbd677868960bca7cd234"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1272,
         "digest": "sha256:0fc8a928f4d793f98257e0284841a994c05ff01c9298baa67155fb4681c00b61"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 646,
         "digest": "sha256:ebd4d0af14d6a24cea79564be1694679af6c06e32152ed8bc0d67d4812bdfddf"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2052,
         "digest": "sha256:981796ad724fba57977b3d8ea7a2f9084ba550ae7d9daf5d2c645f15aaf86466"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 878,
         "digest": "sha256:430670ba00470a36cbe20d55c006edb723c641d806b9dd8f8b243d535af9a39f"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 383,
         "digest": "sha256:8652bf75fb22ecb890e90a7b052fed4dc55ff48caa9697c356ba355c3af4b2da"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:01462724bd3cdeb42dc233ca8a45ffadf4fe39add96fb1e522d5fda621e6bd71"
      }
   ]
}
