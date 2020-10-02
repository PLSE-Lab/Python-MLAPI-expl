# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


R-patched (or R-prerelease, e.g. 4.0.0 alpha/beta/RC)
Solaris 10 on ix64/x64
C, C++, Fortran from Oracle Developer Studio 12.6
Timezone Europe/London

configured with flag --with-internal-tzcode and config.site:
CC='cc -xc99'
CFLAGS='-O -xlibmieee -xlibmil -xtarget=native -xcache=generic -nofstore'
FC=f95
FFLAGS='-O -libmil -xtarget=native -xcache=generic -nofstore'
CXX=CC
CXXSTD="-std=c++11 -library=stdcpp,CrunG3"
CXXFLAGS="-O -xlibmil -xtarget=native -xcache=generic -nofstore"
SAFE_FFLAGS="-O -fstore"
FCLIBS_XTRA=-lfsu /opt/developerstudio12.6/lib/libfui.so.2
R_LD_LIBRARY_PATH="/opt/developerstudio12.6/lib:/usr/local/lib:/opt/csw/lib"

(This is a 32-bit build as that is the compiler default.)

Where these compilers cannot be used, use
GCC 5.2.0 from OpenCSW, with config.site:
CC="/opt/csw//bin/gcc"
CFLAGS=-O2
CPPFLAGS="-I/opt/csw/include -I/usr/local/include"
FC="/opt/csw//bin/gfortran"
FFLAGS=-O2
CXX="/opt/csw//bin/g++"
CXXFLAGS=-O2
LDFLAGS="-L/opt/csw/lib -L/usr/local/lib"
R_LD_LIBRARY_PATH="/usr/local/lib:/opt/csw/lib:/home/ripley/developerstudio12.6/lib:/usr/openwin/lib"

This is used for packages linking to Rcpp/RcppEigen, rgdal/sf
(GDAL is only compilable with g++) and about 40 others.

The version of Java, "1.7.0_65", is the latest for 32-bit Solaris.

Checking is done with the following environment variables:

setenv _R_CHECK_INSTALL_DEPENDS_ TRUE
setenv _R_CHECK_SUGGESTS_ONLY_ TRUE
setenv _R_CHECK_NO_RECOMMENDED_ TRUE
setenv _R_CHECK_DEPRECATED_DEFUNCT_ true
setenv _R_CHECK_REPLACING_IMPORTS_ true
setenv _R_CHECK_S3_METHODS_NOT_REGISTERED_ true
setenv _R_CHECK_OVERWRITE_REGISTERED_S3_METHODS_ true
setenv _R_CHECK_COMPILATION_FLAGS_ true
setenv _R_CHECK_PACKAGES_USED_IN_TESTS_USE_SUBDIRS_ true
setenv _R_CHECK_THINGS_IN_CHECK_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_ true

R-devel
x86_64 Fedora 30 Linux
GCC 9.3 (C, C++, Fortran)
Timezone Europe/London

configured with no options, config.site:
CFLAGS="-g -O2 -Wall -pedantic -mtune=native"
FFLAGS="-g -O2 -mtune=native -Wall -pedantic"
CXXFLAGS="-g -O2 -Wall -pedantic -mtune=native -Wno-ignored-attributes -Wno-deprecated-declarations"

[2019-07-25: pro tem using jre-1.8.0]

The -Wno flags reduce the deluge of warnings from Boost and Eigen headers.

Experimentally, security flags similar to Fedora's are in use:

CFLAGS="-g -O2 -Wall -pedantic -mtune=native -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong -fstack-clash-protection -fcf-protection"

CXXFLAGS="-g -O2 -Wall -pedantic -mtune=native -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-parentheses -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector-strong -fstack-clash-protection -fcf-protection"

Because the Fedora pandoc does not support https://, pandoc 2.9.2 from the 
pandoc distribution site is used.

A local TexLive is used, updated nightly.

This is currently using the enviroment variables

setenv _R_CHECK_INSTALL_DEPENDS_ true
## the next is the default, but --as-cran has true
setenv _R_CHECK_SUGGESTS_ONLY_ false
setenv _R_CHECK_NO_RECOMMENDED_ true
setenv _R_CHECK_DOC_SIZES2_ true
setenv _R_CHECK_DEPRECATED_DEFUNCT_ true
setenv _R_CHECK_SCREEN_DEVICE_ warn
setenv _R_CHECK_REPLACING_IMPORTS_ true
setenv _R_CHECK_TOPLEVEL_FILES_ true
setenv _R_CHECK_DOT_FIRSTLIB_ true
setenv _R_CHECK_RD_LINE_WIDTHS_ true
setenv _R_CHECK_S3_METHODS_NOT_REGISTERED_ true
setenv _R_CHECK_OVERWRITE_REGISTERED_S3_METHODS_ true
setenv _R_CHECK_CODE_USAGE_WITH_ONLY_BASE_ATTACHED_ TRUE
setenv _R_CHECK_NATIVE_ROUTINE_REGISTRATION_ true
setenv _R_CHECK_FF_CALLS_ registration
setenv _R_CHECK_PRAGMAS_ true
setenv _R_CHECK_COMPILATION_FLAGS_ true
setenv _R_CHECK_R_DEPENDS_ true
setenv _R_CHECK_PACKAGES_USED_IN_TESTS_USE_SUBDIRS_ true
setenv _R_CHECK_PKG_SIZES_ false
setenv _R_CHECK_SHLIB_OPENMP_FLAGS_ true

setenv _R_CHECK_LIMIT_CORES_ true
#setenv _R_CHECK_LENGTH_1_CONDITION_ package:_R_CHECK_PACKAGE_NAME_,abort,verbose
setenv _R_S3_METHOD_LOOKUP_BASEENV_AFTER_GLOBALENV_ true
setenv _R_CHECK_COMPILATION_FLAGS_KNOWN_ "-Wno-deprecated-declarations -Wno-ignored-attributes -Wno-parentheses-Werror=format-security -Wp,-D_FORTIFY_SOURCE=2"
setenv _R_CHECK_AUTOCONF_ true
setenv _R_CHECK_THINGS_IN_CHECK_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_EXCLUDE_ "^ompi"
setenv _R_CHECK_BASHISMS_ true
setenv _R_CHECK_DEPENDS_ONLY_DATA_ true
R-devel
x86_64 Fedora 30 Linux
clang 10.0.0 (C, C++), gfortran 9.3
Timezone Europe/London

configured with no options, config.site:
CFLAGS="-g -O3 -Wall -pedantic -mtune=native"
FFLAGS="-g -O2 -mtune=native -Wall -pedantic"
CXXFLAGS="-g -O3 -Wall -pedantic -frtti -Wno-tautological-overlap-compare"
CPPFLAGS="-isystem /usr/local/clang/include"
JAVA_HOME=/usr/lib/jvm/jre-11
LDFLAGS="-L/usr/local/clang/lib64 -L/usr/local/lib64"

Some external C++ libraries compiled with clang are installed under
/usr/local/clang.

The gfortran OpenMP generates calls not in clang's OpenMP runtime.

Because the Fedora pandoc does not support https://, pandoc 2.9.2 from the 
pandoc distribution site is used.

A local TexLive is used, updated nightly.

This is currently using the enviroment variables

setenv _R_CHECK_INSTALL_DEPENDS_ true
setenv _R_CHECK_SUGGESTS_ONLY_ true
setenv _R_CHECK_NO_RECOMMENDED_ true
setenv _R_CHECK_DOC_SIZES2_ true
setenv _R_CHECK_DEPRECATED_DEFUNCT_ true
setenv _R_CHECK_SCREEN_DEVICE_ warn
setenv _R_CHECK_REPLACING_IMPORTS_ true
setenv _R_CHECK_TOPLEVEL_FILES_ true
setenv _R_CHECK_DOT_FIRSTLIB_ true
setenv _R_CHECK_RD_LINE_WIDTHS_ true
setenv _R_CHECK_S3_METHODS_NOT_REGISTERED_ true
setenv _R_CHECK_OVERWRITE_REGISTERED_S3_METHODS_ true
setenv _R_CHECK_CODE_USAGE_WITH_ONLY_BASE_ATTACHED_ TRUE
setenv _R_CHECK_NATIVE_ROUTINE_REGISTRATION_ true
setenv _R_CHECK_FF_CALLS_ registration
setenv _R_CHECK_PRAGMAS_ true
setenv _R_CHECK_COMPILATION_FLAGS_ true
setenv _R_CHECK_R_DEPENDS_ true
setenv _R_CHECK_PACKAGES_USED_IN_TESTS_USE_SUBDIRS_ true
setenv _R_CHECK_SHLIB_OPENMP_FLAGS_ true
setenv _R_CHECK_CODE_ASSIGN_TO_GLOBALENV_ true
setenv _R_CHECK_CODE_DATA_INTO_GLOBALENV_ true
setenv _R_CHECK_PKG_SIZES_ true

setenv _R_CHECK_LIMIT_CORES_ true
#setenv _R_CHECK_LENGTH_1_CONDITION_ package:_R_CHECK_PACKAGE_NAME_,abort,verbose
setenv _R_S3_METHOD_LOOKUP_BASEENV_AFTER_GLOBALENV_ true
setenv _R_CHECK_AUTOCONF_ true
setenv _R_CHECK_THINGS_IN_CHECK_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_EXCLUDE_ "^ompi"
setenv _R_CHECK_BASHISMS_ true
setenv _R_CHECK_ORPHANED_ true
setenv _R_CHECK_DEPENDS_ONLY_DATA_ true
R-devel
x86_64 Fedora 30 Linux
clang 10.0.0 (C, C++), gfortran 9.3
Timezone Europe/London

configured with no options, config.site:
CFLAGS="-g -O3 -Wall -pedantic -mtune=native"
FFLAGS="-g -O2 -mtune=native -Wall -pedantic"
CXXFLAGS="-g -O3 -Wall -pedantic -frtti -Wno-tautological-overlap-compare"
CPPFLAGS="-isystem /usr/local/clang/include"
JAVA_HOME=/usr/lib/jvm/jre-11
LDFLAGS="-L/usr/local/clang/lib64 -L/usr/local/lib64"

Some external C++ libraries compiled with clang are installed under
/usr/local/clang.

The gfortran OpenMP generates calls not in clang's OpenMP runtime.

Because the Fedora pandoc does not support https://, pandoc 2.9.2 from the 
pandoc distribution site is used.

A local TexLive is used, updated nightly.

This is currently using the enviroment variables

setenv _R_CHECK_INSTALL_DEPENDS_ true
setenv _R_CHECK_SUGGESTS_ONLY_ true
setenv _R_CHECK_NO_RECOMMENDED_ true
setenv _R_CHECK_DOC_SIZES2_ true
setenv _R_CHECK_DEPRECATED_DEFUNCT_ true
setenv _R_CHECK_SCREEN_DEVICE_ warn
setenv _R_CHECK_REPLACING_IMPORTS_ true
setenv _R_CHECK_TOPLEVEL_FILES_ true
setenv _R_CHECK_DOT_FIRSTLIB_ true
setenv _R_CHECK_RD_LINE_WIDTHS_ true
setenv _R_CHECK_S3_METHODS_NOT_REGISTERED_ true
setenv _R_CHECK_OVERWRITE_REGISTERED_S3_METHODS_ true
setenv _R_CHECK_CODE_USAGE_WITH_ONLY_BASE_ATTACHED_ TRUE
setenv _R_CHECK_NATIVE_ROUTINE_REGISTRATION_ true
setenv _R_CHECK_FF_CALLS_ registration
setenv _R_CHECK_PRAGMAS_ true
setenv _R_CHECK_COMPILATION_FLAGS_ true
setenv _R_CHECK_R_DEPENDS_ true
setenv _R_CHECK_PACKAGES_USED_IN_TESTS_USE_SUBDIRS_ true
setenv _R_CHECK_SHLIB_OPENMP_FLAGS_ true
setenv _R_CHECK_CODE_ASSIGN_TO_GLOBALENV_ true
setenv _R_CHECK_CODE_DATA_INTO_GLOBALENV_ true
setenv _R_CHECK_PKG_SIZES_ true

setenv _R_CHECK_LIMIT_CORES_ true
#setenv _R_CHECK_LENGTH_1_CONDITION_ package:_R_CHECK_PACKAGE_NAME_,abort,verbose
setenv _R_S3_METHOD_LOOKUP_BASEENV_AFTER_GLOBALENV_ true
setenv _R_CHECK_AUTOCONF_ true
setenv _R_CHECK_THINGS_IN_CHECK_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_EXCLUDE_ "^ompi"
setenv _R_CHECK_BASHISMS_ true
setenv _R_CHECK_ORPHANED_ true
setenv _R_CHECK_DEPENDS_ONLY_DATA_ true

R-devel
x86_64 Fedora 30 Linux
clang 10.0.0 (C, C++), gfortran 9.3
Timezone Europe/London

configured with no options, config.site:
CFLAGS="-g -O3 -Wall -pedantic -mtune=native"
FFLAGS="-g -O2 -mtune=native -Wall -pedantic"
CXXFLAGS="-g -O3 -Wall -pedantic -frtti -Wno-tautological-overlap-compare"
CPPFLAGS="-isystem /usr/local/clang/include"
JAVA_HOME=/usr/lib/jvm/jre-11
LDFLAGS="-L/usr/local/clang/lib64 -L/usr/local/lib64"

Some external C++ libraries compiled with clang are installed under
/usr/local/clang.

The gfortran OpenMP generates calls not in clang's OpenMP runtime.

Because the Fedora pandoc does not support https://, pandoc 2.9.2 from the 
pandoc distribution site is used.

A local TexLive is used, updated nightly.

This is currently using the enviroment variables

setenv _R_CHECK_INSTALL_DEPENDS_ true
setenv _R_CHECK_SUGGESTS_ONLY_ true
setenv _R_CHECK_NO_RECOMMENDED_ true
setenv _R_CHECK_DOC_SIZES2_ true
setenv _R_CHECK_DEPRECATED_DEFUNCT_ true
setenv _R_CHECK_SCREEN_DEVICE_ warn
setenv _R_CHECK_REPLACING_IMPORTS_ true
setenv _R_CHECK_TOPLEVEL_FILES_ true
setenv _R_CHECK_DOT_FIRSTLIB_ true
setenv _R_CHECK_RD_LINE_WIDTHS_ true
setenv _R_CHECK_S3_METHODS_NOT_REGISTERED_ true
setenv _R_CHECK_OVERWRITE_REGISTERED_S3_METHODS_ true
setenv _R_CHECK_CODE_USAGE_WITH_ONLY_BASE_ATTACHED_ TRUE
setenv _R_CHECK_NATIVE_ROUTINE_REGISTRATION_ true
setenv _R_CHECK_FF_CALLS_ registration
setenv _R_CHECK_PRAGMAS_ true
setenv _R_CHECK_COMPILATION_FLAGS_ true
setenv _R_CHECK_R_DEPENDS_ true
setenv _R_CHECK_PACKAGES_USED_IN_TESTS_USE_SUBDIRS_ true
setenv _R_CHECK_SHLIB_OPENMP_FLAGS_ true
setenv _R_CHECK_CODE_ASSIGN_TO_GLOBALENV_ true
setenv _R_CHECK_CODE_DATA_INTO_GLOBALENV_ true
setenv _R_CHECK_PKG_SIZES_ true

setenv _R_CHECK_LIMIT_CORES_ true
#setenv _R_CHECK_LENGTH_1_CONDITION_ package:_R_CHECK_PACKAGE_NAME_,abort,verbose
setenv _R_S3_METHOD_LOOKUP_BASEENV_AFTER_GLOBALENV_ true
setenv _R_CHECK_AUTOCONF_ true
setenv _R_CHECK_THINGS_IN_CHECK_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_ true
setenv _R_CHECK_THINGS_IN_TEMP_DIR_EXCLUDE_ "^ompi"
setenv _R_CHECK_BASHISMS_ true
setenv _R_CHECK_ORPHANED_ true
setenv _R_CHECK_DEPENDS_ONLY_DATA_ true

setenv _R_CHECK_COMPILATION_FLAGS_KNOWN_ -Wno-tautological-overlap-compare

Note the _R_CHECK_SUGGESTS_ONLY_ setting, which is different from the rest of
the Linux runs.


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



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 14966,
      "digest": "sha256:4ee6f3993a1516e5da875d9fcf7d1fc595485753d4d060b7bfc1e60cfebaf73f"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 50382957,
         "digest": "sha256:7e2b2a5af8f65687add6d864d5841067e23bd435eb1a051be6fe1ea2384946b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 222909892,
         "digest": "sha256:59c89b5f9b0c6d94c77d4c3a42986d420aaa7575ac65fcd2c3f5968b3726abfc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 195204532,
         "digest": "sha256:4017849f9f85133e68a4125e9679775f8e46a17dcdb8c2a52bbe72d0198f5e68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1522,
         "digest": "sha256:c8b29d62979a416da925e526364a332b13f8d5f43804ae98964de2a60d47c17a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 717,
         "digest": "sha256:12004028a6a740ac35e69f489093b860968cc37b9668f65b1e2f61fd4c4ad25c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247,
         "digest": "sha256:3f09b9a53dfb03fd34e35d43694c2d38656f7431efce0e6647c47efb5f7b3137"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 408,
         "digest": "sha256:03ed58116b0cb733cc552dc89ef5ea122b6c5cf39ec467f6ad671dc0ba35db0c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 331594702,
         "digest": "sha256:7844554d9ef75bb3f1d224e166ed12561e78add339448c52a8e5679943b229f1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 112665282,
         "digest": "sha256:f15956f7439a8f32fc517e733c3af3104f414f6fccaf6eb901bcb1e233d4f9bf"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 421,
         "digest": "sha256:04daa549c9f7ce91a4b748f66094533c5b54256c7de57023d09e76728bc55ef1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 5494,
         "digest": "sha256:ce39292f31e2ce9cd52a97a59962ac74127458ba3cb69a94373cf6aa4116f0c4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1838,
         "digest": "sha256:3874427bf61ee09a3145aa945f0036f15da21399e24b84cd84c067e930385979"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2451236145,
         "digest": "sha256:97b6e7362f6a297643e996bfae1fe57f228e0c47ed322cd8d0f440c0d1e7da3f"
      }
   ]
}

