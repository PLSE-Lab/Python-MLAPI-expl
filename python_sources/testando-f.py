#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[ ]:


# Shell scripts and Automake sources can't have CRLF line endings
configure.ac	eol=lf
Makefile.am	eol=lf
*.m4		eol=lf
*.sh		eol=lf


# In[ ]:


*~
*.a
*.la
*.lo
*.loT
*.o
*.exe
*.manifest
*.rc

.dirstamp

Makefile
Makefile.in
.deps
.libs

aclocal.m4()
autom4te.cache()
compile()
config.guess()
config.h()
config.h.in()
config.status()
config.sub()
config.log()
configure()
depcomp()
install(-sh)
libtool()
ltmain.sh()
missing()
stamp(-h1)
*.pc()

mingw32(-config.cache)

src(/make-tables)
src(/openslide-tables.c)

test(/_slidedata)
test(/driver)
test(/extended)
test(/mosaic)
test(/parallel)
test(/profile)
test(/query)
test(/test)
test(/try_open)

doc(/html)
doc(/latex)
doc(/man)


# In[ ]:


## Context

**Issue type** (bug report or feature request): 
**Operating system** (e.g. Fedora 24, Mac OS 10.11, Windows 10): 
**Platform** (e.g. 64-bit x86, 32-bit ARM): 
**OpenSlide version**: 
**Slide format** (e.g. SVS, NDPI, MRXS): 

## Details


# In[ ]:


language: c

sudo: required

before_install:
  - sudo apt-get -qq update
  - sudo apt-get install -y libz-dev libjpeg-dev libopenjpeg-dev libtiff-dev libglib2.0-dev libcairo-dev libpng-dev libgdk-pixbuf2.0-dev libxml2-dev libsqlite3-dev valgrind

before_script:
  - autoreconf -i


# In[ ]:


Notable Changes in OpenSlide
============================

Version 3.4.1, 2015-04-20
 * New formats: Philips TIFF, Ventana TIFF
 * Support OpenJPEG 2.1.0
 * Improve performance of JPEG and JP2K decoding
 * Add openslide.region[i].* properties
 * Improve MATLAB compatibility
 * Enable function deprecation warnings with MSVC
 * Many portability fixes
 * aperio: Detect OpenJPEG chroma subsampling breakage during open
 * aperio: Fill in missing tiles with downsampled data
 * aperio: Report MPP for slides scanned in locales with decimal comma
 * hamamatsu: Support NDPI files > 4 GB
 * hamamatsu: Properly detect NDPI slides produced by NDP.toolkit
 * hamamatsu: Support VMS/VMU slides without a NoLayers key
 * hamamatsu: Report MPP for VMS/VMU
 * leica: Support slides with 2010/03/10 XML namespace
 * leica: Base64-decode leica.barcode property in 2010/10/01 namespace
 * sakura: Support slides with multiple focal planes
 * sakura: Support slides without tile table
 * ventana: Support slides with multiple focal planes
 * ventana: Improve positioning of AOIs within level
 * ventana: Fix failure to recognize macro image on some slides

Version 3.4.0, 2014-01-25
 * Major internal restructuring
 * New formats: Hamamatsu NDPI, Sakura SVSLIDE, Ventana BIF (preliminary)
 * Add openslide_detect_vendor()
 * Deprecate openslide_can_open() (not very useful and often misused)
 * Document performance considerations for openslide_open()
 * Add properties giving the bounds of the non-empty region of the slide
 * leica: Support multiple main images if their levels are coplanar
 * leica: Use slide size as level size
 * mirax: Support PNG- and BMP-formatted slides
 * mirax: Fix "Expected first 0 value" error opening some slides
 * mirax: Fix incorrect tile placement on some slides without overlaps
 * mirax: Never synthesize downsampled levels
 * Add OPENSLIDE_DEBUG environment variable (OPENSLIDE_DEBUG=? for help)
 * Fix some crashes in error paths
 * Add tests for many error paths

Version 3.3.3, 2013-04-13
 * Fix inclusion of openslide.h with MSVC
 * Properly handle Aperio JP2K slides with zero-length tiles
 * Support Hamamatsu slides with blank MacroImage key

Version 3.3.2, 2012-12-01
 * Fix seams in MIRAX 2.2 slides (thanks, Agelos Pappas)
 * Fix associated image naming in single-level Aperio slides
 * Stop decoding MIRAX tiles outside requested region
 * Stop decoding unneeded tiles during tile-aligned accesses
 * Increase Hamamatsu VMU tile size to reduce rendering overhead
 * Document performance considerations for openslide_can_open()

Version 3.3.1, 2012-10-14
 * Parallelize concurrent openslide_read_region calls on an openslide_t
 * Eliminate background scanning of tile headers in MIRAX
 * Scan many fewer tiles during first accesses to Hamamatsu VMS
 * Ignore Leica Z-planes other than 0
 * Add experimental tile-size properties
 * Document API thread safety

Version 3.3.0, 2012-09-08
 * Support Leica SCN format (requires libtiff >= 4) (thanks, Agelos Pappas)
 * Allow opening MIRAX 2.2 slides (though there are seams, bug #92)
 * Add standardized microns-per-pixel and objective-power properties
 * Add "macro" associated image in Trestle
 * Rename "layer" to "level" throughout the API (deprecate "layer" functions;
   remove "layer" properties)
 * Report parse errors in openslide_open() by returning an openslide_t in
   error state
 * Deprecate openslide_get_comment()
 * Add openslide_get_version()
 * Improve command-line tools; add manpages
 * Support building with MinGW-w64; drop CMake, MSVC, mingw32
 * Add tests for many error paths

Version 3.2.6, 2012-02-23
 * Support downsampled MIRAX files
 * Improve performance on MIRAX slides without tile overlaps
 * Fix openslide_read_region for large dimensions on layer > 0
   (3.2.5 regression)
 * Correct subpixel error in MIRAX tile placement
 * Fix unlikely use-after-free with Hamamatsu VMU

Version 3.2.5, 2011-12-16
 * Support MIRAX 1.03 files (thanks, Jan Harkes)
 * Fix openslide_read_region for large dimensions
 * Use subpixel precision in all backends
 * Don't keep associated images in memory
 * Disable quickhash-1 for TIFF files with very large top layer
 * Various build fixes (thanks, Jan, Marco Feuerstein, and
   Mathieu Malaterre)
 * Fix some unlikely memory leaks

Version 3.2.4, 2011-03-07
 * Support MIRAX files that do not have non-hierarchical data
   (thanks, Jan Harkes)
 * Fix compilation on Windows (thanks Hauke Heibel)
 * Work around a bug in GKeyFile parser (thanks, Jan)

Version 3.2.3, 2010-09-09
 * Support MIRAX files that use a variant format for tile
   positions (thanks, Hauke Heibel and Marco Feuerstein)
 * Update location of website
 * Add background color property, for slides that have it
 * Update CMake scripts and other Windows fixes
   (thanks Hauke and Marco)
 * Fix some test.c bugs
 * Fix incorrect MIRAX drawing at certain resolutions
   (thanks Hauke and Marco)
 * Support quickhash-1 on older systems (thanks, Jan Harkes)

Version 3.2.2, 2010-06-16
 * Rework some internals of openslide_read_region
 * Support negative coordinates and zero-sized dimensions
   in openslide_read_region
 * Clarify the documentation about openslide_read_region
 * Fix Windows build bug with new NGR support
 * Enable untested BigTIFF support

Version 3.2.1, 2010-06-03
 * Fix crashes on Windows when trying to read Hamamatsu files
 * Fix jpeg 7 problems in read_from_one_jpeg
 * Quiet the error handling system after the first error

Version 3.2.0, 2010-06-01
 * Add experimental CMake support and fixes for building with MSVC
   (thanks to Hauke Heibel!)
 * Enable detecting runtime errors
 * Add initial Hamamatsu Nanozoomer VMU support
   (thanks to Steve Lamont)
 * Add "openslide-write-png" tool

Version 3.1.1, 2010-04-27
 * Fix a crash when reading an invalid VMS file
 * Fix memory leaks when reading an invalid VMS file
 * Accept VMS files that have more than one focal plane (non-0 planes ignored)
 * Fix bug that could cause problems with libtiff 4
 * Relax the required version of Microsoft.VC80.CRT

Version 3.1.0, 2010-04-01
 * Enable large file access on Windows (requires Microsoft.VC80.CRT)
 * Support newer Aperio files (compression 33005)
 * Be more robust in reading raw TIFF tiles
 * Reject invalid TIFF files earlier
 * Fix many memory leaks when probing for TIFF files

Version 3.0.3, 2010-03-01
 * Fix nasty artifacts in some MIRAX files (seen at some zoom levels)

Version 3.0.2, 2010-02-17
 * Restore ability to build with glib 2.12, at the expense
   of not having "quickhash-1" in that configuration

Version 3.0.1, 2010-02-04
 * Fix edge-drawing bug in TIFF backend
 * Ship CHANGELOG.txt

Version 3.0.0, 2010-01-28
 * Switch from GPLv2 to LGPLv2
 * Reduce some unlikely memory leaks
 * Support of more MIRAX files
 * Improve performance of MIRAX rendering, vastly in some cases
 * Reduce appearance of seams in MIRAX
 * Add "quickhash-1" hash property
 * Add "openslide-quickhash1sum" and "openslide-show-properties" tools
 * Rework the API documentation
 * Remove never-implemented prefetch functions from openslide.h (but
   retain with warnings in the library)
 * Start attempting to figure out Trestle tile position files
 * Add some defined property names to the header file


----


Version 2.3.1, 2009-12-14
 * Eliminate Aperio regression introduced in Version 2.3.0

Version 2.3.0, 2009-12-11
 * Support for generic tiled TIFF format (for InterScope files)
 * Bug fixes
 * Reduction of some TIFF error messages
 * Fixes for some build problems
 * Deprecate prefetch functions (never implemented)

Version 2.2.1, 2009-10-23
 * Fixes for thread safety problems in 2.2.0

Version 2.2.0, 2009-09-15
 * Thread safety (lockless with Hamamatsu and MIRAX files)

Version 2.1.0, 2009-08-18
 * Support for MIRAX mrxs

Version 2.0.0, 2009-07-16
 * Support for image metadata and associated images
 * Support Aperio variant
 * Internally rework a lot in preparation for MIRAX
 * Win32 support


----


Version 1.1.1, 2009-02-25
 * Remove never-functional generic JPEG 2000 support
 * Switch Aperio to use the released version of OpenJPEG
 * Be more robust to errors in general

Version 1.1.0, 2008-12-20
 * Greatly speed up Hamamatsu with a tile cache and background
   scanning thread

Version 1.0.0, 2008-12-09
 * Renamed to "OpenSlide"
 * Multi-file Hamamatsu support
 * Switch to 64-bit signed integers in public API where possible


----


Version 0.5.0, 2008-10-21
 * GPLv2 release
 * Working Aperio support
 * More work on generic JPEG 2000

Version 0.4.2, 2008-09-05
 * Documentation updates
 * For Aperio, remove Jasper in lieu of using OpenJPEG
 * Preliminary and non-functional generic JPEG 2000 support

Version 0.4.0, 2008-03-12
 * Update simple test program

Version 0.3.0, 2008-01-31
 * Broken and unusably slow Aperio support

Version 0.2.0, 2008-01-19
 * Using glib
 * Layers are numbered instead of named
 * Actual start of implementation
 * Initial Trestle support
 * Initial Aperio support (without tile codec)
 * Initial slow and incomplete Hamamatsu support
 * Initial test program
 * Documentation updates

Version 0.1.0, 2007-12-06
 * Unreleased, just documentation and headers (called "Wholeslide")


# In[ ]:


Contributing to OpenSlide
=========================

Adam Goode, June 2010



This document is for any contributors to OpenSlide, including those in
the original OpenSlide group at CMU. It includes some guidelines and
advice.

 * Please do not add new API calls without careful thought. Absolutely
   don't add API as unimplemented stubs "just in case". OpenSlide so
   far has been backward compatible in terms of ABI, and part of this
   comes from carefully adding new API only as needed. Removing a bad
   API will come at the cost of binary compatibility.

   I was able to effectively remove the prefetch API without breaking
   ABI by making it a compile time error to link against it, but this
   was only because the API calls were stubs and never implemented and
   never used. Having these unimplemented stubs was a bad idea because
   while the code was unimplemented, no user could possibly test the
   performance implications of using the prefetch; and if we ever
   actually implemented something it would likely hurt performance of
   any user who might have sprinkled prefetch statements around and
   noticed no performance issues with it.

 * Make sure things build on Windows, at least with MinGW. Many users
   of OpenSlide are on Windows. Keep in mind the differences between ELF
   and COFF shared libraries, they can get you. This is especially
   important when an OpenSlide API might consume or return a structure
   from a DLL that OpenSlide links to, in general this is unsafe on Windows.

 * When making a release, try to carefully follow the libtool
   and automake versioning rules. Also add a changelog entry, update
   the website, and post to the announce mailing list.

 * Try to follow the coding style of nearby code.

 * Follow the GError rules for reporting errors.

 * Maintain thread safety. The openslide_t struct is mostly immutable,
   except for the cache (which takes locks) and libtiff (which is sort
   of a disaster).

 * Avoid the midlayer mistake. The grid helpers and cache are there for
   you.

 * Have fun! Graphics are a joy.


# In[ ]:


OpenSlide

Carnegie Mellon University and others

https://openslide.org/

====================

Unless otherwise specified, this code is copyright Carnegie Mellon
University.

This code is licensed under the GNU LGPL version 2.1, not any later
version.  See the file lgpl-2.1.txt for the text of the license.

OpenSlide is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.


# In[ ]:


EXTRA_DIST = README.txt lgpl-2.1.txt LICENSE.txt CHANGELOG.txt doc/Doxyfile CONTRIBUTING.txt

# doxygen
dist-hook:
	cd $(distdir)/doc; doxygen

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = openslide.pc

ACLOCAL_AMFLAGS=-I m4


lib_LTLIBRARIES = src/libopenslide.la

src_libopenslide_la_LIBADD = $(GLIB2_LIBS) $(CAIRO_LIBS) $(SQLITE3_LIBS) 	$(LIBXML2_LIBS) $(OPENJPEG_LIBS) $(LIBTIFF_LIBS) $(LIBPNG_LIBS) 	$(LIBJPEG_LIBS) $(GDKPIXBUF_LIBS) $(ZLIB_LIBS)

src_libopenslide_la_SOURCES = 	src/openslide.c 	src/openslide-cache.c 	src/openslide-decode-gdkpixbuf.c 	src/openslide-decode-jp2k.c 	src/openslide-decode-jpeg.c 	src/openslide-decode-png.c 	src/openslide-decode-sqlite.c 	src/openslide-decode-tiff.c 	src/openslide-decode-tifflike.c 	src/openslide-decode-xml.c 	src/openslide-error.c 	src/openslide-grid.c 	src/openslide-hash.c 	src/openslide-jdatasrc.c 	src/openslide-tables.c 	src/openslide-util.c 	src/openslide-vendor-aperio.c 	src/openslide-vendor-huron.c 	src/openslide-vendor-generic-tiff.c 	src/openslide-vendor-hamamatsu.c 	src/openslide-vendor-leica.c 	src/openslide-vendor-mirax.c 	src/openslide-vendor-philips.c 	src/openslide-vendor-sakura.c 	src/openslide-vendor-trestle.c 	src/openslide-vendor-ventana.c

EXTRA_PROGRAMS = src/make-tables
CLEANFILES = src/make-tables
MAINTAINERCLEANFILES = src/openslide-tables.c
# Depending directly on src/make-tables causes needless regeneration of
# openslide-tables.c.  As the lesser of evils, recursively invoke make.
src/openslide-tables.c: src/make-tables.c
	@$(MAKE) $(AM_MAKEFLAGS) src/make-tables
	$(AM_V_GEN)src/make-tables$(EXEEXT) "$@"

if WINDOWS_RESOURCES
src_libopenslide_la_SOURCES += src/openslide-dll.rc
src/openslide-dll.lo: src/openslide-dll.manifest
endif

.rc.lo:
	$(AM_V_GEN)$(LIBTOOL) $(AM_V_lt) $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --tag=RC --mode=compile $(RC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -i "$<" -o "$@"

src_libopenslide_la_CPPFLAGS = -pedantic -D_OPENSLIDE_BUILDING_DLL 	$(GLIB2_CFLAGS) $(CAIRO_CFLAGS) $(SQLITE3_CFLAGS) $(LIBXML2_CFLAGS) 	$(OPENJPEG_CFLAGS) $(LIBTIFF_CFLAGS) $(LIBPNG_CFLAGS) 	$(GDKPIXBUF_CFLAGS) $(ZLIB_CFLAGS) -DG_LOG_DOMAIN=\"Openslide\" 	-I$(top_srcdir)/src

src_libopenslide_la_LDFLAGS = -version-info 4:1:4 -no-undefined


pkginclude_HEADERS = 	src/openslide.h 	src/openslide-features.h

noinst_HEADERS = 	common/openslide-common.h 	src/openslide-cairo.h 	src/openslide-decode-gdkpixbuf.h 	src/openslide-decode-jp2k.h 	src/openslide-decode-jpeg.h 	src/openslide-decode-png.h 	src/openslide-decode-sqlite.h 	src/openslide-decode-tiff.h 	src/openslide-decode-tifflike.h 	src/openslide-decode-xml.h 	src/openslide-error.h 	src/openslide-hash.h 	src/openslide-private.h


# common program code

noinst_LIBRARIES = common/libopenslide-common.a

common_libopenslide_common_a_CPPFLAGS = $(COMMON_CPPFLAGS)
common_libopenslide_common_a_SOURCES = 	common/openslide-common-cmdline.c 	common/openslide-common-fail.c 	common/openslide-common-fd.c

COMMON_CPPFLAGS = $(GLIB2_CFLAGS) -I$(top_srcdir)/src -I$(top_srcdir)/common
COMMON_LDADD = common/libopenslide-common.a src/libopenslide.la $(GLIB2_LIBS)


# test

noinst_PROGRAMS = test/test test/try_open test/parallel test/query 	test/extended test/mosaic test/profile
noinst_SCRIPTS = test/driver
CLEANFILES += test/driver
EXTRA_DIST += test/driver.in

test_test_CPPFLAGS = $(COMMON_CPPFLAGS) $(CAIRO_CFLAGS) $(VALGRIND_CFLAGS)
test_test_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
# VALGRIND_LIBS not needed
test_test_LDADD = $(COMMON_LDADD) $(CAIRO_LIBS)

test_try_open_CPPFLAGS = $(COMMON_CPPFLAGS)
test_try_open_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
test_try_open_LDADD = $(COMMON_LDADD)

test_parallel_CPPFLAGS = $(COMMON_CPPFLAGS)
test_parallel_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
test_parallel_LDADD = $(COMMON_LDADD)

test_query_CPPFLAGS = $(COMMON_CPPFLAGS)
test_query_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
test_query_LDADD = $(COMMON_LDADD)

test_extended_CPPFLAGS = $(COMMON_CPPFLAGS)
test_extended_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
test_extended_LDADD = $(COMMON_LDADD)

test_mosaic_CPPFLAGS = $(COMMON_CPPFLAGS) $(CAIRO_CFLAGS)
test_mosaic_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
test_mosaic_LDADD = $(COMMON_LDADD) $(CAIRO_LIBS)

test_profile_CPPFLAGS = $(COMMON_CPPFLAGS) $(VALGRIND_CFLAGS)
test_profile_CFLAGS = $(AM_CFLAGS) $(TEST_CFLAGS)
test_profile_LDADD = $(COMMON_LDADD)

if CYGWIN_CROSS_TEST
noinst_PROGRAMS += test/symlink
test_symlink_CFLAGS = $(AM_CFLAGS) -municode
test_symlink_LDFLAGS = $(AM_LDFLAGS) -municode
test_symlink_LDADD = -lkernel32
endif

test/driver: test/driver.in Makefile
	$(AM_V_GEN)sed -e 's:!!SRCDIR!!:$(abs_srcdir)/test:g' 		-e 's:!!BUILDDIR!!:$(abs_builddir)/test:g' 		-e 's:!!VERSION!!:$(VERSION):g' 		-e 's:!!CYGWIN_CROSS_TEST!!:$(CYGWIN_CROSS_TEST):g' 		-e 's:!!FEATURES!!:$(FEATURE_FLAGS):g' "$<" > "$@" && 		chmod +x "$@"


# tools

# show-properties
bin_PROGRAMS = tools/openslide-show-properties
man_MANS = tools/openslide-show-properties.1
tools_openslide_show_properties_CPPFLAGS = $(COMMON_CPPFLAGS)
tools_openslide_show_properties_LDADD = $(COMMON_LDADD)

# quickhash1sum
bin_PROGRAMS += tools/openslide-quickhash1sum
man_MANS += tools/openslide-quickhash1sum.1
tools_openslide_quickhash1sum_CPPFLAGS = $(COMMON_CPPFLAGS)
tools_openslide_quickhash1sum_LDADD = $(COMMON_LDADD)

# write-png
bin_PROGRAMS += tools/openslide-write-png
man_MANS += tools/openslide-write-png.1
tools_openslide_write_png_CPPFLAGS = $(COMMON_CPPFLAGS) $(LIBPNG_CFLAGS)
tools_openslide_write_png_LDADD = $(COMMON_LDADD) $(LIBPNG_LIBS)

# man pages
EXTRA_DIST += $(man_MANS:=.in)


# In[ ]:


OpenSlide

Carnegie Mellon University and others

https://openslide.org/


==========================


get_ipython().set_next_input('What is this');get_ipython().run_line_magic('pinfo', 'this')
=============

This library reads whole slide image files (also known as virtual slides).
It provides a consistent and simple API for reading files from multiple
vendors.


get_ipython().set_next_input('What is the license');get_ipython().run_line_magic('pinfo', 'license')
====================

This code is licensed under the GNU LGPL version 2.1, not any later version.
See the file lgpl-2.1.txt for the text of the license.


Requirements
============

This library requires zlib, libpng, libjpeg, libtiff, OpenJPEG 1.x or >= 2.1,
GDK-PixBuf, libxml2, SQLite >= 3.6.20, cairo >= 1.2, and glib >= 2.16.
Leica and Ventana support require libtiff >= 4.

If you want to run the test suite, you will need PyYAML, python-requests,
xdelta3, cjpeg and djpeg (from libjpeg), a Git checkout of OpenSlide,
at least one installed font, and > 120 GB of disk space.  Valgrind mode
requires Valgrind, plus debug symbols for library dependencies (particularly
glib2) and Fontconfig.  Profile mode requires Valgrind.  Coverage mode
requires gcov and Doxygen.


Features
========

The library can read Aperio, Hamamatsu, Leica, MIRAX, Sakura, Trestle,
and Ventana formats, as well as TIFF files that conform to a simple
convention. (InterScope files tend to be readable as this generic TIFF.)

More information about formats is here:
https://openslide.org/formats/

An openslide_t object can be used concurrently from multiple threads
without locking. (But you must lock or otherwise use memory barriers
when passing the object between threads.)


Properties
==========

The library exposes certain properties as string key-value pairs for
a given virtual slide. (These are accessed by way of the
"openslide_get_property_names" and "openslide_get_property_value" calls.)

These properties are generally uninterpreted data gathered from the
on-disk files. New properties can be added over time in subsequent releases
of OpenSlide. A list of some properties can be found at:
https://openslide.org/properties/

OpenSlide itself creates these properties (for now):

 openslide.background-color
   The background color of the slide, given as an RGB hex triplet.
   This property is not always present.

 openslide.bounds-height
   The height of the rectangle bounding the non-empty region of the slide.
   This property is not always present.

 openslide.bounds-width
   The width of the rectangle bounding the non-empty region of the slide.
   This property is not always present.

 openslide.bounds-x
   The X coordinate of the rectangle bounding the non-empty region of the
   slide. This property is not always present.

 openslide.bounds-y
   The Y coordinate of the rectangle bounding the non-empty region of the
   slide. This property is not always present.

 openslide.comment
   A free-form text comment.

 openslide.mpp-x
   Microns per pixel in the X dimension of level 0. May not be present or
   accurate.

 openslide.mpp-y
   Microns per pixel in the Y dimension of level 0. May not be present or
   accurate.

 openslide.objective-power
   Magnification power of the objective. Often inaccurate; sometimes missing.

 openslide.quickhash-1
   A non-cryptographic hash of a subset of the slide data. It can be used
   to uniquely identify a particular virtual slide, but cannot be used
   to detect file corruption or modification.

 openslide.vendor
   The name of the vendor backend.


Other Documentation
===================

The definitive API reference is in openslide.h. For an HTML version, see
doc/html/openslide_8h.html in this distribution.

Additional documentation is available from the OpenSlide website:
https://openslide.org/

The design and implementation of the library are described in a published
technical note:

 OpenSlide: A Vendor-Neutral Software Foundation for Digital Pathology
 Adam Goode, Benjamin Gilbert, Jan Harkes, Drazen Jukic, M. Satyanarayanan
 Journal of Pathology Informatics 2013, 4:27

 http://download.openslide.org/docs/JPatholInform_2013_4_1_27_119005.pdf

There is also an older technical report:

 CMU-CS-08-136
 A Vendor-Neutral Library and Viewer for Whole-Slide Images
 Adam Goode, M. Satyanarayanan

 http://reports-archive.adm.cs.cmu.edu/anon/2008/abstracts/08-136.html
 http://reports-archive.adm.cs.cmu.edu/anon/2008/CMU-CS-08-136.pdf


Acknowledgements
================

OpenSlide has been supported by the National Institutes of Health and
the Clinical and Translational Science Institute at the University of
Pittsburgh.


get_ipython().set_next_input('How to build');get_ipython().run_line_magic('pinfo', 'build')
=============

./configure
make
make install

(If building from the Git repository, you will first need to install
autoconf, automake, libtool, and pkg-config and run "autoreconf -i".)

Good luck!


# In[ ]:


*()
 *  OpenSlide, a library for reading whole slide image files
 *
 *  Copyright (c) 2007-2012 Carnegie Mellon University
 *  Copyright (c) 2015 Benjamin Gilbert
 *  All rights reserved.
 *
 *  OpenSlide is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, version 2.1.
 *
 *  OpenSlide is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with OpenSlide. If not, see
 *  <http://www.gnu.org/licenses/>.
 *
 */

(/, don't, complain, about, g_option_context_parse_strv(),, which, is, called)
(/, conditionally)
#undef GLIB_VERSION_MAX_ALLOWED
#define GLIB_VERSION_MAX_ALLOWED G_ENCODE_VERSION(2,40)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include "openslide.h"
#include "openslide-common.h"
#include "config.h"

static const char *version_format = "%s " SUFFIXED_VERSION ", "
"using OpenSlide %s\n"
"Copyright (C) 2007-2016 Carnegie Mellon University and others\n"
"\n"
"OpenSlide is free software: you can redistribute it and/or modify it under\n"
"the terms of the GNU Lesser General Public License, version 2.1.\n"
"<http://gnu.org/licenses/lgpl-2.1.html>\n"
"\n"
"OpenSlide comes with NO WARRANTY, to the extent permitted by law.  See the\n"
"GNU Lesser General Public License for more details.\n";


static gboolean show_version;

static const GOptionEntry options[] = {
  {"version", 0, 0, G_OPTION_ARG_NONE, &show_version, "Show version", NULL},
  {NULL, 0, 0, G_OPTION_ARG_NONE, NULL, NULL, NULL}
};


static GOptionContext *make_option_context(const struct common_usage_info *info) {
  GOptionContext *octx = g_option_context_new(info->parameter_string);
  g_option_context_set_summary(octx, info->summary);
  g_option_context_add_main_entries(octx, options, NULL);
  return octx;
}

#if GLIB_CHECK_VERSION(2,40,0)

#define CMDLINE_FREE_ARGS

static char **fixed_argv;

static void free_argv(void) {
  g_strfreev(fixed_argv);
}

void common_fix_argv(int *argc, char ***argv) {
  if (fixed_argv == NULL) {
#ifdef G_OS_WIN32
    fixed_argv = g_win32_get_command_line();
#else
    fixed_argv = g_strdupv(*argv);
#endif
    *argc = g_strv_length(fixed_argv);
    *argv = fixed_argv;
    atexit(free_argv);
  }
}

bool common_parse_options(GOptionContext *ctx,
                          int *argc, char ***argv,
                          GError **err) {
  // properly handle Unicode arguments on Windows
  common_fix_argv(argc, argv);
  bool ret = g_option_context_parse_strv(ctx, argv, err);
  *argc = g_strv_length(*argv);
  return ret;
}

#else

void common_fix_argv(int *argc G_GNUC_UNUSED, char ***argv G_GNUC_UNUSED) {}

bool common_parse_options(GOptionContext *ctx,
                          int *argc, char ***argv,
                          GError **err) {
  return g_option_context_parse(ctx, argc, argv, err);
}

#endif

void common_parse_commandline(const struct common_usage_info *info,
                              int *argc, char ***argv) {
  GError *err = NULL;

  GOptionContext *octx = make_option_context(info);
  common_parse_options(octx, argc, argv, &err);
  g_option_context_free(octx);

  if (err) {
    fprintf(stderr, "%s: %s\n\n", g_get_prgname(), err->message);
    g_error_free(err);
    common_usage(info);

  } else if (show_version) {
    fprintf(stderr, version_format, g_get_prgname(), openslide_get_version());
    exit(0);
  }

  // Remove "--" arguments; g_option_context_parse() doesn't
  for (int i = 0; i < *argc; i++) {
    if (!strcmp((*argv)[i], "--")) {
#ifdef CMDLINE_FREE_ARGS
      free((*argv)[i]);
#endif
      for (int j = i + 1; j <= *argc; j++) {
        (*argv)[j - 1] = (*argv)[j];
      }
      --*argc;
      --i;
    }
  }
}

void common_usage(const struct common_usage_info *info) {
  GOptionContext *octx = make_option_context(info);

  gchar *help = g_option_context_get_help(octx, TRUE, NULL);
  fprintf(stderr, "%s", help);
  g_free(help);

  g_option_context_free(octx);
  exit(2);
}


# In[ ]:


*()
 *  OpenSlide, a library for reading whole slide image files
 *
 *  Copyright (c) 2007-2015 Carnegie Mellon University
 *  All rights reserved.
 *
 *  OpenSlide is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, version 2.1.
 *
 *  OpenSlide is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with OpenSlide. If not, see
 *  <http://www.gnu.org/licenses/>.
 *
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "openslide-common.h"

void common_fail(const char *fmt, ...) {
  va_list ap;

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}


# In[ ]:


*()
 *  OpenSlide, a library for reading whole slide image files
 *
 *  Copyright (c) 2007-2014 Carnegie Mellon University
 *  All rights reserved.
 *
 *  OpenSlide is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, version 2.1.
 *
 *  OpenSlide is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with OpenSlide. If not, see
 *  <http://www.gnu.org/licenses/>.
 *
 */

#include <config.h>

#ifdef WIN32
#define _WIN32_WINNT 0x0600
#include <windows.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <glib.h>

#ifdef HAVE_PROC_PIDFDINFO
(/, Mac, OS, X)
#include <sys/param.h>  // MAXPATHLEN
#include <libproc.h>
#endif

#include "openslide-common.h"

char *common_get_fd_path(int fd) {
  struct stat st;
  if (fstat(fd, &st)) {
    return NULL;
  }

#if defined WIN32
  // Windows
  HANDLE hdl = (HANDLE) _get_osfhandle(fd);
  if (hdl != INVALID_HANDLE_VALUE) {
    DWORD size = GetFinalPathNameByHandle(hdl, NULL, 0, 0);
    if (size) {
      char *path = g_malloc(size);
      DWORD ret = GetFinalPathNameByHandle(hdl, path, size - 1, 0);
      if (ret > 0 && ret <= size) {
        return path;
      }
      g_free(path);
    }
  }
#elif defined HAVE_PROC_PIDFDINFO
  // Mac OS X
  // Ignore kqueues, since they can be opened behind our back for
  // Grand Central Dispatch
  struct kqueue_fdinfo kqi;
  if (proc_pidfdinfo(getpid(), fd, PROC_PIDFDKQUEUEINFO, &kqi, sizeof(kqi))) {
    return NULL;
  }
  char *path = g_malloc(MAXPATHLEN);
  if (!fcntl(fd, F_GETPATH, path)) {
    return path;
  }
  g_free(path);
#else
  (/, Fallback;, works, only, on, Linux)
  char *link_path = g_strdup_printf("/proc/%d/fd/%d", getpid(), fd);
  char *path = g_file_read_link(link_path, NULL);
  g_free(link_path);
  if (path) {
    return path;
  }
#endif

  return g_strdup("<unknown>");
}


# In[ ]:


*()
 *  OpenSlide, a library for reading whole slide image files
 *
 *  Copyright (c) 2007-2014 Carnegie Mellon University
 *  All rights reserved.
 *
 *  OpenSlide is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, version 2.1.
 *
 *  OpenSlide is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with OpenSlide. If not, see
 *  <http://www.gnu.org/licenses/>.
 *
 */

#ifndef OPENSLIDE_COMMON_H
#define OPENSLIDE_COMMON_H

#include <stdbool.h>
#include <glib.h>

(/, cmdline)

struct common_usage_info {
  const char *parameter_string;
  const char *summary;
};

void common_fix_argv(int *argc, char ***argv);

bool common_parse_options(GOptionContext *ctx,
                          int *argc, char ***argv,
                          GError **err);

void common_parse_commandline(const struct common_usage_info *info,
                              int *argc, char ***argv);

void common_usage(const struct common_usage_info *info) G_GNUC_NORETURN;

(/, fail)

void common_fail(const char *fmt, ...) G_GNUC_NORETURN;

(/, fd)

char *common_get_fd_path(int fd);

#endif


# In[ ]:


# Doxyfile 1.8.11

# This file describes the settings to be used by the documentation system
# doxygen (www.doxygen.org) for a project.
#
# All text after a double hash (##) is considered a comment and is placed in
# front of the TAG it is preceding.
#
# All text after a single hash (#) is considered a comment and will be ignored.
# The format is:
# TAG = value [value, ...]
# For lists, items can also be appended using:
# TAG += value [value, ...]
# Values that contain spaces should be placed between quotes (\" \").

#---------------------------------------------------------------------------
# Project related configuration options
#---------------------------------------------------------------------------

# This tag specifies the encoding used for all characters in the config file
# that follow. The default is UTF-8 which is also the encoding used for all text
# before the first occurrence of this tag. Doxygen uses libiconv (or the iconv
# built into libc) for the transcoding. See http://www.gnu.org/software/libiconv
# for the list of possible encodings.
# The default value is: UTF-8.

DOXYFILE_ENCODING      = UTF-8

# The PROJECT_NAME tag is a single word (or a sequence of words surrounded by
# double-quotes, unless you are using Doxywizard) that should identify the
# project for which the documentation is generated. This name is used in the
# title of most generated pages and in a few other places.
# The default value is: My Project.

PROJECT_NAME           = OpenSlide

# The PROJECT_NUMBER tag can be used to enter a project or revision number. This
# could be handy for archiving the generated documentation or if some version
# control system is used.

PROJECT_NUMBER         =

# Using the PROJECT_BRIEF tag one can provide an optional one line description
# for a project that appears at the top of each page and should give viewer a
# quick idea about the purpose of the project. Keep the description short.

PROJECT_BRIEF          =

# With the PROJECT_LOGO tag one can specify a logo or an icon that is included
# in the documentation. The maximum height of the logo should not exceed 55
# pixels and the maximum width should not exceed 200 pixels. Doxygen will copy
# the logo to the output directory.

PROJECT_LOGO           =

# The OUTPUT_DIRECTORY tag is used to specify the (relative or absolute) path
# into which the generated documentation will be written. If a relative path is
# entered, it will be relative to the location where doxygen was started. If
# left blank the current directory will be used.

OUTPUT_DIRECTORY       =

# If the CREATE_SUBDIRS tag is set to YES then doxygen will create 4096 sub-
# directories (in 2 levels) under the output directory of each output format and
# will distribute the generated files over these directories. Enabling this
# option can be useful when feeding doxygen a huge amount of source files, where
# putting all generated files in the same directory would otherwise causes
# performance problems for the file system.
# The default value is: NO.

CREATE_SUBDIRS         = NO

# If the ALLOW_UNICODE_NAMES tag is set to YES, doxygen will allow non-ASCII
# characters to appear in the names of generated files. If set to NO, non-ASCII
# characters will be escaped, for example _xE3_x81_x84 will be used for Unicode
# U+3044.
# The default value is: NO.

ALLOW_UNICODE_NAMES    = NO

# The OUTPUT_LANGUAGE tag is used to specify the language in which all
# documentation generated by doxygen is written. Doxygen will use this
# information to generate all constant output in the proper language.
# Possible values are: Afrikaans, Arabic, Armenian, Brazilian, Catalan, Chinese,
# Chinese-Traditional, Croatian, Czech, Danish, Dutch, English (United States),
# Esperanto, Farsi (Persian), Finnish, French, German, Greek, Hungarian,
# Indonesian, Italian, Japanese, Japanese-en (Japanese with English messages),
# Korean, Korean-en (Korean with English messages), Latvian, Lithuanian,
# Macedonian, Norwegian, Persian (Farsi), Polish, Portuguese, Romanian, Russian,
# Serbian, Serbian-Cyrillic, Slovak, Slovene, Spanish, Swedish, Turkish,
# Ukrainian and Vietnamese.
# The default value is: English.

OUTPUT_LANGUAGE        = English

# If the BRIEF_MEMBER_DESC tag is set to YES, doxygen will include brief member
# descriptions after the members that are listed in the file and class
# documentation (similar to Javadoc). Set to NO to disable this.
# The default value is: YES.

BRIEF_MEMBER_DESC      = YES

# If the REPEAT_BRIEF tag is set to YES, doxygen will prepend the brief
# description of a member or function before the detailed description
#
# Note: If both HIDE_UNDOC_MEMBERS and BRIEF_MEMBER_DESC are set to NO, the
# brief descriptions will be completely suppressed.
# The default value is: YES.

REPEAT_BRIEF           = YES

# This tag implements a quasi-intelligent brief description abbreviator that is
# used to form the text in various listings. Each string in this list, if found
# as the leading text of the brief description, will be stripped from the text
# and the result, after processing the whole list, is used as the annotated
# text. Otherwise, the brief description is used as-is. If left blank, the
# following values are used ($name is automatically replaced with the name of
# the entity):The $name class, The $name widget, The $name file, is, provides,
# specifies, contains, represents, a, an and the.

ABBREVIATE_BRIEF       =

# If the ALWAYS_DETAILED_SEC and REPEAT_BRIEF tags are both set to YES then
# doxygen will generate a detailed section even if there is only a brief
# description.
# The default value is: NO.

ALWAYS_DETAILED_SEC    = NO

# If the INLINE_INHERITED_MEMB tag is set to YES, doxygen will show all
# inherited members of a class in the documentation of that class as if those
# members were ordinary class members. Constructors, destructors and assignment
# operators of the base classes will not be shown.
# The default value is: NO.

INLINE_INHERITED_MEMB  = NO

# If the FULL_PATH_NAMES tag is set to YES, doxygen will prepend the full path
# before files name in the file list and in the header files. If set to NO the
# shortest path that makes the file name unique will be used
# The default value is: YES.

FULL_PATH_NAMES        = NO

# The STRIP_FROM_PATH tag can be used to strip a user-defined part of the path.
# Stripping is only done if one of the specified strings matches the left-hand
# part of the path. The tag can be used to show relative paths in the file list.
# If left blank the directory from which doxygen is run is used as the path to
# strip.
#
# Note that you can specify absolute paths here, but also relative paths, which
# will be relative from the directory where doxygen is started.
# This tag requires that the tag FULL_PATH_NAMES is set to YES.

STRIP_FROM_PATH        =

# The STRIP_FROM_INC_PATH tag can be used to strip a user-defined part of the
# path mentioned in the documentation of a class, which tells the reader which
# header file to include in order to use a class. If left blank only the name of
# the header file containing the class definition is used. Otherwise one should
# specify the list of include paths that are normally passed to the compiler
# using the -I flag.

STRIP_FROM_INC_PATH    =

# If the SHORT_NAMES tag is set to YES, doxygen will generate much shorter (but
# less readable) file names. This can be useful is your file systems doesn't
# support long names like on DOS, Mac, or CD-ROM.
# The default value is: NO.

SHORT_NAMES            = NO

# If the JAVADOC_AUTOBRIEF tag is set to YES then doxygen will interpret the
# first line (until the first dot) of a Javadoc-style comment as the brief
# description. If set to NO, the Javadoc-style will behave just like regular Qt-
# style comments (thus requiring an explicit @brief command for a brief
# description.)
# The default value is: NO.

JAVADOC_AUTOBRIEF      = YES

# If the QT_AUTOBRIEF tag is set to YES then doxygen will interpret the first
# line (until the first dot) of a Qt-style comment as the brief description. If
# set to NO, the Qt-style will behave just like regular Qt-style comments (thus
# requiring an explicit \brief command for a brief description.)
# The default value is: NO.

QT_AUTOBRIEF           = NO

# The MULTILINE_CPP_IS_BRIEF tag can be set to YES to make doxygen treat a
# multi-line C++ special comment block (i.e. a block of //! or /// comments) as
# a brief description. This used to be the default behavior. The new default is
# to treat a multi-line C++ comment block as a detailed description. Set this
# tag to YES if you prefer the old behavior instead.
#
# Note that setting this tag to YES also means that rational rose comments are
# not recognized any more.
# The default value is: NO.

MULTILINE_CPP_IS_BRIEF = NO

# If the INHERIT_DOCS tag is set to YES then an undocumented member inherits the
# documentation from any documented member that it re-implements.
# The default value is: YES.

INHERIT_DOCS           = YES

# If the SEPARATE_MEMBER_PAGES tag is set to YES then doxygen will produce a new
# page for each member. If set to NO, the documentation of a member will be part
# of the file/class/namespace that contains it.
# The default value is: NO.

SEPARATE_MEMBER_PAGES  = NO

# The TAB_SIZE tag can be used to set the number of spaces in a tab. Doxygen
# uses this value to replace tabs by spaces in code fragments.
# Minimum value: 1, maximum value: 16, default value: 4.

TAB_SIZE               = 8

# This tag can be used to specify a number of aliases that act as commands in
# the documentation. An alias has the form:
# name=value
# For example adding
# "sideeffect=@par Side Effects:\n"
# will allow you to put the command \sideeffect (or @sideeffect) in the
# documentation, which will result in a user-defined paragraph with heading
# "Side Effects:". You can put \n's in the value part of an alias to insert
# newlines.

ALIASES                =

# This tag can be used to specify a number of word-keyword mappings (TCL only).
# A mapping has the form "name=value". For example adding "class=itcl::class"
# will allow you to use the command class in the itcl::class meaning.

TCL_SUBST              =

# Set the OPTIMIZE_OUTPUT_FOR_C tag to YES if your project consists of C sources
# only. Doxygen will then generate output that is more tailored for C. For
# instance, some of the names that are used will be different. The list of all
# members will be omitted, etc.
# The default value is: NO.

OPTIMIZE_OUTPUT_FOR_C  = YES

# Set the OPTIMIZE_OUTPUT_JAVA tag to YES if your project consists of Java or
# Python sources only. Doxygen will then generate output that is more tailored
# for that language. For instance, namespaces will be presented as packages,
# qualified scopes will look different, etc.
# The default value is: NO.

OPTIMIZE_OUTPUT_JAVA   = NO

# Set the OPTIMIZE_FOR_FORTRAN tag to YES if your project consists of Fortran
# sources. Doxygen will then generate output that is tailored for Fortran.
# The default value is: NO.

OPTIMIZE_FOR_FORTRAN   = NO

# Set the OPTIMIZE_OUTPUT_VHDL tag to YES if your project consists of VHDL
# sources. Doxygen will then generate output that is tailored for VHDL.
# The default value is: NO.

OPTIMIZE_OUTPUT_VHDL   = NO

# Doxygen selects the parser to use depending on the extension of the files it
# parses. With this tag you can assign which parser to use for a given
# extension. Doxygen has a built-in mapping, but you can override or extend it
# using this tag. The format is ext=language, where ext is a file extension, and
# language is one of the parsers supported by doxygen: IDL, Java, Javascript,
# C#, C, C++, D, PHP, Objective-C, Python, Fortran (fixed format Fortran:
# FortranFixed, free formatted Fortran: FortranFree, unknown formatted Fortran:
# Fortran. In the later case the parser tries to guess whether the code is fixed
# or free formatted code, this is the default for Fortran type files), VHDL. For
# instance to make doxygen treat .inc files as Fortran files (default is PHP),
# and .f files as C (default is Fortran), use: inc=Fortran f=C.
#
# Note: For files without extension you can use no_extension as a placeholder.
#
# Note that for custom extensions you also need to set FILE_PATTERNS otherwise
# the files are not read by doxygen.

EXTENSION_MAPPING      =

# If the MARKDOWN_SUPPORT tag is enabled then doxygen pre-processes all comments
# according to the Markdown format, which allows for more readable
# documentation. See http://daringfireball.net/projects/markdown/ for details.
# The output of markdown processing is further processed by doxygen, so you can
# mix doxygen, HTML, and XML commands with Markdown formatting. Disable only in
# case of backward compatibilities issues.
# The default value is: YES.

MARKDOWN_SUPPORT       = YES

# When enabled doxygen tries to link words that correspond to documented
# classes, or namespaces to their corresponding documentation. Such a link can
# be prevented in individual cases by putting a % sign in front of the word or
# globally by setting AUTOLINK_SUPPORT to NO.
# The default value is: YES.

AUTOLINK_SUPPORT       = YES

# If you use STL classes (i.e. std::string, std::vector, etc.) but do not want
# to include (a tag file for) the STL sources as input, then you should set this
# tag to YES in order to let doxygen match functions declarations and
# definitions whose arguments contain STL classes (e.g. func(std::string);
# versus func(std::string) {}). This also make the inheritance and collaboration
# diagrams that involve STL classes more complete and accurate.
# The default value is: NO.

BUILTIN_STL_SUPPORT    = NO

# If you use Microsoft's C++/CLI language, you should set this option to YES to
# enable parsing support.
# The default value is: NO.

CPP_CLI_SUPPORT        = NO

# Set the SIP_SUPPORT tag to YES if your project consists of sip (see:
# http://www.riverbankcomputing.co.uk/software/sip/intro) sources only. Doxygen
# will parse them like normal C++ but will assume all classes use public instead
# of private inheritance when no explicit protection keyword is present.
# The default value is: NO.

SIP_SUPPORT            = NO

# For Microsoft's IDL there are propget and propput attributes to indicate
# getter and setter methods for a property. Setting this option to YES will make
# doxygen to replace the get and set methods by a property in the documentation.
# This will only work if the methods are indeed getting or setting a simple
# type. If this is not the case, or you want to show the methods anyway, you
# should set this option to NO.
# The default value is: YES.

IDL_PROPERTY_SUPPORT   = YES

# If member grouping is used in the documentation and the DISTRIBUTE_GROUP_DOC
# tag is set to YES then doxygen will reuse the documentation of the first
# member in the group (if any) for the other members of the group. By default
# all members of a group must be documented explicitly.
# The default value is: NO.

DISTRIBUTE_GROUP_DOC   = NO

# If one adds a struct or class to a group and this option is enabled, then also
# any nested class or struct is added to the same group. By default this option
# is disabled and one has to add nested compounds explicitly via \ingroup.
# The default value is: NO.

GROUP_NESTED_COMPOUNDS = NO

# Set the SUBGROUPING tag to YES to allow class member groups of the same type
# (for instance a group of public functions) to be put as a subgroup of that
# type (e.g. under the Public Functions section). Set it to NO to prevent
# subgrouping. Alternatively, this can be done per class using the
# \nosubgrouping command.
# The default value is: YES.

SUBGROUPING            = YES

# When the INLINE_GROUPED_CLASSES tag is set to YES, classes, structs and unions
# are shown inside the group in which they are included (e.g. using \ingroup)
# instead of on a separate page (for HTML and Man pages) or section (for LaTeX
# and RTF).
#
# Note that this feature does not work in combination with
# SEPARATE_MEMBER_PAGES.
# The default value is: NO.

INLINE_GROUPED_CLASSES = NO

# When the INLINE_SIMPLE_STRUCTS tag is set to YES, structs, classes, and unions
# with only public data fields or simple typedef fields will be shown inline in
# the documentation of the scope in which they are defined (i.e. file,
# namespace, or group documentation), provided this scope is documented. If set
# to NO, structs, classes, and unions are shown on a separate page (for HTML and
# Man pages) or section (for LaTeX and RTF).
# The default value is: NO.

INLINE_SIMPLE_STRUCTS  = NO

# When TYPEDEF_HIDES_STRUCT tag is enabled, a typedef of a struct, union, or
# enum is documented as struct, union, or enum with the name of the typedef. So
# typedef struct TypeS {} TypeT, will appear in the documentation as a struct
# with name TypeT. When disabled the typedef will appear as a member of a file,
# namespace, or class. And the struct will be named TypeS. This can typically be
# useful for C code in case the coding convention dictates that all compound
# types are typedef'ed and only the typedef is referenced, never the tag name.
# The default value is: NO.

TYPEDEF_HIDES_STRUCT   = NO

# The size of the symbol lookup cache can be set using LOOKUP_CACHE_SIZE. This
# cache is used to resolve symbols given their name and scope. Since this can be
# an expensive process and often the same symbol appears multiple times in the
# code, doxygen keeps a cache of pre-resolved symbols. If the cache is too small
# doxygen will become slower. If the cache is too large, memory is wasted. The
# cache size is given by this formula: 2^(16+LOOKUP_CACHE_SIZE). The valid range
# is 0..9, the default is 0, corresponding to a cache size of 2^16=65536
# symbols. At the end of a run doxygen will report the cache usage and suggest
# the optimal cache size from a speed point of view.
# Minimum value: 0, maximum value: 9, default value: 0.

LOOKUP_CACHE_SIZE      = 0

#---------------------------------------------------------------------------
# Build related configuration options
#---------------------------------------------------------------------------

# If the EXTRACT_ALL tag is set to YES, doxygen will assume all entities in
# documentation are documented, even if no documentation was available. Private
# class members and static file members will be hidden unless the
# EXTRACT_PRIVATE respectively EXTRACT_STATIC tags are set to YES.
# Note: This will also disable the warnings about undocumented members that are
# normally produced when WARNINGS is set to YES.
# The default value is: NO.

EXTRACT_ALL            = NO

# If the EXTRACT_PRIVATE tag is set to YES, all private members of a class will
# be included in the documentation.
# The default value is: NO.

EXTRACT_PRIVATE        = NO

# If the EXTRACT_PACKAGE tag is set to YES, all members with package or internal
# scope will be included in the documentation.
# The default value is: NO.

EXTRACT_PACKAGE        = NO

# If the EXTRACT_STATIC tag is set to YES, all static members of a file will be
# included in the documentation.
# The default value is: NO.

EXTRACT_STATIC         = NO

# If the EXTRACT_LOCAL_CLASSES tag is set to YES, classes (and structs) defined
# locally in source files will be included in the documentation. If set to NO,
# only classes defined in header files are included. Does not have any effect
# for Java sources.
# The default value is: YES.

EXTRACT_LOCAL_CLASSES  = YES

# This flag is only useful for Objective-C code. If set to YES, local methods,
# which are defined in the implementation section but not in the interface are
# included in the documentation. If set to NO, only methods in the interface are
# included.
# The default value is: NO.

EXTRACT_LOCAL_METHODS  = NO

# If this flag is set to YES, the members of anonymous namespaces will be
# extracted and appear in the documentation as a namespace called
# 'anonymous_namespace{file}', where file will be replaced with the base name of
# the file that contains the anonymous namespace. By default anonymous namespace
# are hidden.
# The default value is: NO.

EXTRACT_ANON_NSPACES   = NO

# If the HIDE_UNDOC_MEMBERS tag is set to YES, doxygen will hide all
# undocumented members inside documented classes or files. If set to NO these
# members will be included in the various overviews, but no documentation
# section is generated. This option has no effect if EXTRACT_ALL is enabled.
# The default value is: NO.

HIDE_UNDOC_MEMBERS     = YES

# If the HIDE_UNDOC_CLASSES tag is set to YES, doxygen will hide all
# undocumented classes that are normally visible in the class hierarchy. If set
# to NO, these classes will be included in the various overviews. This option
# has no effect if EXTRACT_ALL is enabled.
# The default value is: NO.

HIDE_UNDOC_CLASSES     = NO

# If the HIDE_FRIEND_COMPOUNDS tag is set to YES, doxygen will hide all friend
# (class|struct|union) declarations. If set to NO, these declarations will be
# included in the documentation.
# The default value is: NO.

HIDE_FRIEND_COMPOUNDS  = NO

# If the HIDE_IN_BODY_DOCS tag is set to YES, doxygen will hide any
# documentation blocks found inside the body of a function. If set to NO, these
# blocks will be appended to the function's detailed documentation block.
# The default value is: NO.

HIDE_IN_BODY_DOCS      = NO

# The INTERNAL_DOCS tag determines if documentation that is typed after a
# \internal command is included. If the tag is set to NO then the documentation
# will be excluded. Set it to YES to include the internal documentation.
# The default value is: NO.

INTERNAL_DOCS          = NO

# If the CASE_SENSE_NAMES tag is set to NO then doxygen will only generate file
# names in lower-case letters. If set to YES, upper-case letters are also
# allowed. This is useful if you have classes or files whose names only differ
# in case and if your file system supports case sensitive file names. Windows
# and Mac users are advised to set this option to NO.
# The default value is: system dependent.

CASE_SENSE_NAMES       = YES

# If the HIDE_SCOPE_NAMES tag is set to NO then doxygen will show members with
# their full class and namespace scopes in the documentation. If set to YES, the
# scope will be hidden.
# The default value is: NO.

HIDE_SCOPE_NAMES       = NO

# If the HIDE_COMPOUND_REFERENCE tag is set to NO (default) then doxygen will
# append additional text to a page's title, such as Class Reference. If set to
# YES the compound reference will be hidden.
# The default value is: NO.

HIDE_COMPOUND_REFERENCE= NO

# If the SHOW_INCLUDE_FILES tag is set to YES then doxygen will put a list of
# the files that are included by a file in the documentation of that file.
# The default value is: YES.

SHOW_INCLUDE_FILES     = YES

# If the SHOW_GROUPED_MEMB_INC tag is set to YES then Doxygen will add for each
# grouped member an include statement to the documentation, telling the reader
# which file to include in order to use the member.
# The default value is: NO.

SHOW_GROUPED_MEMB_INC  = NO

# If the FORCE_LOCAL_INCLUDES tag is set to YES then doxygen will list include
# files with double quotes in the documentation rather than with sharp brackets.
# The default value is: NO.

FORCE_LOCAL_INCLUDES   = NO

# If the INLINE_INFO tag is set to YES then a tag [inline] is inserted in the
# documentation for inline members.
# The default value is: YES.

INLINE_INFO            = YES

# If the SORT_MEMBER_DOCS tag is set to YES then doxygen will sort the
# (detailed) documentation of file and class members alphabetically by member
# name. If set to NO, the members will appear in declaration order.
# The default value is: YES.

SORT_MEMBER_DOCS       = YES

# If the SORT_BRIEF_DOCS tag is set to YES then doxygen will sort the brief
# descriptions of file, namespace and class members alphabetically by member
# name. If set to NO, the members will appear in declaration order. Note that
# this will also influence the order of the classes in the class list.
# The default value is: NO.

SORT_BRIEF_DOCS        = NO

# If the SORT_MEMBERS_CTORS_1ST tag is set to YES then doxygen will sort the
# (brief and detailed) documentation of class members so that constructors and
# destructors are listed first. If set to NO the constructors will appear in the
# respective orders defined by SORT_BRIEF_DOCS and SORT_MEMBER_DOCS.
# Note: If SORT_BRIEF_DOCS is set to NO this option is ignored for sorting brief
# member documentation.
# Note: If SORT_MEMBER_DOCS is set to NO this option is ignored for sorting
# detailed member documentation.
# The default value is: NO.

SORT_MEMBERS_CTORS_1ST = NO

# If the SORT_GROUP_NAMES tag is set to YES then doxygen will sort the hierarchy
# of group names into alphabetical order. If set to NO the group names will
# appear in their defined order.
# The default value is: NO.

SORT_GROUP_NAMES       = NO

# If the SORT_BY_SCOPE_NAME tag is set to YES, the class list will be sorted by
# fully-qualified names, including namespaces. If set to NO, the class list will
# be sorted only by class name, not including the namespace part.
# Note: This option is not very useful if HIDE_SCOPE_NAMES is set to YES.
# Note: This option applies only to the class list, not to the alphabetical
# list.
# The default value is: NO.

SORT_BY_SCOPE_NAME     = NO

# If the STRICT_PROTO_MATCHING option is enabled and doxygen fails to do proper
# type resolution of all parameters of a function it will reject a match between
# the prototype and the implementation of a member function even if there is
# only one candidate or it is obvious which candidate to choose by doing a
# simple string match. By disabling STRICT_PROTO_MATCHING doxygen will still
# accept a match between prototype and implementation in such cases.
# The default value is: NO.

STRICT_PROTO_MATCHING  = NO

# The GENERATE_TODOLIST tag can be used to enable (YES) or disable (NO) the todo
# list. This list is created by putting \todo commands in the documentation.
# The default value is: YES.

GENERATE_TODOLIST      = YES

# The GENERATE_TESTLIST tag can be used to enable (YES) or disable (NO) the test
# list. This list is created by putting \test commands in the documentation.
# The default value is: YES.

GENERATE_TESTLIST      = YES

# The GENERATE_BUGLIST tag can be used to enable (YES) or disable (NO) the bug
# list. This list is created by putting \bug commands in the documentation.
# The default value is: YES.

GENERATE_BUGLIST       = YES

# The GENERATE_DEPRECATEDLIST tag can be used to enable (YES) or disable (NO)
# the deprecated list. This list is created by putting \deprecated commands in
# the documentation.
# The default value is: YES.

GENERATE_DEPRECATEDLIST= YES

# The ENABLED_SECTIONS tag can be used to enable conditional documentation
# sections, marked by \if <section_label> ... \endif and \cond <section_label>
# ... \endcond blocks.

ENABLED_SECTIONS       =

# The MAX_INITIALIZER_LINES tag determines the maximum number of lines that the
# initial value of a variable or macro / define can have for it to appear in the
# documentation. If the initializer consists of more lines than specified here
# it will be hidden. Use a value of 0 to hide initializers completely. The
# appearance of the value of individual variables and macros / defines can be
# controlled using \showinitializer or \hideinitializer command in the
# documentation regardless of this setting.
# Minimum value: 0, maximum value: 10000, default value: 30.

MAX_INITIALIZER_LINES  = 30

# Set the SHOW_USED_FILES tag to NO to disable the list of files generated at
# the bottom of the documentation of classes and structs. If set to YES, the
# list will mention the files that were used to generate the documentation.
# The default value is: YES.

SHOW_USED_FILES        = YES

# Set the SHOW_FILES tag to NO to disable the generation of the Files page. This
# will remove the Files entry from the Quick Index and from the Folder Tree View
# (if specified).
# The default value is: YES.

SHOW_FILES             = YES

# Set the SHOW_NAMESPACES tag to NO to disable the generation of the Namespaces
# page. This will remove the Namespaces entry from the Quick Index and from the
# Folder Tree View (if specified).
# The default value is: YES.

SHOW_NAMESPACES        = YES

# The FILE_VERSION_FILTER tag can be used to specify a program or script that
# doxygen should invoke to get the current version for each file (typically from
# the version control system). Doxygen will invoke the program by executing (via
# popen()) the command command input-file, where command is the value of the
# FILE_VERSION_FILTER tag, and input-file is the name of an input file provided
# by doxygen. Whatever the program writes to standard output is used as the file
# version. For an example see the documentation.

FILE_VERSION_FILTER    =

# The LAYOUT_FILE tag can be used to specify a layout file which will be parsed
# by doxygen. The layout file controls the global structure of the generated
# output files in an output format independent way. To create the layout file
# that represents doxygen's defaults, run doxygen with the -l option. You can
# optionally specify a file name after the option, if omitted DoxygenLayout.xml
# will be used as the name of the layout file.
#
# Note that if you run doxygen from a directory containing a file called
# DoxygenLayout.xml, doxygen will parse it automatically even if the LAYOUT_FILE
# tag is left empty.

LAYOUT_FILE            =

# The CITE_BIB_FILES tag can be used to specify one or more bib files containing
# the reference definitions. This must be a list of .bib files. The .bib
# extension is automatically appended if omitted. This requires the bibtex tool
# to be installed. See also http://en.wikipedia.org/wiki/BibTeX for more info.
# For LaTeX the style of the bibliography can be controlled using
# LATEX_BIB_STYLE. To use this feature you need bibtex and perl available in the
# search path. See also \cite for info how to create references.

CITE_BIB_FILES         =

#---------------------------------------------------------------------------
# Configuration options related to warning and progress messages
#---------------------------------------------------------------------------

# The QUIET tag can be used to turn on/off the messages that are generated to
# standard output by doxygen. If QUIET is set to YES this implies that the
# messages are off.
# The default value is: NO.

QUIET                  = YES

# The WARNINGS tag can be used to turn on/off the warning messages that are
# generated to standard error (stderr) by doxygen. If WARNINGS is set to YES
# this implies that the warnings are on.
#
# Tip: Turn warnings on while writing the documentation.
# The default value is: YES.

WARNINGS               = YES

# If the WARN_IF_UNDOCUMENTED tag is set to YES then doxygen will generate
# warnings for undocumented members. If EXTRACT_ALL is set to YES then this flag
# will automatically be disabled.
# The default value is: YES.

WARN_IF_UNDOCUMENTED   = YES

# If the WARN_IF_DOC_ERROR tag is set to YES, doxygen will generate warnings for
# potential errors in the documentation, such as not documenting some parameters
# in a documented function, or documenting parameters that don't exist or using
# markup commands wrongly.
# The default value is: YES.

WARN_IF_DOC_ERROR      = YES

# This WARN_NO_PARAMDOC option can be enabled to get warnings for functions that
# are documented, but have no documentation for their parameters or return
# value. If set to NO, doxygen will only warn about wrong or incomplete
# parameter documentation, but not about the absence of documentation.
# The default value is: NO.

WARN_NO_PARAMDOC       = YES

# If the WARN_AS_ERROR tag is set to YES then doxygen will immediately stop when
# a warning is encountered.
# The default value is: NO.

WARN_AS_ERROR          = NO

# The WARN_FORMAT tag determines the format of the warning messages that doxygen
# can produce. The string should contain the $file, $line, and $text tags, which
# will be replaced by the file and line number from which the warning originated
# and the warning text. Optionally the format may contain $version, which will
# be replaced by the version of the file (if it could be obtained via
# FILE_VERSION_FILTER)
# The default value is: $file:$line: $text.

WARN_FORMAT            = "$file:$line: $text"

# The WARN_LOGFILE tag can be used to specify a file to which warning and error
# messages should be written. If left blank the output is written to standard
# error (stderr).

WARN_LOGFILE           =

#---------------------------------------------------------------------------
# Configuration options related to the input files
#---------------------------------------------------------------------------

# The INPUT tag is used to specify the files and/or directories that contain
# documented source files. You may enter file names like myfile.cpp or
# directories like /usr/src/myproject. Separate the files or directories with
# spaces. See also FILE_PATTERNS and EXTENSION_MAPPING
# Note: If this tag is empty the current directory is searched.

INPUT                  = ../src/openslide.h

# This tag can be used to specify the character encoding of the source files
# that doxygen parses. Internally doxygen uses the UTF-8 encoding. Doxygen uses
# libiconv (or the iconv built into libc) for the transcoding. See the libiconv
# documentation (see: http://www.gnu.org/software/libiconv) for the list of
# possible encodings.
# The default value is: UTF-8.

INPUT_ENCODING         = UTF-8

# If the value of the INPUT tag contains directories, you can use the
# FILE_PATTERNS tag to specify one or more wildcard patterns (like *.cpp and
# *.h) to filter out the source-files in the directories.
#
# Note that for custom extensions or not directly supported extensions you also
# need to set EXTENSION_MAPPING for the extension otherwise the files are not
# read by doxygen.
#
# If left blank the following patterns are tested:*.c, *.cc, *.cxx, *.cpp,
# *.c++, *.java, *.ii, *.ixx, *.ipp, *.i++, *.inl, *.idl, *.ddl, *.odl, *.h,
# *.hh, *.hxx, *.hpp, *.h++, *.cs, *.d, *.php, *.php4, *.php5, *.phtml, *.inc,
# *.m, *.markdown, *.md, *.mm, *.dox, *.py, *.pyw, *.f90, *.f, *.for, *.tcl,
# *.vhd, *.vhdl, *.ucf, *.qsf, *.as and *.js.

FILE_PATTERNS          =

# The RECURSIVE tag can be used to specify whether or not subdirectories should
# be searched for input files as well.
# The default value is: NO.

RECURSIVE              = NO

# The EXCLUDE tag can be used to specify files and/or directories that should be
# excluded from the INPUT source files. This way you can easily exclude a
# subdirectory from a directory tree whose root is specified with the INPUT tag.
#
# Note that relative paths are relative to the directory from which doxygen is
# run.

EXCLUDE                =

# The EXCLUDE_SYMLINKS tag can be used to select whether or not files or
# directories that are symbolic links (a Unix file system feature) are excluded
# from the input.
# The default value is: NO.

EXCLUDE_SYMLINKS       = NO

# If the value of the INPUT tag contains directories, you can use the
# EXCLUDE_PATTERNS tag to specify one or more wildcard patterns to exclude
# certain files from those directories.
#
# Note that the wildcards are matched against the file with absolute path, so to
# exclude all test directories for example use the pattern */test/*

EXCLUDE_PATTERNS       =

# The EXCLUDE_SYMBOLS tag can be used to specify one or more symbol names
# (namespaces, classes, functions, etc.) that should be excluded from the
# output. The symbol name can be a fully qualified name, a word, or if the
# wildcard * is used, a substring. Examples: ANamespace, AClass,
# AClass::ANamespace, ANamespace::*Test
#
# Note that the wildcards are matched against the file with absolute path, so to
# exclude all test directories use the pattern */test/*

EXCLUDE_SYMBOLS        =

# The EXAMPLE_PATH tag can be used to specify one or more files or directories
# that contain example code fragments that are included (see the \include
# command).

EXAMPLE_PATH           =

# If the value of the EXAMPLE_PATH tag contains directories, you can use the
# EXAMPLE_PATTERNS tag to specify one or more wildcard pattern (like *.cpp and
# *.h) to filter out the source-files in the directories. If left blank all
# files are included.

EXAMPLE_PATTERNS       =

# If the EXAMPLE_RECURSIVE tag is set to YES then subdirectories will be
# searched for input files to be used with the \include or \dontinclude commands
# irrespective of the value of the RECURSIVE tag.
# The default value is: NO.

EXAMPLE_RECURSIVE      = NO

# The IMAGE_PATH tag can be used to specify one or more files or directories
# that contain images that are to be included in the documentation (see the
# \image command).

IMAGE_PATH             =

# The INPUT_FILTER tag can be used to specify a program that doxygen should
# invoke to filter for each input file. Doxygen will invoke the filter program
# by executing (via popen()) the command:
#
# <filter> <input-file>
#
# where <filter> is the value of the INPUT_FILTER tag, and <input-file> is the
# name of an input file. Doxygen will then use the output that the filter
# program writes to standard output. If FILTER_PATTERNS is specified, this tag
# will be ignored.
#
# Note that the filter must not add or remove lines; it is applied before the
# code is scanned, but not when the output code is generated. If lines are added
# or removed, the anchors will not be placed correctly.
#
# Note that for custom extensions or not directly supported extensions you also
# need to set EXTENSION_MAPPING for the extension otherwise the files are not
# properly processed by doxygen.

INPUT_FILTER           =

# The FILTER_PATTERNS tag can be used to specify filters on a per file pattern
# basis. Doxygen will compare the file name with each pattern and apply the
# filter if there is a match. The filters are a list of the form: pattern=filter
# (like *.cpp=my_cpp_filter). See INPUT_FILTER for further information on how
# filters are used. If the FILTER_PATTERNS tag is empty or if none of the
# patterns match the file name, INPUT_FILTER is applied.
#
# Note that for custom extensions or not directly supported extensions you also
# need to set EXTENSION_MAPPING for the extension otherwise the files are not
# properly processed by doxygen.

FILTER_PATTERNS        =

# If the FILTER_SOURCE_FILES tag is set to YES, the input filter (if set using
# INPUT_FILTER) will also be used to filter the input files that are used for
# producing the source files to browse (i.e. when SOURCE_BROWSER is set to YES).
# The default value is: NO.

FILTER_SOURCE_FILES    = NO

# The FILTER_SOURCE_PATTERNS tag can be used to specify source filters per file
# pattern. A pattern will override the setting for FILTER_PATTERN (if any) and
# it is also possible to disable source filtering for a specific pattern using
# *.ext= (so without naming a filter).
# This tag requires that the tag FILTER_SOURCE_FILES is set to YES.

FILTER_SOURCE_PATTERNS =

# If the USE_MDFILE_AS_MAINPAGE tag refers to the name of a markdown file that
# is part of the input, its contents will be placed on the main page
# (index.html). This can be useful if you have a project on for instance GitHub
# and want to reuse the introduction page also for the doxygen output.

USE_MDFILE_AS_MAINPAGE =

#---------------------------------------------------------------------------
# Configuration options related to source browsing
#---------------------------------------------------------------------------

# If the SOURCE_BROWSER tag is set to YES then a list of source files will be
# generated. Documented entities will be cross-referenced with these sources.
#
# Note: To get rid of all source code in the generated output, make sure that
# also VERBATIM_HEADERS is set to NO.
# The default value is: NO.

SOURCE_BROWSER         = NO

# Setting the INLINE_SOURCES tag to YES will include the body of functions,
# classes and enums directly into the documentation.
# The default value is: NO.

INLINE_SOURCES         = NO

# Setting the STRIP_CODE_COMMENTS tag to YES will instruct doxygen to hide any
# special comment blocks from generated source code fragments. Normal C, C++ and
# Fortran comments will always remain visible.
# The default value is: YES.

STRIP_CODE_COMMENTS    = YES

# If the REFERENCED_BY_RELATION tag is set to YES then for each documented
# function all documented functions referencing it will be listed.
# The default value is: NO.

REFERENCED_BY_RELATION = YES

# If the REFERENCES_RELATION tag is set to YES then for each documented function
# all documented entities called/used by that function will be listed.
# The default value is: NO.

REFERENCES_RELATION    = YES

# If the REFERENCES_LINK_SOURCE tag is set to YES and SOURCE_BROWSER tag is set
# to YES then the hyperlinks from functions in REFERENCES_RELATION and
# REFERENCED_BY_RELATION lists will link to the source code. Otherwise they will
# link to the documentation.
# The default value is: YES.

REFERENCES_LINK_SOURCE = YES

# If SOURCE_TOOLTIPS is enabled (the default) then hovering a hyperlink in the
# source code will show a tooltip with additional information such as prototype,
# brief description and links to the definition and documentation. Since this
# will make the HTML file larger and loading of large files a bit slower, you
# can opt to disable this feature.
# The default value is: YES.
# This tag requires that the tag SOURCE_BROWSER is set to YES.

SOURCE_TOOLTIPS        = YES

# If the USE_HTAGS tag is set to YES then the references to source code will
# point to the HTML generated by the htags(1) tool instead of doxygen built-in
# source browser. The htags tool is part of GNU's global source tagging system
# (see http://www.gnu.org/software/global/global.html). You will need version
# 4.8.6 or higher.
#
# To use it do the following:
# - Install the latest version of global
# - Enable SOURCE_BROWSER and USE_HTAGS in the config file
# - Make sure the INPUT points to the root of the source tree
# - Run doxygen as normal
#
# Doxygen will invoke htags (and that will in turn invoke gtags), so these
# tools must be available from the command line (i.e. in the search path).
#
# The result: instead of the source browser generated by doxygen, the links to
# source code will now point to the output of htags.
# The default value is: NO.
# This tag requires that the tag SOURCE_BROWSER is set to YES.

USE_HTAGS              = NO

# If the VERBATIM_HEADERS tag is set the YES then doxygen will generate a
# verbatim copy of the header file for each class for which an include is
# specified. Set to NO to disable this.
# See also: Section \class.
# The default value is: YES.

VERBATIM_HEADERS       = YES

#---------------------------------------------------------------------------
# Configuration options related to the alphabetical class index
#---------------------------------------------------------------------------

# If the ALPHABETICAL_INDEX tag is set to YES, an alphabetical index of all
# compounds will be generated. Enable this if the project contains a lot of
# classes, structs, unions or interfaces.
# The default value is: YES.

ALPHABETICAL_INDEX     = NO

# The COLS_IN_ALPHA_INDEX tag can be used to specify the number of columns in
# which the alphabetical index list will be split.
# Minimum value: 1, maximum value: 20, default value: 5.
# This tag requires that the tag ALPHABETICAL_INDEX is set to YES.

COLS_IN_ALPHA_INDEX    = 5

# In case all classes in a project start with a common prefix, all classes will
# be put under the same header in the alphabetical index. The IGNORE_PREFIX tag
# can be used to specify a prefix (or a list of prefixes) that should be ignored
# while generating the index headers.
# This tag requires that the tag ALPHABETICAL_INDEX is set to YES.

IGNORE_PREFIX          =

#---------------------------------------------------------------------------
# Configuration options related to the HTML output
#---------------------------------------------------------------------------

# If the GENERATE_HTML tag is set to YES, doxygen will generate HTML output
# The default value is: YES.

GENERATE_HTML          = YES

# The HTML_OUTPUT tag is used to specify where the HTML docs will be put. If a
# relative path is entered the value of OUTPUT_DIRECTORY will be put in front of
# it.
# The default directory is: html.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_OUTPUT            = html

# The HTML_FILE_EXTENSION tag can be used to specify the file extension for each
# generated HTML page (for example: .htm, .php, .asp).
# The default value is: .html.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_FILE_EXTENSION    = .html

# The HTML_HEADER tag can be used to specify a user-defined HTML header file for
# each generated HTML page. If the tag is left blank doxygen will generate a
# standard header.
#
# To get valid HTML the header file that includes any scripts and style sheets
# that doxygen needs, which is dependent on the configuration options used (e.g.
# the setting GENERATE_TREEVIEW). It is highly recommended to start with a
# default header using
# doxygen -w html new_header.html new_footer.html new_stylesheet.css
# YourConfigFile
# and then modify the file new_header.html. See also section "Doxygen usage"
# for information on how to generate the default header that doxygen normally
# uses.
# Note: The header is subject to change so you typically have to regenerate the
# default header when upgrading to a newer version of doxygen. For a description
# of the possible markers and block names see the documentation.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_HEADER            =

# The HTML_FOOTER tag can be used to specify a user-defined HTML footer for each
# generated HTML page. If the tag is left blank doxygen will generate a standard
# footer. See HTML_HEADER for more information on how to generate a default
# footer and what special commands can be used inside the footer. See also
# section "Doxygen usage" for information on how to generate the default footer
# that doxygen normally uses.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_FOOTER            =

# The HTML_STYLESHEET tag can be used to specify a user-defined cascading style
# sheet that is used by each HTML page. It can be used to fine-tune the look of
# the HTML output. If left blank doxygen will generate a default style sheet.
# See also section "Doxygen usage" for information on how to generate the style
# sheet that doxygen normally uses.
# Note: It is recommended to use HTML_EXTRA_STYLESHEET instead of this tag, as
# it is more robust and this tag (HTML_STYLESHEET) will in the future become
# obsolete.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_STYLESHEET        =

# The HTML_EXTRA_STYLESHEET tag can be used to specify additional user-defined
# cascading style sheets that are included after the standard style sheets
# created by doxygen. Using this option one can overrule certain style aspects.
# This is preferred over using HTML_STYLESHEET since it does not replace the
# standard style sheet and is therefore more robust against future updates.
# Doxygen will copy the style sheet files to the output directory.
# Note: The order of the extra style sheet files is of importance (e.g. the last
# style sheet in the list overrules the setting of the previous ones in the
# list). For an example see the documentation.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_EXTRA_STYLESHEET  =

# The HTML_EXTRA_FILES tag can be used to specify one or more extra images or
# other source files which should be copied to the HTML output directory. Note
# that these files will be copied to the base HTML output directory. Use the
# $relpath^ marker in the HTML_HEADER and/or HTML_FOOTER files to load these
# files. In the HTML_STYLESHEET file, use the file name only. Also note that the
# files will be copied as-is; there are no commands or markers available.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_EXTRA_FILES       =

# The HTML_COLORSTYLE_HUE tag controls the color of the HTML output. Doxygen
# will adjust the colors in the style sheet and background images according to
# this color. Hue is specified as an angle on a colorwheel, see
# http://en.wikipedia.org/wiki/Hue for more information. For instance the value
# 0 represents red, 60 is yellow, 120 is green, 180 is cyan, 240 is blue, 300
# purple, and 360 is red again.
# Minimum value: 0, maximum value: 359, default value: 220.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_COLORSTYLE_HUE    = 120

# The HTML_COLORSTYLE_SAT tag controls the purity (or saturation) of the colors
# in the HTML output. For a value of 0 the output will use grayscales only. A
# value of 255 will produce the most vivid colors.
# Minimum value: 0, maximum value: 255, default value: 100.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_COLORSTYLE_SAT    = 100

# The HTML_COLORSTYLE_GAMMA tag controls the gamma correction applied to the
# luminance component of the colors in the HTML output. Values below 100
# gradually make the output lighter, whereas values above 100 make the output
# darker. The value divided by 100 is the actual gamma applied, so 80 represents
# a gamma of 0.8, The value 220 represents a gamma of 2.2, and 100 does not
# change the gamma.
# Minimum value: 40, maximum value: 240, default value: 80.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_COLORSTYLE_GAMMA  = 80

# If the HTML_TIMESTAMP tag is set to YES then the footer of each generated HTML
# page will contain the date and time when the page was generated. Setting this
# to YES can help to show when doxygen was last run and thus if the
# documentation is up to date.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_TIMESTAMP         = NO

# If the HTML_DYNAMIC_SECTIONS tag is set to YES then the generated HTML
# documentation will contain sections that can be hidden and shown after the
# page has loaded.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_DYNAMIC_SECTIONS  = NO

# With HTML_INDEX_NUM_ENTRIES one can control the preferred number of entries
# shown in the various tree structured indices initially; the user can expand
# and collapse entries dynamically later on. Doxygen will expand the tree to
# such a level that at most the specified number of entries are visible (unless
# a fully collapsed tree already exceeds this amount). So setting the number of
# entries 1 will produce a full collapsed tree by default. 0 is a special value
# representing an infinite number of entries and will result in a full expanded
# tree by default.
# Minimum value: 0, maximum value: 9999, default value: 100.
# This tag requires that the tag GENERATE_HTML is set to YES.

HTML_INDEX_NUM_ENTRIES = 100

# If the GENERATE_DOCSET tag is set to YES, additional index files will be
# generated that can be used as input for Apple's Xcode 3 integrated development
# environment (see: http://developer.apple.com/tools/xcode/), introduced with
# OSX 10.5 (Leopard). To create a documentation set, doxygen will generate a
# Makefile in the HTML output directory. Running make will produce the docset in
# that directory and running make install will install the docset in
# ~/Library/Developer/Shared/Documentation/DocSets so that Xcode will find it at
# startup. See http://developer.apple.com/tools/creatingdocsetswithdoxygen.html
# for more information.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

GENERATE_DOCSET        = NO

# This tag determines the name of the docset feed. A documentation feed provides
# an umbrella under which multiple documentation sets from a single provider
# (such as a company or product suite) can be grouped.
# The default value is: Doxygen generated docs.
# This tag requires that the tag GENERATE_DOCSET is set to YES.

DOCSET_FEEDNAME        = "Doxygen generated docs"

# This tag specifies a string that should uniquely identify the documentation
# set bundle. This should be a reverse domain-name style string, e.g.
# com.mycompany.MyDocSet. Doxygen will append .docset to the name.
# The default value is: org.doxygen.Project.
# This tag requires that the tag GENERATE_DOCSET is set to YES.

DOCSET_BUNDLE_ID       = org.doxygen.Project

# The DOCSET_PUBLISHER_ID tag specifies a string that should uniquely identify
# the documentation publisher. This should be a reverse domain-name style
# string, e.g. com.mycompany.MyDocSet.documentation.
# The default value is: org.doxygen.Publisher.
# This tag requires that the tag GENERATE_DOCSET is set to YES.

DOCSET_PUBLISHER_ID    = org.doxygen.Publisher

# The DOCSET_PUBLISHER_NAME tag identifies the documentation publisher.
# The default value is: Publisher.
# This tag requires that the tag GENERATE_DOCSET is set to YES.

DOCSET_PUBLISHER_NAME  = Publisher

# If the GENERATE_HTMLHELP tag is set to YES then doxygen generates three
# additional HTML index files: index.hhp, index.hhc, and index.hhk. The
# index.hhp is a project file that can be read by Microsoft's HTML Help Workshop
# (see: http://www.microsoft.com/en-us/download/details.aspx?id=21138) on
# Windows.
#
# The HTML Help Workshop contains a compiler that can convert all HTML output
# generated by doxygen into a single compiled HTML file (.chm). Compiled HTML
# files are now used as the Windows 98 help format, and will replace the old
# Windows help format (.hlp) on all Windows platforms in the future. Compressed
# HTML files also contain an index, a table of contents, and you can search for
# words in the documentation. The HTML workshop also contains a viewer for
# compressed HTML files.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

GENERATE_HTMLHELP      = NO

# The CHM_FILE tag can be used to specify the file name of the resulting .chm
# file. You can add a path in front of the file if the result should not be
# written to the html output directory.
# This tag requires that the tag GENERATE_HTMLHELP is set to YES.

CHM_FILE               =

# The HHC_LOCATION tag can be used to specify the location (absolute path
# including file name) of the HTML help compiler (hhc.exe). If non-empty,
# doxygen will try to run the HTML help compiler on the generated index.hhp.
# The file has to be specified with full path.
# This tag requires that the tag GENERATE_HTMLHELP is set to YES.

HHC_LOCATION           =

# The GENERATE_CHI flag controls if a separate .chi index file is generated
# (YES) or that it should be included in the master .chm file (NO).
# The default value is: NO.
# This tag requires that the tag GENERATE_HTMLHELP is set to YES.

GENERATE_CHI           = NO

# The CHM_INDEX_ENCODING is used to encode HtmlHelp index (hhk), content (hhc)
# and project file content.
# This tag requires that the tag GENERATE_HTMLHELP is set to YES.

CHM_INDEX_ENCODING     =

# The BINARY_TOC flag controls whether a binary table of contents is generated
# (YES) or a normal table of contents (NO) in the .chm file. Furthermore it
# enables the Previous and Next buttons.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTMLHELP is set to YES.

BINARY_TOC             = NO

# The TOC_EXPAND flag can be set to YES to add extra items for group members to
# the table of contents of the HTML help documentation and to the tree view.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTMLHELP is set to YES.

TOC_EXPAND             = NO

# If the GENERATE_QHP tag is set to YES and both QHP_NAMESPACE and
# QHP_VIRTUAL_FOLDER are set, an additional index file will be generated that
# can be used as input for Qt's qhelpgenerator to generate a Qt Compressed Help
# (.qch) of the generated HTML documentation.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

GENERATE_QHP           = NO

# If the QHG_LOCATION tag is specified, the QCH_FILE tag can be used to specify
# the file name of the resulting .qch file. The path specified is relative to
# the HTML output folder.
# This tag requires that the tag GENERATE_QHP is set to YES.

QCH_FILE               =

# The QHP_NAMESPACE tag specifies the namespace to use when generating Qt Help
# Project output. For more information please see Qt Help Project / Namespace
# (see: http://qt-project.org/doc/qt-4.8/qthelpproject.html#namespace).
# The default value is: org.doxygen.Project.
# This tag requires that the tag GENERATE_QHP is set to YES.

QHP_NAMESPACE          = org.doxygen.Project

# The QHP_VIRTUAL_FOLDER tag specifies the namespace to use when generating Qt
# Help Project output. For more information please see Qt Help Project / Virtual
# Folders (see: http://qt-project.org/doc/qt-4.8/qthelpproject.html#virtual-
# folders).
# The default value is: doc.
# This tag requires that the tag GENERATE_QHP is set to YES.

QHP_VIRTUAL_FOLDER     = doc

# If the QHP_CUST_FILTER_NAME tag is set, it specifies the name of a custom
# filter to add. For more information please see Qt Help Project / Custom
# Filters (see: http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-
# filters).
# This tag requires that the tag GENERATE_QHP is set to YES.

QHP_CUST_FILTER_NAME   =

# The QHP_CUST_FILTER_ATTRS tag specifies the list of the attributes of the
# custom filter to add. For more information please see Qt Help Project / Custom
# Filters (see: http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-
# filters).
# This tag requires that the tag GENERATE_QHP is set to YES.

QHP_CUST_FILTER_ATTRS  =

# The QHP_SECT_FILTER_ATTRS tag specifies the list of the attributes this
# project's filter section matches. Qt Help Project / Filter Attributes (see:
# http://qt-project.org/doc/qt-4.8/qthelpproject.html#filter-attributes).
# This tag requires that the tag GENERATE_QHP is set to YES.

QHP_SECT_FILTER_ATTRS  =

# The QHG_LOCATION tag can be used to specify the location of Qt's
# qhelpgenerator. If non-empty doxygen will try to run qhelpgenerator on the
# generated .qhp file.
# This tag requires that the tag GENERATE_QHP is set to YES.

QHG_LOCATION           =

# If the GENERATE_ECLIPSEHELP tag is set to YES, additional index files will be
# generated, together with the HTML files, they form an Eclipse help plugin. To
# install this plugin and make it available under the help contents menu in
# Eclipse, the contents of the directory containing the HTML and XML files needs
# to be copied into the plugins directory of eclipse. The name of the directory
# within the plugins directory should be the same as the ECLIPSE_DOC_ID value.
# After copying Eclipse needs to be restarted before the help appears.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

GENERATE_ECLIPSEHELP   = NO

# A unique identifier for the Eclipse help plugin. When installing the plugin
# the directory name containing the HTML and XML files should also have this
# name. Each documentation set should have its own identifier.
# The default value is: org.doxygen.Project.
# This tag requires that the tag GENERATE_ECLIPSEHELP is set to YES.

ECLIPSE_DOC_ID         = org.doxygen.Project

# If you want full control over the layout of the generated HTML pages it might
# be necessary to disable the index and replace it with your own. The
# DISABLE_INDEX tag can be used to turn on/off the condensed index (tabs) at top
# of each HTML page. A value of NO enables the index and the value YES disables
# it. Since the tabs in the index contain the same information as the navigation
# tree, you can set this option to YES if you also set GENERATE_TREEVIEW to YES.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

DISABLE_INDEX          = NO

# The GENERATE_TREEVIEW tag is used to specify whether a tree-like index
# structure should be generated to display hierarchical information. If the tag
# value is set to YES, a side panel will be generated containing a tree-like
# index structure (just like the one that is generated for HTML Help). For this
# to work a browser that supports JavaScript, DHTML, CSS and frames is required
# (i.e. any modern browser). Windows users are probably better off using the
# HTML help feature. Via custom style sheets (see HTML_EXTRA_STYLESHEET) one can
# further fine-tune the look of the index. As an example, the default style
# sheet generated by doxygen has an example that shows how to put an image at
# the root of the tree instead of the PROJECT_NAME. Since the tree basically has
# the same information as the tab index, you could consider setting
# DISABLE_INDEX to YES when enabling this option.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

GENERATE_TREEVIEW      = NO

# The ENUM_VALUES_PER_LINE tag can be used to set the number of enum values that
# doxygen will group on one line in the generated HTML documentation.
#
# Note that a value of 0 will completely suppress the enum values from appearing
# in the overview section.
# Minimum value: 0, maximum value: 20, default value: 4.
# This tag requires that the tag GENERATE_HTML is set to YES.

ENUM_VALUES_PER_LINE   = 4

# If the treeview is enabled (see GENERATE_TREEVIEW) then this tag can be used
# to set the initial width (in pixels) of the frame in which the tree is shown.
# Minimum value: 0, maximum value: 1500, default value: 250.
# This tag requires that the tag GENERATE_HTML is set to YES.

TREEVIEW_WIDTH         = 250

# If the EXT_LINKS_IN_WINDOW option is set to YES, doxygen will open links to
# external symbols imported via tag files in a separate window.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

EXT_LINKS_IN_WINDOW    = NO

# Use this tag to change the font size of LaTeX formulas included as images in
# the HTML documentation. When you change the font size after a successful
# doxygen run you need to manually remove any form_*.png images from the HTML
# output directory to force them to be regenerated.
# Minimum value: 8, maximum value: 50, default value: 10.
# This tag requires that the tag GENERATE_HTML is set to YES.

FORMULA_FONTSIZE       = 10

# Use the FORMULA_TRANPARENT tag to determine whether or not the images
# generated for formulas are transparent PNGs. Transparent PNGs are not
# supported properly for IE 6.0, but are supported on all modern browsers.
#
# Note that when changing this option you need to delete any form_*.png files in
# the HTML output directory before the changes have effect.
# The default value is: YES.
# This tag requires that the tag GENERATE_HTML is set to YES.

FORMULA_TRANSPARENT    = YES

# Enable the USE_MATHJAX option to render LaTeX formulas using MathJax (see
# http://www.mathjax.org) which uses client side Javascript for the rendering
# instead of using pre-rendered bitmaps. Use this if you do not have LaTeX
# installed or if you want to formulas look prettier in the HTML output. When
# enabled you may also need to install MathJax separately and configure the path
# to it using the MATHJAX_RELPATH option.
# The default value is: NO.
# This tag requires that the tag GENERATE_HTML is set to YES.

USE_MATHJAX            = NO

# When MathJax is enabled you can set the default output format to be used for
# the MathJax output. See the MathJax site (see:
# http://docs.mathjax.org/en/latest/output.html) for more details.
# Possible values are: HTML-CSS (which is slower, but has the best
# compatibility), NativeMML (i.e. MathML) and SVG.
# The default value is: HTML-CSS.
# This tag requires that the tag USE_MATHJAX is set to YES.

MATHJAX_FORMAT         = HTML-CSS

# When MathJax is enabled you need to specify the location relative to the HTML
# output directory using the MATHJAX_RELPATH option. The destination directory
# should contain the MathJax.js script. For instance, if the mathjax directory
# is located at the same level as the HTML output directory, then
# MATHJAX_RELPATH should be ../mathjax. The default value points to the MathJax
# Content Delivery Network so you can quickly see the result without installing
# MathJax. However, it is strongly recommended to install a local copy of
# MathJax from http://www.mathjax.org before deployment.
# The default value is: http://cdn.mathjax.org/mathjax/latest.
# This tag requires that the tag USE_MATHJAX is set to YES.

MATHJAX_RELPATH        = http://cdn.mathjax.org/mathjax/latest

# The MATHJAX_EXTENSIONS tag can be used to specify one or more MathJax
# extension names that should be enabled during MathJax rendering. For example
# MATHJAX_EXTENSIONS = TeX/AMSmath TeX/AMSsymbols
# This tag requires that the tag USE_MATHJAX is set to YES.

MATHJAX_EXTENSIONS     =

# The MATHJAX_CODEFILE tag can be used to specify a file with javascript pieces
# of code that will be used on startup of the MathJax code. See the MathJax site
# (see: http://docs.mathjax.org/en/latest/output.html) for more details. For an
# example see the documentation.
# This tag requires that the tag USE_MATHJAX is set to YES.

MATHJAX_CODEFILE       =

# When the SEARCHENGINE tag is enabled doxygen will generate a search box for
# the HTML output. The underlying search engine uses javascript and DHTML and
# should work on any modern browser. Note that when using HTML help
# (GENERATE_HTMLHELP), Qt help (GENERATE_QHP), or docsets (GENERATE_DOCSET)
# there is already a search function so this one should typically be disabled.
# For large projects the javascript based search engine can be slow, then
# enabling SERVER_BASED_SEARCH may provide a better solution. It is possible to
# search using the keyboard; to jump to the search box use <access key> + S
# (what the <access key> is depends on the OS and browser, but it is typically
# <CTRL>, <ALT>/<option>, or both). Inside the search box use the <cursor down
# key> to jump into the search results window, the results can be navigated
# using the <cursor keys>. Press <Enter> to select an item or <escape> to cancel
# the search. The filter options can be selected when the cursor is inside the
# search box by pressing <Shift>+<cursor down>. Also here use the <cursor keys>
# to select a filter and <Enter> or <escape> to activate or cancel the filter
# option.
# The default value is: YES.
# This tag requires that the tag GENERATE_HTML is set to YES.

SEARCHENGINE           = NO

# When the SERVER_BASED_SEARCH tag is enabled the search engine will be
# implemented using a web server instead of a web client using Javascript. There
# are two flavors of web server based searching depending on the EXTERNAL_SEARCH
# setting. When disabled, doxygen will generate a PHP script for searching and
# an index file used by the script. When EXTERNAL_SEARCH is enabled the indexing
# and searching needs to be provided by external tools. See the section
# "External Indexing and Searching" for details.
# The default value is: NO.
# This tag requires that the tag SEARCHENGINE is set to YES.

SERVER_BASED_SEARCH    = NO

# When EXTERNAL_SEARCH tag is enabled doxygen will no longer generate the PHP
# script for searching. Instead the search results are written to an XML file
# which needs to be processed by an external indexer. Doxygen will invoke an
# external search engine pointed to by the SEARCHENGINE_URL option to obtain the
# search results.
#
# Doxygen ships with an example indexer (doxyindexer) and search engine
# (doxysearch.cgi) which are based on the open source search engine library
# Xapian (see: http://xapian.org/).
#
# See the section "External Indexing and Searching" for details.
# The default value is: NO.
# This tag requires that the tag SEARCHENGINE is set to YES.

EXTERNAL_SEARCH        = NO

# The SEARCHENGINE_URL should point to a search engine hosted by a web server
# which will return the search results when EXTERNAL_SEARCH is enabled.
#
# Doxygen ships with an example indexer (doxyindexer) and search engine
# (doxysearch.cgi) which are based on the open source search engine library
# Xapian (see: http://xapian.org/). See the section "External Indexing and
# Searching" for details.
# This tag requires that the tag SEARCHENGINE is set to YES.

SEARCHENGINE_URL       =

# When SERVER_BASED_SEARCH and EXTERNAL_SEARCH are both enabled the unindexed
# search data is written to a file for indexing by an external tool. With the
# SEARCHDATA_FILE tag the name of this file can be specified.
# The default file is: searchdata.xml.
# This tag requires that the tag SEARCHENGINE is set to YES.

SEARCHDATA_FILE        = searchdata.xml

# When SERVER_BASED_SEARCH and EXTERNAL_SEARCH are both enabled the
# EXTERNAL_SEARCH_ID tag can be used as an identifier for the project. This is
# useful in combination with EXTRA_SEARCH_MAPPINGS to search through multiple
# projects and redirect the results back to the right project.
# This tag requires that the tag SEARCHENGINE is set to YES.

EXTERNAL_SEARCH_ID     =

# The EXTRA_SEARCH_MAPPINGS tag can be used to enable searching through doxygen
# projects other than the one defined by this configuration file, but that are
# all added to the same external search index. Each project needs to have a
# unique id set via EXTERNAL_SEARCH_ID. The search mapping then maps the id of
# to a relative location where the documentation can be found. The format is:
# EXTRA_SEARCH_MAPPINGS = tagname1=loc1 tagname2=loc2 ...
# This tag requires that the tag SEARCHENGINE is set to YES.

EXTRA_SEARCH_MAPPINGS  =

#---------------------------------------------------------------------------
# Configuration options related to the LaTeX output
#---------------------------------------------------------------------------

# If the GENERATE_LATEX tag is set to YES, doxygen will generate LaTeX output.
# The default value is: YES.

GENERATE_LATEX         = NO

# The LATEX_OUTPUT tag is used to specify where the LaTeX docs will be put. If a
# relative path is entered the value of OUTPUT_DIRECTORY will be put in front of
# it.
# The default directory is: latex.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_OUTPUT           = latex

# The LATEX_CMD_NAME tag can be used to specify the LaTeX command name to be
# invoked.
#
# Note that when enabling USE_PDFLATEX this option is only used for generating
# bitmaps for formulas in the HTML output, but not in the Makefile that is
# written to the output directory.
# The default file is: latex.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_CMD_NAME         = latex

# The MAKEINDEX_CMD_NAME tag can be used to specify the command name to generate
# index for LaTeX.
# The default file is: makeindex.
# This tag requires that the tag GENERATE_LATEX is set to YES.

MAKEINDEX_CMD_NAME     = makeindex

# If the COMPACT_LATEX tag is set to YES, doxygen generates more compact LaTeX
# documents. This may be useful for small projects and may help to save some
# trees in general.
# The default value is: NO.
# This tag requires that the tag GENERATE_LATEX is set to YES.

COMPACT_LATEX          = NO

# The PAPER_TYPE tag can be used to set the paper type that is used by the
# printer.
# Possible values are: a4 (210 x 297 mm), letter (8.5 x 11 inches), legal (8.5 x
# 14 inches) and executive (7.25 x 10.5 inches).
# The default value is: a4.
# This tag requires that the tag GENERATE_LATEX is set to YES.

PAPER_TYPE             = letter

# The EXTRA_PACKAGES tag can be used to specify one or more LaTeX package names
# that should be included in the LaTeX output. The package can be specified just
# by its name or with the correct syntax as to be used with the LaTeX
# \usepackage command. To get the times font for instance you can specify :
# EXTRA_PACKAGES=times or EXTRA_PACKAGES={times}
# To use the option intlimits with the amsmath package you can specify:
# EXTRA_PACKAGES=[intlimits]{amsmath}
# If left blank no extra packages will be included.
# This tag requires that the tag GENERATE_LATEX is set to YES.

EXTRA_PACKAGES         =

# The LATEX_HEADER tag can be used to specify a personal LaTeX header for the
# generated LaTeX document. The header should contain everything until the first
# chapter. If it is left blank doxygen will generate a standard header. See
# section "Doxygen usage" for information on how to let doxygen write the
# default header to a separate file.
#
# Note: Only use a user-defined header if you know what you are doing! The
# following commands have a special meaning inside the header: $title,
# $datetime, $date, $doxygenversion, $projectname, $projectnumber,
# $projectbrief, $projectlogo. Doxygen will replace $title with the empty
# string, for the replacement values of the other commands the user is referred
# to HTML_HEADER.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_HEADER           =

# The LATEX_FOOTER tag can be used to specify a personal LaTeX footer for the
# generated LaTeX document. The footer should contain everything after the last
# chapter. If it is left blank doxygen will generate a standard footer. See
# LATEX_HEADER for more information on how to generate a default footer and what
# special commands can be used inside the footer.
#
# Note: Only use a user-defined footer if you know what you are doing!
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_FOOTER           =

# The LATEX_EXTRA_STYLESHEET tag can be used to specify additional user-defined
# LaTeX style sheets that are included after the standard style sheets created
# by doxygen. Using this option one can overrule certain style aspects. Doxygen
# will copy the style sheet files to the output directory.
# Note: The order of the extra style sheet files is of importance (e.g. the last
# style sheet in the list overrules the setting of the previous ones in the
# list).
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_EXTRA_STYLESHEET =

# The LATEX_EXTRA_FILES tag can be used to specify one or more extra images or
# other source files which should be copied to the LATEX_OUTPUT output
# directory. Note that the files will be copied as-is; there are no commands or
# markers available.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_EXTRA_FILES      =

# If the PDF_HYPERLINKS tag is set to YES, the LaTeX that is generated is
# prepared for conversion to PDF (using ps2pdf or pdflatex). The PDF file will
# contain links (just like the HTML output) instead of page references. This
# makes the output suitable for online browsing using a PDF viewer.
# The default value is: YES.
# This tag requires that the tag GENERATE_LATEX is set to YES.

PDF_HYPERLINKS         = YES

# If the USE_PDFLATEX tag is set to YES, doxygen will use pdflatex to generate
# the PDF file directly from the LaTeX files. Set this option to YES, to get a
# higher quality PDF documentation.
# The default value is: YES.
# This tag requires that the tag GENERATE_LATEX is set to YES.

USE_PDFLATEX           = YES

# If the LATEX_BATCHMODE tag is set to YES, doxygen will add the \batchmode
# command to the generated LaTeX files. This will instruct LaTeX to keep running
# if errors occur, instead of asking the user for help. This option is also used
# when generating formulas in HTML.
# The default value is: NO.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_BATCHMODE        = NO

# If the LATEX_HIDE_INDICES tag is set to YES then doxygen will not include the
# index chapters (such as File Index, Compound Index, etc.) in the output.
# The default value is: NO.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_HIDE_INDICES     = YES

# If the LATEX_SOURCE_CODE tag is set to YES then doxygen will include source
# code with syntax highlighting in the LaTeX output.
#
# Note that which sources are shown also depends on other settings such as
# SOURCE_BROWSER.
# The default value is: NO.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_SOURCE_CODE      = NO

# The LATEX_BIB_STYLE tag can be used to specify the style to use for the
# bibliography, e.g. plainnat, or ieeetr. See
# http://en.wikipedia.org/wiki/BibTeX and \cite for more info.
# The default value is: plain.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_BIB_STYLE        = plain

# If the LATEX_TIMESTAMP tag is set to YES then the footer of each generated
# page will contain the date and time when the page was generated. Setting this
# to NO can help when comparing the output of multiple runs.
# The default value is: NO.
# This tag requires that the tag GENERATE_LATEX is set to YES.

LATEX_TIMESTAMP        = NO

#---------------------------------------------------------------------------
# Configuration options related to the RTF output
#---------------------------------------------------------------------------

# If the GENERATE_RTF tag is set to YES, doxygen will generate RTF output. The
# RTF output is optimized for Word 97 and may not look too pretty with other RTF
# readers/editors.
# The default value is: NO.

GENERATE_RTF           = NO

# The RTF_OUTPUT tag is used to specify where the RTF docs will be put. If a
# relative path is entered the value of OUTPUT_DIRECTORY will be put in front of
# it.
# The default directory is: rtf.
# This tag requires that the tag GENERATE_RTF is set to YES.

RTF_OUTPUT             = rtf

# If the COMPACT_RTF tag is set to YES, doxygen generates more compact RTF
# documents. This may be useful for small projects and may help to save some
# trees in general.
# The default value is: NO.
# This tag requires that the tag GENERATE_RTF is set to YES.

COMPACT_RTF            = NO

# If the RTF_HYPERLINKS tag is set to YES, the RTF that is generated will
# contain hyperlink fields. The RTF file will contain links (just like the HTML
# output) instead of page references. This makes the output suitable for online
# browsing using Word or some other Word compatible readers that support those
# fields.
#
# Note: WordPad (write) and others do not support links.
# The default value is: NO.
# This tag requires that the tag GENERATE_RTF is set to YES.

RTF_HYPERLINKS         = NO

# Load stylesheet definitions from file. Syntax is similar to doxygen's config
# file, i.e. a series of assignments. You only have to provide replacements,
# missing definitions are set to their default value.
#
# See also section "Doxygen usage" for information on how to generate the
# default style sheet that doxygen normally uses.
# This tag requires that the tag GENERATE_RTF is set to YES.

RTF_STYLESHEET_FILE    =

# Set optional variables used in the generation of an RTF document. Syntax is
# similar to doxygen's config file. A template extensions file can be generated
# using doxygen -e rtf extensionFile.
# This tag requires that the tag GENERATE_RTF is set to YES.

RTF_EXTENSIONS_FILE    =

# If the RTF_SOURCE_CODE tag is set to YES then doxygen will include source code
# with syntax highlighting in the RTF output.
#
# Note that which sources are shown also depends on other settings such as
# SOURCE_BROWSER.
# The default value is: NO.
# This tag requires that the tag GENERATE_RTF is set to YES.

RTF_SOURCE_CODE        = NO

#---------------------------------------------------------------------------
# Configuration options related to the man page output
#---------------------------------------------------------------------------

# If the GENERATE_MAN tag is set to YES, doxygen will generate man pages for
# classes and files.
# The default value is: NO.

GENERATE_MAN           = NO

# The MAN_OUTPUT tag is used to specify where the man pages will be put. If a
# relative path is entered the value of OUTPUT_DIRECTORY will be put in front of
# it. A directory man3 will be created inside the directory specified by
# MAN_OUTPUT.
# The default directory is: man.
# This tag requires that the tag GENERATE_MAN is set to YES.

MAN_OUTPUT             = man

# The MAN_EXTENSION tag determines the extension that is added to the generated
# man pages. In case the manual section does not start with a number, the number
# 3 is prepended. The dot (.) at the beginning of the MAN_EXTENSION tag is
# optional.
# The default value is: .3.
# This tag requires that the tag GENERATE_MAN is set to YES.

MAN_EXTENSION          = .3

# The MAN_SUBDIR tag determines the name of the directory created within
# MAN_OUTPUT in which the man pages are placed. If defaults to man followed by
# MAN_EXTENSION with the initial . removed.
# This tag requires that the tag GENERATE_MAN is set to YES.

MAN_SUBDIR             =

# If the MAN_LINKS tag is set to YES and doxygen generates man output, then it
# will generate one additional man file for each entity documented in the real
# man page(s). These additional files only source the real man page, but without
# them the man command would be unable to find the correct page.
# The default value is: NO.
# This tag requires that the tag GENERATE_MAN is set to YES.

MAN_LINKS              = YES

#---------------------------------------------------------------------------
# Configuration options related to the XML output
#---------------------------------------------------------------------------

# If the GENERATE_XML tag is set to YES, doxygen will generate an XML file that
# captures the structure of the code including all documentation.
# The default value is: NO.

GENERATE_XML           = NO

# The XML_OUTPUT tag is used to specify where the XML pages will be put. If a
# relative path is entered the value of OUTPUT_DIRECTORY will be put in front of
# it.
# The default directory is: xml.
# This tag requires that the tag GENERATE_XML is set to YES.

XML_OUTPUT             = xml

# If the XML_PROGRAMLISTING tag is set to YES, doxygen will dump the program
# listings (including syntax highlighting and cross-referencing information) to
# the XML output. Note that enabling this will significantly increase the size
# of the XML output.
# The default value is: YES.
# This tag requires that the tag GENERATE_XML is set to YES.

XML_PROGRAMLISTING     = YES

#---------------------------------------------------------------------------
# Configuration options related to the DOCBOOK output
#---------------------------------------------------------------------------

# If the GENERATE_DOCBOOK tag is set to YES, doxygen will generate Docbook files
# that can be used to generate PDF.
# The default value is: NO.

GENERATE_DOCBOOK       = NO

# The DOCBOOK_OUTPUT tag is used to specify where the Docbook pages will be put.
# If a relative path is entered the value of OUTPUT_DIRECTORY will be put in
# front of it.
# The default directory is: docbook.
# This tag requires that the tag GENERATE_DOCBOOK is set to YES.

DOCBOOK_OUTPUT         = docbook

# If the DOCBOOK_PROGRAMLISTING tag is set to YES, doxygen will include the
# program listings (including syntax highlighting and cross-referencing
# information) to the DOCBOOK output. Note that enabling this will significantly
# increase the size of the DOCBOOK output.
# The default value is: NO.
# This tag requires that the tag GENERATE_DOCBOOK is set to YES.

DOCBOOK_PROGRAMLISTING = NO

#---------------------------------------------------------------------------
# Configuration options for the AutoGen Definitions output
#---------------------------------------------------------------------------

# If the GENERATE_AUTOGEN_DEF tag is set to YES, doxygen will generate an
# AutoGen Definitions (see http://autogen.sf.net) file that captures the
# structure of the code including all documentation. Note that this feature is
# still experimental and incomplete at the moment.
# The default value is: NO.

GENERATE_AUTOGEN_DEF   = NO

#---------------------------------------------------------------------------
# Configuration options related to the Perl module output
#---------------------------------------------------------------------------

# If the GENERATE_PERLMOD tag is set to YES, doxygen will generate a Perl module
# file that captures the structure of the code including all documentation.
#
# Note that this feature is still experimental and incomplete at the moment.
# The default value is: NO.

GENERATE_PERLMOD       = NO

# If the PERLMOD_LATEX tag is set to YES, doxygen will generate the necessary
# Makefile rules, Perl scripts and LaTeX code to be able to generate PDF and DVI
# output from the Perl module output.
# The default value is: NO.
# This tag requires that the tag GENERATE_PERLMOD is set to YES.

PERLMOD_LATEX          = NO

# If the PERLMOD_PRETTY tag is set to YES, the Perl module output will be nicely
# formatted so it can be parsed by a human reader. This is useful if you want to
# understand what is going on. On the other hand, if this tag is set to NO, the
# size of the Perl module output will be much smaller and Perl will parse it
# just the same.
# The default value is: YES.
# This tag requires that the tag GENERATE_PERLMOD is set to YES.

PERLMOD_PRETTY         = YES

# The names of the make variables in the generated doxyrules.make file are
# prefixed with the string contained in PERLMOD_MAKEVAR_PREFIX. This is useful
# so different doxyrules.make files included by the same Makefile don't
# overwrite each other's variables.
# This tag requires that the tag GENERATE_PERLMOD is set to YES.

PERLMOD_MAKEVAR_PREFIX =

#---------------------------------------------------------------------------
# Configuration options related to the preprocessor
#---------------------------------------------------------------------------

# If the ENABLE_PREPROCESSING tag is set to YES, doxygen will evaluate all
# C-preprocessor directives found in the sources and include files.
# The default value is: YES.

ENABLE_PREPROCESSING   = YES

# If the MACRO_EXPANSION tag is set to YES, doxygen will expand all macro names
# in the source code. If set to NO, only conditional compilation will be
# performed. Macro expansion can be done in a controlled way by setting
# EXPAND_ONLY_PREDEF to YES.
# The default value is: NO.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

MACRO_EXPANSION        = NO

# If the EXPAND_ONLY_PREDEF and MACRO_EXPANSION tags are both set to YES then
# the macro expansion is limited to the macros specified with the PREDEFINED and
# EXPAND_AS_DEFINED tags.
# The default value is: NO.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

EXPAND_ONLY_PREDEF     = NO

# If the SEARCH_INCLUDES tag is set to YES, the include files in the
# INCLUDE_PATH will be searched if a #include is found.
# The default value is: YES.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

SEARCH_INCLUDES        = YES

# The INCLUDE_PATH tag can be used to specify one or more directories that
# contain include files that are not input files but should be processed by the
# preprocessor.
# This tag requires that the tag SEARCH_INCLUDES is set to YES.

INCLUDE_PATH           =

# You can use the INCLUDE_FILE_PATTERNS tag to specify one or more wildcard
# patterns (like *.h and *.hpp) to filter out the header-files in the
# directories. If left blank, the patterns specified with FILE_PATTERNS will be
# used.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

INCLUDE_FILE_PATTERNS  =

# The PREDEFINED tag can be used to specify one or more macro names that are
# defined before the preprocessor is started (similar to the -D option of e.g.
# gcc). The argument of the tag is a list of macros of the form: name or
# name=definition (no spaces). If the definition and the "=" are omitted, "=1"
# is assumed. To prevent a macro definition from being undefined via #undef or
# recursively expanded use the := operator instead of the = operator.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

PREDEFINED             =

# If the MACRO_EXPANSION and EXPAND_ONLY_PREDEF tags are set to YES then this
# tag can be used to specify a list of macro names that should be expanded. The
# macro definition that is found in the sources will be used. Use the PREDEFINED
# tag if you want to use a different macro definition that overrules the
# definition found in the source code.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

EXPAND_AS_DEFINED      =

# If the SKIP_FUNCTION_MACROS tag is set to YES then doxygen's preprocessor will
# remove all references to function-like macros that are alone on a line, have
# an all uppercase name, and do not end with a semicolon. Such function macros
# are typically used for boiler-plate code, and will confuse the parser if not
# removed.
# The default value is: YES.
# This tag requires that the tag ENABLE_PREPROCESSING is set to YES.

SKIP_FUNCTION_MACROS   = YES

#---------------------------------------------------------------------------
# Configuration options related to external references
#---------------------------------------------------------------------------

# The TAGFILES tag can be used to specify one or more tag files. For each tag
# file the location of the external documentation should be added. The format of
# a tag file without this location is as follows:
# TAGFILES = file1 file2 ...
# Adding location for the tag files is done as follows:
# TAGFILES = file1=loc1 "file2 = loc2" ...
# where loc1 and loc2 can be relative or absolute paths or URLs. See the
# section "Linking to external documentation" for more information about the use
# of tag files.
# Note: Each tag file must have a unique name (where the name does NOT include
# the path). If a tag file is not located in the directory in which doxygen is
# run, you must also specify the path to the tagfile here.

TAGFILES               =

# When a file name is specified after GENERATE_TAGFILE, doxygen will create a
# tag file that is based on the input files it reads. See section "Linking to
# external documentation" for more information about the usage of tag files.

GENERATE_TAGFILE       =

# If the ALLEXTERNALS tag is set to YES, all external class will be listed in
# the class index. If set to NO, only the inherited external classes will be
# listed.
# The default value is: NO.

ALLEXTERNALS           = NO

# If the EXTERNAL_GROUPS tag is set to YES, all external groups will be listed
# in the modules index. If set to NO, only the current project's groups will be
# listed.
# The default value is: YES.

EXTERNAL_GROUPS        = YES

# If the EXTERNAL_PAGES tag is set to YES, all external pages will be listed in
# the related pages index. If set to NO, only the current project's pages will
# be listed.
# The default value is: YES.

EXTERNAL_PAGES         = YES

# The PERL_PATH should be the absolute path and name of the perl script
# interpreter (i.e. the result of 'which perl').
# The default file (with absolute path) is: /usr/bin/perl.

PERL_PATH              = /usr/bin/perl

#---------------------------------------------------------------------------
# Configuration options related to the dot tool
#---------------------------------------------------------------------------

# If the CLASS_DIAGRAMS tag is set to YES, doxygen will generate a class diagram
# (in HTML and LaTeX) for classes with base or super classes. Setting the tag to
# NO turns the diagrams off. Note that this option also works with HAVE_DOT
# disabled, but it is recommended to install and use dot, since it yields more
# powerful graphs.
# The default value is: YES.

CLASS_DIAGRAMS         = YES

# You can define message sequence charts within doxygen comments using the \msc
# command. Doxygen will then run the mscgen tool (see:
# http://www.mcternan.me.uk/mscgen/)) to produce the chart and insert it in the
# documentation. The MSCGEN_PATH tag allows you to specify the directory where
# the mscgen tool resides. If left empty the tool is assumed to be found in the
# default search path.

MSCGEN_PATH            =

# You can include diagrams made with dia in doxygen documentation. Doxygen will
# then run dia to produce the diagram and insert it in the documentation. The
# DIA_PATH tag allows you to specify the directory where the dia binary resides.
# If left empty dia is assumed to be found in the default search path.

DIA_PATH               =

# If set to YES the inheritance and collaboration graphs will hide inheritance
# and usage relations if the target is undocumented or is not a class.
# The default value is: YES.

HIDE_UNDOC_RELATIONS   = YES

# If you set the HAVE_DOT tag to YES then doxygen will assume the dot tool is
# available from the path. This tool is part of Graphviz (see:
# http://www.graphviz.org/), a graph visualization toolkit from AT&T and Lucent
# Bell Labs. The other options in this section have no effect if this option is
# set to NO
# The default value is: NO.

HAVE_DOT               = NO

# The DOT_NUM_THREADS specifies the number of dot invocations doxygen is allowed
# to run in parallel. When set to 0 doxygen will base this on the number of
# processors available in the system. You can set it explicitly to a value
# larger than 0 to get control over the balance between CPU load and processing
# speed.
# Minimum value: 0, maximum value: 32, default value: 0.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_NUM_THREADS        = 0

# When you want a differently looking font in the dot files that doxygen
# generates you can specify the font name using DOT_FONTNAME. You need to make
# sure dot is able to find the font, which can be done by putting it in a
# standard location or by setting the DOTFONTPATH environment variable or by
# setting DOT_FONTPATH to the directory containing the font.
# The default value is: Helvetica.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_FONTNAME           =

# The DOT_FONTSIZE tag can be used to set the size (in points) of the font of
# dot graphs.
# Minimum value: 4, maximum value: 24, default value: 10.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_FONTSIZE           = 10

# By default doxygen will tell dot to use the default font as specified with
# DOT_FONTNAME. If you specify a different font using DOT_FONTNAME you can set
# the path where dot can find it using this tag.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_FONTPATH           =

# If the CLASS_GRAPH tag is set to YES then doxygen will generate a graph for
# each documented class showing the direct and indirect inheritance relations.
# Setting this tag to YES will force the CLASS_DIAGRAMS tag to NO.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

CLASS_GRAPH            = YES

# If the COLLABORATION_GRAPH tag is set to YES then doxygen will generate a
# graph for each documented class showing the direct and indirect implementation
# dependencies (inheritance, containment, and class references variables) of the
# class with other documented classes.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

COLLABORATION_GRAPH    = YES

# If the GROUP_GRAPHS tag is set to YES then doxygen will generate a graph for
# groups, showing the direct groups dependencies.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

GROUP_GRAPHS           = YES

# If the UML_LOOK tag is set to YES, doxygen will generate inheritance and
# collaboration diagrams in a style similar to the OMG's Unified Modeling
# Language.
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

UML_LOOK               = NO

# If the UML_LOOK tag is enabled, the fields and methods are shown inside the
# class node. If there are many fields or methods and many nodes the graph may
# become too big to be useful. The UML_LIMIT_NUM_FIELDS threshold limits the
# number of items for each type to make the size more manageable. Set this to 0
# for no limit. Note that the threshold may be exceeded by 50% before the limit
# is enforced. So when you set the threshold to 10, up to 15 fields may appear,
# but if the number exceeds 15, the total amount of fields shown is limited to
# 10.
# Minimum value: 0, maximum value: 100, default value: 10.
# This tag requires that the tag HAVE_DOT is set to YES.

UML_LIMIT_NUM_FIELDS   = 10

# If the TEMPLATE_RELATIONS tag is set to YES then the inheritance and
# collaboration graphs will show the relations between templates and their
# instances.
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

TEMPLATE_RELATIONS     = NO

# If the INCLUDE_GRAPH, ENABLE_PREPROCESSING and SEARCH_INCLUDES tags are set to
# YES then doxygen will generate a graph for each documented file showing the
# direct and indirect include dependencies of the file with other documented
# files.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

INCLUDE_GRAPH          = YES

# If the INCLUDED_BY_GRAPH, ENABLE_PREPROCESSING and SEARCH_INCLUDES tags are
# set to YES then doxygen will generate a graph for each documented file showing
# the direct and indirect include dependencies of the file with other documented
# files.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

INCLUDED_BY_GRAPH      = YES

# If the CALL_GRAPH tag is set to YES then doxygen will generate a call
# dependency graph for every global function or class method.
#
# Note that enabling this option will significantly increase the time of a run.
# So in most cases it will be better to enable call graphs for selected
# functions only using the \callgraph command. Disabling a call graph can be
# accomplished by means of the command \hidecallgraph.
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

CALL_GRAPH             = NO

# If the CALLER_GRAPH tag is set to YES then doxygen will generate a caller
# dependency graph for every global function or class method.
#
# Note that enabling this option will significantly increase the time of a run.
# So in most cases it will be better to enable caller graphs for selected
# functions only using the \callergraph command. Disabling a caller graph can be
# accomplished by means of the command \hidecallergraph.
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

CALLER_GRAPH           = NO

# If the GRAPHICAL_HIERARCHY tag is set to YES then doxygen will graphical
# hierarchy of all classes instead of a textual one.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

GRAPHICAL_HIERARCHY    = YES

# If the DIRECTORY_GRAPH tag is set to YES then doxygen will show the
# dependencies a directory has on other directories in a graphical way. The
# dependency relations are determined by the #include relations between the
# files in the directories.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

DIRECTORY_GRAPH        = YES

# The DOT_IMAGE_FORMAT tag can be used to set the image format of the images
# generated by dot. For an explanation of the image formats see the section
# output formats in the documentation of the dot tool (Graphviz (see:
# http://www.graphviz.org/)).
# Note: If you choose svg you need to set HTML_FILE_EXTENSION to xhtml in order
# to make the SVG files visible in IE 9+ (other browsers do not have this
# requirement).
# Possible values are: png, jpg, gif, svg, png:gd, png:gd:gd, png:cairo,
# png:cairo:gd, png:cairo:cairo, png:cairo:gdiplus, png:gdiplus and
# png:gdiplus:gdiplus.
# The default value is: png.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_IMAGE_FORMAT       = png

# If DOT_IMAGE_FORMAT is set to svg, then this option can be set to YES to
# enable generation of interactive SVG images that allow zooming and panning.
#
# Note that this requires a modern browser other than Internet Explorer. Tested
# and working are Firefox, Chrome, Safari, and Opera.
# Note: For IE 9+ you need to set HTML_FILE_EXTENSION to xhtml in order to make
# the SVG files visible. Older versions of IE do not have SVG support.
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

INTERACTIVE_SVG        = NO

# The DOT_PATH tag can be used to specify the path where the dot tool can be
# found. If left blank, it is assumed the dot tool can be found in the path.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_PATH               =

# The DOTFILE_DIRS tag can be used to specify one or more directories that
# contain dot files that are included in the documentation (see the \dotfile
# command).
# This tag requires that the tag HAVE_DOT is set to YES.

DOTFILE_DIRS           =

# The MSCFILE_DIRS tag can be used to specify one or more directories that
# contain msc files that are included in the documentation (see the \mscfile
# command).

MSCFILE_DIRS           =

# The DIAFILE_DIRS tag can be used to specify one or more directories that
# contain dia files that are included in the documentation (see the \diafile
# command).

DIAFILE_DIRS           =

# When using plantuml, the PLANTUML_JAR_PATH tag should be used to specify the
# path where java can find the plantuml.jar file. If left blank, it is assumed
# PlantUML is not used or called during a preprocessing step. Doxygen will
# generate a warning when it encounters a \startuml command in this case and
# will not generate output for the diagram.

PLANTUML_JAR_PATH      =

# When using plantuml, the specified paths are searched for files specified by
# the !include statement in a plantuml block.

PLANTUML_INCLUDE_PATH  =

# The DOT_GRAPH_MAX_NODES tag can be used to set the maximum number of nodes
# that will be shown in the graph. If the number of nodes in a graph becomes
# larger than this value, doxygen will truncate the graph, which is visualized
# by representing a node as a red box. Note that doxygen if the number of direct
# children of the root node in a graph is already larger than
# DOT_GRAPH_MAX_NODES then the graph will not be shown at all. Also note that
# the size of a graph can be further restricted by MAX_DOT_GRAPH_DEPTH.
# Minimum value: 0, maximum value: 10000, default value: 50.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_GRAPH_MAX_NODES    = 50

# The MAX_DOT_GRAPH_DEPTH tag can be used to set the maximum depth of the graphs
# generated by dot. A depth value of 3 means that only nodes reachable from the
# root by following a path via at most 3 edges will be shown. Nodes that lay
# further from the root node will be omitted. Note that setting this option to 1
# or 2 may greatly reduce the computation time needed for large code bases. Also
# note that the size of a graph can be further restricted by
# DOT_GRAPH_MAX_NODES. Using a depth of 0 means no depth restriction.
# Minimum value: 0, maximum value: 1000, default value: 0.
# This tag requires that the tag HAVE_DOT is set to YES.

MAX_DOT_GRAPH_DEPTH    = 0

# Set the DOT_TRANSPARENT tag to YES to generate images with a transparent
# background. This is disabled by default, because dot on Windows does not seem
# to support this out of the box.
#
# Warning: Depending on the platform used, enabling this option may lead to
# badly anti-aliased labels on the edges of a graph (i.e. they become hard to
# read).
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_TRANSPARENT        = NO

# Set the DOT_MULTI_TARGETS tag to YES to allow dot to generate multiple output
# files in one run (i.e. multiple -o and -T options on the command line). This
# makes dot run faster, but since only newer versions of dot (>1.8.10) support
# this, this feature is disabled by default.
# The default value is: NO.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_MULTI_TARGETS      = NO

# If the GENERATE_LEGEND tag is set to YES doxygen will generate a legend page
# explaining the meaning of the various boxes and arrows in the dot generated
# graphs.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

GENERATE_LEGEND        = YES

# If the DOT_CLEANUP tag is set to YES, doxygen will remove the intermediate dot
# files that are used to generate the various graphs.
# The default value is: YES.
# This tag requires that the tag HAVE_DOT is set to YES.

DOT_CLEANUP            = YES


# In[ ]:


GNU LESSER GENERAL PUBLIC LICENSE
     Version 2.1, February 1999

 Copyright (C) 1991, 1999 Free Software Foundation, Inc.
 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

[This is the first released version of the Lesser GPL.  It also counts
 as the successor of the GNU Library Public License, version 2, hence
 the version number 2.1.]

			    Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
Licenses are intended to guarantee your freedom to share and change
free software--to make sure the software is free for all its users.

  This license, the Lesser General Public License, applies to some
specially designated software packages--typically libraries--of the
Free Software Foundation and other authors who decide to use it.  You
can use it too, but we suggest you first think carefully about whether
this license or the ordinary General Public License is the better
strategy to use in any particular case, based on the explanations below.

  When we speak of free software, we are referring to freedom of use,
not price.  Our General Public Licenses are designed to make sure that
you have the freedom to distribute copies of free software (and charge
for this service if you wish); that you receive source code or can get
it if you want it; that you can change the software and use pieces of
it in new free programs; and that you are informed that you can do
these things.

  To protect your rights, we need to make restrictions that forbid
distributors to deny you these rights or to ask you to surrender these
rights.  These restrictions translate to certain responsibilities for
you if you distribute copies of the library or if you modify it.

  For example, if you distribute copies of the library, whether gratis
or for a fee, you must give the recipients all the rights that we gave
you.  You must make sure that they, too, receive or can get the source
code.  If you link other code with the library, you must provide
complete object files to the recipients, so that they can relink them
with the library after making changes to the library and recompiling
it.  And you must show them these terms so they know their rights.

  We protect your rights with a two-step method: (1) we copyright the
library, and (2) we offer you this license, which gives you legal
permission to copy, distribute and/or modify the library.

  To protect each distributor, we want to make it very clear that
there is no warranty for the free library.  Also, if the library is
modified by someone else and passed on, the recipients should know
that what they have is not the original version, so that the original
author's reputation will not be affected by problems that might be
introduced by others.


  Finally, software patents pose a constant threat to the existence of
any free program.  We wish to make sure that a company cannot
effectively restrict the users of a free program by obtaining a
restrictive license from a patent holder.  Therefore, we insist that
any patent license obtained for a version of the library must be
consistent with the full freedom of use specified in this license.

  Most GNU software, including some libraries, is covered by the
ordinary GNU General Public License.  This license, the GNU Lesser
General Public License, applies to certain designated libraries, and
is quite different from the ordinary General Public License.  We use
this license for certain libraries in order to permit linking those
libraries into non-free programs.

  When a program is linked with a library, whether statically or using
a shared library, the combination of the two is legally speaking a
combined work, a derivative of the original library.  The ordinary
General Public License therefore permits such linking only if the
entire combination fits its criteria of freedom.  The Lesser General
Public License permits more lax criteria for linking other code with
the library.

  We call this license the "Lesser" General Public License because it
does Less to protect the user's freedom than the ordinary General
Public License.  It also provides other free software developers Less
of an advantage over competing non-free programs.  These disadvantages
are the reason we use the ordinary General Public License for many
libraries.  However, the Lesser license provides advantages in certain
special circumstances.

  For example, on rare occasions, there may be a special need to
encourage the widest possible use of a certain library, so that it becomes
a de-facto standard.  To achieve this, non-free programs must be
allowed to use the library.  A more frequent case is that a free
library does the same job as widely used non-free libraries.  In this
case, there is little to gain by limiting the free library to free
software only, so we use the Lesser General Public License.

  In other cases, permission to use a particular library in non-free
programs enables a greater number of people to use a large body of
free software.  For example, permission to use the GNU C Library in
non-free programs enables many more people to use the whole GNU
operating system, as well as its variant, the GNU/Linux operating
system.

  Although the Lesser General Public License is Less protective of the
users' freedom, it does ensure that the user of a program that is
linked with the Library has the freedom and the wherewithal to run
that program using a modified version of the Library.

  The precise terms and conditions for copying, distribution and
modification follow.  Pay close attention to the difference between a
"work based on the library" and a "work that uses the library".  The
former contains code derived from the library, whereas the latter must
be combined with the library in order to run.


GNU LESSER GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License Agreement applies to any software library or other
program which contains a notice placed by the copyright holder or
other authorized party saying it may be distributed under the terms of
this Lesser General Public License (also called "this License").
Each licensee is addressed as "you".

  A "library" means a collection of software functions and/or data
prepared so as to be conveniently linked with application programs
(which use some of those functions and data) to form executables.

  The "Library", below, refers to any such software library or work
which has been distributed under these terms.  A "work based on the
Library" means either the Library or any derivative work under
copyright law: that is to say, a work containing the Library or a
portion of it, either verbatim or with modifications and/or translated
straightforwardly into another language.  (Hereinafter, translation is
included without limitation in the term "modification".)

  "Source code" for a work means the preferred form of the work for
making modifications to it.  For a library, complete source code means
all the source code for all modules it contains, plus any associated
interface definition files, plus the scripts used to control compilation
and installation of the library.

  Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running a program using the Library is not restricted, and output from
such a program is covered only if its contents constitute a work based
on the Library (independent of the use of the Library in a tool for
writing it).  Whether that is true depends on what the Library does
and what the program that uses the Library does.

  1. You may copy and distribute verbatim copies of the Library's
complete source code as you receive it, in any medium, provided that
you conspicuously and appropriately publish on each copy an
appropriate copyright notice and disclaimer of warranty; keep intact
all the notices that refer to this License and to the absence of any
warranty; and distribute a copy of this License along with the
Library.

  You may charge a fee for the physical act of transferring a copy,
and you may at your option offer warranty protection in exchange for a
fee.


  2. You may modify your copy or copies of the Library or any portion
of it, thus forming a work based on the Library, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) The modified work must itself be a software library.

    b) You must cause the files modified to carry prominent notices
    stating that you changed the files and the date of any change.

    c) You must cause the whole of the work to be licensed at no
    charge to all third parties under the terms of this License.

    d) If a facility in the modified Library refers to a function or a
    table of data to be supplied by an application program that uses
    the facility, other than as an argument passed when the facility
    is invoked, then you must make a good faith effort to ensure that,
    in the event an application does not supply such function or
    table, the facility still operates, and performs whatever part of
    its purpose remains meaningful.

    (For example, a function in a library to compute square roots has
    a purpose that is entirely well-defined independent of the
    application.  Therefore, Subsection 2d requires that any
    application-supplied function or table used by this function must
    be optional: if the application does not supply it, the square
    root function must still compute square roots.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Library,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Library, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote
it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Library.

In addition, mere aggregation of another work not based on the Library
with the Library (or with a work based on the Library) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may opt to apply the terms of the ordinary GNU General Public
License instead of this License to a given copy of the Library.  To do
this, you must alter all the notices that refer to this License, so
that they refer to the ordinary GNU General Public License, version 2,
instead of to this License.  (If a newer version than version 2 of the
ordinary GNU General Public License has appeared, then you can specify
that version instead if you wish.)  Do not make any other change in
these notices.


  Once this change is made in a given copy, it is irreversible for
that copy, so the ordinary GNU General Public License applies to all
subsequent copies and derivative works made from that copy.

  This option is useful when you wish to copy part of the code of
the Library into a program that is not a library.

  4. You may copy and distribute the Library (or a portion or
derivative of it, under Section 2) in object code or executable form
under the terms of Sections 1 and 2 above provided that you accompany
it with the complete corresponding machine-readable source code, which
must be distributed under the terms of Sections 1 and 2 above on a
medium customarily used for software interchange.

  If distribution of object code is made by offering access to copy
from a designated place, then offering equivalent access to copy the
source code from the same place satisfies the requirement to
distribute the source code, even though third parties are not
compelled to copy the source along with the object code.

  5. A program that contains no derivative of any portion of the
Library, but is designed to work with the Library by being compiled or
linked with it, is called a "work that uses the Library".  Such a
work, in isolation, is not a derivative work of the Library, and
therefore falls outside the scope of this License.

  However, linking a "work that uses the Library" with the Library
creates an executable that is a derivative of the Library (because it
contains portions of the Library), rather than a "work that uses the
library".  The executable is therefore covered by this License.
Section 6 states terms for distribution of such executables.

  When a "work that uses the Library" uses material from a header file
that is part of the Library, the object code for the work may be a
derivative work of the Library even though the source code is not.
Whether this is true is especially significant if the work can be
linked without the Library, or if the work is itself a library.  The
threshold for this to be true is not precisely defined by law.

  If such an object file uses only numerical parameters, data
structure layouts and accessors, and small macros and small inline
functions (ten lines or less in length), then the use of the object
file is unrestricted, regardless of whether it is legally a derivative
work.  (Executables containing this object code plus portions of the
Library will still fall under Section 6.)

  Otherwise, if the work is a derivative of the Library, you may
distribute the object code for the work under the terms of Section 6.
Any executables containing that work also fall under Section 6,
whether or not they are linked directly with the Library itself.


  6. As an exception to the Sections above, you may also combine or
link a "work that uses the Library" with the Library to produce a
work containing portions of the Library, and distribute that work
under terms of your choice, provided that the terms permit
modification of the work for the customer's own use and reverse
engineering for debugging such modifications.

  You must give prominent notice with each copy of the work that the
Library is used in it and that the Library and its use are covered by
this License.  You must supply a copy of this License.  If the work
during execution displays copyright notices, you must include the
copyright notice for the Library among them, as well as a reference
directing the user to the copy of this License.  Also, you must do one
of these things:

    a) Accompany the work with the complete corresponding
    machine-readable source code for the Library including whatever
    changes were used in the work (which must be distributed under
    Sections 1 and 2 above); and, if the work is an executable linked
    with the Library, with the complete machine-readable "work that
    uses the Library", as object code and/or source code, so that the
    user can modify the Library and then relink to produce a modified
    executable containing the modified Library.  (It is understood
    that the user who changes the contents of definitions files in the
    Library will not necessarily be able to recompile the application
    to use the modified definitions.)

    b) Use a suitable shared library mechanism for linking with the
    Library.  A suitable mechanism is one that (1) uses at run time a
    copy of the library already present on the user's computer system,
    rather than copying library functions into the executable, and (2)
    will operate properly with a modified version of the library, if
    the user installs one, as long as the modified version is
    interface-compatible with the version that the work was made with.

    c) Accompany the work with a written offer, valid for at
    least three years, to give the same user the materials
    specified in Subsection 6a, above, for a charge no more
    than the cost of performing this distribution.

    d) If distribution of the work is made by offering access to copy
    from a designated place, offer equivalent access to copy the above
    specified materials from the same place.

    e) Verify that the user has already received a copy of these
    materials or that you have already sent this user a copy.

  For an executable, the required form of the "work that uses the
Library" must include any data and utility programs needed for
reproducing the executable from it.  However, as a special exception,
the materials to be distributed need not include anything that is
normally distributed (in either source or binary form) with the major
components (compiler, kernel, and so on) of the operating system on
which the executable runs, unless that component itself accompanies
the executable.

  It may happen that this requirement contradicts the license
restrictions of other proprietary libraries that do not normally
accompany the operating system.  Such a contradiction means you cannot
use both them and the Library together in an executable that you
distribute.


  7. You may place library facilities that are a work based on the
Library side-by-side in a single library together with other library
facilities not covered by this License, and distribute such a combined
library, provided that the separate distribution of the work based on
the Library and of the other library facilities is otherwise
permitted, and provided that you do these two things:

    a) Accompany the combined library with a copy of the same work
    based on the Library, uncombined with any other library
    facilities.  This must be distributed under the terms of the
    Sections above.

    b) Give prominent notice with the combined library of the fact
    that part of it is a work based on the Library, and explaining
    where to find the accompanying uncombined form of the same work.

  8. You may not copy, modify, sublicense, link with, or distribute
the Library except as expressly provided under this License.  Any
attempt otherwise to copy, modify, sublicense, link with, or
distribute the Library is void, and will automatically terminate your
rights under this License.  However, parties who have received copies,
or rights, from you under this License will not have their licenses
terminated so long as such parties remain in full compliance.

  9. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Library or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Library (or any work based on the
Library), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Library or works based on it.

  10. Each time you redistribute the Library (or any work based on the
Library), the recipient automatically receives a license from the
original licensor to copy, distribute, link with or modify the Library
subject to these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties with
this License.


  11. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Library at all.  For example, if a patent
license would not permit royalty-free redistribution of the Library by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Library.

If any portion of this section is held invalid or unenforceable under any
particular circumstance, the balance of the section is intended to apply,
and the section as a whole is intended to apply in other circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  12. If the distribution and/or use of the Library is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Library under this License may add
an explicit geographical distribution limitation excluding those countries,
so that distribution is permitted only in or among countries not thus
excluded.  In such case, this License incorporates the limitation as if
written in the body of this License.

  13. The Free Software Foundation may publish revised and/or new
versions of the Lesser General Public License from time to time.
Such new versions will be similar in spirit to the present version,
but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number.  If the Library
specifies a version number of this License which applies to it and
"any later version", you have the option of following the terms and
conditions either of that version or of any later version published by
the Free Software Foundation.  If the Library does not specify a
license version number, you may choose any version ever published by
the Free Software Foundation.


  14. If you wish to incorporate parts of the Library into other free
programs whose distribution conditions are incompatible with these,
write to the author to ask for permission.  For software which is
copyrighted by the Free Software Foundation, write to the Free
Software Foundation; we sometimes make exceptions for this.  Our
decision will be guided by the two goals of preserving the free status
of all derivatives of our free software and of promoting the sharing
and reuse of software generally.

			    NO WARRANTY

  15. BECAUSE THE LIBRARY IS LICENSED FREE OF CHARGE, THERE IS NO
WARRANTY FOR THE LIBRARY, TO THE EXTENT PERMITTED BY APPLICABLE LAW.
EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR
OTHER PARTIES PROVIDE THE LIBRARY "AS IS" WITHOUT WARRANTY OF ANY
KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
LIBRARY IS WITH YOU.  SHOULD THE LIBRARY PROVE DEFECTIVE, YOU ASSUME
THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN
WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY
AND/OR REDISTRIBUTE THE LIBRARY AS PERMITTED ABOVE, BE LIABLE TO YOU
FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR
CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
LIBRARY (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING
RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A
FAILURE OF THE LIBRARY TO OPERATE WITH ANY OTHER SOFTWARE), EVEN IF
SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGES.

   END OF TERMS AND CONDITIONS


           How to Apply These Terms to Your New Libraries

  If you develop a new library, and you want it to be of the greatest
possible use to the public, we recommend making it free software that
everyone can redistribute and change.  You can do so by permitting
redistribution under these terms (or, alternatively, under the terms of the
ordinary General Public License).

  To apply these terms, attach the following notices to the library.  It is
safest to attach them to the start of each source file to most effectively
convey the exclusion of warranty; and each file should have at least the
"copyright" line and a pointer to where the full notice is found.

    <one line to give the library's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Also add information on how to contact you by electronic and paper mail.

You should also get your employer (if you work as a programmer) or your
school, if any, to sign a "copyright disclaimer" for the library, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the
  library `Frob' (a library for tweaking knobs) written by James Random Hacker.

  <signature of Ty Coon>, 1 April 1990
  Ty Coon, President of Vice

That's all there is to it!


# In[ ]:


libtool*.m4()
lt*.m4()


# In[ ]:


*(gcc, -o, aperio-bad-jp2k-tiles, aperio-bad-jp2k-tiles.c, -lopenjpeg, -ltiff, */)

*(Check, for, unreadable, JP2K, tiles, in, an, Aperio, slide., */)

*()
 *  OpenSlide, a library for reading whole slide image files
 *
 *  Copyright (c) 2007-2013 Carnegie Mellon University
 *  Copyright (c) 2011 Google, Inc.
 *  All rights reserved.
 *
 *  OpenSlide is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, version 2.1.
 *
 *  OpenSlide is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with OpenSlide. If not, see
 *  <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <tiffio.h>
#include <openjpeg.h>

struct decode_state {
  tdir_t dir;
  ttile_t tile;
  void *buf;
  toff_t size;
};

static void error_callback(const char *msg, void *data) {
  struct decode_state *state = data;
  char filename[80];
  FILE *fp;

  printf("Tile %d error: %s", state->tile, msg);
  snprintf(filename, sizeof(filename), "failed-%d-%d", state->dir,
           state->tile);
  fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Couldn't open file %s\n", filename);
    return;
  }
  if (fwrite(state->buf, state->size, 1, fp) != 1) {
    printf("Couldn't write file %s\n", filename);
  }
  fclose(fp);
}

static void decode_tile(struct decode_state *state)
{
  opj_cio_t *stream = NULL;
  opj_dinfo_t *dinfo = NULL;
  opj_image_t *image = NULL;

  opj_event_mgr_t event_callbacks = {
    .error_handler = error_callback,
  };

  // init decompressor
  opj_dparameters_t parameters;
  dinfo = opj_create_decompress(CODEC_J2K);
  opj_set_default_decoder_parameters(&parameters);
  opj_setup_decoder(dinfo, &parameters);
  stream = opj_cio_open((opj_common_ptr) dinfo, state->buf, state->size);
  opj_set_event_mgr((opj_common_ptr) dinfo, &event_callbacks, state);

  // decode
  image = opj_decode(dinfo, stream);

  // sanity check
  if (image && image->numcomps != 3) {
    error_callback("numcomps != 3", state);
  }

  // clean up
  if (image) opj_image_destroy(image);
  if (stream) opj_cio_close(stream);
  if (dinfo) opj_destroy_decompress(dinfo);
}

int main(int argc, char **argv)
{
  TIFF *tiff;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <tiff>\n", argv[0]);
    return 1;
  }

  tiff = TIFFOpen(argv[1], "r");
  if (!tiff) {
    fprintf(stderr, "Couldn't read TIFF\n");
    return 1;
  }

  do {
    uint16_t compression;
    uint32_t depth;
    ttile_t tiles;
    toff_t *sizes;
    struct decode_state state = {0};

    // ensure tiled and JP2K compression
    if (!TIFFGetField(tiff, TIFFTAG_COMPRESSION, &compression)) {
      fprintf(stderr, "Can't read compression scheme\n");
      TIFFClose(tiff);
      return 1;
    }
    if (!TIFFIsTiled(tiff) || (compression != 33003 && compression != 33005)) {
      continue;
    }

    state.dir = TIFFCurrentDirectory(tiff);
    printf("Directory: %d\n", state.dir);

    (/, check, depth)
    if (TIFFGetField(tiff, TIFFTAG_IMAGEDEPTH, &depth) && depth != 1) {
      printf("Depth != 1: %d\n", depth);
      continue;
    }

    (/, read, tiles)
    tiles = TIFFNumberOfTiles(tiff);
    if (!TIFFGetField(tiff, TIFFTAG_TILEBYTECOUNTS, &sizes)) {
      printf("No tile byte counts\n");
      continue;
    }
    for (state.tile = 0; state.tile < tiles; state.tile++) {
      if (!(state.tile % 50) || state.tile == tiles - 1) {
        fprintf(stderr, "  Reading: %d/%d\r", state.tile, tiles);
      }
      state.size = sizes[state.tile];
      state.buf = malloc(state.size);
      if (TIFFReadRawTile(tiff, state.tile, state.buf, state.size) ==
          state.size) {
        decode_tile(&state);
      } else {
        printf("Tile %d: couldn't read\n", state.tile);
      }
      free(state.buf);
    }
    fprintf(stderr, "\n");
  } while (TIFFReadDirectory(tiff));

  (/, close)
  TIFFClose(tiff);
  return 0;


# In[ ]:


import cairo
import math

def fill_background(cr):
    cr.set_source_rgb(0, 0, 1)
    cr.paint()

def fill_rectangles(cr):
    cr.set_operator(cairo.OPERATOR_SATURATE)
    cr.set_source_rgb(1, 1, 1)

    cr.rectangle(0, 0, 100, 100)
    cr.fill()

    cr.rectangle(100, 0, 100, 100)
    cr.fill()

    cr.rectangle(200, 0, 100, 100)
    cr.fill()

    cr.rectangle(300, 0, 100, 100)
    cr.fill()

def clip(cr):
    cr.rectangle(0, 0, 400, 400)
    cr.clip()

def clear(cr):
    cr.set_operator(cairo.OPERATOR_CLEAR)
    cr.paint()


# init pdf
pdf = cairo.PDFSurface("out.pdf", 400, 400)
cr = cairo.Context(pdf)
cr.rotate(math.pi / 4)

# page 1, push paint pop paint
fill_background(cr)

cr.push_group()
fill_rectangles(cr)
cr.pop_group_to_source()
cr.paint()

cr.show_page()

# page 2, push clear paint pop paint
fill_background(cr)

cr.push_group()
clear(cr)
fill_rectangles(cr)
cr.pop_group_to_source()
cr.paint()

cr.show_page()

# page 3, push clip paint pop paint
fill_background(cr)

cr.push_group()
clip(cr)
fill_rectangles(cr)
cr.pop_group_to_source()
cr.paint()

cr.show_page()

# page 4, push clip clear paint pop paint
fill_background(cr)

cr.push_group()
clip(cr)
clear(cr)
fill_rectangles(cr)
cr.pop_group_to_source()
cr.paint()

cr.show_page()


# done
pdf.finish()


# In[ ]:


#!/usr/bin/python

import struct, sys, os

HEADER_SIZE = 296
STRUCT_SIZE = 9

def is_correct_size(filename):
    filesize = os.path.getsize(filename)
    return (filesize - HEADER_SIZE) % STRUCT_SIZE == 0


def try_reading(f):
    f.seek(HEADER_SIZE)

    while True:
        data = f.read(STRUCT_SIZE)
        if len(data) == 0:
            break

        unpacked = struct.unpack("<iiB", data)

        x = int(unpacked[0]) / 256.0
        y = int(unpacked[1]) / 256.0
        zz = int(unpacked[2])

        print '%25.8f %25.8f %10d' % (x, y, zz)




for filename in sys.argv[1:]:
    if not is_correct_size(filename):
        print "%s: file not of expected size" % (filename)
    else:
        print filename
        f = open(filename, "rb")
        try:
            try_reading(f)
        finally:
            f.close()


# In[ ]:


#!/usr/bin/python

import struct, sys, os

f = open(sys.argv[1])

HEADER_OFFSET = 37

f.seek(HEADER_OFFSET)

filesize = os.stat(sys.argv[1]).st_size
num_items = (filesize - HEADER_OFFSET) / 4

values = []


def get_pointer(num):
    ptr = (num - HEADER_OFFSET) / 4.0
    if int(ptr) == ptr and ptr > 0 and ptr < num_items:
        return int(ptr)
    else:
        return None


try:
    while True:
        values.append(struct.unpack("<i", f.read(4))[0])
except:
    pass

values = (values)
seen = [False] * len(values)


def decode4s(values, ptr):
    firstvals = []

    while True:
        count = values[ptr]
        ptr = ptr + 2
        for i in range(0, count):
            vals = (values[ptr],
                    values[ptr+1],
                    values[ptr+2],
                    values[ptr+3])
            ptr = ptr + 4
            if vals[3] != 4:
                return firstvals
            firstvals.append(vals[0])


# follow all pointers
ptr = 0
while ptr is not None:
    value = values[ptr]
    #print "values[%d] -> %d" % (ptr, value)
    if value == 0:
        #print "  *** %3d" % (value)
        ptr = ptr + 1
        continue
    elif value == 128:
        items = decode4s(values, ptr)
        for z in items:
            width = 216
            print "%d %d" % (z % width, z / width)
        break
    ptr = get_pointer(value)
    

