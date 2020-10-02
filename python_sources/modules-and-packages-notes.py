#!/usr/bin/env python
# coding: utf-8

# This notebook constitutes my notes on David Beazley's [Modules and Packages](https://www.youtube.com/watch?v=0oTh1CXRaQ0) talk (which I watched for the second time, this is very dense material!)
# 
# ### Basics
# * Every variable defined at the top level of a file is scoped to that file.
# * Functions record what module they were created in, and create a copy of globals that were present in that module, which get scoped into the function (just like what JavaScript does).
# * Modules always execute the entire file, even when only loading a subset of names.
# * Module names must be valid Python variable names.
# * Modules are cached to `sys.modules`. An import only happens once over the lifetime of the execution environment.
# * You can purge the cache and explicitly reload by using `import importlib; importlib.reload(<modulename>)`.
# * Modules are marked using an `__init__.py` file placed into the root of the module directory structure. This is an explicit step because the `PYTHONPATH` can point to arbitrary file locations, including locations containing files whose names collide with reserved Python names (e.g. `string`). Explicitly requiring package marking alleviates this issue.
# 
# ### Imports and exports
# * In Python 2 there was the ability to import a module that was located in the same folder as the current module. E.g.:
# 
#     ```
#         p/
#             foo.py
#             spam.py
#     ```
# 
#     And in `spam.py`:
# 
#     ```
#     import foo
#     ```
# 
#     This was disallowed in Python 3.
#     
# * Instead you can use an **absolute import** that references the parent name (which is included in the `PYTHONPATH`):
# 
#     ```
#     from p import foo
#     ```
#     
#     But this is fragile.
# 
# * There are also **explicit relative imports** that use the `dot` operator to mark where the import is coming from. This is recommended.
# * Importing package modules into the top level in `__init__.py` makes those modules top-level with respect to the package. The user does not see the implementation detail of the file path.
# 
# 
# ### Module exports design pattern
# * There is an `__all__` magic that specifies which commands will be exported from a module when that module is imported using `from MODULE import *`. This is always worth specifying in order to avoid injecting sub-imports into the namespace.
# * It also lends itself to a nice design pattern: specifying an `__all__` in the module files, then `import *` in the `__init__.py`. Then specify `__all__ = foo.__all__ + bar.__all__` in `__init__.py`.
# * This basically allows you to manage module namespaces within their files instead of in the `__init__.py` file, which is e.g. what JavaScript does (I heartily approve and will need to do this going forward).
# 
# 
# ### Module subpathing
# * You can import modules nested in deeper levels of heirarchy by connecting names with dots (`.`).
# * Python uses `PYTHONPATH` to look up the top-level name, then switches to using `file.__path__` for subsequent hierarchical names.
# * So you could tell Python to e.g. look in another directory by modifying where `__path__` points.
# 
# 
# ### Runnable files and modules
# * You can pass a flag to the `python` CLI command, `-m`, to run a module as a mainfile. This will execute the internals of the `if name == '__main__'` routine.
# * This is equivalent to executing the file directly: `python file.py`.
# * Except that since the file is installed as a module actually finding the path to it is annoying. The `-m` flag makes this less annoying.
# * You can specify a `__main__.py` file to allow running a main program for a directory (or a module). This makes the entry point explicit and is considered best practice.
# * `__main__.py` works even if the parent directory is not a package root (e.g. there is no `__init__.py`).
# 
# 
# * You can even make zipfiles executable by prepending their contents with a shebang (e.g. `#!usr/bin/env/python3`).
# 
# 
# ### `sys.path`
# * "Most errors having to do with importing things are due to problem with `sys.path`."
# * Conceptually `sys.path` is populated with four kind of things: directories, zip files,  egg files.
# * Egg files are just zip files that following a Python packaging spec dating from 2004.
# * Wheel files are zip files following a different Python packaging spec dating from 2012.
# * Current best practice for distributing pure Python packages is to distribute them as wheels (with e.g. `twine`).
# * A name, e.g. `spam`, may match many different formats: `spam.py`, `spam.zip`, `spam.egg`, `spam.cpython-34m.pyc`...
# 
# 
# ### `sys.prefix`
# * The `prefix` determines where packages go by default.
# * Python by default uses the root directory of the Python installation, but this can be configured in a number of ways.
# * `virtualenv` and `conda` use this hook.
# * Way number one, the `PYTHONPATH` envar.
# * Way number two, Python expects a certain construction of "install landmarks". If these are moved Python will move its pointers to reflect the changes. The landmarks are the `python3` executable, `os.py` (which defines `sys.prefix` root), and `lib-dynload/` (which defines `sys.exec_prefix` root; `exec_prefix` is the root for shared C executables).
# 
# 
# ### Third-party packages
# * Third party pacakges are managed using a `site.py` module built into stdlib.
# * This module handles finding and adding a `site-packages` directory to `sys.path` which contains all dependencies.
# * Interestingly (I never knew this) one of the paths is user-local.
# * A module can create a `.pth` file which contains a list of paths. Those paths will be appended to `sys.path` (1:10).
# * `.pth` files are mainly used by package managers to add `egg` files and similar facilities (making standing up connections to compiled files more explicit).
# * Finally, you can user-define `sitecustomize.py` and `usercustomize.py` modules in site packages which will be exucted by `site.py` on import and may execute arbitrary code.
# * The current working directory is added last.
# * To help debug installation issues you may use the `-E` flag to ignore environment variables and the `-s` flag to ignore site packages (`-I` does both).
# 
# 
# ### Namespace packages
# * Omit `__init__.py` and the package still works. But it now constructs a **namespace** when imported.
# * The regular Python procedure for looking up a name is first match wins. There is a linear iteration order defined by `sys.path` and `os.lsdir`, and if the same package name is defined in two different places, the first one to appear in the linearized sequence wins.
# * If you construct a namespace, what will happen instead is that Python will perform a full scan, and any namespaces with the same name will have their contents merged into a single package.
# * This results in a modified `NamespacePath` on `__path__` which contains .
# * If any of the packages matched have an `__init__` that packages will be pulled instead.
# * That's a lot of magic!
# * Don't do this on a day-to-day basis.
# 
# 
# ### Module versus package
# * A **package** is what is meant to be installed by a user. A **module** is a grouping of code that defines some bit of features that you can import from.
# * In practice the only difference between the two is that a package has two non-null **attributes**: `__package__`, the name of the package, and `__path__`, which is the subcomponent search path.
# 
# 
# ### pyc
# * Modules are compiled into `pyc` files, which is the bytecode object. The file format is magic, mtime, size, marshalled code object (mtime is TTL before recompile).
# * Every time the code is modified, as well as the first time an import is run, a `pyc` file is generated.
# 
# 
# ### Module cache
# * The module cache exists.
# * If an import is located to a function, that module is loaded the first time that function is run, but that copy is retained for the remaining lifetime of the operation.
# * Imports to a function *are* however scoped to that function. They are not injected into all-defined `__globals__`.
# 
# 
# * There are a few ways to create import cycles. Some are handled by the cache and some cannot be resolved.
# * Technically even unresolvable import cycles can be overcome by injecting into the module cache directly: `foo = sys.modules[__package__ + 'foo']`.
# * But don't ever write code that contains import cycles.
# 
# 
# ### Threads
# * Import statements that execute on different threads (`Thread` objects) can get into race conditions because they use the same module cache. They can double-execute a module when they both attempt to import it, and they can attempt use a partially deserialized module that has already been placed into the cache by another thread in the process of an import and fail execution as a result.
# * Python takes a global lock on imports to deal with these potential problems.
# * It uses fine-grained locks tailored to modules and include deadlock detection. A lot of stuff going on there, all of which can be avoided by not using circular dependencies in different threads.
# 
# 
# ### Lazy module loading
# * A design pattern for `__init__` is to set it up using lazy loading. This can significantly speed up the time it takes to load a module initially, because the system doesn't need to disk scan for all of the scattered files that constitute the module at startup. Demo'd at 2:06.
# 
# 
# ### Reloading
# * You can `reload` a module.
# * All `reload` does is re-execute the code. It makes no effort to clean up the content injected by the prior version or versions of the codebase. This has implications like renamed functions still hanging around, etc.; all of the problems of Jupyter actually btw.
# * Objects created from old versions of code are not reloaded, e.g. they will be "old-version" objects.
# * Also, names from within a module imported explicitly (e.g. `from foo import spam`) are not reloaded (e.g. when you run `import foo`).
# 
# 
# * A module can detect if is being reloaded, using `if 'foo' in globals():`.
# * This allows you to reload patch. You can go pretty far here but it's really terrible practice to follow.
# 
# 
# ### `sys.meta_path` and module loaders
# * Every name you attempt to import is presented to the loaders on the `meta_path` in order. The importers gets executed in order with the `find_spec` method and the module name as input, and the first one to find what it's looking for wins.
# * There are three importers by default. A `builtin` importer finds built-in modules, of which there are just a handful: notably, `sys`. The second importer looks for C modules and is named `frozen`. The last one is the `path` loader, which loads all of the things on `sys.path`.
# * This an implementation detail is that it is impossible to pre-empt certain built-in module names like `sys`...unless you reorder this list yourself.
# 
# 
# * A nice feature of loaders is that they allow look-ahead on package names. If the module does not exist, the loader won't be able to find it. This is a neat alternative to try-catching `ImportError`.
# 
# 
# * You can use the loader to define a package in Python code in a just-in-time manner, using `importlib.util.module_from_spec`. Python 3.5 and higher only (2:30).
# * This mechanism is the basis of a true lazy loader mechanism (which has probably hit stdlib already).
# * It also enable e.g. autoinstallers (e.g. attempt to `pip install` something on import).
# 
# 
# ### `sys.path_hooks`
# * `sys.path_hooks` are what are used to determine _how_ to read particular file types.
# * Extending this allows you to e.g. import from URLs.
