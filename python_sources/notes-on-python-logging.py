#!/usr/bin/env python
# coding: utf-8

# ### Basics
# 
# * Python ships with a `logging` module which supports five levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. To raise an error at a certain level use the module method:
# 
#     `import logging; logging.debug(...)`
#     
# * Further configuration of e.g. message format flows from the `basicConfig` method, e.g.: 
# 
#     `logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')`
# 
# * There is a `level` parameter on `basicConfig` that controls what level of messages get written to the log.
# * The other parameters above redirect logging to a file instead of `stdout`.
# * Note the use of precomputed strings to control the write format. Some other paremeters impute based on these strings. For example you can pass an `strformat` string to `datefmt` to change how the date information gets written.
# * To include an error stack trace:
# 
#     `except Exception as e:  logging.error("Exception occurred", exc_info=True)`
# 
# * You can also use the `logging.exception` shortcut which is basically this method with `exc_info=True`.
# 
# ### Object configuration
# * If you use solely the above facilities you wil log to root. You are encouraged to initiate and configure your own logging module objects.
# * The logging module consists of four objects:
#    * `Logger` which has method handlers attached that you use to log.
#    * `LogRecord` table of log messages created.
#    * `Handler` which handles sending the `LogRecord` messages to output formats, e.g. `StreamHandler` versus `FileHandler` versus whatever.
#    * `Formatter` handles the string format.
# * You mostly create new `Logger` objects. E.g. `logging.getLogger('foo')`.
# * It is recommended that we use module-level loggers by passing __name__ as the name parameter to getLogger() to create a logger object as the name of the logger itself would tell us from where the events are being logged.
# * Instead of using `basicConfig` you customize loggers by adding formatters and handlers to them.
# 
# ```python
# # logging_example.py
# 
# import logging
# 
# # Create a custom logger
# logger = logging.getLogger(__name__)
# 
# # Create handlers
# c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler('file.log')
# c_handler.setLevel(logging.WARNING)
# f_handler.setLevel(logging.ERROR)
# 
# # Create formatters and add it to handlers
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)
# f_handler.setFormatter(f_format)
# 
# # Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)
# 
# logger.warning('This is a warning')
# logger.error('This is an error')
# ```
# 
# * Notice that handlers can define their own desired logging levels, for when you e.g. want different levels of errors appearing in different logs.
# * You can also perform this configuration from a file or a `dict` using `fileConfig` or `dictConfig`.
# 
# ### Best practices
# * The accepted best practice in Python is to use the `logging` module to perform logging. This is in contrast to e.g. the consensus in Node, where Bunyan et. al. has emerged as the library of choice. Maybe this is because Python is basically single-threaded due to the GIL so the stdlib implementation is more than good enough?
# * Some good advice in the top comment to [this thread](https://www.reddit.com/r/Python/comments/4wphj3/how_do_you_log_logging_output/).
#   * Don't move off of the INFO log level as the default choice.
#   * Don't log secrets.
#   * Put timing information and who logged in where and when into DEBUG.
#   * System deficiencies that would interest a sysadmin go in WARNING.
#   * ERROR is when you drop a stack trace and ride off into the sunset.
# * Use one logger entity per module.
