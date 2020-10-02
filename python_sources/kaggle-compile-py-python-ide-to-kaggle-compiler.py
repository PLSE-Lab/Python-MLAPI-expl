#!/usr/bin/env python
# coding: utf-8

# # kaggle_compile.py: Python IDE to Kaggle Compiler
# 
# One of the big limitations of Kaggle Kernel Only competitions is that the code needs to be written in a notebook. 
# 
# Whilst this is fine for some usecases, there is a limit to the complexity of code that can be written outside a localhost IDE (I'm a big fan of IntelliJ) with a linter, debugger, type checker and profiler. Code reuse, reorganization, and refactoring is also greatly enhanced using multi-file codebase and git version control. This is the workflow I am familiar with as a professional programmer.
# 
# My first attempt involved manually copy and pasting required functions into a Kaggle Script Notebook, but then I decided that this process could be automated with a script.
# 
# Taking inspiration from my IE6 javascript days, I wrote a python script compiler for Kaggle. It reads in a python executable script, parses the `import` headers for any local include files, then recursively builds a dependency tree and concatenates these into single python file / text-blob. It can be called with either `--save` or `--commit` cli flags to automatically save to disk or commit the result to git.
#   
# # Limitations
# 
# the script only works for `from local.module import function` syntax, and does not support `import local .module` syntax, as calling `module.function()` inside a script would not work with script concatenation. Also the entire python file for the dependency is included, which does not guarantee the absence of namespace conflicts, but given an awareness of good coding practices, it is sufficiently practical for generating Kaggle submissions.
#   
# There are other more robust solutions to this problem such as [stickytape](https://github.com/mwilliamson/stickytape), which allow for module imports, however, the code for dependency files is obfuscated into a single line string variable, which makes for an unreadable Kaggle submission. [kaggle_compile.py](https://github.com/JamesMcGuigan/kaggle-arc/blob/master/submission/kaggle_compile.py) produces readable and easily editable output.
# 
# 
# # Example Notebooks
# - https://www.kaggle.com/jamesmcguigan/bengali-ai-tensorflow-imagedatagenerator-cnn
# - https://www.kaggle.com/jamesmcguigan/arc-oo-framework-xgboost-multimodel-solvers
# 

# # kaggle_compile.sh
# - https://github.com/JamesMcGuigan/kaggle-arc/blob/master/submission/kaggle_compile.sh

# ```
# #!/usr/bin/env bash
# cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}/..")")"  # https://stackoverflow.com/questions/3349105/how-to-set-current-working-directory-to-the-directory-of-the-script/51651602#51651602
# 
# python3 ./submission/kaggle_compile.py ./src/main.py | tee ./submission/submission.py
# time -p python3 ./submission/submission.py           | tee ./submission/submission.log
# ```

# # kaggle_compile.py
# - https://github.com/JamesMcGuigan/kaggle-arc/blob/master/submission/kaggle_compile.py
# 
# NOTE: This script must be run on localhost, and will throw an exception inside a Kaggle Notebook when it tried to run `git remote -v` outside of a localhost git repo.

# In[ ]:


#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from typing import List
from subprocess import PIPE

parser = argparse.ArgumentParser(
    description='Compile a list of python files into a Kaggle compatable script: \n' +
                './kaggle_compile.py [script_files.py] --save'
)
parser.add_argument('files', nargs='+',                                help='list of files to parse' )
parser.add_argument('--python-path', default='.',                      help='directory to search for local namespace imports')
parser.add_argument('--output-dir',  default='./',                     help='directory to write output if --save')
parser.add_argument('--save',        action='store_true',              help='should file be saved to disk')
parser.add_argument('--commit',      action='store_true',              help='should saved file be commited to git')
args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle
if len(args.files) == 0:  parser.print_help(sys.stderr); sys.exit();


module_names = [ name for name in os.listdir(args.python_path)
                 if os.path.isdir(os.path.join(args.python_path, name))
                 and not name.startswith('.') ]
module_regex = '(?:' + "|".join(map(re.escape, module_names)) + ')'
import_regex = re.compile(f'^from\s+({module_regex}.*?)\s+import', re.MULTILINE)


def read_and_comment_file(filename: str) -> str:
    code = open(filename, 'r').read()
    code = re.sub(import_regex, r'# \g<0>', code)
    return code


# TODO: handle "import src.module" syntax
# TODO: handle "from src.module.__init__.py import" syntax
def extract_dependencies_from_file(filename: str) -> List[str]:
    code    = open(filename, 'r').read()
    imports = re.findall(import_regex, code)
    files   = list(map(lambda string: string.replace('.', '/')+'.py', imports))
    return files


def recurse_dependencies(filelist: List[str]) -> List[str]:
    output = filelist
    for filename in filelist:
        dependencies = extract_dependencies_from_file(filename)
        if len(dependencies):
            output = [
                recurse_dependencies(dependencies),
                dependencies,
                output
            ]
    output = flatten(output)
    return output


def flatten(filelist):
    output = []
    for item in filelist:
        if isinstance(item,list):
            if len(item):         output.extend(flatten(item))
        else:                     output.append(item)
    return output


def unique(filelist: List[str]) -> List[str]:
    seen   = {}
    output = []
    for filename in filelist:
        if not seen.get(filename, False):
            seen[filename] = True
            output.append(filename)
    return output


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def savefile():
    savefile = os.path.join( args.output_dir, os.path.basename(args.files[-1]) )  # Assume last provided filename
    return savefile


def compile_script(filelist: List[str]) -> str:
    filelist = unique(filelist)


    shebang = "#!/usr/bin/env python3"
    header = [
        ("\n" + (" ".join(sys.argv)) + "\n"),
        subprocess.run('date --rfc-3339 seconds',     shell=True, stdout=PIPE).stdout.decode("utf-8"),
        subprocess.run('git remote -v',               shell=True, stdout=PIPE).stdout.decode("utf-8"),
        subprocess.run('git branch -v ',              shell=True, stdout=PIPE).stdout.decode("utf-8"),
        subprocess.run('git rev-parse --verify HEAD', shell=True, stdout=PIPE).stdout.decode("utf-8"),
    ]
    if args.save: header += [ f'Wrote: {savefile()}' ]

    header = map(lambda string: string.split("\n"), header )
    header = map(lambda string: '##### ' + string, flatten(header))
    header = "\n".join(flatten(header))

    output_lines = [
        shebang,
        header,
    ]
    for filename in filelist:
        output_lines += [
            f'#####\n##### START {filename}\n#####',
            read_and_comment_file(filename),
            f'#####\n##### END   {filename}\n#####',
        ]
    output_lines += [ header ]
    output_text   = "\n\n".join(output_lines)
    output_text   = reorder_from_future_imports(output_text)
    return output_text

def reorder_from_future_imports(output_text):
    lines   = output_text.split('\n')
    shebangs = [ line for line in lines[:10] if line.startswith('#!/')         ]
    futures  = [ line for line in lines      if '__future__' in line           ]
    other    = [ line for line in lines      if line not in shebangs + futures ]
    output   = "\n".join([ *shebangs, *sorted(set(futures)), *other ])
    return output



if __name__ == '__main__':
    filenames = recurse_dependencies(args.files)
    code      = compile_script(filenames)
    print(code)
    if args.save or args.commit:
        with open(savefile(), 'w') as file:
            file.write(code)
            file.close()
        make_executable(savefile())

        if args.commit:
            while not os.path.exists(savefile()): continue
            command = f'git add {savefile()}; git commit -o {savefile()} -m "kaggle_compile.py | {savefile()}"'
            print(f'$ {command}')
            print( subprocess.check_output(command, shell=True).decode("utf-8") )


# Please leave an upvote if you use this script
