#!/usr/bin/env python
"""
Parts are based on https://github.com/MIC-DKFZ/RegRCNN.git published
under Apache 2.0 license.
"""

from setuptools import find_packages, setup
import os, sys, subprocess

from pathlib import Path

def requirements():
    return [
        'nms-extension',
        'RoIAlign-extension-2D',
        'RoIAlign-extension-3D',
        'batchgenerators==0.19.7'
    ]

def parse_requirements(requirements, exclude=[]):
    return [req for req in requirements if req and not req.startswith("#") and not req.split("==")[0] in exclude]

def pip_install(item):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", item])
    except subprocess.CalledProcessError as e:
        output = e.output
        
def install_custom_ext(setup_path):
    try:
        pip_install(setup_path)
    except Exception as e:
        print("Could not install custom extension {} from source due to error:\n{}\n".format(path, e) +
              "Trying to install from pre-compiled wheel.")
        dist_path = setup_path+"/dist"
        wheel_file = [fn for fn in os.listdir(dist_path) if fn.endswith(".whl")][0]
        pip_install(os.path.join(dist_path, wheel_file))
    
def clean():
    """Custom clean command to tidy up the project root."""
    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')
  
############################################################
# Setup Package
############################################################

if __name__ == "__main__":

    req_file = requirements()
    custom_exts = ["nms-extension", "RoIAlign-extension-2D", "RoIAlign-extension-3D"]
    install_reqs = parse_requirements(req_file, exclude=custom_exts)

    setup(
        name='RegRCNN_Panda',
        version='0.0.1',
        author='C. de Bruyn',
        author_email='boertjie.seun@outlook.com',
        description="Medical Object-Detection Toolkit incl. Regression Capability.",
        classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3.7"
        ],
        packages=find_packages(exclude=['test', 'test.*']),
        install_requires=install_reqs,
    )

    subprocess.call(["cp", "-r", "/kaggle/input/regrcnn-custom-extensions/custom_extensions", "./custom_extensions"])
    
    custom_exts = [
        "./custom_extensions/nms",
        "./custom_extensions/roi_align/2D"
    ]
        
    for path in custom_exts:
        try:
            install_custom_ext(path)
        except Exception as e:
            print("FAILED to install custom extension {} due to Error:\n{}".format(path, e))

    clean()
