def __bootstrap__():
    import sys
    import base64
    import gzip
    import tarfile
    import os
    import io
    import subprocess
    from pathlib import Path
    from tempfile import TemporaryDirectory

    # install required packages
    pkg_dataset_path = Path.cwd().parent / "input" / "kkt-example-requirements"
    pkg_path_list = []
    for p in pkg_dataset_path.glob("*"):
        if p.is_dir():
            pkg_config_files = [str(p.parent) for p in p.glob("**/*") if p.name in ["pyproject.toml", "setup.py"]]
            pkg_root_dir = min(pkg_config_files, key=len)
            pkg_path_list.append(pkg_root_dir)
        else:
            pkg_path_list.append(str(p))
    if 0 < len(pkg_path_list):
        subprocess.run(["pip", "install", "--no-deps", *pkg_path_list])

    if 0 < len(['pandas', 'teimpy']):
        args = ["pip", "install", *['pandas', 'teimpy']]
        subprocess.run(args)

    # this is base64 encoded source code
    tar_io = io.BytesIO(gzip.decompress(base64.b64decode("H4sIALEvEV8C/8vOLolPrUjMLchJ1TXQM9Qz0C2oNNbNy89L1U3Mq9Qrz8hhoBgYAIGZiQmYBgJ02sjYxBjGhogbmhqYmjAoGDDQAZQWlyQWKShQxZPInhsiIMCbmUUESHMAsSJD+TK/F+uALF5GBgZxIJ2NSB368fGZeZkl8fF6BZWpfp5chw149vwWPbRcImj7dbFPDXd+NT+44aX0oe3tlEvXr3NyimixzBKUOuujadGxSmeam4K4s9aHNVNkLTTzm5LVe6eu4tm6OLFHco6DfqOJ2aODC8IObVKrP/HE7MT0Z9cOxcp8N9MwTnm66Kji+sWL52j+CN7v2pjXf6FAc2eDdsGnQt0a57Y0rektVmfm2c349vS+8crP3f+PdbqprdlWej+HT3eC+Q+5kg2y7+fO2mges0EfxYsenTkrvwQAWcFgD6N4EZIB9FIyi0t0M/PS8vXDPVxdfbjPe/ievcjrrat17sz5zUEGV4wfFPmf9db10jmpvylI44S/ruYqlk6mo82fPnuX2HaXMnt1BxUbTV4j4tXlFbzTSehy4yut3tLPnz4HlQp/1tqJGuIeJ9WqVl8CsuqBIa5CwDm+riGOLo4hjlMDfP2Bwb/3efgPjqDsrbIZGq/UlhWmZDBxnhMznrlVTdNT62K9f5OAQt7uG9P+2q3Z/tiF3SvEvrjj0dw2vuivDtNM5/itfMa/6fGBV8/vJHT3Vs1as9QgO2C2pmzthQ2LPdw55eI23fu+Lc3bKGNBwp7D9l+vTWHeV7Va7mftpAMb752fOYn32mpWvdX/tHbnsHcF3mvZ/M+uQvhEVCHbu5MzO3maz/3Wj5op9vxRYaXZ42ff71t3eh72eOcgqHdttQx3XHjt9Ojin/LTIt5YiEecXL/2F9tupt8sqGFy/7Zw1EUgSwMYJkoEwiTI1dk/yKX1jHdREzDPXdibzsmxzCrlouuRhtQFFk4uN1K2tJxn4GzNOHh5d9Gu7PuMcZc1ji33vr695mSEbtbRGffOJ6zRW/312d3jF5Nn/SkuW/XcWWn7Wf/+vBcP20JYnyvtOH1F7PH269OnMYZJb8mosz57wffuhc9VIVG6XDt9vrX8WG8hNcVix3KLB5mLQqwXearcMLBc+4bnHpf694wqU1uLDJGID+fNFQzvPZTY/+pK5zM9nfdFK+x0xVcZ3dn3c+3j+NR97wPuWLWk1sjWavxifWD/feKNh4XsBx5xBngzMokw486YMLCkkQF3NkUzBCPpww1hfExMRkA3Dj3pIowrYiQuIaObiB7xCBPbmIhJBgHerGwg9SxAaAY0YzozwygYBaNgFIyCUTAKRsEoGAWjYBSMglEwCkbBKBgFo2AUjIJRMApGwSgYBaNgFIyCUTAKRsFwBwCkct99ACgAAA==")))
    with TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=tar_io) as tar:
            for member in tar.getmembers():
                pkg_path = Path(temp_dir) / f"{member.name}"
                content_bytes = tar.extractfile(member).read()
                pkg_path.write_bytes(content_bytes)
                os.system("pip install --no-deps {pkg_path}".format(pkg_path=pkg_path))

    sys.path.append("/kaggle/working")
    os.environ.update({})
__bootstrap__()

import kkt_example

submission = kkt_example.load_sample_submission()
for _, row in submission.iterrows():
    row["Label"] = kkt_example.choice()

submission.to_csv("submission.csv")

