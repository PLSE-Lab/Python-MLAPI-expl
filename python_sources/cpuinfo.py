from subprocess import check_output
info = check_output(["cat", "/proc/cpuinfo"]).decode("utf8")
print('Thread count: {}'.format(info.count('GHz')))

print(info)