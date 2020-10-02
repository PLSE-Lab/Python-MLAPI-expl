#!/usr/bin/env python
# coding: utf-8

# # Unix Compression Benchmarks
# 
# Here we compare a range of unix compression techniques against the Hutter Prize Dataset (1,000,000,000 bytes = ~1GB of wikipedia html data)

# In[ ]:


get_ipython().system(' apt-get install -qq time p7zip-full p7zip-rar rar > /dev/null')


# # CLI Commands
# Now lets compare a range of different unix compression techniques

# In[ ]:


get_ipython().run_line_magic('env', 'INPUT_FILE=../input/hutter-prize/enwik9')
get_ipython().run_line_magic('env', 'OUTPUT_FILE=./enwik9')
get_ipython().run_line_magic('env', 'TMPDIR=./')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'benchmark.sh', '#!/usr/bin/env bash\nset -x\n\ntime cp              ${INPUT_FILE}          ${OUTPUT_FILE}\ntime tar         -cf ${OUTPUT_FILE}.tar     ${INPUT_FILE}\ntime tar --gzip  -cf ${OUTPUT_FILE}.tar.gz  ${INPUT_FILE}\ntime tar --bzip2 -cf ${OUTPUT_FILE}.tar.bz2 ${INPUT_FILE}\ntime tar --xz    -cf ${OUTPUT_FILE}.tar.xz  ${INPUT_FILE}\n\ntime gzip         -9 ${INPUT_FILE}     -c > ${OUTPUT_FILE}.gz \ntime bzip2    --best ${INPUT_FILE}     -c > ${OUTPUT_FILE}.bz2\ntime rar           a ${OUTPUT_FILE}.rar     ${INPUT_FILE}\ntime 7z            a ${OUTPUT_FILE}.7z      ${INPUT_FILE}\n\n# # DOCS: https://superuser.com/questions/281573/what-are-the-best-options-to-use-when-compressing-files-using-7-zip\ntime 7z a -t7z -mx=9 -mfb=64  -ms=on -md=32m   -m0=lzma   ${OUTPUT_FILE}.7z.ultra ${INPUT_FILE}\ntime 7z a -t7z -mx=9 -mfb=273 -ms=on -md=1536m -myx=9 -mtm=- -mmt -mmtf -mmf=bt3 -mmc=10000 -mpb=0 -mlc=0 ${OUTPUT_FILE}.7z.ultra+ ${INPUT_FILE}')


# # Timings

# In[ ]:


get_ipython().system(' bash ./benchmark.sh 2>&1 | grep \'^+\\|^real\' | awk \'ORS=NR%2?" ":"\\n"\' | perl -p -e \'s/^(.+)real\\s+(.+)/$2 $1/\' | sort -n')


# # Filesize Results

# This is how we can measure filesize, either in bytes of as human readable

# In[ ]:


get_ipython().system(' ls -lah    ${INPUT_FILE}')
get_ipython().system(' ls -la     ${INPUT_FILE}')
get_ipython().system(' stat -c %s ${INPUT_FILE}')


# In[ ]:


get_ipython().system('ls -laS ${INPUT_FILE}* ${OUTPUT_FILE}*')


# Cleanup

# In[ ]:


rm -f ${OUTPUT_FILE}*

