#!/usr/bin/env python
# coding: utf-8

# # Install MySQL for EDA
# This kernel demonstrate a possibility of using mysql in kaggle kernel. Although most kaggler are using other tools(e.g. pandas) for EDA, this could be help for those who are familiar with SQL.
# 
# The whole work is divided into the following steps: 
# * install dependencies and download mysql-*.tar.gz;
# * install mysql(refer: https://dev.mysql.com/doc/refman/5.7/en/binary-installation.html);
# * start mysql service and create database;
# * dump datasets into mysql;
# * add primary key and index for each table;
# 
# Call for help: there's some problems i can't solve, hope someone could give me some advice, those problems are:
# * i can't preserve mysql-installed status when kernel restart(i.e. all the mysql folder/files and user/groups are cleared when kernel restart);
# * i have tried other install method(e.g. apt-get & dpkg) according to official reference manual, but all failed for various reasons;
# * i can't expose kernel's ip address to internet so i can connect mysql in local machine.
# * i can't interact with mysql in mysql CLI, coz when you run 'mysql' command directly, the notebook hanged there.

# # install dependencies and download mysql-*.tar.gz;

# In[ ]:


# !apt-cache search libaio
get_ipython().system('apt-get -y install libaio1 > /dev/null ')
# !apt-cache search libnuma 
get_ipython().system('apt-get -y install libnuma1 > /dev/null')

get_ipython().system('pip install pymysql')

get_ipython().system('wget http://mysql.mirrors.hoobly.com/Downloads/MySQL-5.6/mysql-5.6.45-linux-glibc2.12-x86_64.tar.gz')


# # install mysql(refer: https://dev.mysql.com/doc/refman/5.7/en/binary-installation.html);

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\ngroupadd mysql\nuseradd -r -g mysql -s /bin/false mysql\ncd /usr/local\ntar zxvf /kaggle/working/mysql-5.6.45-linux-glibc2.12-x86_64.tar.gz > /dev/null\nln -s mysql-5.6.45-linux-glibc2.12-x86_64 mysql\ncd mysql\nmkdir mysql-files\nchown mysql:mysql mysql-files\nchmod 777 mysql-files\n\nchmod -R 777 /tmp\n# official instruction " bin/mysqld --initialize --user=mysql " dosen\'t work, change to "mysql_install_db"\n./scripts/mysql_install_db --user=mysql\n\n# bin/mysqld --initialize --user=mysql \n# bin/mysql_ssl_rsa_setup              \n# bin/mysqld_safe --user=mysql &\n# # Next command is optional\n# cp support-files/mysql.server /etc/init.d/mysql.server')


# # start mysql service and create database;

# In[ ]:


import os
os.environ['PATH'] = ':'.join([os.environ['PATH'], '/usr/local/mysql/bin'])
os.system("mysqld_safe --user=mysql &")

get_ipython().system('sleep 7')

get_ipython().system('mysql -e "create database fraud"')
get_ipython().system('mysql -e "show databases"')


# # dump datasets into mysql;

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


import os
from multiprocessing.pool import Pool

import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

DS_DIR = '../input/ieee-fraud-detection/'
CONN_STR = 'mysql+pymysql://root:@localhost:3306/fraud?charset=utf8'
# carefully handling mysql connection in multiprocess env.
# here, i create a connection for each process, store them in a dict.
PID_CONN = {}


def get_conn(conn_str):
    """get mysql connection for this process, if it dose not exists, create it."""
    pid = os.getpid()
    if pid not in PID_CONN.keys():
        PID_CONN[pid] = {conn_str: create_engine(conn_str)}
    else:
        if conn_str not in PID_CONN[pid].keys():
            PID_CONN[pid][conn_str] = create_engine(conn_str)

    return PID_CONN[pid][conn_str]


class Dump2MySQL():

    def __init__(self, conn_str, index=True, index_label: str = None, if_exists='fail'):
        self.conn_str = conn_str
        self.index = index
        self.index_label = index_label
        self.if_exists = if_exists
        # set in dump_using_mp()
        self.table = None

    def dump_using_mp(self, csv_name, chunksize=1000):
        print("Now dumping: " + csv_name)
        df_chunks = pd.read_csv(os.path.join(DS_DIR, csv_name + '.csv'), chunksize=chunksize)
        self.table = csv_name
        # dump some data to mysql in main process to create table,
        # otherwise, sub-processes will compete creating table, which may cause error.
        self._dump_df(df_chunks.get_chunk(size=chunksize))
        with Pool() as pool:
            _ = list(tqdm(pool.imap(self._dump_df, df_chunks, chunksize=1)))
        return

    def _dump_df(self, df):
        pd.io.sql.to_sql(df,
                         name=self.table,
                         con=get_conn(self.conn_str),
                         index=self.index,
                         index_label=self.index_label,
                         if_exists=self.if_exists)


dumper = Dump2MySQL(conn_str=CONN_STR, index=False, index_label='TransactionID', if_exists='append')

dumper.dump_using_mp('train_identity')
dumper.dump_using_mp('train_transaction')

dumper.dump_using_mp('test_identity')
dumper.dump_using_mp('test_transaction')
dumper.dump_using_mp('sample_submission')


# # add primary key and index for each table;

# In[ ]:


# add primary key and index for train_identity.
sql = """
USE fraud;
ALTER TABLE `train_identity`
MODIFY COLUMN `TransactionID` bigint(20) NOT NULL FIRST,
ADD PRIMARY KEY (`TransactionID`);
""".replace('`', '\`')

get_ipython().system('mysql -e "$sql"')

# add primary key and index for train_transaction.
sql = """
USE fraud;
ALTER TABLE `train_transaction`
MODIFY COLUMN `TransactionID` bigint(20) NOT NULL FIRST,
ADD PRIMARY KEY (`TransactionID`),
MODIFY COLUMN `TransactionDT` bigint(20) NOT NULL,
ADD INDEX (`TransactionID`);
""".replace('`', '\`')

get_ipython().system('mysql -e "$sql"')

# add primary key and index for test_identity.
sql = """
USE fraud;
ALTER TABLE `test_identity`
MODIFY COLUMN `TransactionID` bigint(20) NOT NULL FIRST,
ADD PRIMARY KEY (`TransactionID`);
""".replace('`', '\`')

get_ipython().system('mysql -e "$sql"')

# add primary key and index for test_transaction.
sql = """
USE fraud;
ALTER TABLE `test_transaction`
MODIFY COLUMN `TransactionID` bigint(20) NOT NULL FIRST,
ADD PRIMARY KEY (`TransactionID`),
MODIFY COLUMN `TransactionDT` bigint(20) NOT NULL,
ADD INDEX (`TransactionID`);
""".replace('`', '\`')

get_ipython().system('mysql -e "$sql"')

# add primary key and index for sample_submission.
sql = """
USE fraud;
ALTER TABLE `sample_submission`
MODIFY COLUMN `TransactionID` bigint(20) NOT NULL FIRST,
ADD PRIMARY KEY (`TransactionID`);
""".replace('`', '\`')

get_ipython().system('mysql -e "$sql"')


# In[ ]:


get_ipython().system('mysql -e "use fraud; show tables;"')
# train_identity size:
get_ipython().system('mysql -e "use fraud; select count(TransactionID) as train_identity_size from train_identity"')
# train_transaction size:
get_ipython().system('mysql -e "use fraud; select count(TransactionID) as train_transaction_size from train_transaction"')
# test_identity size:
get_ipython().system('mysql -e "use fraud; select count(TransactionID) as test_identity_size from test_identity"')
# test_transaction:
get_ipython().system('mysql -e "use fraud; select count(TransactionID) as test_transaction_size from test_transaction"')
# sample_submission:
get_ipython().system('mysql -e "use fraud; select count(TransactionID) as sample_submission_size from sample_submission"')


# # do your EDA here, enjoy!

# In[ ]:


def make_sql(where): return f"""
USE fraud;
SELECT
	total,
	cnt_d9,
	( cnt_d9 / total ) AS ratio
FROM(
	SELECT
		count( TransactionID ) AS total,
		count( D9 ) AS cnt_d9 
	FROM
		train_transaction 
	WHERE
	{where}
) AS foo
"""
get_ipython().system("echo 'D9 non-null ratio of whole table:'")
sql = make_sql('1=1')
get_ipython().system('mysql -e "$sql"')
get_ipython().system("echo 'D9 non-null ratio when isFraud = 1:'")
sql = make_sql('isFraud = 1')
get_ipython().system('mysql -e "$sql"')

