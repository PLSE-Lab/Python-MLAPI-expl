#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import s3fs
import dask.dataframe as dd
import pyarrow
import fastparquet
import boto3
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


print(pd.__version__)
print(pyarrow.__version__)
print(fastparquet.__version__)


# In[ ]:


campana_consultora = pd.read_csv('s3://belcorp-20191109/datathon-belcorp-prueba/campana_consultora.csv')
campana_consultora.columns = map(str.lower, campana_consultora.columns)


# In[ ]:


for col in ["codigofactura","evaluacion_nuevas","segmentacion"]:
    print(col)
    dummies = pd.get_dummies(campana_consultora[col], dummy_na=True)
    dummies = dummies.astype("int8")
    dummies.columns = [ col + "_"+ str(dum).lower() for dum in dummies.columns]
    campana_consultora = pd.concat([campana_consultora, dummies],axis=1)


# In[ ]:


col="geografia"
print(col)
dummies = pd.get_dummies(campana_consultora[col], dummy_na=True)
dummies = dummies.astype("int8")
dummies.columns = [ col + "_"+ str(dum).lower()[:2] for dum in dummies.columns]
campana_consultora = pd.concat([campana_consultora, dummies],axis=1)


# In[ ]:


campana_consultora["segmentacion"].unique()


# In[ ]:


campana_consultora["geografia"].unique()


# In[ ]:


campana_consultora.head()


# In[ ]:


campana_consultora.drop("codigocanalorigen",1,inplace=True)
campana_consultora.drop("unnamed: 0",1,inplace=True)


# In[ ]:


campana_consultora["flagdigital"].fillna(value=-1,inplace=True)
campana_consultora["cantidadlogueos"].fillna(value=-1,inplace=True)
campana_consultora["evaluacion_nuevas"].fillna(value="null",inplace=True)
campana_consultora["codigofactura"].fillna(value="null",inplace=True)


# In[ ]:


campana_consultora["campana"] = campana_consultora["campana"].astype("int32")
campana_consultora["cantidadlogueos"] = campana_consultora["cantidadlogueos"].astype("int32")
for col in campana_consultora.columns:
    if "flag" in col:
        campana_consultora[col] = campana_consultora[col].astype("int8")


# In[ ]:


campana_consultora.dtypes


# In[ ]:


campana_consultora.isnull().sum()


# In[ ]:


campana_consultora.drop("geografia",1,inplace=True)
campana_consultora.drop("codigofactura",1,inplace=True)
campana_consultora.drop("evaluacion_nuevas",1,inplace=True)
campana_consultora.drop("segmentacion",1,inplace=True)
campana_consultora.dtypes


# In[ ]:


s3 = boto3.resource("s3")
s3.Object("belcorp-20191109", "features/campana_consultora.parquet.gzip").delete()


# In[ ]:


campana_consultora.head()


# In[ ]:


campana_consultora.to_parquet('s3://belcorp-20191109/features/campana_consultora.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


campana_consultora.head()


# In[ ]:


import pandas as pd
import s3fs
import dask.dataframe as dd
import boto3
import fastparquet
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


maestro_consultora = pd.read_csv('s3://belcorp-20191109/datathon-belcorp-prueba/maestro_consultora.csv')
maestro_consultora.columns = map(str.lower, maestro_consultora.columns)


# In[ ]:


for col in ["estadocivil"]:
    print(col)
    dummies = pd.get_dummies(maestro_consultora[col], dummy_na=True)
    dummies = dummies.astype("int8")
    dummies.columns = [ col + "_"+ str(dum).lower() for dum in dummies.columns]
    maestro_consultora = pd.concat([maestro_consultora, dummies],axis=1)


# In[ ]:


maestro_consultora.head()


# In[ ]:


maestro_consultora["flagcorreovalidad"] = maestro_consultora["flagcorreovalidad"].fillna(value=-1).astype("int8")


# In[ ]:


campana_consultora = pd.read_parquet('s3://belcorp-20191109/features/campana_consultora.parquet.gzip',engine='fastparquet')


# In[ ]:


predict_submission = pd.read_csv('s3://belcorp-20191109/datathon-belcorp-prueba/predict_submission.csv')
predict_submission.columns = map(str.lower, predict_submission.columns)


# In[ ]:


campana_index = campana_consultora[["campana"]].drop_duplicates().reset_index(drop=True)#.query("campana>=201901")


# In[ ]:


consultora_hist = pd.concat([ 
                        pd.merge(campana_index.assign(key=0), 
                                 maestro_consultora[["idconsultora","campanaingreso"]].assign(key=0),
                                 on ='key').drop("key", axis=1).query('campana>campanaingreso')[["campana","idconsultora"]],
                        predict_submission[["idconsultora"]].assign(campana=201907)
                                     ],sort=True)


# In[ ]:


campana_consultora_hist = consultora_hist.merge(campana_consultora[["campana","idconsultora","flagpasopedido","flagactiva","flagpasopedidocuidadopersonal",
"flagpasopedidomaquillaje","flagpasopedidotratamientocorporal", "flagpasopedidotratamientofacial",
"flagpedidoanulado", "flagpasopedidofragancias","flagpasopedidoweb","flagdispositivo",
"flagofertadigital","flagsuscripcion"]],
                            on =['campana','idconsultora'],
                             how = "left")
campana_consultora_hist.fillna(0, inplace=True)
for col in campana_consultora_hist.columns:
    if col[:4]=="flag":
        campana_consultora_hist[col] = campana_consultora_hist[col].astype("int8")
    else:
        campana_consultora_hist[col] = campana_consultora_hist[col].astype("int32")
campana_consultora_hist.dtypes


# In[ ]:


maestro_consultora.columns


# In[ ]:


maestro_consultora.drop("unnamed: 0",1, inplace=True)
maestro_consultora.drop("estadocivil",1, inplace=True)


# In[ ]:


ds_var_maestro_cosultora_01=campana_consultora_hist[["campana","idconsultora"]].merge(maestro_consultora,
                            on ='idconsultora',how = "left")


# In[ ]:


ds_var_maestro_cosultora_01.isnull().sum()


# In[ ]:


ds_var_maestro_cosultora_01.dtypes


# In[ ]:


ds_var_maestro_cosultora_01["antiguedad"]= ds_var_maestro_cosultora_01["campana"].astype(int).apply(lambda x : x//100)*18 + ds_var_maestro_cosultora_01["campana"].astype(int).apply(lambda x : x%100) - ds_var_maestro_cosultora_01["campanaingreso"].astype(int).apply(lambda x : x//100)*18 - ds_var_maestro_cosultora_01["campanaingreso"].astype(int).apply(lambda x : x%100)


# In[ ]:


ds_var_maestro_cosultora_01["campana_ano"]= ds_var_maestro_cosultora_01["campana"].astype(int).apply(lambda x : x//100)
ds_var_maestro_cosultora_01["campana_periodo"]= ds_var_maestro_cosultora_01["campana"].astype(int).apply(lambda x : x%100)


# In[ ]:


ds_var_maestro_cosultora_01["campanaprimerpedido"] = ds_var_maestro_cosultora_01["campanaprimerpedido"].fillna(201907).astype("int32")
ds_var_maestro_cosultora_01["antiguedad_primerpedido"]= ds_var_maestro_cosultora_01["campana"].astype(int).apply(lambda x : x//100)*18 + ds_var_maestro_cosultora_01["campana"].astype(int).apply(lambda x : x%100) - ds_var_maestro_cosultora_01["campanaprimerpedido"].astype(int).apply(lambda x : x//100)*18 - ds_var_maestro_cosultora_01["campanaprimerpedido"].astype(int).apply(lambda x : x%100)


# In[ ]:


for col in ds_var_maestro_cosultora_01.columns:
    if "flag" in col:
        ds_var_maestro_cosultora_01[col] = ds_var_maestro_cosultora_01[col].astype("int8")
    else:
        ds_var_maestro_cosultora_01[col] = ds_var_maestro_cosultora_01[col].astype("int32")


# In[ ]:


ds_var_maestro_cosultora_01.isnull().sum()


# In[ ]:


ds_var_maestro_cosultora_01.dtypes


# In[ ]:


s3 = boto3.resource("s3")
s3.Object("belcorp-20191109", "features/ds_var_maestro_cosultora_01.parquet.gzip").delete()
s3.Object("belcorp-20191109", "features/ds_target_cosultora.parquet.gzip").delete()


# In[ ]:


ds_var_maestro_cosultora_01.to_parquet('s3://belcorp-20191109/features/ds_var_maestro_cosultora_01.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


from pyspark.sql.functions import *#mean,col,split, col, regexp_extract, when, lit,coalesce,max,count,substringb


# In[ ]:


storageAccountScope = "staeu2c001hubddev01"
saRepositorioIbkName = dbutils.secrets.get(scope = storageAccountScope, key = "name")
spark.conf.set("fs.azure.account.key." + saRepositorioIbkName + ".blob.core.windows.net",
  dbutils.secrets.get(scope = storageAccountScope, key = "key1"))
predict_submission = spark.read.format("csv").option("header", True).option("delimiter", ",").load("wasbs://" + "kaggle" + "@" + saRepositorioIbkName + ".blob.core.windows.net/" + "predict_submission.csv")
maestro_consultora = spark.read.format("csv").option("header", True).option("delimiter", ",").load("wasbs://" + "kaggle" + "@" + saRepositorioIbkName + ".blob.core.windows.net/" + "maestro_consultora.csv")
maestro_consultora = maestro_consultora.toDF(*[c.lower() for c in maestro_consultora.columns])                      .withColumn("idconsultora",col("idconsultora").cast("long").cast("string"))
campana_consultora = spark.read.format("csv").option("header", True).option("delimiter", ",").load("wasbs://" + "kaggle" + "@" + saRepositorioIbkName + ".blob.core.windows.net/" + "campana_consultora.csv")                      .withColumn("idconsultora",col("idconsultora").cast("long").cast("string"))
campana_consultora = campana_consultora.toDF(*[c.lower() for c in campana_consultora.columns])

dtt_fvta_cl = spark.read.format("csv").option("header", True).option("delimiter", ",").load("wasbs://" + "kaggle" + "@" + saRepositorioIbkName + ".blob.core.windows.net/" + "dtt_fvta_cl.csv")                      .withColumn("idconsultora",col("idconsultora").cast("long").cast("string"))
dtt_fvta_cl = dtt_fvta_cl.toDF(*[c.lower() for c in dtt_fvta_cl.columns])

maestro_consultora.createOrReplaceTempView("maestro_consultora")
campana_consultora.createOrReplaceTempView("campana_consultora")
predict_submission.createOrReplaceTempView("predict_submission")
dtt_fvta_cl.createOrReplaceTempView("dtt_fvta_cl")


# In[ ]:


maestro_producto = spark.read.format("csv").option("header", True).option("delimiter", ",").load("wasbs://" + "kaggle" + "@" + saRepositorioIbkName + ".blob.core.windows.net/" + "maestro_producto.csv")
maestro_producto = maestro_producto.toDF(*[c.lower() for c in maestro_producto.columns])


# In[ ]:


dtt_fvta_cl2 = dtt_fvta_cl.join(maestro_producto,
          on=["idproducto"],how='left')


# In[ ]:


dtt_fvta_cl_gb = dtt_fvta_cl2.groupBy("campana","idconsultora").agg(
  count("idconsultora").alias("contpedidos"),
  countDistinct("idproducto").alias("contdistidproducto"),
  countDistinct("codigotipooferta").alias("contdistcodigotipooferta"),
  countDistinct("canalingresoproducto").alias("contdistcanalingresoproducto"),
  countDistinct("codigopalancapersonalizacion").alias("contdistcodigopalancapersonalizacion"),
  countDistinct("grupooferta").alias("contdistgrupooferta"),
  sum(when(col("canalingresoproducto")==lit("MIX"),lit(1)).otherwise(0)).alias("contcanalingresoproducto_mix"),
  sum(when(col("canalingresoproducto")==lit("APP"),lit(1)).otherwise(0)).alias("contcanalingresoproducto_app"),
  sum(when(col("canalingresoproducto")==lit("DD"),lit(1)).otherwise(0)).alias("contcanalingresoproducto_dd"),
  sum(when(col("canalingresoproducto")==lit("DIG"),lit(1)).otherwise(0)).alias("contcanalingresoproducto_dig"),
  sum(when(col("canalingresoproducto")==lit("WEB"),lit(1)).otherwise(0)).alias("contcanalingresoproducto_web"),
  sum(when(col("canalingresoproducto").isNull(),lit(1)).otherwise(0)).alias("contcanalingresoproducto_null"),
  sum(when(col("grupooferta")==lit("ARRASTRE"),lit(1)).otherwise(0)).alias("contgrupooferta_arrastre"),
  sum(when(col("grupooferta")==lit("DEMO + GVTAS"),lit(1)).otherwise(0)).alias("contgrupooferta_demo_gvtas"),
  sum(when(col("grupooferta")==lit("NUEVA COLECCION"),lit(1)).otherwise(0)).alias("contgrupooferta_nueva_coleccion"),
  sum(when(col("grupooferta")==lit("PROMOCION USUARIO"),lit(1)).otherwise(0)).alias("contgrupooferta_promocion_usuario"),
  sum(when(col("grupooferta").isNull(),lit(1)).otherwise(0)).alias("contgrupooferta_null"),
  sum(when(col("marca")==lit("CYZONE"),lit(1)).otherwise(0)).alias("contmarca_cyzone"),
  sum(when(col("marca")==lit("ESIKA"),lit(1)).otherwise(0)).alias("contmarca_esika"),
  sum(when(col("marca")==lit("GENERICA"),lit(1)).otherwise(0)).alias("contmarca_generica"),
  sum(when(col("marca")==lit("LBEL"),lit(1)).otherwise(0)).alias("contmarca_lbel"),
  sum(when(col("unidadnegocio")==lit("ACCESORIOS"),lit(1)).otherwise(0)).alias("contunidadnegocio_accesorios"),
  sum(when(col("unidadnegocio")==lit("APOYO"),lit(1)).otherwise(0)).alias("contunidadnegocio_apoyo"),
  sum(when(col("unidadnegocio")==lit("COSMETICOS"),lit(1)).otherwise(0)).alias("contunidadnegocio_cosmeticos"),
  sum(when(col("unidadnegocio")==lit("HOGAR"),lit(1)).otherwise(0)).alias("contunidadnegocio_hogar"),
  sum(when(col("unidadnegocio")==lit("MODA"),lit(1)).otherwise(0)).alias("contunidadnegocio_moda"),
  coalesce(sum("realanulmnneto"),lit(0)).alias("sum_realanulmnneto"),
  coalesce(sum("realdevmnneto"),lit(0)).alias("sum_realdevmnneto"),
  coalesce(sum("realuuanuladas"),lit(0)).alias("sum_realuuanuladas"),
  coalesce(sum("realuudevueltas"),lit(0)).alias("sum_realuudevueltas"),
  coalesce(sum("realuufaltantes"),lit(0)).alias("sum_realuufaltantes"),
  coalesce(sum("realuuvendidas"),lit(0)).alias("sum_realuuvendidas"),
  coalesce(sum("realvtamnfaltneto"),lit(0)).alias("sum_realvtamnfaltneto"),
  coalesce(sum("realvtamnneto"),lit(0)).alias("sum_realvtamnneto"),
  coalesce(sum("realvtamncatalogo"),lit(0)).alias("sum_realvtamncatalogo"),
  coalesce(sum("realvtamnfaltcatalogo"),lit(0)).alias("sum_realvtamnfaltcatalogo"),
  coalesce(sum("descuento"),lit(0)).alias("sum_descuento"),
  coalesce(sum("ahorro"),lit(0)).alias("sum_ahorro"),
  coalesce(sum("preciocatalogo"),lit(0)).alias("sum_preciocatalogo")
)


# In[ ]:


dtt_fvta_cl_gb_pd = dtt_fvta_cl_gb.select("*").toPandas()


# In[ ]:


for col in dtt_fvta_cl_gb_pd.columns:
    print(col)
    if col[:4]=="cont":
        dtt_fvta_cl_gb_pd[col] = dtt_fvta_cl_gb_pd[col].astype("int16")
    elif col[:3]=="sum":
        dtt_fvta_cl_gb_pd[col] = dtt_fvta_cl_gb_pd[col].astype("float32")
    else:
        dtt_fvta_cl_gb_pd[col] = dtt_fvta_cl_gb_pd[col].astype("int32")
dtt_fvta_cl_gb_pd.dtypes


# In[ ]:


dtt_fvta_cl_gb_pd.isnull().sum()


# In[ ]:


import pandas as pd
import pyarrow
import s3fs
import os
import fastparquet
pd.set_option('display.max_columns', None)
os.getcwd()
directory="/root/.aws/"
if not os.path.exists(directory):
    os.mkdir(directory)
    f = open("/root/.aws/config", "w")
    f.write("""[default]
    region = us-east-2""")
    f.close()
    f = open("/root/.aws/credentials", "w")
    f.write("""[default]
    aws_access_key_id = AKIAVRCC7TQFR7R47CFQ
    aws_secret_access_key = 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""")
    f.close()
    os.listdir("/root/.aws")


# In[ ]:


dtt_fvta_cl_gb_pd.to_parquet('s3://belcorp-20191109/features/ds_var_dtt_fvta_cl_01.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


import pandas as pd
import s3fs
import dask.dataframe as dd
import boto3
import fastparquet
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.use_inf_as_na = True


# In[ ]:


pd.__version__


# In[ ]:


maestro_consultora = pd.read_csv('s3://belcorp-20191109/datathon-belcorp-prueba/maestro_consultora.csv')
maestro_consultora.columns = map(str.lower, maestro_consultora.columns)


# In[ ]:


campana_consultora = pd.read_parquet('s3://belcorp-20191109/features/campana_consultora.parquet.gzip',engine='fastparquet')


# In[ ]:


ffvv_consultora = pd.read_parquet('s3://belcorp-20191109/features/ds_var_dtt_fvta_cl_01.parquet.gzip',engine='fastparquet')


# In[ ]:


predict_submission = pd.read_csv('s3://belcorp-20191109/datathon-belcorp-prueba/predict_submission.csv')
predict_submission.columns = map(str.lower, predict_submission.columns)


# In[ ]:


campana_index = campana_consultora[["campana"]].drop_duplicates().reset_index(drop=True)#.query("campana>=201901")


# In[ ]:


consultora_hist = pd.concat([ 
                        pd.merge(campana_index.assign(key=0), 
                                 maestro_consultora[["idconsultora","campanaingreso"]].assign(key=0),
                                 on ='key').drop("key", axis=1).query('campana>campanaingreso')[["campana","idconsultora"]],
                        predict_submission[["idconsultora"]].assign(campana=201907)
                                     ],sort=True)


# In[ ]:


campana_consultora.head()


# In[ ]:


campana_consultora_hist = consultora_hist.merge(campana_consultora,
                            on =['campana','idconsultora'],
                             how = "left")


# In[ ]:


campana_consultora_hist["cantidadlogueos"].max()


# In[ ]:


campana_consultora_hist.dtypes


# In[ ]:


campana_consultora_hist.fillna(0, inplace=True)
for col in campana_consultora_hist.columns:
    if col not in ["campana","idconsultora"]:
        campana_consultora_hist[col] = campana_consultora_hist[col].astype("int8")
    else:
        campana_consultora_hist[col] = campana_consultora_hist[col].astype("int32")
campana_consultora_hist.dtypes


# In[ ]:


len(consultora_hist)


# In[ ]:


ffvv_consultora_hist = consultora_hist.merge(ffvv_consultora,
                            on =['campana','idconsultora'],
                             how = "left")
ffvv_consultora_hist.fillna(0, inplace=True)


# In[ ]:


for col in ffvv_consultora_hist.columns:
    if col[:4]=="cont":
        ffvv_consultora_hist[col] = ffvv_consultora_hist[col].astype("int16")
    elif col[:3]=="sum":
        ffvv_consultora_hist[col] = ffvv_consultora_hist[col].astype("float32")
    else:
        ffvv_consultora_hist[col] = ffvv_consultora_hist[col].astype("int32")
ffvv_consultora_hist.dtypes


# In[ ]:


campana_consultora_hist[["campana","idconsultora","flagpasopedido"]].to_parquet('s3://belcorp-20191109/features/ds_target_cosultora.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


campana_consultora_hist[["campana","idconsultora","flagpasopedido"]].isnull().sum()


# In[ ]:


def ffvv_lag(df,i):
    num_cols_input = len(df.columns)
    df = df.assign(campana_num=lambda x: x.campana //100 *18 + x.campana % 100)    .merge( ffvv_consultora_hist           [["campana","idconsultora",
            "contpedidos",
"contdistgrupooferta",
"contdistcodigotipooferta",

"sum_preciocatalogo",
"sum_realvtamncatalogo",
"sum_descuento"
]]
           .assign(campana_num=lambda x: x.campana //100 *18 + x.campana % 100+ i).drop("campana",1) ,
          on=["campana_num","idconsultora"],how="left").drop("campana_num",1)
    df.columns = list(df.columns[:num_cols_input])                                             +[str(col) + '_lag_'+ str(i) for col in df.columns][num_cols_input:]
    df.fillna(value=0,inplace=True)
    for col in df.columns:
        if col[:4]=="cont":
            df[col] = df[col].astype("int16")
        elif col[:3]=="sum":
            df[col] = df[col].astype("float32")
        else:
            df[col] = df[col].astype("int32")
    return df


# In[ ]:


ffvv_consultora_hist_lag_f = campana_consultora_hist[["campana","idconsultora"]]
for i in range(1,5):
    ffvv_consultora_hist_lag_f= ffvv_lag(ffvv_consultora_hist_lag_f,i)
ffvv_consultora_hist_lag_f.dtypes


# In[ ]:


def campana_lag(df,i):
    num_cols_input = len(df.columns)
    df = df.assign(campana_num=lambda x: x.campana //100 *18 + x.campana % 100)    .merge( campana_consultora_hist[["campana","idconsultora","flagpasopedido",
                                    "flagpasopedidoweb",
                                    "flagpasopedidomaquillaje"]]\
           .assign(campana_num=lambda x: x.campana //100 *18 + x.campana % 100+ i).drop("campana",1) ,
          on=["campana_num","idconsultora"],how="left").drop("campana_num",1)
    df.columns = list(df.columns[:num_cols_input])                                             +[str(col) + '_lag_'+ str(i) for col in df.columns][num_cols_input:]
    df.fillna(value=0,inplace=True)
    for col in df.columns:
        if col[:4]=="flag":
            df[col] = df[col].astype("int8")
        else:
            df[col] = df[col].astype("int32")
    return df


# In[ ]:


def campana_lag2(df,i):
    num_cols_input = len(df.columns)
    df = df.assign(campana_num=lambda x: x.campana //100 *18 + x.campana % 100)    .merge( campana_consultora_hist           .assign(campana_num=lambda x: x.campana //100 *18 + x.campana % 100+ i).drop("campana",1) ,
          on=["campana_num","idconsultora"],how="left").drop("campana_num",1)
    df.columns = list(df.columns[:num_cols_input])                                             +[str(col) + '_lag_'+ str(i) for col in df.columns][num_cols_input:]
    df.fillna(value=0,inplace=True)
    for col in df.columns:
        if col[:4]=="flag":
            df[col] = df[col].astype("int8")
        else:
            df[col] = df[col].astype("int32")
    return df


# In[ ]:


campana_consultora_hist_lag_f = campana_consultora_hist[["campana","idconsultora"]]
campana_consultora_hist_lag_f= campana_lag2(campana_consultora_hist_lag_f,1)
for i in range(2,10):
    campana_consultora_hist_lag_f= campana_lag(campana_consultora_hist_lag_f,i)
campana_consultora_hist_lag_f.dtypes


# In[ ]:


campana_consultora_hist_lag_final = campana_consultora_hist_lag_f.merge(ffvv_consultora_hist_lag_f,on=["campana","idconsultora"],how="left")


# In[ ]:


campana_consultora_hist_lag_final = campana_consultora_hist_lag_final.query("campana>=201816")


# In[ ]:


ratios = [("contdistgrupooferta","sum_preciocatalogo"),
 ("contdistgrupooferta","sum_realvtamncatalogo"),
 ("sum_realvtamncatalogo","sum_preciocatalogo"),
 ("sum_realvtamncatalogo","sum_descuento")
]


# In[ ]:


for col1,col2 in ratios:
    for i in range(1,5):
        campana_consultora_hist_lag_final[col1 + "_" + col2 + "_ratio_lag_"+str(i)] =         campana_consultora_hist_lag_final[col1 + "_lag_"+str(i)].astype("float32").divide(
        campana_consultora_hist_lag_final[col2 + "_lag_"+str(i)].astype("float32"),fill_value=0)


# In[ ]:


import time


# In[ ]:


list_cols = ["flagpasopedido","flagpasopedidoweb","flagpasopedidomaquillaje"]
for col in list_cols:
    for i in [3,6,9]:
        campana_consultora_hist_lag_final[col + "_lag_totalcamp_"+str(i)] =         (campana_consultora_hist_lag_final[[col + "_lag_"+str(n) for n in range(1,i+1)]].sum(axis=1)).astype("int8")
        campana_consultora_hist_lag_final[col + "_lag_primercamp_"+str(i)] =         (campana_consultora_hist_lag_final[[col + "_lag_"+str(n) for n in range(1,i+1)]].mul(range(1,i+1), axis=1).max(axis=1)).astype("int8")
        campana_consultora_hist_lag_final[col + "_lag_ultimacamp_"+str(i)] =         (campana_consultora_hist_lag_final[[col + "_lag_"+str(n) for n in range(1,i+1)]]         .mul(range(1,i+1), axis=1).replace(0, 10).min(axis=1)).replace(10, 0).astype("int8")
        campana_consultora_hist_lag_final[col + "_lag_ratio_tot_pri_"+str(i)] =         campana_consultora_hist_lag_final[col + "_lag_totalcamp_"+str(i)]         .divide(campana_consultora_hist_lag_final[col + "_lag_primercamp_"+str(i)] ,fill_value=0).astype("float32")
        campana_consultora_hist_lag_final[col + "_lag_ratio_ult_pri_"+str(i)] =         campana_consultora_hist_lag_final[col + "_lag_ultimacamp_"+str(i)]         .divide(campana_consultora_hist_lag_final[col + "_lag_primercamp_"+str(i)],fill_value=0).astype("float32")


# In[ ]:


campana_consultora_hist_lag_final.fillna(value=0,inplace=True)


# In[ ]:


list_cols =["contpedidos",
"contdistgrupooferta",
"contdistcodigotipooferta",

"sum_preciocatalogo",
"sum_realvtamncatalogo",
"sum_descuento"
]
for col in list_cols:
    print(col)
    #start = time.time()
    for i in range(2,5):
        campana_consultora_hist_lag_final[col + "_lag_prom_"+str(i)] =         (campana_consultora_hist_lag_final[[col + "_lag_"+str(n) for n in range(1,i+1)]].sum(axis=1)/i).astype("float32")
        #campana_consultora_hist_lag_final[col + "_lag_max_"+str(i)] = \
        #(campana_consultora_hist_lag_final[[col + "_lag_"+str(i) for i in range(1,i+1)]].max(axis=1)).astype("float32")
        campana_consultora_hist_lag_final[col + "_lag_min_"+str(i)] =         (campana_consultora_hist_lag_final[[col + "_lag_"+str(n) for n in range(1,i+1)]].min(axis=1)).astype("float32")
    #campana_consultora_hist_lag_final[col + "_lag_slope"] = \
    #(campana_consultora_hist_lag_final[[col + "_lag_"+str(i) for i in range(1,5)]].apply(lambda x: - np.polyfit(range(0,4), x, 1)[0]
    #                                                                                       ,axis=1)).astype("float32")
for col1,col2 in ratios:  
    for i in range(2,5):
        campana_consultora_hist_lag_final[col1 + "_" + col2 + "_ratio_lag_prom_"+str(i)] =         (campana_consultora_hist_lag_final[[col1 + "_" + col2 + "_ratio_lag_"+str(n) for n in range(1,i+1)]].sum(axis=1)/i).astype("float32")
    #campana_consultora_hist_lag_final[col + "_" + col2 + "_ratio_lag_max_"+str(i)] = \
    #(campana_consultora_hist_lag_final[[col + "_" + col2 + "_ratio_lag_"+str(i) for i in range(1,i+1)]].max(axis=1)).astype("float32")
    #campana_consultora_hist_lag_final[col + "_" + col2 + "_ratio_lag_min_"+str(i)] = \
    #(campana_consultora_hist_lag_final[[col + "_" + col2 + "_ratio_lag_"+str(i) for i in range(1,i+1)]].min(axis=1)).astype("float32")
                
for col in list_cols:   
    campana_consultora_hist_lag_final.drop([col + "_lag_"+str(i) for i in range(2,5)],1,inplace=True)
for col1,col2 in ratios:  
    campana_consultora_hist_lag_final.drop([col1 + "_" + col2 + "_ratio_lag_"+str(i) for i in range(2,5)],1,inplace=True)
    #elapsed = time.time() - start
    #print(elapsed)


# In[ ]:


campana_consultora_hist_lag_final.fillna(value=0,inplace=True)


# In[ ]:


campana_consultora_hist_lag_final.columns


# In[ ]:


campana_consultora_hist_lag_final.to_parquet('s3://belcorp-20191109/features/ds_var_campana_consultora_hist_final_1.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


import pandas as pd
import s3fs
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import time
import fastparquet
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


ds_var_maestro_cosultora_01 = pd.read_parquet('s3://belcorp-20191109/features/ds_var_maestro_cosultora_01.parquet.gzip',
                                              engine ='fastparquet')
ds_target_cosultora = pd.read_parquet('s3://belcorp-20191109/features/ds_target_cosultora.parquet.gzip',
                                     engine ='fastparquet')

ds=ds_target_cosultora.merge(ds_var_maestro_cosultora_01,on=["campana","idconsultora"],how="left")
ds=ds.query("campana>=201816")
print(len(ds))


# In[ ]:


ds.dtypes


# In[ ]:


campana_consultora_hist_lag_6 = pd.read_parquet('s3://belcorp-20191109/features/ds_var_campana_consultora_hist_final_1.parquet.gzip',
                                              engine ='fastparquet')


# In[ ]:


campana_consultora_hist_lag_6.dtypes


# In[ ]:


ds=ds.merge(campana_consultora_hist_lag_6,on=["campana","idconsultora"],how="left")
#ds=ds.query("campana>=201816")
print(len(ds))


# In[ ]:


from sklearn.linear_model import LogisticRegression
#name_col="flagpasopedido_lag_"
def apply_coef_logit(ds_i,name_col):
    cols = [name_col+str(i) for i in range(1,10)]
    print(cols)
    logit = LogisticRegression(n_jobs=-1)
    logit.fit(ds_i[cols],ds_i["flagpasopedido"])
    print(logit.coef_[0])
    for i in list(range(1, 10)):
        ds_i[name_col+str(i)]=        ds_i[name_col+str(i)]*logit.coef_[0][i-1]
    return ds_i


# In[ ]:


ds = apply_coef_logit(ds,"flagpasopedido_lag_")
ds = apply_coef_logit(ds,"flagpasopedidoweb_lag_")
ds = apply_coef_logit(ds,"flagpasopedidomaquillaje_lag_")


# In[ ]:


ds["flagpasopedido_suma_ponderada"] =     ds[["flagpasopedido_lag_"+str(i) for i in range(1,10)]].sum(axis=1)
ds["flagpasopedidoweb_suma_ponderada"] =     ds[["flagpasopedidoweb_lag_"+str(i) for i in range(1,10)]].sum(axis=1)
ds["flagpasopedidomaquillaje_suma_ponderada"] =     ds[["flagpasopedidomaquillaje_lag_"+str(i) for i in range(1,10)]].sum(axis=1)


# In[ ]:


ds["flagpasopedido_lag_1"] = [1 if i>0 else 0 for i in ds["flagpasopedido_lag_1"]]
ds["flagpasopedidoweb_lag_1"] = [1 if i>0 else 0 for i in ds["flagpasopedidoweb_lag_1"]]
ds["flagpasopedidomaquillaje_lag_1"] = [1 if i>0 else 0 for i in ds["flagpasopedidomaquillaje_lag_1"]]
ds["flagpasopedido_lag_1"] = ds["flagpasopedido_lag_1"].astype("int8")
ds["flagpasopedidoweb_lag_1"] = ds["flagpasopedido_lag_1"].astype("int8")
ds["flagpasopedidomaquillaje_lag_1"] = ds["flagpasopedido_lag_1"].astype("int8")


# In[ ]:


ds.drop(["flagpasopedido_lag_"+str(i) for i in range(2,10)],1,inplace=True)
ds.drop(["flagpasopedidoweb_lag_"+str(i) for i in range(2,10)],1,inplace=True)
ds.drop(["flagpasopedidomaquillaje_lag_"+str(i) for i in range(2,10)],1,inplace=True)

#ds.drop(["sum_realvtamncatalogo_lag_"+str(i) for i in range(2,4)],1,inplace=True)
#ds.drop(["contdistcodigotipooferta_lag_"+str(i) for i in range(2,13)],1,inplace=True)
#ds.drop(["sum_preciocatalogo_lag_"+str(i) for i in range(2,13)],1,inplace=True)

#ds.drop(["sum_realvtamncatalogo_lag_prom_"+str(i) for i in range(4,13)],1,inplace=True)
#ds.drop(["contdistcodigotipooferta_lag_prom_"+str(i) for i in range(4,13)],1,inplace=True)
#ds.drop(["sum_preciocatalogo_lag_prom_"+str(i) for i in range(4,13)],1,inplace=True)


# In[ ]:


ds.fillna(value=0,inplace=True)
for col in ds.select_dtypes("float64").columns:
    ds[col] = ds[col].astype("float32")


# In[ ]:


ds.to_parquet('s3://belcorp-20191109/features/ds_lag_9_final.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


len(ds.columns)


# In[ ]:


#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import fastparquet
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV,train_test_split
#from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


import joblib
import gzip
import s3fs
fs = s3fs.S3FileSystem(anon=False)
def save_model(model_name,model,auc_cv,X_train,y_train):
    toBePersisted = dict({
        'model_name': model_name,
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'AUC_CV': auc_cv    
    })
    with fs.open("belcorp-20191109/models2/"+model_name+".joblib",'wb') as s3_file:
        joblib.dump(toBePersisted, s3_file, compress=('gzip', 3) )


# In[ ]:


import tempfile
import boto3
import joblib
import s3fs
import pandas as pd
fs = s3fs.S3FileSystem(anon=False)
def read_model(model_path):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(model_path.split("/")[0])
    object = bucket.Object("/".join(model_path.split("/")[1:]))
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        object.download_fileobj(f)
    obj_reloaded = joblib.load(tmp.name)
    return obj_reloaded


# In[ ]:


ds = pd.read_parquet('s3://belcorp-20191109/features/ds_lag_9_final.parquet.gzip',
                                              engine ='fastparquet')


# In[ ]:


ds_test =ds.query("campana==201907")
ds_train =ds.query("campana<=201906")


# In[ ]:


for camp  in ds_train["campana"].unique():
    predict = [ 1 if i>0 else 0 for i in ds_train[ds_train["campana"]==camp]["flagpasopedido_lag_1"]]
    print("{0} AUC Score : {1:.4g} ".format(camp,
                                            metrics.roc_auc_score( ds_train[ds_train["campana"]==camp]["flagpasopedido"] ,
                                                                  predict)))


# In[ ]:


ds_fpp_lag_1_1 = ds_train[ds_train["flagpasopedido_lag_1"]>0]
ds_fpp_lag_1_0 = ds_train[ds_train["flagpasopedido_lag_1"]==0]


# In[ ]:


target = 'flagpasopedido'
idcols = ['campana','idconsultora']
predictors = [x for x in ds.columns if x not in idcols + [target] ]


# In[ ]:


alg=XGBClassifier(n_jobs=-1)
alg.fit(ds_train.query("campana<=201906")[predictors], 
         ds_train.query("campana<=201906")[target],
         eval_metric="auc", verbose=True)


# In[ ]:


print("{0} AUC Score : {1:.6g} ".format(1,
metrics.roc_auc_score(ds_train.query("campana==201906")[target],
                      alg.predict_proba(ds_train.query("campana==201906")[predictors])[:,1])))


# In[ ]:


pd_imp = pd.DataFrame({ 'col' : predictors,
              'importance' : alg.feature_importances_})\
        .sort_values(by='importance',ascending=False)
pd_imp.head(12)


# In[ ]:


pd_imp.to_csv("features6.csv")


# In[ ]:


["campana","idconsultora","flagpasopedido"]+list(pd_imp["col"][:60])


# In[ ]:


ds[["campana","idconsultora","flagpasopedido"]+list(pd_imp["col"][:60])].to_csv("dataset_arnold.csv",index=False)


# In[ ]:


ds[["campana","idconsultora","flagpasopedido"]+list(pd_imp["col"][:60])].to_parquet('s3://belcorp-20191109/features/ds_lag_9_feature_selected60.parquet.gzip', 
                              engine ='fastparquet',compression='gzip')


# In[ ]:


import lightgbm
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


ds = pd.read_parquet('s3://belcorp-20191109/features/ds_lag_9_feature_selected60.parquet.gzip',
                                              engine ='fastparquet')


# In[ ]:


ds_test =ds.query("campana==201907")
ds_train =ds.query("campana<=201906")


# In[ ]:


target = 'flagpasopedido'
idcols = ['campana','idconsultora']
predictors = [x for x in ds.columns if x not in idcols + [target] ]


# In[ ]:


X_train = ds_train[predictors]
X_test = ds_test[predictors]
y_train = ds_train[target]
y_test = ds_test[target]


# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(ds_train[predictors], ds_train[target], 
#                                                    test_size=0.2, random_state=0)


# In[ ]:


# 1st level model
def stacking(model,n_folds,X_train,y_train,X_test):
    #model = RandomForestClassifier()
    # Number of folds
    #n_folds = 3
    # Empty array to store out-of-fold predictions (single column)
    S_train_A_scratch = np.zeros((X_train.shape[0], 1))
    # Empty array to store temporary test set predictions made in each fold
    S_test_temp = np.zeros((X_test.shape[0], n_folds))
    # Empty list to store scores from each fold
    scores = []
    # Split initialization
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Loop across folds
    for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):

        # Split data and target
        X_tr = X_train.iloc[tr_index]
        y_tr = y_train.iloc[tr_index]
        X_te = X_train.iloc[te_index]
        y_te = y_train.iloc[te_index]

        # Fit
        _ = model.fit(X_tr, y_tr)

        # Predict out-of-fold part of train set
        S_train_A_scratch[te_index, :] = model.predict_proba(X_te)[:,1].reshape(-1, 1)

        # Predict test set
        S_test_temp[:, fold_counter] = model.predict_proba(X_test)[:,1]

        # Print score of current fold
        score = metrics.roc_auc_score(y_te, S_train_A_scratch[te_index, :])
        scores.append(score)
        print('fold %d: [%.8f]' % (fold_counter, score))

    # Compute mean of temporary test set predictions to get final test set prediction
    S_test_A_scratch = np.mean(S_test_temp, axis=1).reshape(-1, 1)

    # Mean OOF score + std
    print('\nMEAN:   [%.8f] + [%.8f]' % (np.mean(scores), np.std(scores)))

    # Full OOF score
    # !!! FULL score slightly differs from MEAN score because folds contain
    # different number of examples (404 can't be divided by 3)
    # If we set n_folds=4 scores will be identical for given metric
    print('FULL:   [%.8f]' % (metrics.roc_auc_score(y_train, S_train_A_scratch)))
    return S_train_A_scratch,S_test_A_scratch


# In[ ]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(random_state=1)

train_pred1 , test_pred1 =stacking(model=model1,n_folds=5,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred1_pd=pd.DataFrame(train_pred1)
test_pred1_pd=pd.DataFrame(test_pred1)


# In[ ]:


model2 = XGBClassifier(n_jobs=-1)

train_pred2 , test_pred2 =stacking(model=model2,n_folds=5,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred2_pd=pd.DataFrame(train_pred2)
test_pred2_pd=pd.DataFrame(test_pred2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_jobs=-1)

train_pred3 ,test_pred3=stacking(model=model3,n_folds=5,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred3_pd=pd.DataFrame(train_pred3)
test_pred3_pd=pd.DataFrame(test_pred3)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model4 = ExtraTreesClassifier(n_jobs=-1)

train_pred4 ,test_pred4=stacking(model=model4,n_folds=5,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred4_pd=pd.DataFrame(train_pred4)
test_pred4_pd=pd.DataFrame(test_pred4)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model5 = AdaBoostClassifier()

train_pred5 ,test_pred5=stacking(model=model5,n_folds=5,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred5_pd=pd.DataFrame(train_pred5)
test_pred5_pd=pd.DataFrame(test_pred5)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model7= KNeighborsClassifier()

train_pred7 ,test_pred7=stacking(model=model7,n_folds=5,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred7_pd=pd.DataFrame(train_pred7)
test_pred7_pd=pd.DataFrame(test_pred7)


# In[ ]:


"""from sklearn import svm
model8 = svm.SVC(gamma='scale')
train_pred8 ,test_pred8=stacking(model=model8,n_folds=3,X_train=X_train,X_test=X_test,y_train=y_train)

train_pred8_pd=pd.DataFrame(train_pred8)
test_pred8_pd=pd.DataFrame(test_pred8)"""


# In[ ]:


df = pd.concat([train_pred1_pd, train_pred2_pd ,train_pred3_pd,train_pred4_pd,train_pred5_pd,train_pred7_pd], axis=1)
df_test = pd.concat([test_pred1_pd,test_pred2_pd , test_pred3_pd,test_pred4_pd,test_pred5_pd,test_pred7_pd], axis=1)


# In[ ]:


model = LogisticRegression(random_state=1)
model.fit(df,y_train)
#model.score(df_test, y_test)
print('Final prediction score train: [%.8f]' % metrics.roc_auc_score(y_train, 
                                                         model.predict_proba(df)[:,1]))
#print('Final prediction score test: [%.8f]' % metrics.roc_auc_score(y_test, 
#                                                         model.predict_proba(df_test)[:,1]))


# In[ ]:


X_train_copy = X_train.copy()
X_test_copy = X_test.copy()


# In[ ]:


X_train_copy["prediction"]=model.predict_proba(df)[:,1]
X_train_copy.loc[X_train_copy["campanaultimopedido"]==(X_train_copy["campana_ano"]*100+X_train_copy["campana_periodo"]),"prediction"]=1
X_train_copy.loc[X_train_copy["campanaultimopedido"]<(X_train_copy["campana_ano"]*100+X_train_copy["campana_periodo"]),"prediction"]=0
X_test_copy["prediction"]=model.predict_proba(df_test)[:,1]
X_test_copy.loc[X_test_copy["campanaultimopedido"]==(X_test_copy["campana_ano"]*100+X_test_copy["campana_periodo"]),"prediction"]=1
X_test_copy.loc[X_test_copy["campanaultimopedido"]<(X_test_copy["campana_ano"]*100+X_test_copy["campana_periodo"]),"prediction"]=0


# In[ ]:


print('Final prediction score train: [%.8f]' % metrics.roc_auc_score(y_train, 
                                                         X_train_copy["prediction"]))
#print('Final prediction score test: [%.8f]' % metrics.roc_auc_score(y_test, 
#                                                         X_test_copy["prediction"]))


# In[ ]:


pd_submit=ds_test[["idconsultora","flagpasopedido"]].copy()
pd_submit.loc[:,"flagpasopedido"]=X_test_copy["prediction"]
pd_submit.to_csv("predict_stacking_2.csv",index=False)

