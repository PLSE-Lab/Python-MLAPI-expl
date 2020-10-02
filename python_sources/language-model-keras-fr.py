# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:05:55 2018

"""
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import collections,random,string,datetime,re, unicodedata,os,time,functools,itertools
import typing,logging,glob,gzip,lzma,shutil
from pathlib import Path

logging.basicConfig(filename='out.log',level=logging.DEBUG)


date_depart=datetime.datetime.now()
########## parametres
#ascii Start of text control
start_char=chr(3)
#ascii End of medium control
end_char=chr(25)


unit_enc=3000
embed_b=128
l_max=200
batch=8
dropout=0.2
dropout_s=0.2


fichier_modele="languagemodel_fr.h5"


load_pretrained=True

loss_weights=[0.2,1,1]




fichier_modele_base,fichier_modele_ext= os.path.os.path.splitext(fichier_modele)


pretrained_paths=[fichier_modele,
                  "../input/language-model-fr-h5/languagemodel_fr.h5",
                  "../input/language-model-keras-fr/languagemodel_fr_u3000e128v724_.h5"

                ]



chemins_textes=["../input",
                r"D:\temp\fr"
                ]


regul_rnn_enc= keras.regularizers.l1_l2(0,0.05)
embeddings_regularizer=keras.regularizers.l1_l2(0.001,0.005)

duree_max=datetime.timedelta(hours=5)
date_limite= date_depart+duree_max




def optimal_gru(*args ,**kvargs):

    if tf.test.is_gpu_available():

        return keras.layers.CuDNNGRU(*args ,**kvargs)

    else:

        return keras.layers.GRU(*args ,reset_after=True,**kvargs)


def split_lines(text):
    lines = ["".join(s).strip() for s in re.findall(r"^(.{10,}[?:!\.])|([?:!\.]{10,})$",text,flags=re.MULTILINE)]


    return lines
def split_phrases(text,skip_start=24,skip_end=5):
    lines = ["".join(s).strip() for s in re.findall(r"^(.{5,}[?:!\.])|([?:!\.]{5,})$",text,flags=re.MULTILINE)]
    lines=lines[skip_start:-skip_end]

    gen_phrases=( re.findall(r".*?[!:.](?=\s[-A-Z]\w+ )",t) for t in lines)
    phrases=[s for s in itertools.chain.from_iterable( gen_phrases) ]


    return phrases
def gen_filereader(filelist):
    filelist=list(filelist)
    while True:
        random.shuffle(filelist)
        max_read=int(2e9)
        for p in filelist:
            s="nul"
            try:
                if p.endswith(".txt.bz2"):
                    with bz2.open(p,"rt",encoding="utf8",errors="ignore") as f:
                        while s!="":
                            s=f.read(max_read).expandtabs()
                            yield unicodedata.normalize('NFKD',s)
                if p.endswith(".txt.gz"):
                    with gzip.open(p,"rt",encoding="utf8",errors="ignore") as f:
                        while s!="":
                            s=f.read(max_read).expandtabs()
                            yield unicodedata.normalize('NFKD',s)
                elif p.endswith(".txt.xz"):
                    with lzma.open(p,"rt",encoding="utf8",errors="ignore") as f:
                        while s!="":
                            s=f.read(max_read).expandtabs()
                            yield unicodedata.normalize('NFKD',s)
                elif p.endswith(".txt"):
                    with open(p,"r",encoding="utf8",errors="ignore") as f:
                        while s!="":
                            s=f.read(max_read).expandtabs()
                            yield unicodedata.normalize('NFKD',s)
            except KeyboardInterrupt:
                break
            except:
                logging.exception("gen_filereader")


def collect_files(chemins_textes):
    textfiles=[]
    for chemin_texte in chemins_textes:
        chemin_texte=os.path.expanduser(chemin_texte)
        chemin_texte=os.path.abspath(chemin_texte)
        textfiles+=glob.glob(os.path.join(chemin_texte,"**/*.txt"),
                             recursive=True)
        textfiles+=glob.glob(os.path.join(chemin_texte,"**/*.txt.gz"),
                             recursive=True)
        textfiles+=glob.glob(os.path.join(chemin_texte,"**/*.txt.xv"),
                             recursive=True)
        textfiles+=glob.glob(os.path.join(chemin_texte,"**/*.txt.bz2"),
                             recursive=True)
    return textfiles

def generateur_triplets(textfiles):
    if not isinstance(textfiles,typing.Generator):
        gen_file=gen_filereader(textfiles)
    else:
        gen_file=textfiles
    for text in gen_file:
        phrases=split_phrases(text)
        n=len(phrases)-2
        if n>1:
            for i in  np.random.choice(n,n,False):
                yield phrases[i+1], phrases[i], phrases[i+2]






set_chars=set(chr(c) for c in  range(0x2A0) if unicodedata.category(chr(c))[0] in "NLPSZ")
set_chars.add("\n")
set_chars.update(chr(c) for c in  range(0x1F600,0x1F650))
set_chars.update(chr(c) for c in  range(0x1F345,0x1F380))


set_chars.update("")
set_chars="".join(c for c in  sorted(set_chars) )
set_chars=chr(0)+set_chars+start_char+end_char
set_chars=unicodedata.normalize('NFKC',set_chars)
set_chars="".join(sorted(set(set_chars)))
vocab=len(set_chars)
dico_dec={a:b for a,b in enumerate(set_chars)}
dico_enc={b:a for a,b in enumerate(set_chars)}

chaine_en_config=f"_u{unit_enc}e{embed_b}v{vocab}_"


copy_search_paths=[f"../input/**/{fichier_modele_base}{chaine_en_config}.h5",
f"../input/*/{fichier_modele_base}.h5"  ]

for p in copy_search_paths:
    for f in glob.iglob(p, recursive=True):
        shutil.copy(f,".")

pretrained_paths+=glob.glob(

                  f"../input/**/{fichier_modele_base}{chaine_en_config}.h5",
                  recursive=True)



def encode_string(s,dico_enc=dico_enc):

    return  [dico_enc.get(c,0) for c in s]
def encode_string_list(texts,dico_enc=dico_enc):
    seq_list=[]
    for s in  texts:
        s=start_char+s+end_char

        seqs=   [encode_string(s[1:],dico_enc=dico_enc),
        encode_string(s[:-1],dico_enc=dico_enc)]
        seq_list.append(seqs)
    return seq_list




def gen_batch_tri(textfiles,batch=16,dico_enc=dico_enc):
    gen_file=gen_filereader(textfiles)
    gen=generateur_triplets(gen_file)
    while True:
        entrees=[]
        sortie=[]

        for g in range(batch):
            texts=next(gen)
            seqs=encode_string_list(texts,dico_enc=dico_enc)
            entrees.append([s[0] for s in seqs])
            sortie.append([s[1] for s in seqs])


        yield entrees,sortie



textfiles=collect_files(chemins_textes)
print("fichiers presents:",len(textfiles))
gen=gen_batch_tri(textfiles,batch=batch)
liste=next(gen)
#print(liste[:4])


################ rnn
phrase=keras.Input(shape=(None,),dtype='int16',name='phrase')

embed_l=keras.layers.Embedding(vocab,embed_b,
                               embeddings_initializer="orthogonal",
                               name="embed",
                               embeddings_regularizer=embeddings_regularizer
                               )
embed=embed_l(phrase)
embed=keras.layers.SpatialDropout1D(dropout)(embed)

gru_enc=optimal_gru(unit_enc, return_sequences=True,
                    return_state=False,
                    name="gru_encode",
                    kernel_regularizer=regul_rnn_enc
                    )
enc_p_i=gru_enc(embed)
seqmax=keras.layers.GlobalMaxPool1D( name="seqmax")

enc_status= seqmax(enc_p_i)

dense_o=keras.layers.Dense(vocab,activation='softmax',name="fc_soft")(enc_p_i)

encodeur=keras.Model(inputs=phrase,outputs=[dense_o,enc_status],name="encodeur")




precedente=keras.Input(shape=(None,),dtype='int16',name='precedente')

suivante=keras.Input(shape=(None,),dtype='int16',name='suivante')



enc_p,enc_val=encodeur(phrase)




gru_dec_suiv=optimal_gru(unit_enc, return_sequences=True,
                         return_state=False,name="gru_dec_suiv")
gru_dec_prec=optimal_gru(unit_enc, return_sequences=True,
                         return_state=False,name="gru_dec_prec")


embed_suiv=embed_l(suivante)
embed_suiv=keras.layers.SpatialDropout1D(dropout)(embed_suiv)

embed_prec=embed_l(precedente)
embed_prec=keras.layers.SpatialDropout1D(dropout)(embed_prec)

enc_val=keras.layers.Dropout(dropout_s)(enc_val)
gruo_s=gru_dec_suiv(embed_suiv,initial_state=enc_val)
gruo_s=keras.layers.SpatialDropout1D(dropout)(gruo_s)
gruo_p=gru_dec_prec(embed_prec,initial_state=enc_val)
gruo_p=keras.layers.SpatialDropout1D(dropout)(gruo_p)

dec_s_direct=keras.layers.Dense(vocab,
                                activation='softmax',name="fc_dec_s_direct")(gruo_s)
dec_p_direct=keras.layers.Dense(vocab,
                                activation='softmax',name="fc_dec_p_direct")(gruo_p)

model=keras.Model(inputs=[phrase,precedente,suivante],
                  outputs=[enc_p,dec_p_direct,dec_s_direct],
                  name="lanagage_model_direct")


model.compile('adam',
              "sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"],
              loss_weights=loss_weights
              )

encodeur.summary()
model.summary()

def batch_gen(gen):
    while True:
        try:
            entrees,sorties=next(gen)
            batch= len(entrees)
            entrees=[s for s in itertools.chain.from_iterable(entrees)]
            entrees1=keras.preprocessing.sequence.pad_sequences(entrees,dtype='int16')
            entrees1= entrees1.reshape(batch,3,-1).swapaxes(0,1)
            entrees1=[t  for t in entrees1]
            sorties=[s for s in itertools.chain.from_iterable(sorties)]
            sorties1=keras.preprocessing.sequence.pad_sequences(entrees,dtype='int16')
            sorties1= sorties1.reshape(batch,3,-1,1).swapaxes(0,1)
            sorties1=[t  for t in sorties1]
            yield entrees1,sorties1

        except KeyboardInterrupt:
            break
        except:
            logging.exception("batch_gen")
class termination_date(keras.callbacks.Callback ):
    def __init__(self,end_date):
        self.end_date=end_date
    def on_epoch_end(self, batch, logs=None):
        if datetime.datetime.now()>self.end_date:
            self.model.stop_training = True
            logging.info("end date")



gen1=batch_gen(gen)
entrees,sorties=next(gen1)

gen_val=gen_batch_tri(textfiles,batch=8)
gen_val=batch_gen(gen_val)
gen_test=gen_batch_tri(textfiles,batch=1)
gen_test=batch_gen(gen_test)



fichier_modele_config=f"{fichier_modele_base}{chaine_en_config}{fichier_modele_ext}"


fichier_checkpoints=fichier_modele_base+chaine_en_config
fichier_checkpoints+="{epoch:02d}"
fichier_checkpoints+=fichier_modele_ext
callbacks= [keras.callbacks.ReduceLROnPlateau(
                               patience=6,
                                      min_delta=0.005,
                                      verbose=1,),
           keras.callbacks.ModelCheckpoint(

                                        filepath=fichier_modele_config, verbose=1,
                                    save_best_only=True,period=20),
           keras.callbacks.EarlyStopping(

                                             min_delta=0.001,
                                             patience=15
                                            ),
#           keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger('training.csv',
                                      append=True),
            termination_date(date_limite)
                            ]


if load_pretrained:
    for p in pretrained_paths:
        if os.path.isfile(p):

            model.load_weights(p,by_name=True,skip_mismatch=True)
            print("chargement réussi:",p)

initial_epoch=0
while datetime.datetime.now()<date_limite:
    try:
        history=model.fit_generator(gen1,
                                    initial_epoch=initial_epoch,
                                 steps_per_epoch=100,
                                 epochs=100,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=gen_val,
                                 validation_steps=6

                                 )

        chaine_date_fichier=date_depart.strftime("_%Y_%m_%d_")





        model.save(fichier_modele)
        model.save(f"{fichier_modele_base}{chaine_en_config}{fichier_modele_ext}")
        initial_epoch=history.epoch[-1]



        evalf="eval_status.log"
        try:
            os.remove(evalf+".bak")
        except FileNotFoundError:
            pass

        if os.path.isfile(evalf):
            os.rename(evalf,evalf+".bak")

        with open(evalf,"w") as f:
            evaluations= model.evaluate_generator(gen_test,steps=10)
            for met,ev in zip(model.metrics_names,evaluations):
                        print(f"{met}:{ev} ")
                        print(f"{met}:{ev} ",file=f)




            entree,sorties=next(gen_test)
            preds=model.predict(entree)
            preds_s=["".join(dico_dec[ c] for c in np.argmax(p[0],axis=-1)) for p in  preds]
            for sortie,pre in zip( ("autoencode:","precedent:","suivant:"),preds_s):
                print(sortie,file=f)
                print (pre,file=f)
                print(sortie)
                print (pre)
    except KeyboardInterrupt:
        break
    except:
        logging.exception("train error")

chaine_date_fichier=date_depart.strftime("_%Y_%m_%d_")

model.get_layer("encodeur").save(f"{fichier_modele_base}_encodeur{chaine_en_config}{chaine_date_fichier}.h5")
model.get_layer("encodeur").save(f"{fichier_modele_base}_encodeur{chaine_en_config}.h5")