
import tensorflow as tf, sys
import os

img_path = '../input/10-monkey-species/validation/validation'

with tf.gfile.FastGFile('../input/retrained-inception-model/retrained_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,name='')


predictions = []
true_value = []
with tf.Session() as sess:
    for i in os.listdir(img_path):
        for j in os.listdir(img_path+'/'+i):
            img_data = tf.gfile.FastGFile(img_path+'/'+i+'/'+j,'rb').read()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            true_value.append(int(i[-1]))
            predictions.append(sess.run(softmax_tensor,{'DecodeJpeg/contents:0':img_data}).argmax())


from sklearn.metrics import confusion_matrix
print(confusion_matrix(true_value,predictions))


from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(true_value,predictions)*100,'%')

