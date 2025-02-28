import tensorflow as tf
import numpy as np
import glob
from scipy.special import softmax
import matplotlib.pyplot as plt
#import BChargeTagger.train as tr


from featureDict import featureDict

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("Inputs",inputs)
    print("Outputs",outputs)

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        '''
        for layer in layers:
            print(layer)
        print("-" * 50)
        '''

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))
'''
with tf.io.gfile.GFile("frozenModel.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

featureGroups = [
    "cpf","cpf_charge",
    "muon","muon_charge",
    "electron","electron_charge",
    "npf","sv","global"
]

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(
    graph_def=graph_def,
    inputs=[x+":0" for x in featureGroups],
    outputs=["prediction:0"],
    print_graph=True
)

#This whole snippet is taken from the train.py because it may be ok to use the full dataset
#-------------------------------------------------------------------------------------------------------------
files = glob.glob("/nfs/dust/cms/user/mkomm/WbWbX/BChargeTagger/samples/unpacked*.tfrecord")
print ("files: ",len(files))
#files = files[:500]
files = files[:1]
#baggingFraction = 0.5

features = {
    "xcharge": tf.io.FixedLenFeature([1], tf.float32),
    "bPartonCharge": tf.io.FixedLenFeature([1], tf.float32),
    "bHadronCharge": tf.io.FixedLenFeature([1], tf.float32),
    "cHadronCharge": tf.io.FixedLenFeature([1], tf.float32),
    "bDecay": tf.io.FixedLenFeature([7], tf.float32),
}

#This looks like it is adding the jet components to the existing list of features
for name,featureGroup in featureDict.items():
    if 'max' in featureGroup.keys():
        features[name] = tf.io.FixedLenFeature([featureGroup['max']*len(featureGroup['branches'])], tf.float32)
    else:
        features[name] = tf.io.FixedLenFeature([len(featureGroup['branches'])], tf.float32)

'''

'''
def decode_data(raw_data):
    decoded_data = tf.io.parse_example(raw_data,features)
    print("Features before reshaping",features)
    print()
    for name,featureGroup in featureDict.items():
        print("Name",name)
        print("Feature group from raw data",featureGroup)
        if 'max' in featureGroup.keys():
            decoded_data[name] = tf.reshape(decoded_data[name],[-1,featureGroup['max'],len(featureGroup['branches'])])
            #decoded_data[name] = tf.reshape(decoded_data[name],[1,featureGroup['max'],len(featureGroup['branches'])])
    return decoded_data

def setup_pipeline(fileList):
    ds = tf.data.Dataset.from_tensor_slices(fileList)
    batch_array = []
    ds.shuffle(len(fileList),reshuffle_each_iteration=True)
    blk_len = 100
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(
            x, compression_type='GZIP', buffer_size=100000000
        ),
        cycle_length=6, 
        block_length=blk_len, 
        num_parallel_calls=6
    )


    #ds = ds.batch(250) #decode in batches (match block_length?)
    ds = ds.batch(blk_len) #decode in batches (match block_length?)
    ds = ds.map(decode_data, num_parallel_calls=6)
    print("Data after decoding")
    print(ds)
    print()
    num_batch = 0
    num_jet = 0
    b_neg = []
    b0_bar = []
    b0 = []
    b_pos = []

    for y in ds: #This is probably each batch
        num_batch +=1
        
        print("Data label")
        print("Length of batch",len(y))
        print("Batch number",num_batch)
        
        for i in range(len(y)):
            #if num_jet > 20:
            #    break
            num_jet +=1
            print("Overall jet number",num_jet)

            #print(y['cpf_charge'][i])
            batch = {}
            batch_og = {}
            for featureGroup in featureGroups:
                batch[featureGroup] = y[featureGroup][i]
                shape_og = batch[featureGroup].shape

                if 'max' in featureDict[featureGroup].keys():
                    batch[featureGroup] = tf.reshape(batch[featureGroup],[1,shape_og[0],shape_og[1]])
                else:
                    batch[featureGroup] = tf.reshape(batch[featureGroup],[1,shape_og[0]])

            for feature_og in features:
                batch_og[feature_og] = y[feature_og][i]

            #print(y[i])
            logit_jet = frozen_func(**batch)
            prob = softmax(logit_jet)

            
            print("Flags for B mesons evaluated")
            print("Logit values")
            print(logit_jet)
            print("Probability values?")
            print(prob.shape) #Very weird shape, what was the need to add another extra dimension
            print(prob[0][0])
            print("Truth values?")
            print("Xcharge",batch_og['xcharge'])
            print("bPartonCharge",batch_og['bPartonCharge'])
            print("bHadronCharge",batch_og['bHadronCharge'])
            print("cHadronCharge",batch_og['cHadronCharge'])
            print("bdecay",batch_og['bDecay'])

            b_neg.append(prob[0][0][0])
            b0_bar.append(prob[0][0][1])
            b0.append(prob[0][0][2])
            b_pos.append(prob[0][0][3])

            
            #batch_array.append(batch)
    #print(ds['cpf'])
    #print(type(ds))
    #print()
    print("Total number of jets",num_jet)
    
    print("Four B meson arrays")
    print("B-")
    print(b_neg)

    print("B0_bar")
    print(b0_bar)

    print("B0")
    print(b0)

    print("B+")
    print(b_pos)

    ds = ds.unbatch()
    ds = ds.shuffle(50000,reshuffle_each_iteration=True)
    ds = ds.batch(10000)
    ds = ds.prefetch(5)
    

    #print("Final Pipeline setup function data",ds)
    #return ds
    return ds, b_neg, b0, b0_bar, b_pos



ds = tf.data.Dataset.from_tensor_slices(files)
print("Data before decoding")
print(ds)
print()
ds = ds.batch(50) #decode in batches (match block_length?)
dsTest = ds.map(decode_data, num_parallel_calls=6)
print("Data after decoding")
print(dsTest)
print()
print("Number of jets?",len(dsTest))


dsTest, b_neg, b0, b0_bar, b_pos = setup_pipeline(files)
print("Length of entries (number of jets)",len(b_neg),len(b0),len(b0_bar),len(b_pos))

#This is sort of copied from the train.py script as well
fig = plt.figure(figsize=[6.4, 5.8],dpi=300)
plt.hist(b_neg, bins=50, density=True,alpha=0.25,color='blue',label="B-")
plt.hist(b0, bins=50, density=True,alpha=0.25,color='red',label="B0")
plt.hist(b0_bar, bins=50, density=True,alpha=0.25,color='orange',label="B0_bar")
plt.hist(b_pos, bins=50, density=True,alpha=0.25,color='green',label="B+")
plt.legend()
plt.savefig("/nfs/dust/cms/user/paranjpe/Probabilities.png")

#print("Length of batch parameters (number of jets in batch)",len(batch_test))

#---------------------------------------------------------------------------------------------------------------

batch = {}
for featureGroup in featureGroups:
    print("Feature group",featureGroup)
    if 'max' in featureDict[featureGroup].keys():
        batch[featureGroup]=tf.constant(np.zeros((1,featureDict[featureGroup]['max'],len(featureDict[featureGroup]['branches'])),dtype=np.float32))
    else:
        batch[featureGroup]=tf.constant(np.zeros((1,len(featureDict[featureGroup]['branches'])),dtype=np.float32))
    
    print(batch[featureGroup])
print (frozen_func(**batch))
'''
