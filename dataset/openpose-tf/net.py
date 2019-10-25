import tensorflow as tf

# get tensors
inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

# forward pass
with tf.Session() as sess:
    heatmaps, pafs = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
        inputs: image
    })
