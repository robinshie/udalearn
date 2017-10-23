import tensorflow as tf
'''
weights = tf.Variable(tf.truncated_normal([2,3]))
biases = tf.Variable(tf.truncated_normal([3]))

save_path = './model.ckpt'

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(biases))
    
    saver.save(sess,save_path)
'''
tf.reset_default_graph()
save_path = './model.ckpt'
weights = tf.Variable(tf.truncated_normal([2,3]))
biases = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,save_path)
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(biases))