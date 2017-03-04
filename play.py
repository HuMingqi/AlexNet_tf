'''
Author: hiocde
Email: hiocde@gmail.com
Start: 2.27.17
Completion: 
Original: 
Domain: Use a trained model to do something such as feature extract.
'''

import tensorflow as tf
import json
#import sys
import input
import alexnet

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', 'G:/machine_learning/dataset/Alexnet_tf/paris',
                            """Data input directory when using a product level model(trained and tested).""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'G:/machine_learning/models/Alexnet_tf/log',
							"""Model checkpoint dir path.""")
tf.app.flags.DEFINE_string('output', 'G:/machine_learning/models/Alexnet_tf/output', 
							'''Model output dir, stores image features''')
MOVING_AVERAGE_DECAY = 0.9999     # equals the value used in training.


def build_feature_lib():
	imageset = input.ImageSet(FLAGS.input, False)	#pre-load datalist
	# open a graph
	with tf.Graph().as_default() as g:
		# Build model(graph)
			# First build a input pipeline(Model's input subgraph).
		images, labels, ids = imageset.next_batch(FLAGS.batch_size)	# Dont need like alexnet.FLAGS.batch_size
		logits = alexnet.inference(images)
		
		# Use our model
		fc1= g.get_tensor_by_name("fc1/fc1:0")	#*** use full name: variable_scope name + var/op name + output index
		fc2= g.get_tensor_by_name("fc2/fc2:0")
		softmax= tf.nn.softmax(logits)	#softmax = exp(logits) / reduce_sum(exp(logits), dim), dim=-1 means add along line.

		# Run our model
		steps= int(imageset.num_exps/FLAGS.batch_size)
			# Restore the moving average version of the learned variables for better effect.
		EMA = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
		variables_to_restore = EMA.variables_to_restore()
		# for name in variables_to_restore:
		# 	print(name)
		saver = tf.train.Saver(variables_to_restore)
		
		with tf.Session() as sess:
			# Restore model from checkpoint.
			# Note!: checkpoint file not a single file, so don't use like this:
			# saver.restore(sess, '/path/to/model.ckpt-1000.index')
				# Don't forget launch queue.
			tf.train.start_queue_runners(sess=sess)

			ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
				# model_checkpoint_path looks something like: /path/to/model.ckpt-1000	
			if ckpt and ckpt.model_checkpoint_path:      		
	  			saver.restore(sess, ckpt.model_checkpoint_path)
	  			# print(ckpt.model_checkpoint_path)
      		
			fc1_list=fc2_list=softmax_list=[]
			id_list=[]
			for step in range(steps):			
				_ids, _fc1, _fc2, _softmax= sess.run([ids,fc1,fc2,softmax])	#return nd-array
				put_2darray(_fc1, fc1_list)
				put_2darray(_fc2, fc2_list)
				put_2darray(_softmax, softmax_list)
				print(_ids)
				for id in _ids.tolist():
					id_list.append(id)
				
		save(id_list, fc1_list, FLAGS.output+'/fc1_features.json')
		save(id_list, fc2_list, FLAGS.output+'/fc2_features.json')
		save(id_list, softmax_list, FLAGS.output+'/softmax_features.json')


# def extract_batch_features():
# 	images, labels, ids = imageset.next_batch(FLAGS.batch_size)	# Dont need like alexnet.FLAGS.batch_size
# 	logits = alexnet.inference(images)
# 	softmax= tf.nn.softmax(logits)	#softmax = exp(logits) / reduce_sum(exp(logits), dim), dim=-1 means add along line.
# 	fc1= tf.get_default_graph().get_tensor_by_name("fc1:0")
# 	fc2= tf.get_default_graph().get_tensor_by_name("fc2:0")
# 	return fc1, fc2, softmax, ids

def put_2darray(_2darray, li):
	_li= _2darray.tolist()
	for line in _li:
		li.append(line)

def save(id_list, feature_list, file_n):
    '''save as json obj like {id:feature, ...}, A feature is a vector'''
    dic = dict(list(zip(id_list, feature_list)))
    file = open(file_n, 'w')
    file.write(json.dumps(dic))
    file.close()


def main(argv=None):
	build_feature_lib()

if __name__ == '__main__':	
	# ...About the argv parse, is a XuanXue.
	# I only know you can script it like this:
	# python3 play.py --argv1=val1 --argv2==val2 ...
	# to run it and the FLAGS is right.
	# if you're interested, to see the source of tf.app.run.

	tf.app.run()
	# or tf.app.run(argv=sys.argv)
	# or tf.app.run(build_feature_lib) PS:Add a useless arg in build_feature_lib.
	# or tf.app.run(build_feature_lib, argv=sys.argv)
	# All ok, so weird!