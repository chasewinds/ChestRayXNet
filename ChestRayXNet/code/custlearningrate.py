import tensorflow as tf
import logging
import numpy as np

class CustLearningRate:
	@classmethod
	def create_policy(cls, config_str, global_step, steps_per_epoch):
		name_, params_ = config_str.split("::")
		logging.info("Learning rate policy: %s" % name_)
		if name_ == "Interval":
			epochs_lr = np.array(eval(params_))
			return cls.IntervalLearningRate(epochs_lr, global_step, steps_per_epoch)
		elif name_ == "ExpDecay":
			params_ =  eval(params_)
			return cls.ExpDecayLearningRate(float(params_['init_lr']),
											global_step,
											steps_per_epoch,
											float(params_['decay_rate']))
		elif name_ == "PowerDecay":
			params_ = eval(params_)
			init_lr = float(params_['init_lr'])
			decay_rate = float(params_['decay_rate'])

			decay_per_epochs = int(params_['decay_per_epochs'])
			cur_epochs = tf.cast(tf.round(tf.div(global_step, steps_per_epoch)), tf.float32)
			decay_stairs = tf.cast(tf.round(tf.div(cur_epochs, decay_per_epochs)), tf.float32)
			return tf.multiply(init_lr, tf.pow(decay_rate, decay_stairs))
		else:
			raise ValueError("Unknow is learning policy: %s" % name_)



	@classmethod
	def IntervalLearningRate(cls, epochs_lr, global_step, steps_per_epoch):
		a = [x[0] for x in epochs_lr]
		epochs_invervals = map(lambda i: sum(a[:i+1]),  range(len(a)))
		lr_list = [x[1] for x in epochs_lr]
		logging.info("epochs_invervals: %s, lr_list: %s" % (epochs_invervals, lr_list))

		epochs_invervals= tf.constant(epochs_invervals, tf.float32)
		logging.info("epochs_invervals.shape: %s" %  epochs_invervals.get_shape())
		lr_list = tf.constant(lr_list, tf.float32)

		epoch = tf.div(tf.cast(global_step, dtype=tf.float32), tf.constant(steps_per_epoch, dtype=tf.float32))
		false_const = tf.constant(False, tf.bool)

		# def get_threshold_idx(self, thres, v):
		# 	for i, t in enumerate(thres):
		# 		if v < t:
		# 			return i
		# 	return len(thres)-1

		i = tf.constant(0, tf.int32)
		cond = lambda i, x, t: tf.cond(tf.less(i, tf.size(x)), lambda: tf.less(x[i], t), lambda: false_const)
		while_op = tf.while_loop(cond,
							 lambda i, t, x: (tf.add(i, 1), t, x),
							 [i, epochs_invervals, epoch])

		idx = while_op[0]

		return lr_list[idx]

	@classmethod
	def ExpDecayLearningRate(cls, init_lr,  global_step, steps_per_epoch, decay_rate):
		logging.info("ExpDecayLearningRate: init_lr: %f, steps_per_epoch: %s, decay_rate: %s", init_lr, steps_per_epoch, decay_rate)
		n_epochs = tf.cast(tf.round(tf.div(global_step, steps_per_epoch)), tf.float32)
		return tf.multiply(init_lr, tf.exp(
							tf.multiply(
								-1.0*decay_rate,
								n_epochs)))
