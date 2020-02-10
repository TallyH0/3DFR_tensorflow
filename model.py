import tensorflow as tf
import time
import os
import cv2
import numpy as np
from data_loader import Data_loader_sequence


class Net_3DFR():
  def __init__(self, h, w, c, num_history=50):
    # self.data_loader = Data_loader_sequence(dir_image, h, w, True)
    self.img_h = h
    self.img_w = w
    self.img_c = c
    self.num_history = num_history
    self.loaders = []
    gpu_option = tf.GPUOptions(allow_growth=True)
    self.config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_option)


  def build(self, batch_size, max_epoch, train=True, loaders=None):
    if train:
      self.max_epoch = max_epoch
      if not loaders:
        print('For training, there should be data_loader')
        exit(-1)
      else:
        self.loaders = loaders
      
      self.step_per_epoch = 0
      for loader in loaders:
        self.step_per_epoch += loader.size

      self.placeholder_history = tf.placeholder(tf.float32, [batch_size, self.num_history, self.img_h, self.img_w, self.img_c], name='history')
      self.placeholder_temporal_mean = tf.placeholder(tf.float32, [batch_size, 1, self.img_h, self.img_w, self.img_c], name='median')
      self.placeholder_current = tf.placeholder(tf.float32, [batch_size, 1, self.img_h, self.img_w, self.img_c], name='current')
      self.placeholder_label = tf.placeholder(tf.float32, [batch_size, 1, self.img_h, self.img_w, 1], name='label')
      self.global_step = tf.Variable(0, False)
      
      n_c = 8
      with tf.variable_scope('Avalanche_Feat', reuse=tf.AUTO_REUSE):
        branch0_1 = tf.nn.relu(tf.layers.conv3d(self.placeholder_history, n_c, [5, 1, 1], [5, 1, 1], 'same'))
        branch0_2 = tf.nn.relu(tf.layers.conv3d(self.placeholder_history, n_c, [5, 3, 3], [5, 1, 1], 'same'))
        branch0_3 = tf.nn.relu(tf.layers.conv3d(self.placeholder_history, n_c, [5, 5, 5], [5, 1, 1], 'same'))
        
        branch0_mean = tf.reduce_mean(branch0_1 + branch0_2 + branch0_3, axis=-1, keepdims=True)
        branch0_final = tf.layers.conv3d(branch0_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
        branch1_1 = tf.nn.relu(tf.layers.conv3d(branch0_final, n_c, [5, 1, 1], [5, 1, 1], 'same'))
        branch1_2 = tf.nn.relu(tf.layers.conv3d(branch0_final, n_c, [5, 3, 3], [5, 1, 1], 'same'))
        branch1_3 = tf.nn.relu(tf.layers.conv3d(branch0_final, n_c, [5, 5, 5], [5, 1, 1], 'same'))
        
        branch1_mean = tf.reduce_mean(branch1_1 + branch1_2 + branch1_3, axis=-1, keepdims=True)
        branch1_final = tf.layers.conv3d(branch1_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
        branch2_1 = tf.nn.relu(tf.layers.conv3d(branch1_final, n_c, [2, 1, 1], [2, 1, 1], 'same'))
        branch2_2 = tf.nn.relu(tf.layers.conv3d(branch1_final, n_c, [2, 3, 3], [2, 1, 1], 'same'))
        branch2_3 = tf.nn.relu(tf.layers.conv3d(branch1_final, n_c, [2, 5, 5], [2, 1, 1], 'same'))
        
        branch2_mean = tf.reduce_mean(branch2_1 + branch2_2 + branch2_3, axis=-1, keepdims=True)
        branch2_final = tf.layers.conv3d(branch2_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
      with tf.variable_scope('Con_Feat', reuse=tf.AUTO_REUSE):
        con_branch0_1 = tf.layers.conv3d(self.placeholder_current, n_c, [1, 1, 1], [1, 1, 1], 'same')
        con_branch0_2 = tf.layers.conv3d(self.placeholder_current, n_c, [1, 3, 3], [1, 1, 1], 'same')
        con_branch0_3 = tf.layers.conv3d(self.placeholder_current, n_c, [1, 5, 5], [1, 1, 1], 'same')
        con_branch0_mean= tf.reduce_mean(con_branch0_1 + con_branch0_2 + con_branch0_3, axis=-1, keepdims=True)
        con_branch0_final = tf.layers.conv3d(con_branch0_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
      msfeat = tf.concat([branch2_final, con_branch0_final, self.placeholder_temporal_mean], axis=-1)
      
      with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        nc_enc = 8
        enc0_1 = tf.layers.conv3d(msfeat, nc_enc, [1, 3, 3], [1, 1, 1], 'same')
        enc0_2 = tf.layers.conv3d(enc0_1, nc_enc * 2, [1, 3, 3], [1, 1, 1], 'same')
        enc0_pool = tf.layers.max_pooling3d(enc0_2, [1, 2, 2], [1, 2, 2], 'same')
        enc0_final = tf.nn.relu(enc0_pool)
        
        enc1_1 = tf.layers.conv3d(enc0_final, nc_enc * 2, [1, 3, 3], [1, 1, 1], 'same')
        enc1_2 = tf.layers.conv3d(enc1_1, nc_enc * 4, [1, 3, 3], [1, 1, 1], 'same')
        enc1_pool = tf.layers.max_pooling3d(enc1_2, [1, 2, 2], [1, 2, 2], 'same')
        enc1_final = tf.nn.relu(enc1_pool)
        
        enc2_1 = tf.layers.conv3d(enc1_final, nc_enc * 4, [1, 3, 3], [1, 1, 1], 'same')
        enc2_2 = tf.layers.conv3d(enc2_1, nc_enc * 8, [1, 3, 3], [1, 1, 1], 'same')
        enc2_pool = tf.layers.max_pooling3d(enc2_2, [1, 2, 2], [1, 2, 2], 'same')
        enc2_final = tf.nn.relu(enc2_pool)
        
      with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
        op_upsample = tf.keras.layers.UpSampling3D((1, 2, 2))
        dec0_up = op_upsample(enc2_final)
        dec0_1 = tf.layers.conv3d_transpose(dec0_up, nc_enc*8, [1, 3, 3], [1, 1, 1], 'same')
        dec0_2 = tf.nn.relu(tf.layers.conv3d_transpose(dec0_1, nc_enc*4, [1, 3, 3], [1, 1, 1], 'same'))
        
        dec1_up = op_upsample(dec0_2)
        dec1_1 = tf.layers.conv3d_transpose(dec1_up, nc_enc*4, [1, 3, 3], [1, 1, 1], 'same')
        dec1_2 = tf.nn.relu(tf.layers.conv3d_transpose(dec1_1, nc_enc*2, [1, 3, 3], [1, 1, 1], 'same'))
        
        dec2_up = op_upsample(dec1_2)
        dec2_1 = tf.layers.conv3d_transpose(dec2_up, nc_enc*2, [1, 3, 3], [1, 1, 1], 'same')
        dec2_2 = tf.nn.relu(tf.layers.conv3d_transpose(dec2_1, nc_enc, [1, 3, 3], [1, 1, 1], 'same'))
        
        dec_final = tf.layers.conv3d_transpose(dec2_2, 1, [1, 1, 1], [1, 1, 1], 'same')
        
      
      with tf.name_scope('loss'):
        preprocess_label = tf.div(self.placeholder_label, 255)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_final, labels=preprocess_label))
        
      lr = tf.train.piecewise_constant(self.global_step, [self.step_per_epoch * 20, self.step_per_epoch * 40, self.step_per_epoch * 60], [6e-4, 4e-4, 2e-4, 1e-4])
      with tf.variable_scope('opt'):
        opt = tf.train.GradientDescentOptimizer(lr)
        self.train_op = opt.minimize(loss, global_step = self.global_step)
        
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('lr', lr)
      tf.summary.image('gt', tf.squeeze(self.placeholder_label, axis=1))
      tf.summary.image('out', tf.squeeze(tf.sigmoid(dec_final), axis=1))
      tf.summary.image('frame', tf.squeeze(self.placeholder_current, axis=1))
      
      self.merged = tf.summary.merge_all()
      self.saver = tf.train.Saver(max_to_keep=1)
      self.sess = tf.Session(config=self.config)

    else:
      self.placeholder_history = tf.placeholder(tf.float32, [batch_size, self.num_history, self.img_h, self.img_w, self.img_c], name='history')
      self.placeholder_temporal_mean = tf.placeholder(tf.float32, [batch_size, 1, self.img_h, self.img_w, self.img_c], name='median')
      self.placeholder_current = tf.placeholder(tf.float32, [batch_size, 1, self.img_h, self.img_w, self.img_c], name='current')
      
      n_c = 8
      with tf.variable_scope('Avalanche_Feat', reuse=tf.AUTO_REUSE):
        branch0_1 = tf.nn.relu(tf.layers.conv3d(self.placeholder_history, n_c, [5, 1, 1], [5, 1, 1], 'same'))
        branch0_2 = tf.nn.relu(tf.layers.conv3d(self.placeholder_history, n_c, [5, 3, 3], [5, 1, 1], 'same'))
        branch0_3 = tf.nn.relu(tf.layers.conv3d(self.placeholder_history, n_c, [5, 5, 5], [5, 1, 1], 'same'))
        
        branch0_mean = tf.reduce_mean(branch0_1 + branch0_2 + branch0_3, axis=-1, keepdims=True)
        branch0_final = tf.layers.conv3d(branch0_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
        branch1_1 = tf.nn.relu(tf.layers.conv3d(branch0_final, n_c, [5, 1, 1], [5, 1, 1], 'same'))
        branch1_2 = tf.nn.relu(tf.layers.conv3d(branch0_final, n_c, [5, 3, 3], [5, 1, 1], 'same'))
        branch1_3 = tf.nn.relu(tf.layers.conv3d(branch0_final, n_c, [5, 5, 5], [5, 1, 1], 'same'))
        
        branch1_mean = tf.reduce_mean(branch1_1 + branch1_2 + branch1_3, axis=-1, keepdims=True)
        branch1_final = tf.layers.conv3d(branch1_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
        branch2_1 = tf.nn.relu(tf.layers.conv3d(branch1_final, n_c, [2, 1, 1], [2, 1, 1], 'same'))
        branch2_2 = tf.nn.relu(tf.layers.conv3d(branch1_final, n_c, [2, 3, 3], [2, 1, 1], 'same'))
        branch2_3 = tf.nn.relu(tf.layers.conv3d(branch1_final, n_c, [2, 5, 5], [2, 1, 1], 'same'))
        
        branch2_mean = tf.reduce_mean(branch2_1 + branch2_2 + branch2_3, axis=-1, keepdims=True)
        branch2_final = tf.layers.conv3d(branch2_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
      with tf.variable_scope('Con_Feat', reuse=tf.AUTO_REUSE):
        con_branch0_1 = tf.layers.conv3d(self.placeholder_current, n_c, [1, 1, 1], [1, 1, 1], 'same')
        con_branch0_2 = tf.layers.conv3d(self.placeholder_current, n_c, [1, 3, 3], [1, 1, 1], 'same')
        con_branch0_3 = tf.layers.conv3d(self.placeholder_current, n_c, [1, 5, 5], [1, 1, 1], 'same')
        con_branch0_mean= tf.reduce_mean(con_branch0_1 + con_branch0_2 + con_branch0_3, axis=-1, keepdims=True)
        con_branch0_final = tf.layers.conv3d(con_branch0_mean, n_c, [2, 1, 1], [2, 1, 1], 'same')
        
      msfeat = tf.concat([branch2_final, con_branch0_final, self.placeholder_temporal_mean], axis=-1)
      
      with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        nc_enc = 8
        enc0_1 = tf.layers.conv3d(msfeat, nc_enc, [1, 3, 3], [1, 1, 1], 'same')
        enc0_2 = tf.layers.conv3d(enc0_1, nc_enc * 2, [1, 3, 3], [1, 1, 1], 'same')
        enc0_pool = tf.layers.max_pooling3d(enc0_2, [1, 2, 2], [1, 2, 2], 'same')
        enc0_final = tf.nn.relu(enc0_pool)
        
        enc1_1 = tf.layers.conv3d(enc0_final, nc_enc * 2, [1, 3, 3], [1, 1, 1], 'same')
        enc1_2 = tf.layers.conv3d(enc1_1, nc_enc * 4, [1, 3, 3], [1, 1, 1], 'same')
        enc1_pool = tf.layers.max_pooling3d(enc1_2, [1, 2, 2], [1, 2, 2], 'same')
        enc1_final = tf.nn.relu(enc1_pool)
        
        enc2_1 = tf.layers.conv3d(enc1_final, nc_enc * 4, [1, 3, 3], [1, 1, 1], 'same')
        enc2_2 = tf.layers.conv3d(enc2_1, nc_enc * 8, [1, 3, 3], [1, 1, 1], 'same')
        enc2_pool = tf.layers.max_pooling3d(enc2_2, [1, 2, 2], [1, 2, 2], 'same')
        enc2_final = tf.nn.relu(enc2_pool)
        
      with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
        op_upsample = tf.keras.layers.UpSampling3D((1, 2, 2))
        dec0_up = op_upsample(enc2_final)
        dec0_1 = tf.layers.conv3d_transpose(dec0_up, nc_enc*8, [1, 3, 3], [1, 1, 1], 'same')
        dec0_2 = tf.nn.relu(tf.layers.conv3d_transpose(dec0_1, nc_enc*4, [1, 3, 3], [1, 1, 1], 'same'))
        
        dec1_up = op_upsample(dec0_2)
        dec1_1 = tf.layers.conv3d_transpose(dec1_up, nc_enc*4, [1, 3, 3], [1, 1, 1], 'same')
        dec1_2 = tf.nn.relu(tf.layers.conv3d_transpose(dec1_1, nc_enc*2, [1, 3, 3], [1, 1, 1], 'same'))
        
        dec2_up = op_upsample(dec1_2)
        dec2_1 = tf.layers.conv3d_transpose(dec2_up, nc_enc*2, [1, 3, 3], [1, 1, 1], 'same')
        dec2_2 = tf.nn.relu(tf.layers.conv3d_transpose(dec2_1, nc_enc, [1, 3, 3], [1, 1, 1], 'same'))
        
        dec_final = tf.layers.conv3d_transpose(dec2_2, 1, [1, 1, 1], [1, 1, 1], 'same')
        self.result = tf.sigmoid(dec_final)
      
      self.saver = tf.train.Saver(max_to_keep=1)
      self.sess = tf.Session(config=self.config)

  def train(self, dir_model):
    checkpoint = tf.train.latest_checkpoint(dir_model)
    summary_writer = tf.summary.FileWriter(dir_model)
    if checkpoint:
      self.saver.restore(self.sess, checkpoint)
    else:
      self.sess.run(tf.global_variables_initializer())
      self.saver.save(self.sess, os.path.join(dir_model, 'model.ckpt'))

    cnt = 0
    while cnt < self.max_epoch * self.step_per_epoch:
      try:
        for loader in self.loaders:
          cnt += 1
          b_image, b_current, b_label, b_median = loader.batch(1)
          feed_dict = {
            self.placeholder_history : b_image,
            self.placeholder_label : [b_label],
            self.placeholder_current : [b_current],
            self.placeholder_temporal_mean : [b_median]
          }
          _, summary, step = self.sess.run([self.train_op, self.merged, self.global_step], feed_dict = feed_dict)
          if step % 97 == 0:
            summary_writer.add_summary(summary, step)
          
          if step % 9999 == 0:
            self.saver.save(self.sess, os.path.join(dir_model, 'model.ckpt'), global_step = self.global_step)

      except Exception as e:
        print(e)
        break

    self.saver.save(self.sess, os.path.join(dir_model, 'model.ckpt'), global_step = self.global_step)

  def load_model(self, dir_model):
    checkpoint = tf.train.latest_checkpoint(dir_model)
    if not checkpoint:
      print('No checkpoint')
      exit(-1)
    self.saver.restore(self.sess, checkpoint)

  def test(self, loader, dir_output):
    for index in range(loader.size):
      b_image, b_current, b_label, b_median = loader.batch(1, index)
      feed_dict = {
        self.placeholder_history : b_image,
        self.placeholder_current : [b_current],
        self.placeholder_temporal_mean : [b_median]
      }

      result = self.sess.run(self.result, feed_dict = feed_dict)
      colormap = np.array(np.squeeze(result * 255), np.uint8)
      colormap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)
      b_current_rgb = cv2.cvtColor(np.array(b_current[0], np.uint8), cv2.COLOR_GRAY2BGR)
      combine = np.hstack((b_current_rgb, colormap))
      fname = os.path.join(dir_output, '%04d.png' % index)
      cv2.imwrite(fname, combine)

      # return np.array(np.squeeze(result * 255), np.uint8)


if __name__ == '__main__':
  pass