#!/usr/local/bin/python
import cv2
import scipy
import numpy
import random
import math
import scipy.sparse
import scipy.sparse.linalg

class HazeRemoval(object):
  def __init__(self, patch_size, omega, t_lb, lamb, window_size, epsilon):
    print "init a haze_removal"
    self.patch_size = patch_size
    self.omega = omega
    self.t_lb = t_lb
    self.lamb = lamb
    self.ws = window_size
    self.epsilon = epsilon

  def get_dark_channel(self, in_img):
    if len(in_img.shape) != 3:
      print "error, input mat doesn't have an image-style shape"
      exit(-1)
    height = in_img.shape[0]
    width = in_img.shape[1]
    channel = in_img.shape[2]
    out_mat = numpy.zeros(shape=(height, width))
    for i in xrange(height):
      for j in xrange(width):
        r1 = self.patch_size/2
        r2 = self.patch_size - r1
        out_mat[i, j] = in_img[max(0, i-r1):min(height, i+r2), max(0,j-r1):min(width, j+r2), :].min()
    return out_mat
  
  def get_atmospheric_light(self, dark_channel_img, in_img):
    flat_img = in_img.reshape((dark_channel_img.size, in_img.shape[2]))
    top_idx_list = numpy.argsort(dark_channel_img.flatten())[-int(dark_channel_img.size*0.001):]
    top_intensity = flat_img[top_idx_list.astype(int), :].sum(1)
    return flat_img[top_idx_list[numpy.argsort(top_intensity)[-1]], :]*1.

  def get_transmission(self, A, in_img):
    norm_img = numpy.divide(in_img.reshape((in_img.shape[0]*in_img.shape[1], in_img.shape[2])), A*1.).reshape(in_img.shape)
    return 1.-self.omega*self.get_dark_channel(norm_img)

  def soft_mat(self, t, in_img):
    height, width, channel = in_img.shape
    rad1 = self.ws/2
    rad2 = self.ws - rad1
    idx_mat = numpy.arange(0, height*width).reshape((height, width))
    partial_element_buffer = numpy.zeros((height*width*self.ws**4))
    h_idx_buffer = numpy.zeros((height*width*self.ws**4)).astype(long)
    w_idx_buffer = numpy.zeros((height*width*self.ws**4)).astype(long)
    U = numpy.eye(channel)*self.epsilon
    buffer_end = 0
    for i in xrange(height):
      for j in xrange(width):
        h_range_start = max(i-rad1, 0)
        h_range_end = min(i+rad2, height)
        w_range_start = max(j-rad1, 0)
        w_range_end = min(j+rad2, width)
        pixs = in_img[h_range_start: h_range_end, w_range_start: w_range_end, :].reshape(((h_range_end-h_range_start)*(w_range_end-w_range_start), channel))
        pixs_num = pixs.shape[0]
        pixs_submean = pixs-numpy.tile(pixs.mean(0).flatten(), pixs_num).reshape((pixs_num, channel))
        pixs_covar_inv = numpy.linalg.inv(numpy.dot(pixs_submean.T, pixs_submean) + U)*pixs_num
        L_partial_element = numpy.eye(pixs_num)-1./pixs_num*(1.+numpy.dot(numpy.dot(pixs_submean, pixs_covar_inv), pixs_submean.T))
        #print L_partial_element.shape, pixs_num**2, buffer_end, partial_element_buffer.size
        partial_element_buffer[buffer_end: buffer_end+pixs_num**2] = L_partial_element.flatten()
        tmp = numpy.tile(idx_mat[h_range_start: h_range_end, w_range_start: w_range_end].flatten(), pixs_num).reshape((pixs_num, pixs_num))
        h_idx_buffer[buffer_end: buffer_end+pixs_num**2] = tmp.T.flatten()
        w_idx_buffer[buffer_end: buffer_end+pixs_num**2] = tmp.flatten()
        buffer_end += pixs_num**2
    L = scipy.sparse.csc_matrix((partial_element_buffer, (h_idx_buffer, w_idx_buffer)))
    b = t.flatten()*self.lamb
    T = scipy.sparse.linalg.spsolve(L + self.lamb*scipy.sparse.identity(b.size), b)
    return T.reshape(t.shape)
 
  def get_scene_radiance(self, t, A, in_img):
    flat_img = in_img.reshape((in_img.shape[0]*in_img.shape[1], in_img.shape[2]))
    flat_img1 = numpy.subtract(flat_img, A)
    t_with_lb_expand = numpy.repeat(numpy.maximum(t, self.t_lb*numpy.ones(shape=in_img.shape[1])), in_img.shape[2]).reshape((in_img.shape[0]*in_img.shape[1], in_img.shape[2]))
    return numpy.add(flat_img1/t_with_lb_expand, A).reshape(in_img.shape)

  def remove_haze(self, img, name):
    d_img = hr.get_dark_channel(img)
    cv2.imwrite("d_"+img_name, d_img.astype(int))
    A = self.get_atmospheric_light(d_img, img)
    t_img = self.get_transmission(A, img)
    t_img = self.soft_mat(t_img, img)
    cv2.imwrite("t_"+img_name, (t_img*255).astype(int))
    out_img = self.get_scene_radiance(t_img, A, img)
    cv2.imwrite("out_"+img_name, out_img)

if __name__ == "__main__":
  hr = HazeRemoval(patch_size=15, omega=0.99, t_lb=0.1, lamb=0.0001, window_size=3, epsilon=0.1)
  img_name = "mountain.jpg"
  img = cv2.imread("data/" + img_name)
  out_img = hr.remove_haze(img, img_name)
