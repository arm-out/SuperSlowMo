import numpy as np
import tensorflow as tf
import layers


class SloMoNet(tf.keras.Model):
    def __init__(self, n_frames=12, name="SloMoNet", **kwargs):
        super(SloMoNet, self).__init__(name=name, **kwargs)
        self.t_slices = tf.constant(np.linspace(0, 1, n_frames))
        self.flow_comp_layer = layers.UNet(4, name="flow_comp")
        self.optical_flow = layers.OpticalFlow(name="optical_flow")
        self.output_layer = layers.Output(name="predictions")
        self.warp_layers = [layers.BackWarp()] * 2

    def call(self, inputs, training=False, **kwargs):
        frames_0, frames_1, frames_i = inputs

        # extract frame t coefficient
        t_indices = tf.gather(self.t_slices, frames_i)
        t_indices = tf.cast(t_indices, dtype=tf.float32)
        t_indices = t_indices[:, tf.newaxis, tf.newaxis, tf.newaxis]

        # compute flow
        flow_input = tf.concat([frames_0, frames_1], axis=3)
        flow_out = self.flow_comp_layer(flow_input)

        # optical flow
        flow_01, flow_10 = flow_out[:, :, :, :2], flow_out[:, :, :, 2:]
        optical_input = [frames_0, frames_1, flow_01, flow_10, t_indices]
        f_t0, v_t0, f_t1, v_t1, g_i0_ft0, g_i1_ft1 = self.optical_flow(
            optical_input)

        # compute predictions
        preds_input = [frames_0, f_t0, v_t0, frames_1, f_t1, v_t1, t_indices]
        predictions = self.output_layer(preds_input)

        # backwarp for loss
        warp0 = self.warp_layers[1]([frames_1, flow_01])
        warp1 = self.warp_layers[0]([frames_0, flow_10])
        losses_output = [flow_01, flow_10, warp0, warp1, g_i0_ft0, g_i1_ft1]
        return predictions, losses_output
