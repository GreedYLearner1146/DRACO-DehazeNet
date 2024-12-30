from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import layers

### Hyperparameters ###
num_epochs = 80  
step = 0

tf.keras.backend.clear_session() # Clear the session first.
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Optimizer.
######################################################

for epoch in range(num_epochs):
    print("starting epoch ",epoch)
    for rp, rn in zip(train_clear, train_haze):
      with tf.GradientTape(persistent=True) as tape:      # Model gradient tape.
        ################################## Network ###########################################
        haze = tf.expand_dims(rn, axis=0)
        clear = tf.expand_dims(rp,axis=0)
        dehazed,_,_,_,_ = DPE_Net_contrastive([haze,clear])   # The embedded outputs of the anchor, positive and negative.
        loss_mae = tf.keras.ops.mean(tf.abs(dehazed - clear), axis=-1)               # mae loss definition.
        ssim_loss = - tf.reduce_mean(tf.image.ssim(dehazed, clear, 2.0))  # ssim loss definition.
        _, preds1, preds2, preds3, preds4 = DPE_Net_contrastive([haze,clear])   # The embedded outputs of the anchor, positive and negative.
        # Quadruplet Loss Function.
        anchor, anchor2, positive, negative = preds1, preds2, preds3, preds4   # Preds1 is anchor, preds2 is positive, preds3 is negative.
        p_dist = tf.reduce_mean(tf.abs(intermediate_layer_model(anchor) - intermediate_layer_model(positive)), axis=-1)
        n_dist = tf.reduce_mean(tf.abs(intermediate_layer_model(anchor) - intermediate_layer_model(negative)), axis=-1)
        n_dist2 = tf.reduce_mean(tf.abs(intermediate_layer_model(anchor2) - intermediate_layer_model(negative)), axis=-1)
        n_dist3 = tf.reduce_mean(tf.abs(intermediate_layer_model(anchor2) - intermediate_layer_model(anchor)), axis=-1)
        loss_Quadruplet = tf.reduce_mean((1/32)*(p_dist /(n_dist + n_dist2 + n_dist3 + 1e-7)))  # Contrast loss dehazing. Rmb to include the very small term in the denominator.
        ################################## Training ############################################
        loss = loss_mae + 0.1*loss_Quadruplet       # combined loss function.
      grads = tape.gradient(loss, DPE_Net_contrastive.trainable_variables)   # combined gradient.
      opt.apply_gradients(zip(grads, DPE_Net_contrastive.trainable_variables))  # combined opt.
      #################################### Steps for training ######################################
      step += 1
      if step % 1 == 0:
          # measure other metrics if needed
          print("loss: ", tf.keras.ops.sum(loss))  # Need this to output only 1 values instead of a tensor-shaped value.
    print("Epoch ", epoch)
