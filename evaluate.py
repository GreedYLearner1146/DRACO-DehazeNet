################################ Prediction #######################################################

data_test_dehazed,_,_,_,_ = DPE_Net_contrastive.predict([np.asarray(test_haze), np.asarray(test_clear)])

############################### Compute PSNR and SSIM. #######################################
from math import log10, sqrt
import numpy as np

from skimage.metrics import structural_similarity

def PSNR(clean_image, predicted_image):
    mse = np.mean((clean_image - predicted_image)**2)
    if (mse ==0):  # MSE = 0 => no noise present in image, PSNR has no importance.
        return 100
    max_pixel = 1   # Because we normalized our images.
    psnr = 20*log10(max_pixel/sqrt(mse))
    return psnr

print('PSNR:',PSNR(data_test_dehazed, test_clear))

#Use compare_ssim built in functions.

ssim = []
for i in range (len(data_test_dehazed)):
  score = tf.image.ssim(np.asarray(data_test_dehazed[i]), np.asarray(test_clear[i]), max_val=2.0)
  ssim.append(score)

# Print the score.
print("SSIM: {}".format(np.average(ssim)))
