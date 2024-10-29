# DRACO-DehazeNet
This is the code repository for our work ``DRACO-DehazeNet: An Efficient Image Dehazing Network Combining Detail Recovery and a Novel Contrastive Learning Paradigm" by Gao Yu Lee, Tanmoy Dam, Md. Meftahul Ferdaus, Daniel Puiu Poenar, and Vu Duong, currently appeared as an ArXiV preprint.

**Abstract**: Image dehazing is crucial for clarifying images obscured by haze or fog, but current learning-based approaches is dependent on large volumes of training data and hence consumed significant computational power. Additionally, their performance is often inadequate under non-uniform or heavy haze. To address these challenges, we developed the Detail Recovery And Contrastive DehazeNet, which facilitates efficient and effective dehazing via a dense dilated inverted residual block and an attention-based detail recovery network that tailors enhancements to specific dehazed scene contexts. A major innovation is its ability to train effectively with limited data, achieved through a novel quadruplet loss-based contrastive dehazing paradigm. This approach distinctly separates hazy and clear image features while also distinguish lower-quality and higher-quality dehazed images obtained from each sub-modules of our network, thereby refining the dehazing process to a larger extent. Extensive tests on a variety of benchmarked haze datasets demonstrated the superiority of our approach.

All codes that is and will be shown here are presented in Tensorflow Keras format.

The link to our paper can be found at https://arxiv.org/abs/2410.14595 

**(This repo may be updated at various times.)**

# Preliminary Results 

We evalauted our approach on 4 benchmarked dataset, namely RESIDE [1], NH-HAZE [2], DENSE-HAZE [3], and O-HAZE [4]. \
NH-HAZE and DENSE-HAZE are comprised of generated real haze that are of non-homogeneous and dense nature, while O-HAZE is comprised of generated real haze that ensures homogenity in the image captured. For RESIDE, we used the SOTS subset for evaluation.


| Datasets| PSNR | SSIM| 
| ------ | ------| ------| 
| RESIDE (SOTS) [1] | 38.08 | 0.9906 | 
| O-HAZE [4] | 22.94 | 0.9000 | 
| NH-HAZE [2] | 20.82 | 0.7582 | 
| DENSE-HAZE [3] | 14.25 | 0.6028 | 


# Code Instructions

# Training weights 

# Citation Information

Please cite the following preprint if it is useful in your research:

G. Y. Lee, T. Dam, M. M. Ferdaus, D. P. Poenar, and V. Duong, “Draco-dehazenet: An efficient image dehazing network combining detail recovery
and a novel contrastive learning paradigm,” arXiv preprint arXiv:2410.14595,2024

@article{lee2024draco, \
  title={DRACO-DehazeNet: An Efficient Image Dehazing Network Combining Detail Recovery and a Novel Contrastive Learning Paradigm}, \
  author={Lee, Gao Yu and Dam, Tanmoy and Ferdaus, Md Meftahul and Poenar, Daniel Puiu and Duong, Vu}, \
  journal={arXiv preprint arXiv:2410.14595}, \
  year={2024}
}

# Relevant References

