# CAIN: Video Stabilization through Deep Frame Interpolation

This is a pytorch implementation of Video Stabilization using [DIFRINT](https://arxiv.org/abs/1909.02641).

![Video Stabilization Example](https://github.com/btxviny/Video-Stabilization-through-Frame-Interpolation-using-CAIN/blob/main/result.gif)

## Inference Instructions

Follow these instructions to perform video stabilization using the pretrained model:
1. **Set up custom layers for PWC-Net.**
    -Detailed instructions in [pwc-net.pytorch](https://github.com/vt-vl-lab/pwc-net.pytorch).
    I provide an alternative implementation that runs using [RAFT](https://pytorch.org/vision/main/models/raft.html) which need no custom layers but is much slower.
1. **Download Pretrained Model:**
   - Download the pretrained model [weights](https://drive.google.com/drive/folders/1CeeOBN1gYuQv_9Oj73c7y056Wus8A012?usp=sharing).
   - Extract the file and place the folder inside the root directory.
   - Download PWC pretrained [weights](https://drive.google.com/drive/folders/14wYcYymTatXWPFSvGpxJ_kkH5A0SQ_lP?usp=sharing).
   - Extract the file and place the folder inside the root directory.

3. **Run the Stabilization Script:**
   - Run the following command:
     ```bash
     python stabilize_difrint.py --in_path input_path --out_path output_path
     ```
   - Replace ` input_path` with the path to your input unstable video.
   - Replace `output_path` with the desired path for the stabilized output video.
   - For the implementation using RAFT run the following command:
     ```bash
     python stabilize_difrint_raft.py --in_path input_path --out_path output_path
     ```
## Training
I provide the notebook `train.ipynb` which were used for training DIFRINT on the [DAVIS](https://davischallenge.org/) dataset as proposed in the original paper.
I provide the notebooks `train_unet_vimeo.ipynb`, `train_resnet_vimeo.ipynb` which were used for finetuning on the [Vimeo-Triplet](http://toflow.csail.mit.edu/) dataset.
 
