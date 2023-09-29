# CLIPMIA
This is an official repository for Practical Membership Inference Attacks Against Large-Scale Multi-Modal Models: A Pilot Study (ICCV2023).

In this paper, we take a first step towards developing practical MIAs against large-scale multi-modal models.
We introduce a simple baseline strategy by thresholding the cosine similarity between text and image features of a target point and propose further enhancing the baseline by aggregating
cosine similarity across transformations of the target. We also present a new weakly supervised attack method that leverages ground-truth non-members (e.g., obtained by using the publication date of a target model and the timestamps of the open data) to further enhance the attack.

1. Install Anaconda Environment
  
2. Download Dataset
   Use the following repository to download the dataset interested such LAION, CC12M, and CC3M: https://github.com/rom1504/img2dataset/tree/main

3. Check Duplication between Datasets such as text, URLs
   Run the "Inspect_image_overlapping.ipynb" or "Inspect_image_overlapping.ipynb" for the corresponding datasets.

4. Run main.py to initiate the attack.


For any 
