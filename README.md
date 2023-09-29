# CLIPMIA
This is an official repository for Practical Membership Inference Attacks Against Large-Scale Multi-Modal Models: A Pilot Study (ICCV2023).

In this paper, we take a first step towards developing practical MIAs against large-scale multi-modal models.
We introduce a simple baseline strategy by thresholding the cosine similarity between text and image features of a target point and propose further enhancing the baseline by aggregating
cosine similarity across transformations of the target. We also present a new weakly supervised attack method that leverages ground-truth non-members (e.g., obtained by using the publication date of a target model and the timestamps of the open data) to further enhance the attack.

Steps to Reproduce the Attacks
1. Install Anaconda Environment
Ensure Anaconda is installed to manage packages and dependencies.

2. Download Dataset
Obtain datasets like LAION, CC12M, and CC3M using the repository: img2dataset.

3. Check Duplication
Utilize "Inspect_image_overlapping.ipynb" to inspect overlapping text, URLs, etc., in the selected datasets.

4. Hyperparameter Setting.
Non-Train Datasets
args.val_data-nontrain-1: Path for non-train dataset 1.
args.val_num_samples-nontrain: Number of samples in non-train dataset 1.
Repeat for other non-train datasets.

Pseudo-Train Datasets
args.val_data-train: Path for pseudo-train dataset 1.
args.val_num_samples-nontrain: Number of samples in pseudo-train dataset 1.
Repeat for other pseudo-train datasets.

Evaluation Datasets
args.train_data-1: Path for the primary training dataset.
—train_num_samples-1: Number of samples in the training dataset.
args.val_data-1: Path for the first non-training dataset.
—val_num_samples-1: Number of samples in the first non-training dataset.
Repeat for other evaluation datasets.

5. Execute the Attack
Run main.py --model ViT-B-32

For any questions, please contact myeongseob@vt.edu
