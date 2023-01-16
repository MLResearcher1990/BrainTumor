# BrainTumor
Brain Tumor Segmentation and Classification
This framework consists of multiple steps. First, brain MRI images are enhanced to improve quality, reduce noise, and clarify tumor area detection in the initial
preprocessing stage. In the next step, a CNN-based approach is proposed to detect tumor region to focus on the tumor area instead of the whole image and reduce the input size of the multitask network. After passing these two steps, 
MRI images are entered into an integrated and end-to-end network called Multi-scale Cascaded Multi-Task network. The
proposed network is based on multitask learning that improves performance compared to single-task networks and
simultaneously allows brain tumor segmentation and classification
