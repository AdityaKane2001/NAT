# Experiments on NA, DiNA with Token Merging

This document serves as a summary of experiments and other notes I found during training these models on ImageNet-1k.

1. **Windowed tome with reduction from 4x to 1x** <br>
This experiment aimed to reduce 4x4 to 2x2 by merging 12 tokens into 4 tokens. Run [here](https://wandb.ai/compyle/wintome-dinat-s-IN1k/runs/20231006-232237-wintome_dinat_s_tiny-224). 

    Some thoughts on the same:
    1. The current block architecture is as in the image attached. Note that this is for the last block in each level only. In summary, the new architecture is NAT-Swin + WinToMe, since dilation=1 everywhere. 
    2. Looking at the comparison of train and val loss, I feel that the model is over-regularized. Reducing augmentations is something that can be done off the bat, but I am not sure about the effect of that on ViT-based models. HP-tuning is something that seems very lucrative, but this might not be the best time for this, so like you said last time, I think I should focus on other components at this time.
    3. One improvement for ToMe that comes to my mind is that instead of merging 12 tokens into 4, we can try repeated ToMe, i.e. 16 -> 8 and 8 -> 4. The intuition is that all tokens will be able to "see" all other tokens. Another idea that you suggested was to split the 4x4 window into 4 2x2 windows and choosing the most similar token in each sub-window, which can be tried out as well.
    4. A note on skip connections: I had shared an image earlier (https://files.slack.com/files-tmb/TP34CK0KH-F05UYRBASLV-9f0e1a0b46/wintome_nat_arch_720.jpg), where there was no residual connection around LN + MLP. This turned out to be extremely detrimental, since the loss was exploding from the get-go. Due to that, I modified the architecture a bit to include an additional MLP block, which does the depthwise upsampling.

    Following are the things I think I should try (maybe in that order):
    1. ToMe improvement: Devising a way where we merge the closest tokens in the pool (maybe all-to-all comparisons?). The current method seems an issue since the source tokens (12 tokens, which are merged into 4 destination tokens) never "see" themselves, and hence we are at the risk of incorrectly merging tokens. Simply running the ToMe algorithm twice seems to be an intuitive way to go, since we'd be making a slightly more informed decision about merging tokens.
    2. Merging inside the attention module: As I did with the reduction block, we can merge tokens right after attention before the projection, and add an upsampling MLP block after the projection. This will make sure that the keys are more relevant. For merging, I will use whichever method gives approach performance in the first point.

