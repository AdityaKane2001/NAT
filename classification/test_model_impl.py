import torch
from torchsummary import summary

# from wintome_nat import (
#     wintome_nat_tiny,
#     wintome_nat_mini,
#     wintome_nat_small,
#     wintome_nat_base,
# )


from smooth_wintome_dinats import smooth_wintome_dinat_s_tiny

# from wintome_dinats import (
#     wintome_nat_s_tiny,
#     wintome_nat_s_small,
#     wintome_nat_s_base,
#     wintome_nat_s_large,
#     wintome_nat_s_large_21k,
#     wintome_nat_s_large_384,
#     wintome_dinat_s_tiny,
#     wintome_dinat_s_small,
#     wintome_dinat_s_base,
#     wintome_dinat_s_large,
#     wintome_dinat_s_large_21k,
#     wintome_dinat_s_large_384,
# )

# from dinats import dinat_s_large_384
from extras import get_gflops, get_mparams

# from nat import nat_tiny, nat_mini, nat_small, nat_base


model_clss = [
    smooth_wintome_dinat_s_tiny,
#     wintome_nat_s_tiny,
#     # dinat_s_tiny,
#     wintome_nat_s_small,
#     wintome_nat_s_base,
#     wintome_nat_s_large,
#     wintome_nat_s_large_21k,
#     wintome_nat_s_large_384,
#     wintome_dinat_s_tiny,
#     wintome_dinat_s_small,
#     wintome_dinat_s_base,
#     wintome_dinat_s_large,
#     wintome_dinat_s_large_21k,
#     wintome_dinat_s_large_384,
#     dinat_s_large_384
]


for model_cls in model_clss:
    print(model_cls.__name__)
    model = model_cls()
    
    # print(model)
    print(model(torch.rand(2, 3, 224, 224)).shape)
    # break

    # summary(model, input_data=(3, 224, 224), device="cuda:0")
    # print(f"WinTomeNAT flops: ", get_gflops(model, device="cuda:0"))
    # print(f"WinTomeNAT params: ", get_mparams(model, device="cuda:0"))
    # print("=" * 80)
    # break

# model = dinat_s_tiny()
# # print(model(torch.rand(2, 3, 224, 224)).shape)
# summary(model, input_data=(3, 224, 224), device="cuda:0")

# print(f"NAT flops: ", get_gflops(model, device="cuda:0"))
# print(f"NAT params: ", get_mparams(model, device="cuda:0"))
