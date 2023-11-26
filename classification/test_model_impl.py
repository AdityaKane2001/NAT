import torch
from torchsummary import summary

from wintome_nat import (
    wintome_nat_tiny,
    wintome_nat_mini,
    wintome_nat_small,
    wintome_nat_base,
)


from wintome_dinats import (
    wintome_nat_s_tiny,
    wintome_nat_s_small,
    wintome_nat_s_base,
    wintome_nat_s_large,
    wintome_nat_s_large_21k,
    wintome_nat_s_large_384,
    wintome_dinat_s_tiny,
    wintome_dinat_s_small,
    wintome_dinat_s_base,
    wintome_dinat_s_large,
    wintome_dinat_s_large_21k,
    wintome_dinat_s_large_384,
)
from hybrid_dinat import hybrid_dinat_s_tiny
from hybrid_vit import hybrid_dinat_rpb_isotropic_small
from dinats import dinat_s_tiny
from extras import get_gflops, get_mparams
from isotropic import vitrpb_small, dinat_isotropic_small
from nat import nat_tiny, nat_mini, nat_small, nat_base


model_clss = [
    hybrid_dinat_rpb_isotropic_small, # 81.18
    dinat_isotropic_small, # 80.8
    vitrpb_small, # 81.2
    # dinat_s_tiny
    # wintome_nat_s_tiny,
    # # dinat_s_tiny,
    # wintome_nat_s_small,
    # wintome_nat_s_base,
    # wintome_nat_s_large,
    # wintome_nat_s_large_21k,
    # wintome_nat_s_large_384,
    # wintome_dinat_s_tiny,
    # wintome_dinat_s_small,
    # wintome_dinat_s_base,
    # wintome_dinat_s_large,
    # wintome_dinat_s_large_21k,
    # wintome_dinat_s_large_384,
    # dinat_s_large_384
]


for model_cls in model_clss:
    print(model_cls.__name__)
    model = model_cls()
    model.to("cuda:0")

    # print(model)
    print(model(torch.rand(2, 3, 224, 224).to("cuda:0")).shape)

    # summary(model, input_data=(3, 224, 224), device="cuda:0")
    print(f"\tflops: ", get_gflops(model, device="cuda:0"))
    print(f"\tparams: ", get_mparams(model, device="cuda:0"))
    # print("=" * 80)
    # break

# model = dinat_s_tiny()
# # print(model(torch.rand(2, 3, 224, 224)).shape)
# summary(model, input_data=(3, 224, 224), device="cuda:0")

# print(f"NAT flops: ", get_gflops(model, device="cuda:0"))
# print(f"NAT params: ", get_mparams(model, device="cuda:0"))
