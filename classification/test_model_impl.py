import torch
from torchsummary import summary


from dinats import dinat_s_large_384, dinat_s_tiny
from extras import get_gflops, get_mparams
from isotropic import nat_isotropic_small, dinat_isotropic_small, vitrpb_small
from nat import nat_tiny, nat_mini, nat_small, nat_base
from vit_pretrained import attnprune_vit_small_patch16_224_augreg_in21k, vit_small_patch16_224_augreg_in21k
from dinats_sa import dinat_s_sa_tiny

model_clss = [
    # attnprune_vit_small_patch16_224_augreg_in21k,
    # vit_small_patch16_224_augreg_in21k,
    # wintome_nat_s_tiny,
    dinat_s_tiny,
    dinat_s_sa_tiny
    # nat_isotropic_small, 
    # dinat_isotropic_small, 
    # vitrpb_small,
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
    
    # print(model)
    # print(model(torch.rand(2, 3, 224, 224)).shape)
    model.to("cuda:0")
    # print(model)

    # summary(model, input_data=(3, 224, 224), device="cuda:0")
    print(f"flops: ", get_gflops(model, device="cuda:0"))
    print(f"params: ", get_mparams(model, device="cuda:0"))
    print("=" * 80)
    # break

# model = dinat_s_tiny()
# # print(model(torch.rand(2, 3, 224, 224)).shape)
# summary(model, input_data=(3, 224, 224), device="cuda:0")

# print(f"NAT flops: ", get_gflops(model, device="cuda:0"))
# print(f"NAT params: ", get_mparams(model, device="cuda:0"))
