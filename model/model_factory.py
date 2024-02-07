from functools import partial

from torch import nn
from model import vit_autoenc
from model.varivit import VisionTransformer3D
from model.vit import VisionTransformer3D_vit

def get_models(model_name, args, attn_drop_rate=0.0, drop_rate=0.0):

    if model_name == 'vit':
        print(f"Number of channels is {args.in_channels}")
        return VisionTransformer3D_vit(volume_size=args.volume_size, in_chans=args.in_channels, num_classes=args.nb_classes,
                                   patch_size=args.patch_size, global_pool=args.global_pool, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   drop_path_rate=args.drop_path)
       
    elif model_name == 'varivit':
        print(f"Number of channels is {args.in_channels}")
        return VisionTransformer3D(max_volume_size=args.volume_size, in_chans=args.in_channels, num_classes=args.nb_classes,
                                   patch_size=args.patch_size, global_pool=args.global_pool, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   drop_path_rate=args.drop_path)#, attn_drop_rate=attn_drop_rate, drop_rate=drop_rate)


    else:
        raise NotImplementedError("Only AE model supported till now")
