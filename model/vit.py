import copy
import math
from collections import OrderedDict
from functools import partial
from einops.layers.torch import Rearrange
import torch
from model.model_utils.vit_helpers import get_3d_sincos_pos_embed
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers.weight_init import lecun_normal_, trunc_normal_

from model.model_utils.vit_helpers import _load_weights


def traid(t):
    return t if isinstance(t, tuple) else (t, t, t)

# We have to adapt the different layers for 3D and try to align this with the `timm` implementations
class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, volume_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        volume_size = traid(volume_size)
        patch_size = traid(patch_size)
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = (volume_size[0] // patch_size[0], volume_size[1] // patch_size[1], volume_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L, H, W = x.shape
        assert L == self.volume_size[0] and H == self.volume_size[1] and W == self.volume_size[2], \
            f"Volume image size ({L}*{H}*{W}) doesn't match model ({self.volume_size[0]}*{self.volume_size[1]}*{self.volume_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Mlp3D(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# The Attention is always applied to the sequences. Thus, at this point, it should be the same model
# whether we apply it in NLP, ViT, speech or any other domain :)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_weights = attn.clone()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0]  # Access only the output, not the attention weights
        x = x + self.mlp(self.norm2(x))
        return x, self.attn(self.norm1(x))[1]  # Return both output and attention weights


class VisionTransformer3D_vit(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, volume_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed3D, norm_layer=None,
                 act_layer=None, weight_init='', global_pool=False):
        """
        Args:
            volume_size (int, triad): input image size
            patch_size (int, triad): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            volume_size=volume_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.head_dist = None
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        num_patches = self.patch_embed.num_patches

        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], round(num_patches ** (1 / 3)), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
    
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        outputs = []
        attention_maps = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, attn_map = blk(x)  # Retrieve both output and attention map
            outputs.append(x)  # Store outputs for potential use
            attention_maps.append(attn_map)  # Store attention maps

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, attention_maps

    def forward(self, x):
        outcome, attention_maps = self.forward_features(x)  # Unpack returned values

        if self.head_dist is not None:
            x, x_dist = self.head(outcome), self.head_dist(outcome)  # Use 'outcome' for both heads
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(outcome)

        return x, attention_maps



class VisionTransformer3D_vitContrastive(VisionTransformer3D_vit):
    def __init__(self, volume_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed3D, norm_layer=None,
                 act_layer=None, weight_init='', global_pool=False, use_proj=False):
        super(VisionTransformer3D_vitContrastive, self).__init__(volume_size=volume_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size, distilled=distilled,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, norm_layer=norm_layer,
                 act_layer=act_layer, weight_init=weight_init, global_pool=global_pool)
        # build a 3-layer projector
        self.use_proj = use_proj
        if use_proj:
            self.projection_head = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                                            nn.BatchNorm1d(self.embed_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                                            nn.BatchNorm1d(self.embed_dim),
                                            nn.ReLU(inplace=True), # second layer
                                            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                                            nn.BatchNorm1d(self.embed_dim, affine=False)) # output layer
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(self.embed_dim, self.embed_dim)  # output layer
        )


    def forward(self, x1, x2):
        # call the parent function for the two views
        z1 = super(VisionTransformer3D_vitContrastive, self).forward(x1)
        z2 = super(VisionTransformer3D_vitContrastive, self).forward(x2)
        if self.use_proj:
            z1, z2 = self.projection_head(z1), self.projection_head(z2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()



if __name__ == '__main__':
    image_size = (64, 64, 64)
    sample_img = torch.randn(8, 3, 64, 64, 64)
    # model = ViT(image_size=image_size, num_classes=2, in_channels=5)
    # # Put things to cuda and check
    # model.cuda()
    # sample_img = sample_img.cuda()
    # output = model(sample_img)
    # print(output.shape)
    embed = VisionTransformer3D_vitContrastive(volume_size=image_size, in_chans=3, num_classes=-1, use_proj=True)
    output, _, _, _ = embed(sample_img, sample_img)
    print(output.shape)
    (1-output).sum().backward()
