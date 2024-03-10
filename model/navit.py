from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

#helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 1024

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-3:]
        pl, ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (pl * ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

#normalization
#they use layernorm without bias, something that pytorch does not offer
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

#they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

#feedforward
def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

#attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#transformer block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(nn.Module):
    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        channels = 4, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0., 
        token_dropout_prob = None
    ):
        super().__init__()
        image_length = image_height = image_width = image_size

        #what percent of tokens to dropout
        #if int or float given, then assume constant dropout prob
        #otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda *args: token_dropout_prob

        #calculate patching related stuff

        assert divisible_by(image_length, patch_size) and divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_length_dim, patch_height_dim, patch_width_dim = (image_length // patch_size),(image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 3)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )
        self.pos_embed_length = nn.Parameter(torch.randn(patch_length_dim, dim))
        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        #final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        #output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        sample,
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)
        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        #extract from dict
        patches, patch_positions, attn_mask, key_pad_mask, batched_image_ids, num_images = sample['patches'], sample['patch_positions'], sample['attn_mask'], sample['key_pad_mask'], sample['batched_image_ids'], sample['num_images']     

        #to patches
        x = self.to_patch_embedding(patches.to(self.device))        

        #factorized 3d absolute positional embedding

        l_indices, h_indices, w_indices = patch_positions.unbind(dim = -1)
        l_pos = self.pos_embed_length[l_indices]
        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + l_pos + h_pos + w_pos

        #embed dropout

        x = self.dropout(x)

        #attention

        x = self.transformer(x, attn_mask = attn_mask)

        #do attention pooling at the end

        max_queries = num_images.amax().item()

        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])

        #attention pool mask

        image_id_arange = arange(max_queries)

        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')

        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')

        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

        #attention pool

        x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries

        x = rearrange(x, 'b n d -> (b n) d')

        #each batch element may not have same amount of images

        is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        is_images = rearrange(is_images, 'b n -> (b n)')

        x = x[is_images]

        #project out to logits

        x = self.to_latent(x)

        return self.mlp_head(x)