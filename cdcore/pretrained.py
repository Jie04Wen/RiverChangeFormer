# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn.functional as F


def _load_raw(path):
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'state_dict' in ck and not any(
            k.startswith(('encoder.', 'backbone.')) for k in ck.keys()):
        ck = ck['state_dict']
    return ck


def _cat_kv(k_w, v_w, k_b=None, v_b=None):
    w = torch.cat([k_w, v_w], dim=0)
    b = None
    if k_b is not None and v_b is not None:
        b = torch.cat([k_b, v_b], dim=0)
    return w, b


def build_encoder_state_from_hf(sd):

    out = {}

    def take(suffix):
        w = sd.get('encoder.' + suffix + '.weight')
        if w is None:
            return None, None
        b = sd.get('encoder.' + suffix + '.bias')
        return w, b

    for i in range(4):
        s = i + 1
        # patch embedding
        w, b = take('patch_embeddings.%d.proj' % i)
        if w is not None:
            out['patch_embed%d.proj.weight' % s] = w
            if b is not None:
                out['patch_embed%d.proj.bias' % s] = b
        w, b = take('patch_embeddings.%d.layer_norm' % i)
        if w is not None:
            out['patch_embed%d.norm.weight' % s] = w
            if b is not None:
                out['patch_embed%d.norm.bias' % s] = b
        # stage final norm
        w, b = take('layer_norm.%d' % i)
        if w is not None:
            out['norm%d.weight' % s] = w
            if b is not None:
                out['norm%d.bias' % s] = b

    # blocks
    i = 0
    while True:
        pre = 'encoder.block.%d.' % i
        if not any(k.startswith(pre) for k in sd):
            break
        s = i + 1
        j = 0
        while True:
            bp = 'block.%d.%d.' % (i, j)
            if not any(k.startswith('encoder.' + bp) for k in sd):
                break
            t = 'block%d.%d.' % (s, j)
            for src, dst in [('layer_norm1', 'norm1'), ('layer_norm2', 'norm2')]:
                w, b = take(bp + src)
                if w is not None:
                    out[t + dst + '.weight'] = w
                    if b is not None:
                        out[t + dst + '.bias'] = b
            # q / k / v -> q 与 kv(k||v)
            qw, qb = take(bp + 'attention.self.q')
            kw, kb = take(bp + 'attention.self.k')
            vw, vb = take(bp + 'attention.self.v')
            if qw is not None:
                out[t + 'attn.q.weight'] = qw
                if qb is not None:
                    out[t + 'attn.q.bias'] = qb
                kvw, kvb = _cat_kv(kw, vw, kb, vb)
                out[t + 'attn.kv.weight'] = kvw
                if kvb is not None:
                    out[t + 'attn.kv.bias'] = kvb
            # spatial reduction conv + its norm
            w, b = take(bp + 'attention.self.sr')
            if w is not None:
                out[t + 'attn.sr.weight'] = w
                if b is not None:
                    out[t + 'attn.sr.bias'] = b
            w, b = take(bp + 'attention.self.layer_norm')
            if w is not None:
                out[t + 'attn.norm.weight'] = w
                if b is not None:
                    out[t + 'attn.norm.bias'] = b
            # attention output projection
            w, b = take(bp + 'attention.output.dense')
            if w is not None:
                out[t + 'attn.proj.weight'] = w
                if b is not None:
                    out[t + 'attn.proj.bias'] = b
            # MLP: dense1->fc1, dwconv.dwconv, dense2->fc2
            w, b = take(bp + 'mlp.dense1')
            if w is not None:
                out[t + 'mlp.fc1.weight'] = w
                if b is not None:
                    out[t + 'mlp.fc1.bias'] = b
            w, b = take(bp + 'mlp.dwconv.dwconv')
            if w is not None:
                out[t + 'mlp.dwconv.dwconv.weight'] = w
                if b is not None:
                    out[t + 'mlp.dwconv.dwconv.bias'] = b
            w, b = take(bp + 'mlp.dense2')
            if w is not None:
                out[t + 'mlp.fc2.weight'] = w
                if b is not None:
                    out[t + 'mlp.fc2.bias'] = b
            j += 1
        i += 1
    return out


def _squeeze_conv1x1(w):

    return w.squeeze(-1).squeeze(-1) if (w is not None and w.dim() == 4
                                         and w.shape[-1] == 1 and w.shape[-2] == 1) else w


def build_encoder_state_from_mmseg1x(sd):

    out = {}

    def put(dst, w, b=None):
        if w is not None:
            out[dst + '.weight'] = w
            if b is not None:
                out[dst + '.bias'] = b

    for i in range(4):
        s = i + 1
        proj_w = sd.get('layers.%d.0.projection.weight' % i)
        if i >= 1:  # stage2-4：源为 3x3 下采样卷积，本仓库为 7x7 -> 居中补零
            proj_w = _center_pad_conv3_to_7(proj_w)
        put('patch_embed%d.proj' % s, proj_w,
            sd.get('layers.%d.0.projection.bias' % i))
        put('patch_embed%d.norm' % s,
            sd.get('layers.%d.0.norm.weight' % i),
            sd.get('layers.%d.0.norm.bias' % i))
        put('norm%d' % s,
            sd.get('layers.%d.2.weight' % i),
            sd.get('layers.%d.2.bias' % i))
        j = 0
        while ('layers.%d.1.%d.norm1.weight' % (i, j)) in sd:
            p = 'layers.%d.1.%d.' % (i, j)
            t = 'block%d.%d.' % (s, j)
            put(t + 'norm1', sd.get(p + 'norm1.weight'), sd.get(p + 'norm1.bias'))
            put(t + 'norm2', sd.get(p + 'norm2.weight'), sd.get(p + 'norm2.bias'))
            # nn.MultiheadAttention 的 in_proj 按 [q;k;v] 行打包 -> q 与 kv(k||v)
            ipw = sd.get(p + 'attn.attn.in_proj_weight')
            ipb = sd.get(p + 'attn.attn.in_proj_bias')
            if ipw is not None:
                C = ipw.shape[1]
                out[t + 'attn.q.weight'] = ipw[0:C]
                out[t + 'attn.kv.weight'] = ipw[C:3 * C]
                if ipb is not None:
                    out[t + 'attn.q.bias'] = ipb[0:C]
                    out[t + 'attn.kv.bias'] = ipb[C:3 * C]
            put(t + 'attn.proj',
                sd.get(p + 'attn.attn.out_proj.weight'),
                sd.get(p + 'attn.attn.out_proj.bias'))
            if (p + 'attn.sr.weight') in sd:
                put(t + 'attn.sr', sd.get(p + 'attn.sr.weight'),
                    sd.get(p + 'attn.sr.bias'))
                put(t + 'attn.norm', sd.get(p + 'attn.norm.weight'),
                    sd.get(p + 'attn.norm.bias'))
            # FFN：layers.0=fc1(1x1conv)、layers.1=DWConv、layers.4=fc2(1x1conv)
            put(t + 'mlp.fc1',
                _squeeze_conv1x1(sd.get(p + 'ffn.layers.0.weight')),
                sd.get(p + 'ffn.layers.0.bias'))
            put(t + 'mlp.dwconv.dwconv',
                sd.get(p + 'ffn.layers.1.weight'),
                sd.get(p + 'ffn.layers.1.bias'))
            put(t + 'mlp.fc2',
                _squeeze_conv1x1(sd.get(p + 'ffn.layers.4.weight')),
                sd.get(p + 'ffn.layers.4.bias'))
            j += 1
    return out


def _center_pad_conv3_to_7(w):

    if w is not None and w.dim() == 4 and w.shape[-1] == 3 and w.shape[-2] == 3:
        return F.pad(w, (2, 2, 2, 2))
    return w


def build_encoder_state_from_mmseg(sd):
    out = {}
    if any(k.startswith('backbone.') for k in sd):
        keys = [k[len('backbone.'):] for k in sd if k.startswith('backbone.')]
        src = {k[len('backbone.'):]: v for k, v in sd.items()
               if k.startswith('backbone.')}
    else:
        keys = [k for k in sd if k.split('.')[0]
                in ('patch_embed1', 'patch_embed2', 'patch_embed3', 'patch_embed4',
                    'block1', 'block2', 'block3', 'block4', 'norm1', 'norm2',
                    'norm3', 'norm4')]
        src = {k: v for k, v in sd.items() if k in keys}
    prefix_keys = set(keys)

    def has(name):
        return name in prefix_keys

    def get(name):
        return src.get(name)

    for s in range(1, 5):
        for pe in ('proj', 'norm'):
            dst = 'patch_embed%d.%s' % (s, pe)
            src = 'patch_embed%d.%s' % (s, pe)
            if has(src):
                out[dst + '.weight'] = get(src + '.weight')
                if get(src + '.bias') is not None:
                    out[dst + '.bias'] = get(src + '.bias')
        if get('norm%d.weight' % s) is not None:
            out['norm%d.weight' % s] = get('norm%d.weight' % s)
            if get('norm%d.bias' % s) is not None:
                out['norm%d.bias' % s] = get('norm%d.bias' % s)
        j = 0
        while has('block%d.%d.norm1.weight' % (s, j)):
            t = 'block%d.%d.' % (s, j)
            for srcn, dstn in [('norm1', 'norm1'), ('norm2', 'norm2')]:
                out[t + dstn + '.weight'] = get(t + srcn + '.weight')
                if get(t + srcn + '.bias') is not None:
                    out[t + dstn + '.bias'] = get(t + srcn + '.bias')
            # qkv -> q and kv
            qkv_w = get(t + 'attn.qkv.weight')
            if qkv_w is not None:
                C = qkv_w.shape[1]
                out[t + 'attn.q.weight'] = qkv_w[0:C]
                out[t + 'attn.kv.weight'] = qkv_w[C:3 * C]
                qkv_b = get(t + 'attn.qkv.bias')
                if qkv_b is not None:
                    out[t + 'attn.q.bias'] = qkv_b[0:C]
                    out[t + 'attn.kv.bias'] = qkv_b[C:3 * C]
            for leaf in ('attn.sr', 'attn.norm', 'attn.proj',
                         'mlp.fc1', 'mlp.dwconv.dwconv', 'mlp.fc2'):
                w = get(t + leaf + '.weight')
                if w is not None:
                    out[t + leaf + '.weight'] = w
                    b = get(t + leaf + '.bias')
                    if b is not None:
                        out[t + leaf + '.bias'] = b
            j += 1
    return out


def load_mit_b2(encoder, weights_path, verbose=True):
    sd = _load_raw(weights_path)
    bare_mit = any(k.startswith(('patch_embed1.', 'block1.', 'norm1.')) for k in sd)
    if any(k.startswith('layers.') for k in sd):
        fmt, mapped = 'mmseg1.x/MiT', build_encoder_state_from_mmseg1x(sd)
    elif any(k.startswith('encoder.') for k in sd):
        fmt, mapped = 'HuggingFace', build_encoder_state_from_hf(sd)
    elif any(k.startswith('backbone.') for k in sd) or bare_mit:
        fmt, mapped = 'mmseg/MiT', build_encoder_state_from_mmseg(sd)
    else:
        fmt, mapped = 'plain', {k: v for k, v in sd.items()}

    model_sd = encoder.state_dict()
    usable, skipped = {}, []
    for k, v in mapped.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            usable[k] = v
        else:
            skipped.append(k)
    missing, unexpected = encoder.load_state_dict(usable, strict=False)

    total_t = len(model_sd)
    cover_t = len(usable)
    total_p = sum(t.numel() for t in model_sd.values())
    cover_p = sum(v.numel() for k, v in usable.items() if k in model_sd)
    report = {
        'format': fmt,
        'source_tensors': len(sd),
        'mapped_tensors': len(mapped),
        'loaded_tensors': cover_t,
        'encoder_tensors': total_t,
        'tensor_coverage': round(cover_t / total_t, 4),
        'param_coverage': round(cover_p / total_p, 4),
        'shape_skipped': skipped,
        'missing': sorted(missing),
        'unexpected': sorted(unexpected),
    }
    if verbose:
        print('[pretrain] MiT-B2 (%s) -> encoder: tensors %d/%d (%.2f%%), '
              'params %.2f%%; shape-skipped %d, missing %d, unexpected %d'
              % (fmt, cover_t, total_t, 100.0 * cover_t / total_t,
                 100.0 * cover_p / total_p, len(skipped),
                 len(missing), len(unexpected)), flush=True)
        if skipped:
            print('[pretrain] shape-skipped keys:', skipped[:10], flush=True)
        real_missing = [m for m in missing if not m.endswith(('attn.sr.weight',
                        'attn.sr.bias', 'attn.norm.weight', 'attn.norm.bias'))]
        if real_missing:
            print('[pretrain] NOTE non-SR missing keys (%d):' % len(real_missing),
                  real_missing[:12], flush=True)
    return report


def main():
    from .models import ChangeFormerV6
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    a = p.parse_args()
    net = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256)
    rep = load_mit_b2(net.Tenc_x2, a.weights)
    print(rep)


if __name__ == '__main__':
    main()
