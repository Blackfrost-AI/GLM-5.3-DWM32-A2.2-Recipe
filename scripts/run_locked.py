#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Locked alpha-2.2 BF16 DWM followed by NVIDIA-layout ModelOpt quantization.

No inference server or CUDA device is initialized. Activation calibration is
transplanted from NVIDIA's pinned checkpoint, rather than recomputed after DWM.

This portable form is derived from the historical script whose SHA-256 is
2b857ededdb3cc2caaeb173a4d99879d4ee221f2daa71295055aecf4fad9dad0.
The numerical projection and serialization core is unchanged.
"""
import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import pathlib
import re
import shutil
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONNOUSERSITE'] = '1'
import torch
import modelopt
from safetensors import safe_open
from safetensors.torch import save_file
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor

if not __debug__:
    raise RuntimeError('This fail-closed recipe cannot run with Python optimization (-O)')
if not sys.flags.no_user_site:
    raise RuntimeError('Run with python3 -s so user site-packages cannot shadow the locked environment')

RUN = None
SOURCE = None
OUTPUT = None
REF = None
DIRECTIONS = None
CONFIG = None
ALPHA = 2.2
WORKERS = 8
THREADS_PER_WORKER = 8
SOURCE_REVISION = '9d2398f478cab2de883137db3a36ad2c96205e24'
REFERENCE_REVISION = 'e3b8f9f2fad29727dfa95e766c5cac18ea1414bf'
DIRECTION_SHA256 = 'e12d8fee3e2f75dc7e269bc09c810f079eeb1abce258376ce65f811192db27f8'
DIRECTION_BYTES = 193_612_804
MODELOPT_INIT_SHA256 = '96647188e0afa8dcdcadbe42178feeb4cc19113d9ffcbaf633bd62414bd6d453'
MODELOPT_NVFP4_SHA256 = 'e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e'
MODELOPT_FP8_SHA256 = 'ec9caf56fc140720a55732f827dbdd41b9fdf8fcb961a1e37028d4d4b69b7fae'
LOCKED_HASHES = {
    'source_config': 'ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca',
    'source_index': '5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e',
    'reference_config': 'b0b6b9ab922a82e3688bc39a4117bc443e106fcf88969ef48b3627778f4bdf7b',
    'reference_quant_config': '13799d4fed353c20dde2264fbdd7f68a780b514782f8d80bba96ac5e44aa9bba',
    'reference_index': '157752617100cefaf25a468812e14abefca24c356e6e2f7ae55bce8c3b511863',
}
SOURCE_AUX_HASHES = {
    'chat_template.jinja': '3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c',
    'generation_config.json': 'ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55',
    'tokenizer.json': '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d',
    'tokenizer_config.json': '98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc',
}
PATTERNS = {
    'o_proj': re.compile(r'^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$'),
    'dense_down': re.compile(r'^model\.layers\.(\d+)\.mlp\.down_proj\.weight$'),
    'expert_down': re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.\d+\.down_proj\.weight$'),
    'shared_down': re.compile(r'^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$'),
}
EXPECTED = {'o_proj': 76, 'dense_down': 1, 'expert_down': 19200, 'shared_down': 75}
EXPECTED_DATA_BYTES = 464_115_789_896
EXPECTED_OUTPUT_WEIGHT_BYTES = 464_143_550_648
EXPECTED_PRESERVED_BF16_F32 = 1_825
DTYPES = {'BF16': torch.bfloat16, 'F32': torch.float32, 'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn}
PLAN = None
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_LFS_CANONICAL_SHA256 = 'd7e8f6f8f6d6b03413e7fe5bdefa65c1a4b8d6dd19f30dd963498595ea8c84c2'
SOURCE_TOTAL_BYTES = 1_506_667_387_408

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def configure(config):
    global CONFIG, RUN, SOURCE, OUTPUT, REF, DIRECTIONS
    CONFIG = dict(config)
    RUN = pathlib.Path(CONFIG['run']).resolve()
    SOURCE = pathlib.Path(CONFIG['source']).resolve()
    OUTPUT = pathlib.Path(CONFIG['output']).resolve()
    REF = pathlib.Path(CONFIG['reference']).resolve()
    DIRECTIONS = pathlib.Path(CONFIG['directions']).resolve()
    assert len({SOURCE, OUTPUT, REF, RUN}) == 4
    assert not RUN.is_relative_to(OUTPUT) and not OUTPUT.is_relative_to(RUN)
    assert not DIRECTIONS.is_relative_to(OUTPUT)
    for immutable in (SOURCE, REF):
        assert not OUTPUT.is_relative_to(immutable) and not immutable.is_relative_to(OUTPUT)
        assert not RUN.is_relative_to(immutable) and not immutable.is_relative_to(RUN)
    assert all(path.is_dir() for path in [SOURCE, REF]), (SOURCE, REF)
    assert DIRECTIONS.is_file()
    RUN.mkdir(parents=True, exist_ok=True)

def verify_locked_inputs():
    checks = {
        SOURCE / 'config.json': LOCKED_HASHES['source_config'],
        SOURCE / 'model.safetensors.index.json': LOCKED_HASHES['source_index'],
        REF / 'config.json': LOCKED_HASHES['reference_config'],
        REF / 'hf_quant_config.json': LOCKED_HASHES['reference_quant_config'],
        REF / 'model.safetensors.index.json': LOCKED_HASHES['reference_index'],
    }
    for path, expected in checks.items():
        assert path.is_file() and sha256_file(path) == expected, path
    for name, expected in SOURCE_AUX_HASHES.items():
        path = SOURCE / name
        assert path.is_file() and not path.is_symlink() and sha256_file(path) == expected, path
    assert DIRECTIONS.stat().st_size == DIRECTION_BYTES
    assert sha256_file(DIRECTIONS) == DIRECTION_SHA256
    modelopt_init = pathlib.Path(inspect.getfile(modelopt)).resolve()
    nvfp4_source = pathlib.Path(inspect.getfile(NVFP4QTensor)).resolve()
    fp8_source = pathlib.Path(inspect.getfile(FP8QTensor)).resolve()
    assert sha256_file(modelopt_init) == MODELOPT_INIT_SHA256, modelopt_init
    assert sha256_file(nvfp4_source) == MODELOPT_NVFP4_SHA256, nvfp4_source
    assert sha256_file(fp8_source) == MODELOPT_FP8_SHA256, fp8_source
    reference_receipt = json.loads((REF / 'REFERENCE_RECEIPT.json').read_text())
    assert reference_receipt['complete'] is True
    assert reference_receipt['repo'] == 'nvidia/GLM-5.3-NVFP4'
    assert reference_receipt['revision'] == REFERENCE_REVISION
    assert reference_receipt['reference_shards'] == 50
    assert reference_receipt['indexed_tensors'] == 232403
    assert reference_receipt['expected_data_bytes'] == EXPECTED_DATA_BYTES
    assert reference_receipt['calibrated_scalars'] == 115218
    reference_index = json.loads((REF / 'model.safetensors.index.json').read_text())
    reference_shards = sorted(set(reference_index['weight_map'].values()))
    expected_cache = set(LOCKED_HASHES_NAME for LOCKED_HASHES_NAME in (
        'config.json', 'hf_quant_config.json', 'model.safetensors.index.json',
        'calibrated-scalars.json',
    ))
    expected_cache.update(f'{shard}.header.json' for shard in reference_shards)
    expected_cache.update(f'{shard}.scalars.json' for shard in reference_shards)
    assert set(reference_receipt['cache_sha256']) == expected_cache
    assert {path.name for path in REF.glob('*.header.json')} == {f'{shard}.header.json' for shard in reference_shards}
    assert {path.name for path in REF.glob('*.scalars.json')} == {f'{shard}.scalars.json' for shard in reference_shards}
    for name, expected in reference_receipt['cache_sha256'].items():
        path = REF / name
        assert path.is_file() and not path.is_symlink() and sha256_file(path) == expected, path

    source_manifest = json.loads((PACKAGE_ROOT / 'evidence/source-lfs-manifest.json').read_text())
    canonical = ''.join(
        f"{entry['path']}\t{entry['bytes']}\t{entry['sha256']}\n"
        for entry in source_manifest['entries']
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == SOURCE_LFS_CANONICAL_SHA256
    assert source_manifest['shard_count'] == 282
    assert source_manifest['total_bytes'] == SOURCE_TOTAL_BYTES
    source_entries = {entry['path']: entry for entry in source_manifest['entries']}
    source_receipt = json.loads((RUN / 'source-verification.json').read_text())
    assert source_receipt['complete'] is True
    assert source_receipt['repository'] == 'zai-org/GLM-5.3-BF16'
    assert source_receipt['revision'] == SOURCE_REVISION
    assert pathlib.Path(source_receipt['source_root']).resolve() == SOURCE
    assert source_receipt['manifest_canonical_sha256'] == SOURCE_LFS_CANONICAL_SHA256
    assert source_receipt['shard_count'] == 282
    assert source_receipt['total_bytes'] == SOURCE_TOTAL_BYTES
    assert set(source_receipt['files']) == set(source_entries)
    for name, recorded in source_receipt['files'].items():
        expected = source_entries[name]
        assert recorded['bytes'] == expected['bytes']
        assert recorded['sha256'] == expected['sha256']
        path = SOURCE / name
        stat = path.stat()
        assert not path.is_symlink()
        assert (stat.st_size, stat.st_mtime_ns) == (recorded['bytes'], recorded['mtime_ns']), path

def recipe_binding():
    return {
        'schema': 'blackfrost.glm53-a2p2.binding.v1',
        'recipe_id': 'glm-5.3-bf16-dwm32-a2p2-nvfp4',
        'alpha': ALPHA,
        'passes': 1,
        'rank': 1,
        'layers': [2, 77],
        'excluded_mtp_layer': 78,
        'source_revision': SOURCE_REVISION,
        'reference_revision': REFERENCE_REVISION,
        'direction_sha256': DIRECTION_SHA256,
        'modelopt_commit': '022767c7ab3d7d36211affd85e5c496770cde768',
    }

def input_receipt_binding():
    return {
        'recipe_binding': recipe_binding(),
        'source_verification_sha256': sha256_file(RUN / 'source-verification.json'),
        'reference_receipt_sha256': sha256_file(REF / 'REFERENCE_RECEIPT.json'),
        'direction_sha256': DIRECTION_SHA256,
    }

def log(event, **values):
    print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **values}), flush=True)

def atomic_json(path, value):
    tmp = pathlib.Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)

def header(path):
    with path.open('rb') as f:
        count = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(count))

def classify(key):
    for kind, pattern in PATTERNS.items():
        match = pattern.match(key)
        if match and 2 <= int(match.group(1)) < 78:
            return kind, int(match.group(1))
    return None

def initialize(config=None, threads=THREADS_PER_WORKER, verify_inputs=True):
    global PLAN
    if config is not None:
        configure(config)
    assert CONFIG is not None
    if verify_inputs:
        verify_locked_inputs()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    source_document = json.loads((SOURCE / 'model.safetensors.index.json').read_text())
    source_index = source_document['weight_map']
    reference_index = json.loads((REF / 'model.safetensors.index.json').read_text())
    scalars = json.loads((REF / 'calibrated-scalars.json').read_text())
    with safe_open(DIRECTIONS, framework='pt', device='cpu') as f:
        dirs = f.get_tensor('dirs').float().contiguous()
        direction_meta = f.metadata()
    assert tuple(dirs.shape) == (78, 6144)
    assert bool(torch.isfinite(dirs).all())
    assert torch.allclose(dirs.norm(dim=1), torch.ones(78), atol=2e-5, rtol=2e-5)
    assert direction_meta['template_applied'] == 'false'
    assert direction_meta['revision'] == REFERENCE_REVISION
    assert direction_meta['schema'] == 'blackfrost.glm53_nvfp4.raw32_native_capture.v1'
    assert direction_meta['repo'] == 'nvidia/GLM-5.3-NVFP4'
    assert direction_meta['runtime'] == 'sglang_native_modelopt_mixed'
    assert direction_meta['primary_site'] == 'post_layer_residual_sum'
    assert direction_meta['generated_tokens'] == '0'
    assert direction_meta['weights_modified'] == 'false'
    assert direction_meta['alpha_selected'] == 'none'
    assert not json.loads((SOURCE / 'config.json').read_text()).get('quantization_config')
    counts = collections.Counter(classify(k)[0] for k in source_index if classify(k))
    assert dict(counts) == EXPECTED, counts
    assert sum(k.startswith('model.layers.78.') for k in source_index) == 791
    assert len(source_index) == 59585
    assert len(set(source_index.values())) == 282
    source_manifest = json.loads((PACKAGE_ROOT / 'evidence/source-lfs-manifest.json').read_text())
    assert set(source_index.values()) == {entry['path'] for entry in source_manifest['entries']}
    assert len(reference_index['weight_map']) == 232403
    assert len(scalars) == 115218
    snapshot = {name: {'size': (SOURCE / name).stat().st_size, 'mtime_ns': (SOURCE / name).stat().st_mtime_ns}
                for name in set(source_index.values())}
    PLAN = {'source_index': source_index, 'reference_index': reference_index,
            'scalars': scalars, 'dirs': dirs, 'snapshot': snapshot, 'direction_metadata': direction_meta}
    assert not torch.cuda.is_initialized()

class Reader:
    def __init__(self):
        self.stack = contextlib.ExitStack()
        self.handles = {}
    def get(self, key):
        shard = PLAN['source_index'][key]
        if shard not in self.handles:
            self.handles[shard] = self.stack.enter_context(safe_open(SOURCE / shard, framework='pt', device='cpu'))
        return self.handles[shard].get_tensor(key)
    def close(self):
        self.stack.close()

@torch.inference_mode()
def projected(weight, key, alpha=ALPHA):
    target = classify(key)
    if not target:
        return weight, None
    kind, layer = target
    assert weight.dtype == torch.bfloat16 and weight.ndim == 2 and weight.shape[0] == 6144
    r = PLAN['dirs'][layer]
    work = weight.float()
    before = r @ work
    # Single FP32 projection, followed by BF16 storage precision. No norm
    # restoration or post-quantization correction/iteration is performed.
    work.sub_(alpha * r[:, None] * before[None, :])
    edited = work.to(torch.bfloat16)
    after = r @ edited.float()
    denom = before.square().sum().clamp_min(1e-30)
    signed = float((before * after).sum() / denom)
    error = float((after - (1 - alpha) * before).norm() / before.norm().clamp_min(1e-30))
    assert bool(torch.isfinite(edited).all()), key
    assert abs(signed - (1 - alpha)) < 0.03, (key, signed)
    assert error < 0.05, (key, error)
    audit = {'key': key, 'kind': kind, 'layer': layer, 'prequant_signed_component': signed,
             'prequant_component_relative_error': error}
    return edited, audit

def gate_up_scale(reader, key, divisor, cache):
    prefix = re.sub(r'\.(gate|up)_proj\.weight$', '', key)
    cache_key = (prefix, divisor)
    if cache_key not in cache:
        gate = reader.get(prefix + '.gate_proj.weight')
        up = reader.get(prefix + '.up_proj.weight')
        value = torch.maximum(gate.abs().amax(), up.abs().amax()).float() / divisor
        assert bool(torch.isfinite(value)) and value.item() > 0
        cache[cache_key] = value
    return cache[cache_key]

@torch.inference_mode()
def convert(reader, key, expected, cache, alpha=ALPHA):
    weight = reader.get(key)
    assert weight.dtype == torch.bfloat16, (key, weight.dtype)
    edited, audit = projected(weight, key, alpha)
    kind = expected['dtype']
    module = key[:-len('.weight')] if key.endswith('.weight') else key
    if kind == 'BF16':
        return {key: edited.contiguous()}, audit
    assert key.endswith('.weight'), key
    if kind == 'U8':
        if key.endswith(('.gate_proj.weight', '.up_proj.weight')):
            scale2 = gate_up_scale(reader, key, 6 * 448, cache)
        else:
            scale2 = NVFP4QTensor.get_weights_scaling_factor_2(edited)
        quantized, block_scale, scale2 = NVFP4QTensor.quantize(edited, 16, weights_scaling_factor_2=scale2)
        assert bool(torch.isfinite(block_scale.float()).all()) and bool((block_scale.float() > 0).all()), key
        values = {key: quantized._quantized_data.contiguous(), key + '_scale': block_scale.contiguous(),
                  key + '_scale_2': scale2.reshape(()).float().clone().contiguous()}
        if audit:
            # Measure the stored representation as well as the BF16 operator;
            # quantization error is reported and never used to alter alpha.
            dequantized = quantized.dequantize(dtype=torch.float32, scale=block_scale,
                                               double_scale=scale2, block_sizes={-1: 16})
            r = PLAN['dirs'][audit['layer']]
            before = r @ weight.float()
            component = r @ dequantized
            audit['stored_signed_component'] = float((before * component).sum() / before.square().sum().clamp_min(1e-30))
            audit['weight_relative_quantization_error'] = float((dequantized - edited.float()).norm() / edited.float().norm().clamp_min(1e-30))
            assert math.isfinite(audit['stored_signed_component']) and audit['weight_relative_quantization_error'] < 0.20, audit
    elif kind == 'F8_E4M3':
        scale = (gate_up_scale(reader, key, 448, cache) if key.endswith(('.gate_proj.weight', '.up_proj.weight'))
                 else edited.float().abs().amax() / 448)
        quantized, scale = FP8QTensor.quantize(edited.float(), scales=scale)
        assert bool(torch.isfinite(quantized._quantized_data.float()).all()), key
        values = {key: quantized._quantized_data.contiguous(), key + '_scale': scale.reshape(()).float().clone().contiguous()}
        if audit:
            r = PLAN['dirs'][audit['layer']]
            before = r @ weight.float()
            component = r @ (values[key].float() * scale)
            audit['stored_signed_component'] = float((before * component).sum() / before.square().sum().clamp_min(1e-30))
    else:
        raise RuntimeError(f'Unexpected dtype {kind}: {key}')
    activation = module + '.input_scale'
    values[activation] = torch.tensor(PLAN['scalars'][activation], dtype=torch.float32)
    return values, audit

def check_shapes(values, expected):
    assert set(values) == {k for k in expected if k != '__metadata__'}
    for key, tensor in values.items():
        e = expected[key]
        assert list(tensor.shape) == e['shape'] and tensor.dtype == DTYPES[e['dtype']], (key, tensor.shape, tensor.dtype, e)

def worker(shard):
    final = OUTPUT / shard
    manifest = RUN / 'shard-reports' / (shard + '.json')
    if final.exists() and manifest.exists():
        report = json.loads(manifest.read_text())
        assert report['alpha'] == ALPHA and report['status'] == 'complete'
        assert report['binding'] == recipe_binding()
        assert final.stat().st_size == report['file_bytes']
        assert sha256_file(final) == report['file_sha256']
        return report
    expected = json.loads((REF / (shard + '.header.json')).read_text())
    values = {}
    audits = []
    cache = {}
    reader = Reader()
    started = time.monotonic()
    try:
        source_keys = sorted(k for k in expected if k in PLAN['source_index'])
        for position, key in enumerate(source_keys, 1):
            if expected[key]['dtype'] in ['BF16', 'F32']:
                # Clone avoids retaining source mmap lifetimes while writing.
                weight = reader.get(key)
                value, audit = projected(weight, key)
                values[key] = value.clone().contiguous()
            else:
                new_values, audit = convert(reader, key, expected[key], cache)
                # Companion scales can cross reference shard boundaries. Keep
                # only entries assigned to this shard; the scalar scale values
                # for other shards are recorded for deterministic regeneration.
                for name, value in new_values.items():
                    if name in expected:
                        values[name] = value
            if audit:
                audits.append(audit)
            if position % 200 == 0:
                log('shard_progress', shard=shard, tensors=position, total=len(source_keys), elapsed_seconds=round(time.monotonic()-started))
        # Every calibration input tensor is copied from NVIDIA's original run.
        for key, entry in expected.items():
            if key == '__metadata__' or key in values:
                continue
            if key.endswith('.input_scale') and key in PLAN['scalars']:
                values[key] = torch.tensor(PLAN['scalars'][key], dtype=torch.float32).reshape(entry['shape'])
            elif key.endswith(('_scale', '_scale_2')):
                # NVIDIA's 10-GB shard boundaries occasionally split a weight
                # from its scale. Regenerate that companion deterministically
                # from the same BF16 input and operator, without editing its
                # original weight twice or counting an additional target.
                suffix = '_scale_2' if key.endswith('_scale_2') else '_scale'
                weight_key = key[:-len(suffix)]
                other_shard = PLAN['reference_index']['weight_map'][weight_key]
                other_header = json.loads((REF / (other_shard + '.header.json')).read_text())
                regenerated, _ = convert(reader, weight_key, other_header[weight_key], cache)
                values[key] = regenerated[key]
            else:
                raise RuntimeError(f'Unresolved output tensor {key}')
        check_shapes(values, expected)
        tmp = OUTPUT / ('.' + shard + '.tmp')
        save_file(values, tmp, metadata={'format': 'pt'})
        os.replace(tmp, final)
        report = {'status': 'complete', 'alpha': ALPHA, 'shard': shard, 'file_bytes': final.stat().st_size,
                  'file_sha256': sha256_file(final),
                  'tensor_count': len(values), 'edited_counts': dict(collections.Counter(a['kind'] for a in audits)),
                  'audits': audits, 'elapsed_seconds': round(time.monotonic()-started, 3),
                  'cuda_initialized': torch.cuda.is_initialized(), 'binding': recipe_binding()}
        assert not report['cuda_initialized']
        atomic_json(manifest, report)
        log('shard_complete', shard=shard, seconds=report['elapsed_seconds'], bytes=report['file_bytes'])
        return report
    finally:
        reader.close()

def validate_output_tree(shards):
    allowed_root_files = set(shards) | set(SOURCE_AUX_HASHES) | {
        'config.json', 'hf_quant_config.json', 'model.safetensors.index.json',
        'DWM_BUILD_PROVENANCE.json', 'OFFLINE_VERIFICATION.json',
        'SERIALIZED_SCALE_VERIFICATION.json',
    }
    for path in OUTPUT.rglob('*'):
        assert not path.is_symlink(), path
        relative = path.relative_to(OUTPUT).as_posix()
        if path.is_dir():
            assert relative == 'dwm-research', relative
        elif '/' not in relative:
            assert relative in allowed_root_files, relative
        else:
            assert relative == 'dwm-research/dwm32-directions.safetensors', relative

def probe():
    initialize()
    reader = Reader()
    cache = {}
    all_headers = {}
    for path in REF.glob('*.header.json'):
        all_headers.update(json.loads(path.read_text()))
    reports = []
    try:
        for layer in [2, 3, 40, 77]:
            keys = [f'model.layers.{layer}.self_attn.o_proj.weight']
            keys += ([f'model.layers.{layer}.mlp.down_proj.weight'] if layer == 2 else
                     [f'model.layers.{layer}.mlp.experts.0.down_proj.weight', f'model.layers.{layer}.mlp.experts.0.gate_proj.weight',
                      f'model.layers.{layer}.mlp.experts.0.up_proj.weight', f'model.layers.{layer}.mlp.shared_experts.down_proj.weight'])
            for key in keys:
                for alpha in [0.0, ALPHA]:
                    started = time.monotonic()
                    values, audit = convert(reader, key, all_headers[key], cache, alpha)
                    for name, value in values.items():
                        e = all_headers[name]
                        assert value.dtype == DTYPES[e['dtype']] and list(value.shape) == e['shape'], name
                    if key.endswith(('.gate_proj.weight', '.up_proj.weight')):
                        sibling = key.replace('.gate_proj.', '.up_proj.') if '.gate_proj.' in key else key.replace('.up_proj.', '.gate_proj.')
                        sibling_values, _ = convert(reader, sibling, all_headers[sibling], cache, alpha)
                        suffix = '_scale_2' if values[key].dtype == torch.uint8 else '_scale'
                        assert torch.equal(values[key+suffix], sibling_values[sibling+suffix])
                    reports.append({'key': key, 'alpha': alpha, 'seconds': time.monotonic()-started, 'audit': audit})
        assert not torch.cuda.is_initialized()
        atomic_json(RUN / 'preflight-probes.json', {
            'passed': True,
            'cases': reports,
            'cuda_initialized': False,
            'input_binding': input_receipt_binding(),
        })
        log('probe_complete', cases=len(reports), cuda_initialized=False)
    finally:
        reader.close()

def build(workers):
    assert workers == WORKERS
    initialize(CONFIG)
    preflight_path = RUN / 'preflight-probes.json'
    assert preflight_path.is_file()
    preflight = json.loads(preflight_path.read_text())
    assert preflight['passed'] is True and preflight['cuda_initialized'] is False
    assert preflight['input_binding'] == input_receipt_binding()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = sorted(set(PLAN['reference_index']['weight_map'].values()))
    assert len(shards) == 50
    binding_path = RUN / 'RECIPE_BINDING.json'
    binding = recipe_binding()
    if binding_path.exists():
        assert json.loads(binding_path.read_text()) == binding
        validate_output_tree(shards)
    else:
        assert not any(OUTPUT.iterdir())
        assert not (RUN / 'shard-reports').exists()
        atomic_json(binding_path, binding)
    (RUN / 'shard-reports').mkdir(exist_ok=True)
    available = os.statvfs(OUTPUT).f_bavail * os.statvfs(OUTPUT).f_frsize
    needed = sum(v['data_offsets'][1]-v['data_offsets'][0] for p in REF.glob('*.header.json') for k,v in json.loads(p.read_text()).items() if k!='__metadata__')
    assert needed == EXPECTED_DATA_BYTES, needed
    existing = sum(p.stat().st_size for p in OUTPUT.glob('*.safetensors'))
    assert available + existing > needed + 100_000_000_000, (available, needed)
    plan_record = {'alpha': ALPHA, 'passes': 1, 'rank': 1, 'skip_early': 2, 'exclude_mtp': 78,
                   'source': str(SOURCE), 'source_repo': 'zai-org/GLM-5.3-BF16',
                   'source_revision': SOURCE_REVISION,
                   'direction_artifact': str(DIRECTIONS), 'direction_metadata': PLAN['direction_metadata'],
                   'activation_calibration': 'NVIDIA original calibrated scales reused; no fresh post-DWM activation calibration',
                   'nvidia_reference_revision': REFERENCE_REVISION,
                   'weight_scales': 'recomputed after DWM; gate/up scales share per-expert joint amax',
                   'arithmetic': 'FP32 projection followed by BF16 rounding and ModelOpt quantization',
                   'norm_restoration': False, 'quantization_correction_iterations': 0,
                   'modelopt_source': '0.48.0dev-022767c7', 'modelopt_commit': '022767c7ab3d7d36211affd85e5c496770cde768',
                   'expected_edits': EXPECTED, 'source_snapshot': PLAN['snapshot'],
                   'expected_output_data_bytes': needed, 'workers': workers,
                   'cpu_threads_per_worker': THREADS_PER_WORKER, 'recipe_binding': binding}
    plan_record['preflight_receipt_sha256'] = sha256_file(preflight_path)
    atomic_json(RUN / 'build-plan.json', plan_record)
    log('build_started', workers=workers, expected_shards=50, expected_data_bytes=needed)
    # Spawn avoids forking a running Torch thread pool. Each worker imports
    # its own CPU-only ModelOpt codec and read-only metadata bank.
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=initialize,
            initargs=(CONFIG, THREADS_PER_WORKER, False)) as pool:
        futures = [pool.submit(worker, shard) for shard in shards]
        reports = []
        for future in concurrent.futures.as_completed(futures):
            reports.append(future.result())
            log('build_progress', completed_shards=len(reports), total_shards=len(shards))
    counts = collections.Counter()
    for report in reports:
        counts.update(report['edited_counts'])
    assert dict(counts) == EXPECTED, counts
    # Preserve source tokenizer/generation files; NVIDIA's exact runtime and
    # quantization metadata define the serialized checkpoint's loading format.
    # Copy only the four pinned runtime files. Never sweep a source directory:
    # it may contain credentials, logs, local notes, symlinks, or stale files.
    for name, expected in SOURCE_AUX_HASHES.items():
        path = SOURCE / name
        assert path.is_file() and not path.is_symlink() and sha256_file(path) == expected
        shutil.copy2(path, OUTPUT / name)
    shutil.copy2(REF / 'config.json', OUTPUT / 'config.json')
    shutil.copy2(REF / 'hf_quant_config.json', OUTPUT / 'hf_quant_config.json')
    shutil.copy2(REF / 'model.safetensors.index.json', OUTPUT / 'model.safetensors.index.json')
    research = OUTPUT / 'dwm-research'
    research.mkdir(exist_ok=True)
    shutil.copy2(DIRECTIONS, research / 'dwm32-directions.safetensors')
    # Research tensors stay outside the model loader's root shard glob.
    public_provenance = {k:v for k,v in plan_record.items()
                         if k not in {'source', 'direction_artifact', 'source_snapshot'}}
    public_provenance['direction_asset'] = {
        'name': DIRECTIONS.name, 'bytes': DIRECTION_BYTES, 'sha256': DIRECTION_SHA256,
    }
    atomic_json(OUTPUT / 'DWM_BUILD_PROVENANCE.json', public_provenance)
    verify()

def verify():
    global PLAN
    if PLAN is None:
        initialize(CONFIG)
    plan = json.loads((RUN / 'build-plan.json').read_text())
    assert plan['recipe_binding'] == recipe_binding()
    assert plan['expected_output_data_bytes'] == EXPECTED_DATA_BYTES
    assert plan['preflight_receipt_sha256'] == sha256_file(RUN / 'preflight-probes.json')
    reader = Reader()
    counts = collections.Counter()
    mtp = preserved = input_scales = 0
    stored_components = []
    relative_errors = []
    try:
        shards = sorted(set(PLAN['reference_index']['weight_map'].values()))
        assert len(shards) == 50
        validate_output_tree(shards)
        assert {p.name for p in OUTPUT.glob('*.safetensors')} == set(shards)
        for shard in shards:
            expected = json.loads((REF / (shard + '.header.json')).read_text())
            actual = header(OUTPUT / shard)
            assert set(actual)-{'__metadata__'} == set(expected)-{'__metadata__'}, shard
            for key in actual:
                if key != '__metadata__':
                    assert actual[key]['dtype'] == expected[key]['dtype'] and actual[key]['shape'] == expected[key]['shape'], key
            report = json.loads((RUN / 'shard-reports' / (shard + '.json')).read_text())
            assert report['binding'] == recipe_binding()
            assert report['file_sha256'] == sha256_file(OUTPUT / shard)
            counts.update(report['edited_counts'])
            for audit in report['audits']:
                if 'stored_signed_component' in audit:
                    stored_components.append(audit['stored_signed_component'])
                if 'weight_relative_quantization_error' in audit:
                    relative_errors.append(audit['weight_relative_quantization_error'])
            with safe_open(OUTPUT / shard, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key.endswith('.input_scale'):
                        tensor=f.get_tensor(key)
                        assert tensor.item() == PLAN['scalars'][key], key
                        input_scales += 1
                    elif key in PLAN['source_index'] and expected[key]['dtype'] in ['BF16','F32'] and not classify(key):
                        # Byte comparisons cover clean MTP, attention inputs,
                        # routing, norms, embedding and all unaffected BF16.
                        assert torch.equal(f.get_tensor(key).reshape(-1).view(torch.uint8), reader.get(key).reshape(-1).view(torch.uint8)), key
                        preserved += 1
                        mtp += key.startswith('model.layers.78.')
            log('verify_shard', shard=shard)
        assert dict(counts)==EXPECTED and mtp==791 and input_scales==57609, (counts,mtp,input_scales)
        assert preserved == EXPECTED_PRESERVED_BF16_F32, preserved
        assert json.loads((OUTPUT/'config.json').read_text())==json.loads((REF/'config.json').read_text())
        assert json.loads((OUTPUT/'hf_quant_config.json').read_text())==json.loads((REF/'hf_quant_config.json').read_text())
        assert sha256_file(OUTPUT/'config.json') == LOCKED_HASHES['reference_config']
        assert sha256_file(OUTPUT/'hf_quant_config.json') == LOCKED_HASHES['reference_quant_config']
        assert sha256_file(OUTPUT/'model.safetensors.index.json') == LOCKED_HASHES['reference_index']
        for name, expected in SOURCE_AUX_HASHES.items():
            assert sha256_file(OUTPUT/name) == expected, name
        embedded_direction = OUTPUT/'dwm-research'/'dwm32-directions.safetensors'
        assert embedded_direction.stat().st_size == DIRECTION_BYTES
        assert sha256_file(embedded_direction) == DIRECTION_SHA256
        public_provenance = json.loads((OUTPUT/'DWM_BUILD_PROVENANCE.json').read_text())
        assert 'source' not in public_provenance and 'direction_artifact' not in public_provenance
        assert public_provenance['recipe_binding'] == recipe_binding()
        assert public_provenance['direction_asset'] == {
            'name': DIRECTIONS.name, 'bytes': DIRECTION_BYTES, 'sha256': DIRECTION_SHA256,
        }
        for name, expected_stat in plan['source_snapshot'].items():
            stat=(SOURCE/name).stat()
            assert {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}==expected_stat, name
        assert not torch.cuda.is_initialized()
        disk=os.statvfs(OUTPUT)
        weight_file_bytes = sum((OUTPUT/s).stat().st_size for s in shards)
        assert weight_file_bytes == EXPECTED_OUTPUT_WEIGHT_BYTES, weight_file_bytes
        result={'complete':True,'alpha':ALPHA,'output':str(OUTPUT),'weight_shards':50,
                'weight_index_tensors':232403,'edited_counts':dict(counts),'mtp_tensors_unchanged':mtp,
                'unmodified_bf16_and_f32_tensors_byte_equal':preserved,'nvidia_activation_scales_preserved':input_scales,
                'nvidia_layout_matches':True,'source_unchanged':True,'cuda_initialized':False,
                'weight_file_bytes':weight_file_bytes,
                'available_bytes':disk.f_bavail*disk.f_frsize,'loaded':False,'transferred_to_b200':False,
                'fresh_post_dwm_activation_calibration':False,'generation_tested':False,
                'stored_dwm_signed_component':{'count':len(stored_components),'min':min(stored_components),'max':max(stored_components),'mean':sum(stored_components)/len(stored_components)},
                'quantization_weight_relative_error':{'count':len(relative_errors),'min':min(relative_errors),'max':max(relative_errors),'mean':sum(relative_errors)/len(relative_errors)},
                'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        atomic_json(RUN/'build-complete.json',result)
        public_result = {k:v for k,v in result.items() if k not in {'output', 'available_bytes'}}
        atomic_json(OUTPUT/'OFFLINE_VERIFICATION.json',public_result)
        log('build_complete',**result)
    finally:
        reader.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['probe','build','verify'])
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--directions',required=True)
    parser.add_argument('--reference-dir',required=True)
    args=parser.parse_args()
    configure({'source':args.source, 'output':args.output, 'run':args.run_dir,
               'directions':args.directions, 'reference':args.reference_dir})
    if args.mode=='probe':probe()
    elif args.mode=='build':build(WORKERS)
    else:verify()
