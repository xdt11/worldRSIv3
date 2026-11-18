"""Declarative harness selection and application defaults; no runtime singletons."""
from dataclasses import dataclass, field, asdict
import copy
import hashlib
import json
import math
import re

DEFAULT_CONFIG = {
    'base_url': 'https://llmgate.shop/v1',
    'models': ['claude-opus-4-8', 'gpt-5.6-sol', 'gpt-6-astra',
               'glm-5.3', 'kimi-k3', 'kimi-k3.1-pro', 'qwen3.8-max'],
    'model': None,
    'critic_model': None,
    'max_runs': 3,
    'max_calls_per_run': 50,
    'max_output_tokens': None,
    'max_total_tokens_per_run': 3_000_000,
    'run_critic_max_calls': 6,
    'run_critic_max_total_tokens': 200_000,
    'max_repeated_tool_outcomes': 3,
    'critic_visual_frames': 6,
    'seed': None,
    'request_timeout_s': 240,
    'runtime': {'parallel_tool_calls': True, 'tool_output_chars':4096,
                'overview_frames_per_video':3,
                'protocol_repair_attempts':1,
                # A configurable working budget, not a claim about provider/model capacity.
                'context_window_tokens': 65536, 'context_summarization_cutoff': 0.75},
    'sandbox': {'image': 'worldrsi:newton', 'cpus': 4, 'memory': '12g',
                'pids_limit': 256, 'gpus': 'auto', 'startup_timeout_s': 120, 'cleanup_timeout_s': 60},
    'live': {'device': 'auto', 'max_sessions': 3, 'idle_timeout_s': 60, 'max_fps': 24, 'width': 768, 'startup_timeout_s': 120, 'max_states': 1000,
             'max_state_bytes': 512 * 1024 * 1024, 'max_message_bytes': 16 * 1024 * 1024, 'render_timeout_s': 30},
    'harness': {},
    'evaluation': {
        'source': 'data/resources/benchmark/4DCodeBench',
        'root': 'data/resources/benchmark',
        'python_packages': 'data/resources/scorer/packages',
        'runtime': 'cuda', 'gpus': 'auto',
        'image': 'worldrsi:local', 'types': ['all'],
        'cpus': 4, 'memory': '12g', 'timeout_s': 1200,
        'video_frame_policy': 'all_reference_frames', 'video_timestamps_s': None,
        'video_max_timestamp_error_s': 0.000001,
        'vqa_enabled': True, 'vqa_model': 'kimi-k3', 'vqa_timeout_s':1000,
    },
    'adoption_policy': None,
    'outer_rsi': True,
    'trial_enabled': False,
    'rollout': {'enabled': True, 'duration_s': 160, 'max_duration_s': 160, 'render_fps': 10,
                'renderer_url': 'http://127.0.0.1:8786', 'render_timeout_s': 1800,
                'agent_max_calls': 50, 'agent_max_tokens': 300000, 'memory': '16g'},
    'scheduler': {'max_cpus':128,'max_memory':None,'memory_fraction':0.85,'max_gpus':8,
                  'min_gpu_free_memory':'8g','max_jobs':128},
    'campaign': {'outer_rsi': True, 'trial_enabled': True, 'auto_adopt': True,
                 'test_workers': 5, 'max_parallel_tasks': 6, 'benchmark_manifest_sha256': None},
}

def merge_config(base, patch):
    """Merge nested settings without modifying defaults or a historical record."""
    result = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def normalize_config(value=None):
    result = merge_config(DEFAULT_CONFIG, value or {})
    # Migrate only the controller/new-task defaults. Frozen historical task
    # records are never rewritten and their old backend is not executed.
    if result['sandbox'].get('image') == 'worldrsi:local':
        result['sandbox']['image'] = 'worldrsi:newton'
        result['sandbox'].pop('resource_identity_sha256', None)
    if not isinstance(result.get('runtime'),dict):
        raise ValueError('Runtime policy must be a mapping')
    for key in ('image_history_limit','image_anchor_limit','initial_image_limit'):
        result['runtime'].pop(key,None)
    if result['runtime'].get('context_window_tokens') is None:
        result['runtime']['context_window_tokens']=DEFAULT_CONFIG['runtime']['context_window_tokens']
    if result['max_runs'] is None:
        result['max_runs'] = 3
    return result


def normalize_threat_config(value=None, *, defaults=None):
    """Scene editing has its own new-task budget, retaining explicit choices."""
    base = normalize_config(defaults)
    base.update(max_runs=1, max_calls_per_run=50, max_total_tokens_per_run=None,
                run_critic_max_total_tokens=None)
    return normalize_config(merge_config(base, value or {}))


def validate_config(cfg):
    if not isinstance(cfg, dict) or not isinstance(cfg.get('sandbox'), dict) or not isinstance(cfg.get('evaluation'), dict):
        raise ValueError('Configuration requires sandbox and evaluation mappings')
    scheduling=cfg.get('scheduler',DEFAULT_CONFIG['scheduler'])
    if not isinstance(scheduling,dict) or set(scheduling)-set(DEFAULT_CONFIG['scheduler']):
        raise ValueError('Unknown scheduler configuration fields')
    for key in ('max_cpus','max_jobs'):
        value=scheduling.get(key,DEFAULT_CONFIG['scheduler'][key])
        if type(value) is not int or value<1:raise ValueError(f'scheduler.{key} must be a positive integer')
    value=scheduling.get('max_gpus',8)
    if type(value) is not int or value<0:raise ValueError('scheduler.max_gpus must be a nonnegative integer')
    fraction=scheduling.get('memory_fraction',.85)
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<fraction<=1:
        raise ValueError('scheduler.memory_fraction must be in (0, 1]')
    from orchestration.resources import memory_bytes
    for key in ('max_memory','min_gpu_free_memory'):
        value=scheduling.get(key,DEFAULT_CONFIG['scheduler'][key])
        if key=='min_gpu_free_memory' and value is None:
            raise ValueError('scheduler.min_gpu_free_memory must be a positive memory size')
        if value is not None:memory_bytes(value)
    if not isinstance(cfg.get('models'), list) or any(not isinstance(m, str) for m in cfg['models']):
        raise ValueError('models must be a list of model names')
    for key in ('model', 'critic_model'):
        if cfg.get(key) is not None and cfg[key] not in cfg['models']:
            raise ValueError(f'{key} must be null or one of the configured models')
    if not isinstance(cfg.get('base_url'), str):
        raise ValueError('base_url must be a string')
    if type(cfg.get('request_timeout_s')) not in (int, float) or cfg['request_timeout_s'] <= 0:
        raise ValueError('request_timeout_s must be positive')
    for key in ('startup_timeout_s', 'cleanup_timeout_s'):
        setting=cfg['sandbox'].get(key,DEFAULT_CONFIG['sandbox'][key])
        if type(setting) not in (int,float) or not math.isfinite(setting) or setting<=0:
            raise ValueError(f'sandbox.{key} must be positive and finite')
    runtime = cfg.get('runtime',DEFAULT_CONFIG['runtime'])
    if not isinstance(runtime,dict) or set(runtime)-set(DEFAULT_CONFIG['runtime']):
        raise ValueError('Unknown runtime policy fields')
    if type(runtime.get('parallel_tool_calls',True)) is not bool:
        raise ValueError('runtime.parallel_tool_calls must be boolean')
    window = runtime.get('context_window_tokens')
    cutoff = runtime.get('context_summarization_cutoff', 0.75)
    if type(window) is not int or window <= 0:
        raise ValueError('runtime.context_window_tokens must be a positive working budget')
    if type(cutoff) not in (int, float) or not 0 < cutoff <= 1:
        raise ValueError('runtime.context_summarization_cutoff must be in (0, 1]')
    output=cfg.get('max_output_tokens') or 8192  # Space reserved for context planning
    if type(output) is int and output>0 and window<=output+min(output,2048):
        raise ValueError('Working context budget must exceed max_output_tokens plus summary output reserve')
    for key in ('overview_frames_per_video','tool_output_chars','protocol_repair_attempts'):
        setting = runtime.get(key,DEFAULT_CONFIG['runtime'][key])
        if type(setting) is not int or setting < (0 if key=='protocol_repair_attempts' else 1):
            raise ValueError(f'runtime.{key} must be a positive integer (repair attempts may be zero)')
    if cfg.get('seed') is not None and (type(cfg['seed']) is not int or not 0 <= cfg['seed'] <= 2**32 - 1):
        raise ValueError('seed must be null or uint32')
    missing = [key for key in ('model', 'max_runs') if not cfg.get(key)]
    for key in ('max_runs', 'max_calls_per_run', 'max_output_tokens', 'max_total_tokens_per_run',
                'run_critic_max_calls', 'run_critic_max_total_tokens', 'max_repeated_tool_outcomes'):
        val = cfg.get(key)
        if val is not None and (isinstance(val, bool) or not isinstance(val, int) or val <= 0):
            raise ValueError(f'{key} must be a positive integer or null for an unlimited call/token budget')
    repeat = cfg.get('max_repeated_tool_outcomes')
    if repeat is not None and repeat < 2:
        raise ValueError('max_repeated_tool_outcomes must be at least two, or null to disable')
    if type(cfg.get('critic_visual_frames', 6)) is not int or cfg.get('critic_visual_frames', 6) <= 0:
        raise ValueError('critic_visual_frames must be a positive integer')
    vqa_timeout=cfg['evaluation'].get('vqa_timeout_s',1000)
    if type(vqa_timeout) not in (int,float) or not math.isfinite(vqa_timeout) or vqa_timeout<=0:
        raise ValueError('evaluation.vqa_timeout_s must be positive and finite')
    hz = cfg['evaluation'].get('video_sample_hz')
    if type(cfg['evaluation'].get('vqa_enabled',True)) is not bool:raise ValueError('evaluation.vqa_enabled must be boolean')
    if cfg['evaluation'].get('vqa_model','kimi-k3')!='kimi-k3':raise ValueError('The VQA judge is fixed to kimi-k3')
    if hz is not None and (type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0):
        raise ValueError('evaluation.video_sample_hz must be positive and finite, or null')
    live = cfg.get('live', {})
    if not isinstance(live, dict) or set(live) - set(DEFAULT_CONFIG['live']):
        raise ValueError('Unknown live rendering configuration')
    device = live.get('device', 'auto')
    if not isinstance(device, str):
        raise ValueError('live.device must be a string')
    if device not in {'auto', 'cpu'}:
        if not re.fullmatch(r'\d+|GPU-[a-fA-F0-9-]+', device):
            raise ValueError('live.device must be auto, cpu, GPU index or UUID')
    for key, low, high, default in [('max_sessions', 1, 8, 3), ('idle_timeout_s', 5, 600, 60),
                                   ('max_fps', 1, 60, 24), ('width', 128, 1280, 768)]:
        setting = live.get(key, default)
        if type(setting) is not int or not low <= setting <= high:
            raise ValueError(f'live.{key} must be an integer in [{low}, {high}]')
    for key in ('startup_timeout_s','max_states','max_state_bytes','max_message_bytes','render_timeout_s'):
        value = live.get(key, DEFAULT_CONFIG['live'][key])
        if type(value) is not int or value <= 0:
            raise ValueError(f'live.{key} must be a positive integer')
    for key in ('outer_rsi', 'trial_enabled'):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if not isinstance(cfg.get('campaign', {}), dict):
        raise ValueError('campaign must be a mapping')
    manifest = cfg.get('campaign', {}).get('benchmark_manifest_sha256')
    if manifest is not None and (not isinstance(manifest, str) or not re.fullmatch('[0-9a-f]{64}', manifest)):
        raise ValueError('campaign.benchmark_manifest_sha256 must be a SHA-256 string or null')
    for key in ('outer_rsi', 'trial_enabled', 'auto_adopt'):
        if key in cfg.get('campaign', {}) and type(cfg['campaign'][key]) is not bool:
            raise ValueError(f'campaign.{key} must be boolean')
    for key in ('test_workers', 'max_parallel_tasks'):
        if type(value := cfg.get('campaign',{}).get(key,DEFAULT_CONFIG['campaign'][key])) is not int or value < 1:
            raise ValueError(f'campaign.{key} must be a positive integer')
    if not cfg.get('base_url', '').startswith(('https://', 'http://localhost', 'http://127.0.0.1')):
        missing.append('valid base_url')
    # Metrics are reference evidence. Missing seed/model weights/protocol/policy
    # stay explicit in evaluator output and do not veto manual/experiment mode.
    return missing


@dataclass(frozen=True)
class HarnessSpec:
    plugins: dict = field(default_factory=lambda: {
        'llm': 'responses', 'sandbox': 'docker', 'renderer': 'newton-warp',
        'world': 'yaml', 'critic': 'evidence', 'evaluator': '4dcodebench', 'context': 'world',
        'acceptance': 'manual', 'viewer': 'newton_live'})
    tools: tuple = ('bash', 'probe_media', 'extract_frames', 'crop_image', 'estimate_depth', 'segment_image',
                    'view_image', 'set_world_state', 'submit_scene', 'scene_state', 'observe_scene')
    prompts: dict = field(default_factory=lambda: {
        'agent': 'harness/prompts/reconstruction.md', 'run_critic': 'harness/prompts/run_critic.md',
        'task_critic': 'harness/prompts/task_critic.md',
        'threat_agent': 'harness/prompts/threat.md',
        'threat_run_critic': 'harness/prompts/threat_run_critic.md',
        'threat_task_critic': 'harness/prompts/threat_task_critic.md',
        'rollout_evaluator': 'harness/prompts/rollout_evaluator.md'})
    documents: tuple = ('docs/world-state.md', 'docs/newton-runtime.md')

    @classmethod
    def parse(cls, value=None):
        value = {} if value is None else value
        if not isinstance(value, dict) or set(value) - {'plugins', 'tools', 'prompts', 'documents'}:
            raise ValueError('Unknown harness spec fields')
        if any(key in value and not isinstance(value[key],dict) for key in ('plugins','prompts')):
            raise ValueError('Harness plugins and prompts must be mappings')
        if any(key in value and (not isinstance(value[key],(list,tuple)) or
                                any(not isinstance(item,str) for item in value[key])) for key in ('tools','documents')):
            raise ValueError('Harness tools and documents must be lists of names')
        default = cls()
        spec = cls(plugins={**default.plugins, **value.get('plugins', {})},
                   tools=tuple(value.get('tools', default.tools)),
                   prompts={**default.prompts, **value.get('prompts', {})},
                   documents=tuple(value.get('documents', default.documents)))
        if spec.plugins != default.plugins:
            raise ValueError('Execution components are fixed; unsupported plugin overrides are not allowed')
        if not spec.tools or len(spec.tools) != len(set(spec.tools)) or any(not isinstance(v, str) for v in spec.tools):
            raise ValueError('Harness tools must be unique plugin names')
        if spec.tools != default.tools:
            raise ValueError('Role toolsets are fixed; unsupported tool overrides are not allowed')
        if set(spec.prompts) != set(default.prompts) or any(not isinstance(p,str) or not re.fullmatch(r'harness/prompts/[A-Za-z0-9_.-]+\.md', p) for p in spec.prompts.values()):
            raise ValueError('Harness prompts must be Markdown files under harness/prompts/')
        if any(not re.fullmatch(r'docs/[A-Za-z0-9_.-]+\.md', p) for p in spec.documents):
            raise ValueError('Context documents must be Markdown files under docs/')
        return spec

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
