#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-8B',cache_dir = '/autodl-tmp/models')