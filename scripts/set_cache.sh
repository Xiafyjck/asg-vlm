# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# cz 环境缓存重定向:按 XDG Base Directory 规范把缓存与数据目录指到共享卷,
# 实现"4090 下载,H200 直读"。
#
# 用法:两台机器都在 ~/.bashrc 末尾加一行(bash 子进程 export 传不回父 shell,必须 source):
#   source /绝对路径/asg_vlm/scripts/set_cache.sh
# 下面两个路径必须先改成共享卷上的绝对路径,两机一致。

export XDG_CACHE_HOME=/CHANGE_ME/xdg/cache
export XDG_DATA_HOME=/CHANGE_ME/xdg/data

# 守 XDG 的工具自动跟随:uv 的包缓存($XDG_CACHE_HOME/uv)、
# uv 托管的 python 解释器($XDG_DATA_HOME/uv/python)、huggingface($XDG_CACHE_HOME/huggingface)。
# modelscope 不守 XDG,单独指过去。
export MODELSCOPE_CACHE="$XDG_CACHE_HOME/modelscope"

# 两机统一用 uv 托管解释器,不落各机系统 python
export UV_PYTHON_PREFERENCE=only-managed

if [ ! -d "$XDG_CACHE_HOME" ]; then
  echo "set_cache.sh: XDG_CACHE_HOME=$XDG_CACHE_HOME 不存在,先修改本文件里的路径并创建目录" >&2
fi
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<