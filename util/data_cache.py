from pathlib import Path
import pickle


class DataCache:
    """
    数据缓存工具类
    用于缓存数据，避免重复计算
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(__file__).parent / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def save(self, key: str, data):
        with open(self._get_cache_path(key), "wb") as f:
            pickle.dump(data, f)

    def load(self, key: str):
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def exists(self, key: str) -> bool:
        return self._get_cache_path(key).exists()

    def clear(self, key: str):
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()


# 全局数据缓存实例
dc = DataCache("cache")


def cache_wrapper(cache_key, regenerate_flag, func, after_load=None):
    def wrapper(*args, **kwargs):
        if regenerate_flag or not dc.exists(cache_key):
            result = func(*args, **kwargs)
            dc.clear(cache_key)
            dc.save(cache_key, result)
            print(f"数据已保存到缓存: {cache_key}")
            if after_load:
                result = after_load(result)
            return result
        else:
            loaded = dc.load(cache_key)
            print(f"从缓存加载数据: {cache_key}")
            if after_load:
                loaded = after_load(loaded)
            return loaded

    return wrapper
