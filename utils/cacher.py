import os
import pickle
import hashlib
from functools import wraps
# import sys
import time

def create_directory_for_file(file_path):
    """为指定的文件路径生成存储的文件夹"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(f"Directory {directory} already exists.")

def check_nonsence(obj) -> bool:
    '''
    """check an obkect is none or empty string

    Returns:
        bool: nonsence or not
    """    
    '''
    if obj is None:
        return True
    if isinstance(obj, str) and len(obj) == 0:
        return True
    else:
        return False

class Cacher():
    def __init__(self, cache_dir='./utils/cached', auto_clear_days=30):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        
        self.auto_clear_days = auto_clear_days
        self.auto_clear_timestamp = int(time.time()) - auto_clear_days * 24 * 3600  # 在这个之前的都要被清除
        
        # if not os.path.exists(os.path.join(self.cache_dir, 'key2timestamp.pickle')):
        #     self.key2timestamp = dict()
        #     with open(os.path.join(self.cache_dir, 'key2timestamp.pickle'), 'wb') as f:
        #         pickle.dump(self.key2timestamp, f)
        # else:
        #     with open(os.path.join(self.cache_dir, 'key2timestamp.pickle'), 'rb') as f:
        #         self.key2timestamp = pickle.load(f)
        
        # self.auto_clear_days = auto_clear_days
        

    def rm_cache(self):
        assert not self.cache_dir.startswith('/'), 'Cache directory cannot start with /'
        assert '..' not in self.cache_dir, 'Cache directory cannot go up to the parent directory for safety issues'
        os.system('rm -rf ' + self.cache_dir)
        print(f'All cache files removed from {self.cache_dir}')
        
class CacheDecorator(Cacher):
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a unique cache file name based on function name and arguments
            hash_key = f'{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}'  # 32 位的 key
            cache_key = f"{func.__name__}_{hash_key}"
            sub_file = f'{hash_key[:3]}.pickle'  
            
            if 'reload' not in kwargs or kwargs['reload'] == False:  # allow reload                
                # Use the first 3 characters， 这样可以把存储的文件限制在几千个（16**3），如果是4 一下子又上万了 
                # TODO 这样会无法处理多进程内的冲突了，不过进程不安全的结果只是丢失cache，可以接受
                cache_file = os.path.join(self.cache_dir, sub_file)
                
                # Check if cache file exists
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        # print(f'Loading result from cache file: {cache_file}')
                        cached_d = pickle.load(f)
                        if cache_key in cached_d:
                            res = cached_d[cache_key]['result']
                            if not check_nonsence(res):
                                print(f'load cached results from {cache_file}')
                                # print(f'Loaded result from cache file: {cache_file}')
                                return res
                            else:
                                print('loaded result is nonsence, reload')
                                
            # Call the original function
            if 'reload' in kwargs:
                del kwargs['reload']
            result = func(*args, **kwargs)
            
            # Write result to cache file
            cached_file = os.path.join(self.cache_dir, sub_file)
            
            if not os.path.exists(cached_file):  # store the cache file if not exists
                with open(cached_file, 'wb') as f:
                    cache_d = dict()
                    cache_d[cache_key] = {'time': int(time.time()), 'result': result}
                    pickle.dump(cache_d, f)
                    # print(f'Saving result to cache file: {cache_file}')
            else:     # store and remove outdated cache
                with open(cached_file, 'rb') as f:
                    cache_d = pickle.load(f)
                for k, v in cache_d.items():
                    if v['time'] < self.auto_clear_timestamp:
                        del cache_d[k]
                cache_d[cache_key] = {'time': int(time.time()), 'result': result}
                with open(cached_file, 'wb') as f:
                    pickle.dump(cache_d, f)
                    # print(f'Saving result to cache file: {cache_file}')
            return result
        
        return wrapper
    
    
    

class VariableCacher(Cacher):
    def cacheVar(self, data, cache_key):
        # 生成 cache_key 的哈希值
        hash_object = hashlib.sha256(cache_key.encode())
        hashed_key = hash_object.hexdigest()
        
        cache_file = os.path.join(self.cache_dir, hashed_key)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
            print(f'Saving result to cache file: {cache_file}')
    
    def loadVar(self, cache_key):
        # 生成 cache_key 的哈希值
        hash_object = hashlib.sha256(cache_key.encode())
        hashed_key = hash_object.hexdigest()
        
        cache_file = os.path.join(self.cache_dir, hashed_key)
        if not os.path.exists(cache_file):
            raise Exception(f'Cache file not found for key: {cache_key}')
        
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            print(f'Loaded result from cache file: {cache_file}')
            return data