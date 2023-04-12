# import time
# import logging
# import subprocess
import math
import os
import sys
# import numpy as np
from tqdm import tqdm
import shutil
import types
# import tempfile
# import pickle
import inspect
import importlib
from threading import Lock
import traceback
import concurrent.futures
import multiprocessing as mp
import multiprocessing.dummy as mp_dummy
import queue
# import argparse
from .logger_utils import setup_logger

__all__ = ["parallel_wrapper"]



logger = setup_logger('/tmp/parallel_manager/parallel.log', name=__name__)

_RUNNER_MAIN_MODULE_NAME = '__parallel_executor_original_main__'
_import_lock = Lock()
def _import_module_in_runner(module_name, file_name):
    """Import a module in runner."""
    if module_name == '__main__':
        module_name = _RUNNER_MAIN_MODULE_NAME
        with _import_lock:
            module = sys.modules.get(module_name, None)
            if module is not None:
                return module
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module
            return module
    else:
        return importlib.import_module(module_name)


def batch_wrapper(array, split_num=None, sum_count=None):
    """
        return:
            if with_enum:
                i, begin_index, end_index, array[begin_index:end_index]
            else:
                begin_index, end_index, array[begin_index:end_index]
    """
    if split_num is None and sum_count is None:
        raise Exception('You must set split_num or sum_count')
    num_array = len(array)
    if split_num is None:
        split_num = math.ceil(num_array * 1.0 / sum_count)
    sum_count = math.ceil(num_array * 1.0 / split_num)

    for ith in range(sum_count):
        begin_index = ith * split_num
        end_index = min((ith + 1) * split_num, num_array)
        yield ith, begin_index, end_index, array[begin_index:end_index]


def is_lab_notebook():
    import re
    try:
        import psutil
        return any(re.search('jupyter-lab', x)
                   for x in psutil.Process().parent().cmdline())
    except ImportError:
        return False


def get_signature():
    import hashlib
    main_path = os.path.abspath(sys.argv[0])
    point_dir_path = '_'.join(main_path.split('/')[-2:])
    to_hash = [main_path]+[p for p in sys.argv[1:] if not p.startswith('-')]
    md5 = hashlib.md5(''.join(to_hash).encode(encoding='UTF-8')).hexdigest()[:8]
    final_sig = point_dir_path+'_'+str(md5)
    return final_sig

class _Func:
    def __init__(self, fn):
        self.in_lab = is_lab_notebook()
        if self.in_lab:
            self.fn = fn
        else:
            self.module_name = fn.__module__
            self.file_name = sys.modules[fn.__module__].__file__
            self.name = fn.__name__

    def load_in_runner(self):
        if self.in_lab:
            return self.fn
        else:
            mod=_import_module_in_runner(self.module_name, self.file_name)
            return getattr(mod, self.name)


class QueueSignal(object):
    def __init__(self, info):
        self.info = info

def queue_to_iterable(queue):
    while True:
        logger.debug('queue_to_iterable ID %s',id(queue))
        item = queue.get()
        if isinstance(item, QueueSignal):
            if item.info =='end':
                break
            continue
        yield item

def load_pickle(picklename):
    import pickle
    try:
        with open(picklename, 'rb') as f:
            pickledata = pickle.load(f)
        return pickledata
    except Exception as e:
        print('无法载入pickle文件，可能是文件不小心损坏了，手动删掉吧： {}'.format(picklename))
        raise e

def dump_pickle(data, picklename):
    import pickle
    assert isinstance(picklename, str), 'picklename must be a string, but it is a {}'.format(type(picklename))
    with open(picklename, 'wb') as f:
        pickle.dump(data, f)

def run_wrapper_v2(run_func, args, kwargs, taskinfo):
    index = taskinfo['index_task']
    cache_pkl = taskinfo['cache_pkl']
    data_repalced_with_queue = taskinfo['data_repalced_with_queue']
    for argname in data_repalced_with_queue:
        kwargs[argname] = queue_to_iterable(kwargs[argname])
    logger.info(f'Running Task {index}.')
    if cache_pkl:
        if os.path.exists(cache_pkl):
            try:
                ret = load_pickle(cache_pkl)
                # logger.info(f'Task {index} use cached result: {cache_pkl}')
                return ret
            except:
                # logger.error(f'cache file {cache_pkl} not legal')
                pass

    func = run_func.load_in_runner()
    try:
        ret = func(*args, **kwargs)
    except Exception as e:
        logger.fatal(f'Task {index} got error.')
        logger.fatal(traceback.format_exc())
        return e

    if cache_pkl:
        os.makedirs(os.path.dirname(cache_pkl), exist_ok=True)
        dump_pickle(ret, cache_pkl)
    logger.info(f'Run done Task {index}.')
    return ret

def data_put_in_queue(queue, data, end_num):
    logger.debug('data_put_in_queue ID %s',id(queue))
    for d in data:
        queue.put(d)
    for _ in range(end_num):
        queue.put(QueueSignal(info='end'))

import inspect
import ctypes
def _async_raise(tid, exctype):
  """raises the exception, performs cleanup if needed"""
  tid = ctypes.c_long(tid)
  if not inspect.isclass(exctype):
    exctype = type(exctype)
  res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
  if res == 0:
    raise ValueError("invalid thread id")
  elif res != 1:
    # """if it returns a number greater than one, you're in trouble,
    # and you should call it again with exc=NULL to revert the effect"""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
    raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
  _async_raise(thread.ident, SystemExit)

def run_parallel_function(function_arg_pair_list, num_parallel, thread_or_process = 'process'):
    rets = []

    if thread_or_process =='process':
        pool_executor = concurrent.futures.ProcessPoolExecutor
    elif thread_or_process =='thread':
        pool_executor = concurrent.futures.ThreadPoolExecutor

    with pool_executor(num_parallel) as executor:
        futures = [executor.submit(function_arg_pair[0], *function_arg_pair[1]) for function_arg_pair in
                   function_arg_pair_list]
        for _, future in enumerate(tqdm(futures)):
            try:
                ret = future.result()
                rets.append(ret)
            except Exception as e:
                # print('大概率是内存不够被Killed了')
                logger.fatal('Process died abrupt and Cannot be catched. \
                    大概率是内存不够被Killed了。\
                    这种情况多进程wrapper完全没办法应对(会报一个queue.full的错误)。建议检查内存消耗，或者使用黑科技（开启cache功能）：while true; do; your command; done')
                logger.fatal(e)
                logger.fatal(traceback.format_exc())
                # pid = os.getpid()
                # kill_child_processes(pid)
                # kill_process(pid)
                # os.system(f'kill -9 {pid}')
                # exit(1)
                rets.append(e)
    return rets

class ParallelManager(object):
    """
        加入新的功能：
        1.能断点续跑  (基于文件)
        2.调试功能/主进程
        3.防止内存炸掉
        4.提供默认配置，简化申请流程
        5.自动划分任务
    """

    def __init__(self, num_parallel, thread_or_process='process', cache=False, cache_dir='/tmp/parallelmanager/'):
        self.cache = cache
        self.cache_dir = os.path.join(cache_dir, get_signature())
        self.num_parallel = num_parallel
        self.thread_or_process = thread_or_process
        self.tasks = []
        self.outs = None

        self.data_generator = {}

    def clear_cache(self):
        logger.warning('clear old cache: %s', self.cache_dir)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        return self

    def _new_queue(self):
        if self.thread_or_process == 'thread':
            return queue.Queue()
        elif self.thread_or_process == 'process':
            return mp.Manager().Queue()
        else:
            raise NotImplementedError

    def add(self, run_func, num_task, run_func_args=None, run_func_kwargs=None, parallel_argnames=(0,),
            extra_taskinfo=False, replace_data_with_queue=False):
        run_func_args = [] if run_func_args is None else run_func_args
        run_func_kwargs = {} if run_func_kwargs is None else run_func_kwargs

        # 获取args数量
        run_func_argnames, _, _, _ = inspect.getargspec(run_func)

        # 把run_func_args全部变成run_func_kwargs
        run_func_args_kw = {run_func_argnames[i_arg]: arg
                            for i_arg, arg in enumerate(run_func_args)
                            }
        run_func_kwargs.update(run_func_args_kw)
        if extra_taskinfo:
            assert '_taskinfo' in run_func_argnames, 'If you want to use extra task info, you should add _taskinfo in your func\'s args'
        parallel_argnames = [pa if isinstance(pa, str) else run_func_argnames[pa] \
                             for pa in parallel_argnames]  # 把数字翻译成函数字段

        data_repalced_with_queue = []
        if len(parallel_argnames) > 0:
            first_argname = list(parallel_argnames)[0]
            first_arg = run_func_kwargs[first_argname]

            if replace_data_with_queue:
                assert len(parallel_argnames) == 1, '对于queue替代输入，暂时只支持一个输入'
                replaced_input_queue = self._new_queue()
                logger.debug('ID %s',id(replaced_input_queue))
                self.data_generator[first_argname] = {'arg': first_arg, 'replaced_queue': replaced_input_queue,
                                                      'num_task': num_task}
                run_func_kwargs[first_argname] = replaced_input_queue
                parallel_argnames = []
                num_data = num_task
                data_repalced_with_queue = [first_argname]
            else:
                if isinstance(first_arg, types.GeneratorType):
                    first_arg = list(first_arg)
                    run_func_kwargs[first_argname] = first_arg
                num_data = len(first_arg)
                for pa in parallel_argnames[1:]:
                    arg = run_func_kwargs[pa]
                    if isinstance(arg, types.GeneratorType):
                        arg = list(arg)
                        run_func_kwargs[pa] = arg
                    assert len(arg) == num_data, f'参数长度不一致，请检查{first_argname} {len(first_arg)}  vs {pa} {len(arg)}'
        else:
            num_data = num_task
        if num_data == 0:
            logger.error('No task to be run.')
            return self
        assert num_task < 50000, '太多的任务会导致文件数量超过单文件夹数量上限，建议减少一下哈'

        for ith, begin_index, end_index, _ in batch_wrapper(range(num_data), sum_count=num_task):
            batch_run_func_args = []
            batch_run_func_kwargs = {}
            for kwarg_name in run_func_kwargs:
                if kwarg_name in parallel_argnames:
                    batch_run_func_kwargs[kwarg_name] = run_func_kwargs[kwarg_name][begin_index:end_index]
                else:
                    batch_run_func_kwargs[kwarg_name] = run_func_kwargs[kwarg_name]

            if not self.cache:
                cache_pkl = None
            else:
                cache_pkl = os.path.join(self.cache_dir, f'{run_func.__name__}_{num_task}', f'{len(self.tasks)}.pkl')

            taskinfo = {'offset_data': begin_index, 'num_data': (end_index - begin_index), 'total_data': num_data,
                        'index_task': ith, 'total_task': num_task, 'cache_pkl': cache_pkl,
                        'data_repalced_with_queue': data_repalced_with_queue}
            if extra_taskinfo:
                batch_run_func_kwargs['_taskinfo'] = taskinfo
            self.tasks.append([run_wrapper_v2, [_Func(run_func), batch_run_func_args, batch_run_func_kwargs, taskinfo]])
        return self

    def _start_data_process(self):
        # 启动守护进程供给数据
        for dg_name, dg in self.data_generator.items():
            if self.thread_or_process == 'thread':
                mp_ctx = mp_dummy
            elif self.thread_or_process == 'process':
                mp_ctx = mp
            else:
                raise NotImplementedError

            dg['process'] = mp_ctx.Process(target=data_put_in_queue,
                                           args=(dg['replaced_queue'], dg['arg'], dg['num_task']))
            dg['process'].start()

    def _stop_data_process(self):
        for dg_name, dg in self.data_generator.items():
            try:
                logger.warning('Stoping the data process')
                stop_thread(dg['process'])
            except:
                pass

    def run(self, merge=False, ):
        self._start_data_process()

        self.outs = run_parallel_function(self.tasks, self.num_parallel, thread_or_process=self.thread_or_process)

        self._stop_data_process()
        if merge:
            self.check_outs()
            return self.merge_output(self.outs)
        else:
            return self.outs

    @staticmethod
    def merge_output(outs):
        """ 如果输出的结果是list和dict的话，这儿可以帮你合并起来。
        """
        type_check = {type(out) for out in outs}
        assert len(type_check) == 1, 'Type not consistent, please check: {}, or use cache=recache to rerun'.format(
            type_check)
        merge_type = type_check.pop()
        assert merge_type in (
            dict, list, type(None)), 'Return Type not support: {}, you may set merge=False.'.format(merge_type)
        if merge_type == dict:
            merged_result = {}
            for out in outs:
                for k in out:
                    if k in merged_result:
                        assert isinstance(merged_result[k], list) and isinstance(out[k],
                                                                                 list), 'Key {} conflict'.format(k)
                        merged_result[k].extend(out[k])
                    else:
                        merged_result[k] = out[k]
        elif merge_type == list:
            merged_result = []
            for out in outs:
                merged_result.extend(out)
        elif merge_type == type(None):
            logger.warning('Function return None, have you forgot to return the result?')
            merged_result = None

        return merged_result

    def debug_one(self, index=0, clear_cache=False):
        """ 在主进程调试某一个特定任务(有时候你需要申请一块有gpu的机器)。你可以在函数里面使用embed了。
        """
        if clear_cache:
            self.clear_cache()
        self._start_data_process()
        if index >= len(self.tasks):
            logger.fatal('Index exceed task maxlimit.')
            return
        out = self.tasks[index][0](*self.tasks[index][1])
        self._stop_data_process()
        if isinstance(out, Exception):
            raise out
        else:
            return out

    def check_outs(self):
        """ 检查结果是否有异常，有的话抛出，并告诉你是哪一个子任务。这时候你需要用debug_one来调试一下子任务。
        """
        for i, o in enumerate(self.outs):
            if isinstance(o, Exception):
                logger.fatal(
                    f'Task {i} is error. Please set debug_one={i} to check the reason. Or rerun it with cache.')
                raise o


def parallel_wrapper(num_parallel,  # 多少个进程或线程
                     num_task=None,  # 分成多少个任务，默认为线程数。大于线程数会排队进行
                     parallel_argnames=(0,),  # 函数中的哪个变量需要被拆分，可以为index或者 函数名
                     cache='from_command',
                     # True或者'cache'或者'recache'会启动cache机制，其中recache会清理掉之前的结果。  from_command检查shell参数'--cache'或'--recache'。
                     debug_one=None,  # 用主进程/主线程 调试某一个特定任务
                     extra_taskinfo=False,  # 是否传入特殊的_taskinfo变量，以获取当前的一些元任务信息，要求函数有_taskinfo参数
                     merge_output=True,  # 输出如果是list或dict，它会智能合并，否则会返回num_task个结果的list。
                     thread_or_process='process',  # process or thread
                     replace_data_with_queue=False,
                     # 设置这个之后，会启动一个专门的process，parallel_argnames里面的数据，会被不停地put到一个queue里面，而每个子进程会不停地获取。一般input是无限的时候使用
                     cache_dir='/tmp/parallelmanager/',
                     no_check_main=False,
                     ):
    """
    Example:
        # 我们会把这个函数的参数parallel_argnames分成num_task份，用num_parallel个进程/线程 跑下面的函数结果，并把结果返回合并。
        # 想断点续跑，只需要运行的时候加入--cache
        @parallel_wrapper(60,120,('arg3',))
        def func(arg1,arg2,arg3,.....):
            pass
        if __name__ == '__main__':
            out = func(arg1,arg2,arg3,...)
    """

    def __run_wrapper(func):
        def __run(*args, **kwargs):
            if func.__module__ == '__main__' or no_check_main:

                if 'DEBUG_ONE' in os.environ:
                    _debug_one = int(os.environ['DEBUG_ONE'])
                else:
                    _debug_one = debug_one
                if 'CACHE' in os.environ:
                    _cache = os.environ['CACHE']
                else:
                    _cache = cache
                _num_task = num_task if num_task else num_parallel
                _cache = '--cache' in sys.argv or '--recache' in sys.argv if _cache == 'from_command' else _cache

                simple_parallel_manager = ParallelManager(num_parallel=num_parallel,
                                                          thread_or_process=thread_or_process, cache=_cache,
                                                          cache_dir=cache_dir)
                if (_cache == 'from_command' and '--recache' in sys.argv) or _cache == 'recache':
                    simple_parallel_manager.clear_cache()
                simple_parallel_manager.add(func, _num_task, args, kwargs, parallel_argnames=parallel_argnames,
                                            extra_taskinfo=extra_taskinfo,
                                            replace_data_with_queue=replace_data_with_queue)
                if _debug_one is None:
                    out = simple_parallel_manager.run(merge=True, )
                else:
                    out = simple_parallel_manager.debug_one(int(_debug_one))
                return out
            else:
                return func(*args, **kwargs)

        return __run

    return __run_wrapper





