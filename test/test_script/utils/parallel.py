from redpy.utils_redpy.parallel_utils import parallel_wrapper
import time


@parallel_wrapper(5, 10000, cache='recache')
def say(say_list):
    for info in say_list:
        time.sleep(1)
        print(info)


if __name__ == '__main__':
    start = time.time()
    say_list = [i for i in range(50)]
    say(say_list)

    print(time.time()-start)

