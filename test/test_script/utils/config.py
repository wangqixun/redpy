from redpy.utils_redpy.config_utils import Config






if __name__ == '__main__':
    cfg = Config().fromfile('/share/wangqixun/workspace/github_project/redpy/test/data/cfg_test.py')
    print(cfg)

    print(cfg.filename)

    print(cfg.pretty_text)

    print(cfg.keys())
