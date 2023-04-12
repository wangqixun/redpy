from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from pyapollo import ApolloClient
from .logger_utils import setup_logger
import os

__all__ = ['CosFileClient']

logger = setup_logger()



class CosFileClient(object):
    def __init__(self):
        self.CONFIG_SERVER_MAP = {
            'dev': 'http://apollocc-cs-dev.sit.xiaohongshu.com',
            'test': 'http://apollocc-cs-tst.sit.xiaohongshu.com',
            'sit': 'http://apollocc-cs-sit.sit.xiaohongshu.com',
            'beta': 'http://apollocc-cs.int.beta.xiaohongshu.com',
            'prod': 'http://apollocc-cs.int.xiaohongshu.com'
        }
    
    def init_cos_client(self, bucket):
        # 无需修改    
        if bucket == "recr-img-1251524319":
            # print("vproject bucket")
            sid     = "kms.cos.secretId.t1646121605152"
            skey    = "kms.cos.secretKey.t1646121605152"
        else:
            # print("our own bucket")
            sid     = "kms.cos.secretId.t1645524302357"
            skey    = 'kms.cos.secretKey.t1645524302357'

        kclient = ApolloClient(app_id="vproject", cluster="default", config_server_url=self.CONFIG_SERVER_MAP['sit'])
        secret_id = kclient.get_value(key=sid, namespace='kms')
        secret_key = kclient.get_value(key=skey, namespace='kms')
        region = 'ap-shanghai'
        token = None              
        scheme = 'https'           
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
        client = CosS3Client(config)
        return client


    def upload_to_cos(self, key, file_path, bucket="sns-media-ai-1251524319"):
        '''
        vproject: recr-img-1251524319
        media-ai: sns-media-ai-1251524319
        '''
        # 上传
        client  = self.init_cos_client(bucket)
        with open(file_path, 'rb') as fp:
            response = client.put_object(
                Bucket = bucket,
                Body=fp,
                Key=key,
                StorageClass='STANDARD',
                EnableMD5=False
            )
        return key


    def download_from_cos(self, key, file_path, bucket='sns-media-ai-1251524319'):
        '''
        vproject: recr-img-1251524319
        media-ai: sns-media-ai-1251524319
        '''
        if key.startswith("http"):
            os.system("wget '%s' -O %s"%(key, file_path))
            return file_path

        client  = self.init_cos_client(bucket)
        response = client.get_object(
            Bucket=bucket,
            Key=key,
        )
        response['Body'].get_stream_to_file(file_path)
        return file_path



