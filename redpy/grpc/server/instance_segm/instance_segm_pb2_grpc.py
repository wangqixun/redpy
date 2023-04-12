# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import instance_segm_pb2 as instance__segm__pb2


class InstanceSegmStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.infer_one_img = channel.unary_unary(
                '/InstanceSegm/infer_one_img',
                request_serializer=instance__segm__pb2.ISOneImgRequest.SerializeToString,
                response_deserializer=instance__segm__pb2.ISOneImgReply.FromString,
                )


class InstanceSegmServicer(object):
    """Missing associated documentation comment in .proto file."""

    def infer_one_img(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InstanceSegmServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'infer_one_img': grpc.unary_unary_rpc_method_handler(
                    servicer.infer_one_img,
                    request_deserializer=instance__segm__pb2.ISOneImgRequest.FromString,
                    response_serializer=instance__segm__pb2.ISOneImgReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'InstanceSegm', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class InstanceSegm(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def infer_one_img(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/InstanceSegm/infer_one_img',
            instance__segm__pb2.ISOneImgRequest.SerializeToString,
            instance__segm__pb2.ISOneImgReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
