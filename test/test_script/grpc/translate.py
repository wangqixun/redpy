from redpy.grpc import TranslateClient


if __name__ == '__main__':
    text = ['hello, what day is today?']
    client = TranslateClient(
        ip='10.4.200.42',
        port='30201',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    res = client.run(
        text,
    )
    print(res)
