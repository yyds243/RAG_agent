import time

from pymilvus import MilvusClient
from knowledge.utils.bge_m3_embedding_utils import get_beg_m3_embedding_model


if __name__ == '__main__':
    #1. 定义milvus对象
    client = MilvusClient(uri="http://192.168.10.128:19530")

    # 2.创建集合
    if client.has_collection(collection_name="test_create_collection001"):
        client.drop_collection(collection_name="test_create_collection001")

    client.create_collection(
        collection_name="test_create_collection001",
        dimension=1024,
        auto_id=True
    )

    docs = [
        "小明喜欢学习python"
    ]

    embedding_model = get_beg_m3_embedding_model()
    vector = embedding_model.encode_documents(docs)
    # 3. 构建数据
    data = [
        {"vector": vector['dense'][i].tolist(), "text": docs[i], "subject": "history"}
        for i in range(len(docs))
    ]

    res = client.insert(
        collection_name="test_create_collection001",  # target collection
        data=data,  # query vectors
    )
    time.sleep(2)
    query_vector = embedding_model.encode_queries(["谁喜欢学python"])

    res = client.search(
        collection_name="test_create_collection001",
        data = [query_vector['dense'][0].tolist()],
        limit = 1,
        output_fields=['text']
    )

    print(res)