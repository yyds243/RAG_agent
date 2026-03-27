import time

from pymilvus import MilvusClient
from pymilvus import DataType
from knowledge.utils.bge_m3_embedding_utils import get_beg_m3_embedding_model
from knowledge.utils.milvus_utils import get_milvus_client

if __name__ == '__main__':
    #1. 定义milvus对象
    client = get_milvus_client()

    # 2.创建集合
    if client.has_collection(collection_name="aaa_test111"):
        client.drop_collection(collection_name="aaa_test111")

    #  2.1 创建schema
    schema = client.create_schema(enable_dynamic_field=True)
    # 添加主键约束
    schema.add_field(
        field_name="my_id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    # 添加向量约束
    schema.add_field(
        field_name="my_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,
    )
    # 添加标量约束
    schema.add_field(
        field_name="my_varchar",
        datatype=DataType.VARCHAR,
        max_length=512,
    )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="my_vector",  # Name of the vector field to be indexed
        index_type="IVF_FLAT",  # 要建立的索引类型，可以设置为AUTOINDEX，自动选择
        index_name="vector_index",  # Name of the index to create
        metric_type="COSINE",  # 计算向量之间的距离
        params={
            "nlist": 64,  # Number of clusters for the index
        }  # Index building params
    )

    # 创建标量索引
    index_params.add_index(
        field_name="my_varchar",  # Name of the vector field to be indexed
        index_type="INVERTED",  # 要建立的索引类型，可以设置为AUTOINDEX，自动选择
        index_name="inverted_index",  # Name of the index to create
    )
    # 创建集合
    client.create_collection(
        collection_name="aaa_test111",
        schema=schema,
        index_params=index_params,
    )

    docs = [
        "小明喜欢学习python"
    ]

    embedding_model = get_beg_m3_embedding_model()
    vector = embedding_model.encode_documents(docs)
    # 3. 构建数据
    data = [
        {"my_vector": vector['dense'][i].tolist(), "my_varchar": docs[i], "subject": "history"}
        for i in range(len(docs))
    ]

    res = client.insert(
        collection_name="aaa_test111",  # target collection
        data=data,  # query vectors
    )
    # print(res)


    time.sleep(2)
    query_vector = embedding_model.encode_queries(["谁喜欢学python"])

    res1 = client.search(
        collection_name="test_create_collection002",  # Collection name
        anns_field="my_vector",
        data=[query_vector['dense'][0].tolist()],  # Query vector
        limit=3,  # TopK results to return
        output_fields=["my_varchar","subject"]
    )

    print(res1)

    # res = client.search(
    #     collection_name="test_create_collection002",
    #     data = [query_vector['dense'][0].tolist()],
    #     limit = 1,
    #     output_fields=['my_varchar']
    # )