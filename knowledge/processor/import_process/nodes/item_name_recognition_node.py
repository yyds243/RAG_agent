from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError,EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.milvus_utils import get_milvus_client

from knowledge.utils.bge_m3_embedding_utils import get_beg_m3_embedding_model
from knowledge.processor.import_process.prompts.item_name_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE,ITEM_NAME_SYSTEM_PROMPT


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks,config = self._validate_inputs(state)

        # 2. 构建LLM上下文（提取商品名）
        item_name_context = self._prepare_item_name_context(file_title,chunks, config)

        # 3. 调用LLM模型
        item_name = self._recognition_item_name_by_llm(file_title, item_name_context)

        # 4. 稠密向量：计算语义相似度，负责语义的理解。稀疏向量：负责关键词匹配
        dense_vec, sparse_vec = self._embedding_item_name(item_name)

        # 5. 存储到Milvus数据库中
        self._save_milvus(file_title,item_name,dense_vec, sparse_vec,config)

        # 6.回填到state以及CHUNK中去
        self._fill_item_name(item_name,state,chunks)

        return state
    def _validate_inputs(self, state : ImportGraphState):
        self.log_step("step1","校验输入参数")
        config = get_config()
        # 1. 获取state中的file_title以及chunks
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        # 2. 判断提取到的参数是否为空
        if not file_title:
            raise ValidationError("文件标题为空",self.name)

        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunk为空或者无效",self.name)

        item_name_chunk_k = config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError("chunk为空或者无效",self.name)


        self.logger.info(f"检测到文件：{file_title},对应的切片长度：{len(chunks)}")
        # 3. 返回
        return file_title, chunks, config

    def _prepare_item_name_context(self, file_title:str,chunks:List[Dict[str,Any]], config):
        self.log_step("step2","构建商品名字提取的上下文")

        result = []
        total = 0
        # 前五块中留下内容的字符串中的字符数不能超过2000个字符长度
        for index, chunk in enumerate(chunks[:config.item_name_chunk_k]):
            # 1. 判断chunk的类型
            if not isinstance(chunk, dict):
                continue

            ##构建上下文：标题+body(content:标题 +\n\n+body)
            # 2. 提取
            content = chunk.get('content')

            spices = f"切片-{index+1} - {content}"

            result.append(spices)

            # 3.计算长度
            total += len(spices)
            result.append(spices)

            # 4. 判断收集到的长度是否超过阈值
            if total > config.item_name_chunk_size:
                break

            # if total + len(spices) > config.item_name_chunk_size:
            #     break
            # total += len(spices)
            # result.append(spices)
        # 强制返回不超过chunk-size的长度
        return "\n\n".join(result)[:config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title:str, item_name_context:str):
        self.log_step("step3", "LLM识别商品名")
        # 1. 实例化llm客户端
        llm_client = get_llm_client()
        if llm_client is None:
            self.logger.error(f"LLM初始化失败，商品名安全回退到标题：{file_title}")
        # 2. diaoyong
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title , context=item_name_context)


        # 3. 调用模型( 可以传的对象# , str ,[], promptValue)
        try:
            llm_response=llm_client.invoke([
                SystemMessage(content = ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content = prompt)
            ])

            # 4.获取模型的输出内容
            # content = llm_response.content
            # getattr作用就是查看第一个对象中是否有第二个元素的值，有就返回这个元素的值，否则返回""
            item_name = getattr(llm_response, "content","").strip()

            # 5.p判断
            if not item_name or item_name.upper() == 'UNKNOWN':
                self.logger.warning(f"LLM无法提取有效的商品名,安全回退到标题名：{file_title}")

                return file_title
            # 6. 真正返回提取到的商品名
            self.logger.info(f"提取到的商品名字：{item_name}")
            return item_name
        except Exception as e:
            self.logger.error(f"LLM无法提取有效的商品名,安全回退到标题名：{file_title}")
            return file_title

    def _embedding_item_name(self, item_name: str) -> Optional[Tuple[list, dict[Any, Any]]]:
        self.log_step("step4", "embedding模型嵌入商品名")

        try:
            # 1.获取嵌入模型
            embedding_model = get_beg_m3_embedding_model()

            # 2.嵌入llm得到的结果
            embedding_result = embedding_model.encode_documents([item_name])

            # 3. 获取稠密和稀疏向量
            dense = embedding_result['dense'][0].tolist()
            start_index = embedding_result['sparse'].indptr[0]
            end_index = embedding_result['sparse'].indptr[1]
            weights = embedding_result['sparse'].data[start_index:end_index]
            tokenids = embedding_result['sparse'].indices[start_index:end_index]

            sparse = dict(zip(tokenids, weights))
            return dense, sparse
        except Exception as e:
            self.log_step(f"嵌入商品名：{item_name}失败，原因是{str(e)}")
            return EmbeddingError(f"嵌入商品名：{item_name}失败，原因是{str(e)}",self.name)

    def _save_milvus(self, file_title,item_name,dense_vec:List[float], sparse_vec:Dict[str,Any],config):
        self.log_step("step5","保存到向量数据库")
        # 1. 判断向量是否存在
        if not dense_vec or not sparse_vec:
            self.logger.warning(f"{item_name}向量生成不完成，无法进库")
            return
        # 2.操作milvus
        try:
            # 2.1 创建集合
            milvus_client = get_milvus_client()
            collection_name = config.item_name_collection
            # 2.2 如果不存在集合就创建一个
            if not milvus_client.has_collection(collection_name=collection_name):
               self._create_item_name_collection(milvus_client,collection_name)
            # 2.3 构建数据
            data = {
                'file_title': file_title,
                'item_name': item_name,
                'dense_vector': dense_vec,
                'sparse_vector': sparse_vec,
            }
            # 2.4. 插入到milvus并保存{'insert_count':1,"ids":[100,101,102]}
            res = milvus_client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已成功保存到milvus，ID={res['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus操作完全失败：{str(e)}")

    def _create_item_name_collection(self, client, collection_name):
        self.logger.info(f"正在创建集合{collection_name}")
        # 创建约束
        schema = client.create_schema()
        # 主键约束
        schema.add_field(field_name="pk",datatype=DataType.VARCHAR,is_primary=True,auto_id=True,max_length=100)
        # 标量约束
        schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535)

        # 向量约束
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 建立索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        # 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(f"集合{collection_name}创建成功并添加了索引")

    def _fill_item_name(self, item_name:str, state:ImportGraphState, chunks:List[Dict[str,Any]]):
        self.log_step("step6", "回填商品名信息")
        for chunk in chunks:
            chunk['item_name'] = item_name

        # 程序员自己使用
        state['item_name'] = item_name

if __name__ == '__main__':
    import json
    chunk_json = r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\chunks.json"
    with open(chunk_json,'r',encoding='utf-8') as f:
        chunk_content = json.load(f)

    state = {
        'file_title':'万用表的使用',
        'chunks':chunk_content
    }

    # 实例化节点
    item_name_recognition_node=ItemNameRecognitionNode()

    #process
    res = item_name_recognition_node.process(state)

    print(json.dumps(res,ensure_ascii=False,indent=4))