
"""
知识图谱查询节点
"""
import json,re,logging
from typing import Dict, List, Tuple, Any
from json import JSONDecodeError
from langchain_core.messages import HumanMessage,SystemMessage
from pymilvus import MilvusClient

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.utils.bge_m3_embedding_utils import get_beg_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.llm_client import get_llm_client
from knowledge.processor.query_process.exceptions import  StateFieldError
from knowledge.processor.query_process.prompts.query_prompt import _ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.milvus_utils import milvus_client, get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query
from knowledge.utils.neo4j_utils import get_neo4j_driver

# 常量
_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN = 0.5

# -------------------------------------------------
# Neo4J的信息
# -------------------------------------------------
ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]


# Neo4j的Cypher语句
#精确匹配（不需要 LIMIT，因为 `(name, item_name)` 唯一）：
_CYPHER_EXACT_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name=$item_name AND n.name=$name
RETURN  n.item_name as item_name,n.name as name
LIMIT 1
"""

# 模糊匹配（保留 LIMIT，因为 CONTAINS 可能命中多条）：
_CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($entity_name)
      AND n.item_name = $item_name
RETURN n.name AS name, n.item_name AS item_name
LIMIT $limit
"""

# 查询种子节点的一跳关系
_CYPHER_ONE_HOP_RELATIONS = """

MATCH (seed:Entity {name:$name,item_name:$item_name})-[r]-(nbr:Entity)

WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$item_name

RETURN 
  CASE WHEN startNode(r)=seed  THEN  seed.name  ELSE nbr.name END AS head
  type(r) as rel
  CASE WHEN  startNode(r)=seed  THEN nbr.name ELSE seed.name END AS tail

limit $limit
"""
# 根据带权重的节点查询chunk_id
_CYPHER_LOOKUP_CHUNK = """

UNWIND $weighted_nodes as n

MATCH (e:Entity{e.name=n.entity_name,e.item_name=n.item_name})-[r:MENTIONED_IN]->(c:Chunk{c.item_name=n.item_name})

WITH c,sum(n.weight) AS score, count(e) AS cnt

RETURN c.id AS chunk_id, c.item_name AS item_name, score, cnt

ORDER BY score DESC, cnt DESC,chunk_id DESC

LIMIT $limit

"""

SEED_NODE_WEIGHT = 2.0
NER_NODE_WEIGHT = 1.0


def _clean_parse_llm_content(llm_response_content:str)->List[str]:
    # 1. 判断llm输出内容是否为空
    if not llm_response_content:
        return []
    # 2. 清洗Json代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", llm_response_content)
    re_sub = re.sub(r"\s*```$", "", text)
    # 3. 反序列化
    try:
        deserialized_result: Dict[str, Any] = json.loads(re_sub)
    except JSONDecodeError as e:
        logging.error(f"JSON 反序列失败，原因: {str(e)}")
        return []

    # 4. 获取提取后的实体名
    entities_name = deserialized_result.get('entities',[])
    if not entities_name or not isinstance(entities_name,list):
        return []

    # 4.3 遍历所有实体名
    seen = set()
    entities_name_result = []
    for entity_name in entities_name:
        if not entity_name or not isinstance(entity_name,str):
            continue
        # 3 .实体名是否过长
        truncated_entity_name = truncate_entity_name_length(entity_name)

        # 4.去重
        if truncated_entity_name not in seen:
            seen.add(truncated_entity_name)
            entities_name_result.append(truncated_entity_name)
    return entities_name_result

def truncate_entity_name_length(entity_name:str)->str:
    """

    Args:
        entity_name:

    Returns:

    """
    name = entity_name.strip()
    return name[:_ENTITY_NAME_MAX_LENGTH] if len(name) > _ENTITY_NAME_MAX_LENGTH else name

def _item_name_filter_expr(item_names:List[str])->str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"

def _clean_seed_rows(rows: List[Dict[str,Any]])->List[EntitySeedNode]:
    """
    清洗查询种子节点的数据
    Args:
        rows:
    Returns:
    """
    if not rows:
        return []
    clean_seeds_result = []
    # 遍历
    for row in rows:
        item_name = row.get('item_name','').strip()
        entity_name = row.get('entity_name','').strip()
        if not (entity_name and item_name):
            continue
        # 封装
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name,
        })
    return clean_seeds_result


class _EntityExtractor:
    """
    实体提取器：利用LLM从查询问题中提取实体
    """
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def _extract(self,user_query:str)->List[str]:
        #1. 获取Llm客户端
        llm_client = get_llm_client(mode_name="kimi-k2.5")
        if llm_client is None:
            return []
        #3. 获取提示词
        entities_name_extract_system_prompt = _ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            MAX_ENTITY_NAME_LENGTH=_ENTITY_NAME_MAX_LENGTH
        )

        # 4. 调用LLM
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f"用户问题：{user_query}")
            ])
            # 4.1 获取模型结果
            llm_response_result = getattr(llm_response,'content','').strip()

            # 4.2 清洗以及解析
            entities_name = _clean_parse_llm_content(llm_response_result)
            self._logger.info("实体抽取完成: %d 个 %s", len(entities_name), entities_name)
            return entities_name
        except Exception as e:
            self._logger.error(f"LLM调用失败：{str(e)}")
            return []


class _EntityAligner:
    """
    实体对齐器：根据LLm提取到的实体名，去查询MIlvus,获取真正的实体名（对齐后的才能在neo4j中进行查询)
    """
    def __init__(self,collection_name:str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name
    def _align(self,entity_names:List[str],item_names:list[str]):
        """
        Args:
            entity_names:LLM提取的实体名
            item_names: 商品名
        Returns:
        Dict[str,Any]:该字典准备封装两个Key
        第一个key: entities_aligned:[]所有对齐后的实体名
        第二个Key:entity_elements[]:所有对齐后的实体信息[source_id,distance,origin,aligned,content]
        """
        fallback_result = {"entities_aligned":[],"entity_elements":[]}
        # 1. 判断实体名是否有
        if not entity_names:
            return fallback_result
        # 2.获取嵌入模型
        embedding_model = get_beg_m3_embedding_model()
        if not embedding_model:
            self._logger.error("嵌入模型不存在")
        # 3. 获取milvus客户端
        milvus_client = get_milvus_client()
        if not milvus_client:
            self._logger.error("Milvus客户端不存在")
            return fallback_result
        # 4. 向量化实体名
        embedding_result = generate_hybrid_embeddings(embedding_model=embedding_model,embedding_documents=entity_names)

        # 5. 检验嵌入结果
        if embedding_result is None:
            self._logger.error("嵌入结果无法获取")
            return fallback_result

        # 5.1 获取嵌入后的稠密向量
        embedding_result_dense = embedding_result['dense']
        embedding_result_sparse = embedding_result['sparse']

        # 6. 获取item_name 的表达式
        item_name_filtered_expr = _item_name_filter_expr(item_names)

        # 7. 遍历所有的实体名字
        aligned_entities_name: List[str] = [] # 存放所有实体名字
        aligned_entity_elements: List[Dict[str,Any]] = []  # 存放所有实体详细信息
        seen = set()
        for index, entity_name in enumerate(entity_names):
            # 7.1 先看对齐一个实体
            align_one_result: List[Dict[str, Any]]= self._align_one(milvus_client,self._collection_name,item_name_filtered_expr,
                                                              embedding_result_dense,embedding_result_sparse,index,entity_name)

            aligned_entity_elements.extend(align_one_result)

            # 7.3 遍历商品下的对齐结果
            for detail in align_one_result:
                # a) 获取对齐后的实体名
                aligned_name = detail.get("aligned")

                # b) 获取商品名
                item_name = detail.get("item_name")

                # c) 判断对齐名是否有
                if aligned_name:

                    # 去重 同名实体在不同商品下都保留
                    key = (item_name, aligned_name)
                    if key not in seen:
                        seen.add(key)
                        aligned_entities_name.append(aligned_name)

        self._logger.info(f"对齐后的实体个数 {len(aligned_entities_name)} 实体的名字：{aligned_entities_name}")
        return {
            "entities_aligned_name": aligned_entities_name,
            "entities_aligned_elements": aligned_entity_elements
        }

    def _align_one(self, milvus_client:MilvusClient, _collection_name:str, item_name_filtered_expr:str, embedding_result_dense:List,
                   embedding_result_sparse:List, index:int, entity_name:str) -> List[Dict[str,Any]]:
        """
        对齐指定的实体名
        Returns:

        """
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]
        # 1. 判断实体名的稠密和稀疏向量
        if not dense_vector or not sparse_vector:
            return [{"original": entity_name, "aligned": "", "context": "", "reason": "vector values is not exist "}]

        # 2. 创建混合实体搜索请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector = dense_vector,
            sparse_vector = sparse_vector,
            expr = item_name_filtered_expr,
            limit = 5
        )
        # 3. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client = milvus_client,
            collection_name = _collection_name,
            search_requests = hybrid_search_requests,
            ranker_weights=(0.4,0.6),
            norm_score = True,
            limit = 5,
            output_fields = ["source_chunk_id","item_name","context","entity_name"]
        )
        # 4.解析结果
        hits = reps[0] if reps else []
        if not hits:
            return [{"original":entity_name,"aligned":"","context":"","reason":"search result is Empty"}]

        # 4.1  按 item_name 分组，每组取最高分
        best_by_item: Dict[str, Dict] = {}
        for hit in hits:
            # a) 获取实体
            entity = hit.get("entity")

            # b) 从实体中获取
            item_name = entity.get("item_name").strip()

            # c) 只保留每个 item_name 下的第一个（即最高分）
            if item_name not in best_by_item:
                best_by_item[item_name] = hit

        # 4.2 是否有最好的item_name
        if not best_by_item:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_item_name"}]

        # 4.3  item_name 分组输出结果，过滤低于阈值的
        results: List[Dict[str, Any]] = []
        for item_name, best in best_by_item.items():
            # a) 获取最好的那一个分数
            score = best.get("distance")

            # b) 判断分数值
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN):
                continue

            # c) 获取实体信息
            ent = best.get("entity")

            # d) 将不同商品下最好的实体名添加到结果集中
            results.append({
                "original": entity_name,
                "aligned": ent.get("entity_name"),
                "score": score,
                "item_name": item_name,
                "source_chunk_id": ent.get("source_chunk_id"),
                "reason": "top1_per_item",
            })

        # 4.4 全部低于阈值时返回未命中
        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results

    def _pick_best_entity_name(self, search_entities_name_result: List[Dict[str,Any]])->Dict[str,Any]:
        """
        从返回的5个实体名中留下一个实体名
        Args:

        Returns:
        """
        # 1.判断是否检索到了
        if not search_entities_name_result:
            return None
        # 2.获取第一个
        first_entity = search_entities_name_result[0]
        if not first_entity:
            return None

        # 3. 获取第一个实体名的分数值
        first_entity_name_score = first_entity.get('distance')
        if not first_entity_name_score:
            return None
        # 返回第一个且分数超过阈值的
        return first_entity if first_entity_name_score > _DEFAULT_ENTITY_NAME_ALIGN else None

class _Neo4jGraphReader:
    """
    对Neo4j进行读操作
    1. 种子节点的查询(1.精确查询，2.降级走模糊查询）
    2. 查询种子节点的一跳关系
    3. 根据所有节点来查询chunk（item_name,id)
    4. 根据所有chunk_id来查询milvus中，进而得到所有的chunk
    """
    def __init__(self,
                 database: str,
                 # kg_max_seed_candidates: int,
                 # kg_max_total_seeds: int,
                 # kg_max_triples_per_seed: int,
                 # kg_max_total_triples: int,
                 # kg_max_total_chunks: int
                 ):
        self._database = database
        # self._kg_max_seed_candidates = kg_max_seed_candidates
        # self._kg_max_total_seeds = kg_max_total_seeds
        # self.kg_max_triples_per_seed = kg_max_triples_per_seed
        # self._kg_max_total_triples = kg_max_total_triples
        # self._kg_max_total_chunks = kg_max_total_chunks
        self._logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        # 1. 获取驱动
        neo4j_driver = get_neo4j_driver()
        if neo4j_driver is None:
            raise RuntimeError("neo4j驱动获取失败")
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self,pairs: List[ItemEntityPair]) -> List[EntitySeedNode]:
        """
        根据item_name和entity_name来查询种子节点
        Args:
            pairs:

        Returns:
        """
        # 1. 检查pair对是否存在
        if not pairs:
            return []
        final_seeds_result: List[EntitySeedNode] = []
        # 2. 遍历
        for pair in pairs:
            # 2.1 获取item_name
            item_name = pair.get('item_name','').strip()
            # 2.2获取entity_name
            entity_name = pair.get('entity_name','').strip()
            # 2.3 过滤
            if not item_name or not entity_name:
                continue
            # 2.4 执行cypher语句，
            try:
                with self._session() as session:
                    # 2.5 执行种子节点查询
                    candidates_seed_nodes = self._execute_seed_nodes(session,item_name,entity_name,
                                                                     self._kg_max_seed_candidates)

                    # 2.6将查询到的种子节点加入到最终列表中
                    final_seeds_result.extend(candidates_seed_nodes)

                    # 2.7 截取种子节点的个数(一般不会超）
                    if len(final_seeds_result)>self._kg_max_total_seeds:
                        final_seeds_result = final_seeds_result[:self._kg_max_total_seeds]
                        break
            except Exception as e:
                self._logger.error(f"获取种子节点失败,原因 :{str(e)}")
                return []
        self._logger.info(f"获取种子节点成功，有{len(final_seeds_result)}个")
        return final_seeds_result

    def _execute_seed_nodes(self, session, item_name, entity_name, _kg_max_seed_candidates):
        """
        执行种子节点查询
        Args:
            session:
            item_name:
            entity_name:
            _kg_max_seed_candidates:

        Returns:
        """
        # 1、精确查询
        exact_rows = session.execute_read(
            lambda tx:tx.run(
                _CYPHER_EXACT_SEEDS, item_name=item_name,name=entity_name
            ).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)
        # 2.模糊查询
        fuzzy_rows = session.execute_read(
            lambda tx:tx.run(
                _CYPHER_FUZZY_SEEDS,item_name=item_name,name=entity_name,limit=_kg_max_seed_candidates
            ).data()
        )
        return _clean_seed_rows(fuzzy_rows)

    # def find_oue_hop_relations(self,seed_nodes: List[EntitySeedNode]):
    #     """
    #     职责： 根据种子节点查询一跳的关系（双向），并且过滤掉MENTIONED_IN 关系的节点
    #     注意：1. 去重（不允许同一条边出现多次）只能出现一次。 2.图谱中存储的节点和关系结构是什么 查询的时候一定要和存储的我结构保证一致 3. 邻居节点可以是你在一跳范围内指向的节点也可以别人在一跳范围内指向你的节点
    #     比如：A->B(类型：认识) A->B(类型：认识) B->A(类型：认识)
    #     Args:
    #         seed_nodes:
    #     Returns:
    #     """
    #     # 1.判断种子节点
    #     if not seed_nodes:
    #         return []
    #     seen = set()
    #     one_hop_relations_final=[]
    #     for seed_node in seed_nodes:
    #         item_name = seed_node.get('item_name', '').strip()
    #         seed_name = seed_node.get('entity_name', '').strip()
    #         if not item_name or not seed_node:
    #             continue
    #
    #     # 2. 执行cypher语句
    #     try:
    #         with self._session() as session:
    #             # 2.1 查询种子的一跳节点
    #             seed_one_hop_relations: List[OneHopRelation] = self._execute_one_hop_relations(session, item_name,
    #                                                                                                    seed_name,
    #                                                                                                    self.kg_max_triples_per_seed)
    #             if not seed_one_hop_relations:
    #                 return []
    #
    #             # b 去重，同一条边不能查两次
    #             for seed_one_hop_relation in seed_one_hop_relations:
    #                 # b.1 获取头
    #                 head = seed_one_hop_relation.get('head')
    #                 # b.2 获取rel
    #                 rel = seed_one_hop_relation.get('rel')
    #                 # b.3 获取tail
    #                 tail = seed_one_hop_relation.get('tail')
    #                 # b.4 获取item_name
    #                 item_name = seed_one_hop_relation.get('item_name')
    #
    #                 key = (item_name, head, rel, tail)
    #                 if key not in seen:
    #                     seen.add(key)
    #                     one_hop_relations_final.append(seed_one_hop_relation)
    #
    #             # c) 截取 种子节点的关系，防止超过LLM窗口阈值
    #             if len(one_hop_relations_final) > self._kg_max_total_triples:
    #                 one_hop_relations_final = one_hop_relations_final[:self._kg_max_total_triples]
    #                 break
    #          # d) 返回
    #     except Exception as e:
    #         self._logger.error(f"查询 {seed_name} 种子节点的一跳关系失败: {str(e)}")
    #         return []
    #     self._logger.info(f"查询 {len(seed_nodes)} 个种子节点对应的关系:{len(one_hop_relations_final)}条")
    #     return one_hop_relations_final


    # def _execute_one_hop_relations(self, session, item_name, seed_name, kg_max_triples_per_seed) -> List[OneHopRelation]:
    #     """
    #     Args:
    #         session:
    #         item_name:商品名
    #         seed_name:种子名
    #         kg_max_triples_per_seed:种子节点的最大关系数
    #     Returns:
    #     """
    #     # 1. 根据session执行查询方法
    #     one_hop_relations = session.execute_read(
    #         lambda tx: tx.run(
    #             _CYPHER_ONE_HOP_RELATIONS, item_name=item_name, name=seed_name, limit=kg_max_triples_per_seed
    #         ).data()
    #     )
    #     if not one_hop_relations:
    #         return []
    #
    #     #3. 遍历所有的一跳关系
    #     one_hop_result = []
    #     for one_hop_relation in one_hop_relations:
    #         head = one_hop_relation.get('head', '').strip()
    #         rel = one_hop_relations.get('rel','').strip()
    #         tail = one_hop_relation.get('tail', '').strip()
    #
    #         if not (head and rel and tail):
    #             continue
    #
    #         one_hop_result.append({
    #             "head": head,
    #             "rel": rel,
    #             "tail": tail,
    #             "item_name": item_name
    #         })
    #     return one_hop_result

def _build_item_entity_pairs(aligned_entities_info: List[Dict[str,Any]])->List[ItemEntityPair]:
    """
     职责：从对齐后的实体详情中获取商品名+实体名的pair对
    去重：因为同一个商品名下的实体名只留一个, 不同商品名下的实体名都留
    Args:
        aligned_entities_info:

    Returns:

    """
    # 1. 判断
    if not aligned_entities_info:
        return []
    seen = set()
    item_entity_pairs = []
    # 2. 遍历对齐后的实体详情
    for entity_elements in aligned_entities_info:
        # 获取对齐后的实体名和item_name

        aligned_entity_name = entity_elements.get('aligned',"").strip()
        item_name = entity_elements.get('item_name',"").strip()

        if not(item_name and aligned_entity_name):
            continue
        # 2.2去重
        key = (item_name,aligned_entity_name)
        if key not in seen:
            seen.add(key)
            item_entity_pairs.append({
                "item_name": item_name,
                "entity_name": aligned_entity_name
            })

    return item_entity_pairs
class KnowledgeGraphSearchNode(BaseNode):
    """
    按pipeline的顺序进行
    实体抽取->对齐实体*>neo4j查询->回填Chunk
    """

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.参数校验
        validated_query,validated_item_name  = self._validate_inputs(state)

        # 2. 执行流水线
        result = self._run_pipeline(validated_query,validated_item_name)
        return  result


    def _validate_inputs(self, state:QueryGraphState)-> Tuple[str, List[str]]:
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query','')
        item_names = state.get('item_names','')

        # 2.判断是否存在
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3. 从重写的问题中踢掉商品名：(商品名是没有作用的查询
        pattern = '|'.join(re.escape(name) for name in item_names)
        user_query = re.sub(pattern,'',rewritten_query).strip()

        return user_query.strip(), item_names


    def _run_pipeline(self, validated_query, validated_item_name) -> Tuple[str, List[str]]:
        # 1. 初始化
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(collection_name=self.config.entity_name_collection)
        neo4j_graph_reader = _Neo4jGraphReader(database=self.config.neo4j_database)


        # 2.利用提取器提取实体，只需要核心的实体名字
        entities_name = entity_extractor._extract(user_query=validated_query)
        entities_name_aligned = entity_aligner._align(entities_name,item_names=validated_item_name)
        # 2.1 获取对齐后的实体名和实体详情
        aligned_entities_name = entities_name_aligned.get('entities_aligned_name')
        aligned_entities_info = entities_name_aligned.get('entities_aligned_elements')

        # 3.构建item_name + eneity_name
        item_entity_paris: List[ItemEntityPair] = _build_item_entity_pairs(aligned_entities_info)

        #4. Neo4j操作
        # 4.1 根据商品名和实体名的pairs来查询种子节点
        seed_nodes:List[EntitySeedNode]  = neo4j_graph_reader.find_seed_nodes(item_entity_paris)

        # return entities_name_aligned,entities_name
        return seed_nodes

if __name__ == '__main__':
    kg_search_node = KnowledgeGraphSearchNode()
    state = {
        "rewritten_query": "RS-12 数字万用表更换电池需要注意什么",
        "item_names": ["RS-12 数字万用表"]
    }
    result = kg_search_node.process(state)
    print(result)