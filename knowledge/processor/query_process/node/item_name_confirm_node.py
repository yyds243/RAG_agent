import json,re
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json,re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple
from knowledge.processor.query_process.config import QueryConfig

from langchain_core.messages import HumanMessage,SystemMessage
from knowledge.utils.milvus_utils import get_milvus_client,create_hybrid_search_requests,execute_hybrid_search_query
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.bge_m3_embedding_utils import generate_hybrid_embeddings,get_beg_m3_embedding_model

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.prompts.item_name_extract_prompt import  ITEM_NAME_EXTRACT_TEMPLATE

class ItemNameAligner():
    """
    主要职责：1. 查询向量数据库。2. 评分对齐。3. 分数差异过滤
    """
    def match_align_filter(self,item_names:List[str])->Tuple[List[str],List[str]]:
        # 1. 查询向量数据库
        search_result = self._match_vector(item_names)

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_result)

        # 3. 分数差异过滤
        if len(confirmed)>1:
            confirmed = self._item_name_score_filter(confirmed,search_result)

        return [],[]
    def _match_vector(self,item_names:List[str]):
        """
        根据LLM提取的商品名，查询向量数据库
        Args:
            item_names: LLM提取的商品名
        Returns: List[dict[str,any]]：每一个item_name 下的查询结果
        Dict[str,Any]:{"extracted_name":"LLM提取出来的商品名字“，
        """
        # 1 定义最终搜索结果
        search_result = []

        # 2. 获取Milvus_client
        milvus_client = get_milvus_client()
        if milvus_client is None:
            return []

        # 3 获取嵌入模型
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            logger.error("获取嵌入模型失败")
            return []
        # 获取稠密、稀疏向量
        hybrid_embedding_result = generate_hybrid_embeddings(embedding_model,item_names)

        # 4. 遍历LLM提取的所有商品名
        for index,extract_item_name in enumerate(item_names):
            #4.1 混合向量检索
            hybrid_search_request = create_hybrid_search_requests(
                dense_vector = hybrid_embedding_result['dense'][index],
                sparse_vector=hybrid_embedding_result['sparse'][index],
            )
            # 4.2 创建混合检索请求
            # (milvus集成bgem3嵌入向量只会对稠密向量归一化)
            #(WeightRanker是权重融合排序器，norm_score属性会对稠密和稀疏向量检索的结果分数值进行归一化。为了公平的最后计算两者权重时
            hybrid_search_result = execute_hybrid_search_query(milvus_client,
                                                               collection_name=QueryConfig.item_name_collection,
                                                               search_requests=hybrid_search_request,
                                                                ranker_weights=(0.5,0.5),
                                                               norm_score=True,output_fields=["item_name"])
            # 4.3 解析混合检索请求的结果对象
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    # [
                    #     {"item_name": "iPhone15", "score": 0.92},
                    #     {"item_name": "iPhone15 Pro", "score": 0.85},
                    #     {"item_name": "iPhone14", "score": 0.72}
                    # ]
                    {"item_name": h["entity"]["item_name"], "score":h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_request else [])
                ]
            }
            # 4.4 将构建好的查询结果放入到最终检索结果中去
            search_result.append(item_name_search_result)
        return search_result

    def _item_name_score_align(self, search_result:List[Dict[str,Any]])->Tuple[List[str], List[str]]:
        """
        职责：根据向量数据库检索到的商品名，将其放到对应的confirmed或options
        Args:
            search_result:
        Returns:
              分数阈值的规则：confirm：0.75   options:0.6
            分数阈值作为放到confirmed或者options的条件。

            返回值：confirmed有，将confirmed中的商品名 传给下游四路检索
            返回值：options有，确认下一步，询问到底在咨询哪一款商品。
            返回值：confirmed没有 options没有，直接告诉没有找到具体的商品名
            返回值：confirmed有 options有，至少确定了一个商品名，没有必要让用户在次确认这个商品。

            注意：
            1. 如果像confirmed列表中添加某一次遍历向量数据库查询到的商品名时，发现confirmed已经有该商品名了。            3. 如果confirmed中已经有某一个商品从向量数据库返回的某个对应的item_name，那么下一次从另外一个商品名中根据向量数据库中返回的同一个item_name 既不能加到confirmed（重复） 也不能加入options中
            4.如果options中已经有某一个商品从向量数据库返回的某个对应的item_name，那么下一次从另外一个商品名中根据向量数据库中返回的同一个item_name 不能加到options中（重复） 但是可以加入confirm中
            所以去重的方向是单向的
        """
        #1.定义两个容器,options阈值为0.7
        confirmed, options = [], []
        # 2. 遍历向量数据库中所有LLM提取到跟商品名相关的相似性结果
        for item_name_search_result in search_result:
            # 2.1 获取LLM的商品名
            extracted_name = item_name_search_result.get('extracted_name')
            # 2.2 对某一个商品名下的item_name进行降序（从高到低排序）
            matches = sorted(item_name_search_result.get('matches'),key=lambda x: x['score'],reverse=True )
            # 2.3 获取mathes中较高的分数值
            high = [  m for m in matches if m.get('score')>=0.7]

            #询问能否进入到confirmed中
            if high:
                # 3.1找最精准的
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)

                # 场景A:刚好等于向量数据库中的检索结果（唯一匹配）
                if extract:
                    picked = extract['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景B：只有一个高分候选
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                # C：有多个高分候选但不精确匹配
                else:
                    #
                    for h in high[:3]:
                        picked = h.get('item_name')
                        if picked not in options and picked not in confirmed:
                            options.append(picked)
            # 是否能进入到options中
            else:
                mid = [m for m in matches if m['score']>=0.6 and m.get('item_name') not in options and m.get('item_name') not in confirmed]
                if mid:
                    for m in mid:
                        picked = m.get('item_name')
                        options.append(picked)
        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed:List[str], search_result:List[Dict[str,Any]]):
        """
        item_names:有三个item_name
        item_name1:0.9 （最相似的（基准））
        item_name2:0.88（真实比对）
        item_name3:0.66（可能误判）
        Args:
            confirmed:
            search_result:
        Returns:
        """
        # 1.定义字典容器（存储confirmed中item_name在向量数据库中的分数值）
        item_name_score={}
        """
        {        
                商品名: 最高相似度 例如：  ("iPhone15",0.90),
                                        ("iPhone15 Pro",0.88),
                                        ("iPhone14",0.66)
        }
        """
        for search_res in search_result:
            # 获取matches
            matches = search_res.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    # 一个商品名可能在多个query中出现，
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        #
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x:x[1],reverse=True)

        #3. 取出分数值最大的
        max_item_name_score = sorted_item_name_score[0][1]
        #，只保留与最高分差值 ≤ 0.15 的商品
        return [name for name,score in item_name_score.items()
                if max_item_name_score - score <= 0.15]

class ItemNameExtractor:
    """
    目的： 基于用户的原始问题+用户的历史对话（暂时不看）去提取用户真正想问的商品名
    询问的场景：1、（单级询问）请问RS12-万用表如何测量电阻。LLM->商品:[RS万用表,万用表测量电阻(假)]如果从向量数据库中找到了这个假的也会进入到confirm中(误判)
    2、(多级询问) 请问RS12-万用表和13万用表分别如何测量电阻。商品:[RS12-万用表RS12,RS13万用表]->加入到confirm中去
    """
    def extract_item_name(self,original_query:str):
        """
        LLM根据用户原始问题来提取商品名
        Args:
            original_query:
        Returns:
        """
        result:Dict[str,Any] = {"item_name":[],"rewritten_query":""}
        history = ""
        # 1. 获取LLM客户端
        llm_client = get_llm_client(mode_name="kimi-k2.5",response_format=True)
        if llm_client is None:
            return result
        # 2. 定义提示词(用户级别)
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history if history else "暂无上下文", query=original_query)
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息"

        # 3.LLM调用
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content = human_prompt)
        ])
        llm_content = llm_response.content.strip()
        if not llm_content.strip():
            return result

        # 4.清洗和解析
        try:
            parsed_result = self._clean_parse(llm_content)
            result['rewritten_query'] = parsed_result.get('rewritten_query') or original_query
            result['item_name'] = parsed_result.get('item_name')

        except Exception as e:
            logger.error(f'清洗以及解析LLM的输出失败:{str(e)}')
        return result

    def _clean_parse(self,response:str)->Dict[str,Any]:

        # 1 清洗json代码围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        # 2. 反序列化
        try:
            parsed_llm_result = json.loads(content)

            #2.1 清洗item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names,list):
                clean_item_names = []
            else:# 过滤掉空字符串
                clean_item_names = [ raw_item for raw_item in rwa_item_names if raw_item.strip() ]

            # 2.2 清洗rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query,str) else raw_rewritten_query.strip()
            return {"item_name":clean_item_names,"rewritten_query":clean_rewritten_query}


        except JSONDecodeError as e:
            raise JSONDecodeError(msg=f"反序列LLM输出失败:{str(e)}")


class ItemNameConfirmNode(BaseNode):

    name = "item_name_confirm_node"

    def __init__(self):
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()

    def process(self,state:QueryGraphState)->QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state.get("original_query")

        #2. 调用LLM来提取商品名(原因是：如果直接基于用户的原始问题进行检索，质量很差）
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query)
        # 2.1 获取item_names
        item_names = clean_llm_result.get('item_names')
        # 2.2 获取rewritten_query
        rewritten_query = clean_llm_result.get('rewritten_query')
        if item_names:
            # 3. 查询向量数据库并过滤1（评分对齐&分数差异过滤）
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [],[]

        # 4. 决定state的key值
        self._decide(state,item_names,confirmed,options,rewritten_query)


        return state

    def _decide(self,state:QueryGraphState,item_names:List[str],confirmed:List[str],
                options:List[str],rewritten_query:str):
        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = item_names

        elif options:
            state['answer'] = (f"我不确定您指的是哪款产品。"
                               f"您是在询问以下产品吗：{'、'.join(options)}？")
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


_node_instance = ItemNameConfirmNode()


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)


    test_state = {
        "session_id": "test_123",
        "original_query": "你们店里那款苏伯尔RS-12数字万用表怎么测电压？",
        "item_names": [],
        "rewritten_query": "",
        "answer": "",
        "history": [],
    }

    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_test = ItemNameConfirmNode()

    result = node_test.process(test_state)
    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")