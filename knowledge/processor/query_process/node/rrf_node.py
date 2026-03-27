from typing import List, Dict, Any

from knowledge.processor.query_process.base  import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RffNode(BaseNode):
    """
    一种经典的排名融合算法。这是一个重新排名的策略，基于文档的排名。被多路命中的同一个文档未来计算得到的得分更高，顺序也就更靠前.
    原因：由于不同检索路内部评价分数所采用的方式不同，造成量纲差异。采用按分数直接排序或者按chunk_id进行累加都是不可行的
        而rrf就消除不同路的量纲差异（评分度量类型不一样）。只看排名，不看原始评分，多路共识>单路评分，一个chunk被三路检索都认可一定比一路排第一更可信
    """
    name = "RFF_node"
    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self,state:QueryGraphState)->QueryGraphState:

        # 1.拿各路搜索的结果(排除网络搜索:rerank做)。本质：因为网络搜索的结果并没有chunk_id,也就是没有在milvus中查询(回填操作)，而其他三路都有chunk_id
        # 1.1 获取向量检索路的结果
        vector_search_result = state.get('embedding_chunks',[])
        # 1.2 获取hyde向量检索路的结果
        hyde_search_result = state.get("hyde_embedding_chunks",[])
        # 1.3 获取Kg图谱检索的结果
        kg_search_result = state.get("kg_chunks",[])

        # 2. 为不同路的搜索结果设置权重
        search_source = {
            "vector_search_result":(self._normalize_input(vector_search_result),0.9),
            "hyde_search_result":(self._normalize_input(hyde_search_result),0.9),
            "kg_search_result":(self._normalize_input(kg_search_result),self.config.rrf_kg_weight)
        }

        # 3.构建rrf_inputs,提取结果以及权重
        rrf_inputs = list(search_source.values())

        # 4. 利用RRF计算公式去获取所有路查询所有chunk对应的score
        rrf_merge_results = self._rrf_merge(rrf_inputs,_rrf_k=self.config.rrf_k,_top_k=self.config.rrf_max_results)

        # 5. 获取rrf_chunks(只取文档，不要分数）
        rrf_chunks = [doc for doc,_ in rrf_merge_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7.更新state
        state['rrf_chunks'] = rrf_chunks

        return state


    def _normalize_input(self,rrf_input):
        """
        统一处理各路检索到的最终结果
        Args:
            rrf_input:

        Returns:
        """
        diff_path_result = []
        # 1.
        if not rrf_input:
            return []
        # 2.遍历该路的所有结果
        for doc in rrf_input:
            # 2.1 判断是否有效
            if not  isinstance(doc,dict):
                continue

            # 2.2 获取entity
            entity = doc.get('entity') or None
            if not entity:
                continue

            diff_path_result.append(entity)
        return diff_path_result

    def _rrf_merge(self,rrf_inputs:List[Dict[str,Any]],_rrf_k,_top_k):
        """
        根据rrf公式来计算每一个文档的总得分，最后对总得分进行排序
        Args:
            rrf_inputs: 各路的搜索结果以及各路在rrf中的权重
            _rrf_k: 平滑参数，一般取60.如果想让头部排名的文档影响更大，k调小，如果要兼容多个排名的文档，
            _tok_k: 合并完之后，返回的文档数
        Returns:
                合并以及排序后的文档
        """
        chunk_scores = {} # 存放所有chunk（多路命中的chunk只留一份）的rrf计算后的分数值，chunk_id: score
        chunk_data = {} # 存放所有chunk的原始内容，chunk_id: doc。"chunk_1": {"chunk_id": "chunk_1", "entity": {...}},
        for rrf_input,weight in rrf_inputs:
            # rrf_input是每一路的检索结果，weight是每一路的权重
            for i, doc in enumerate(rrf_input,1):
                # i：从每一路排名第一的文档开始遍历，
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue
                # rrf公式： score = Σ weight/(smoothing_k+rank(k)),文档k在第i路中的排名位置（从1开始）
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id,float(0)) + weight / (_rrf_k + i)
                chunk_data.setdefault(chunk_id,doc)

            # 按得分降序排序，截取前top_k条
            """
            最终得到的结构是
            [ (chunk_data[cid], score)，--->({"chunk_id": "c1", ...}, 0.032)
                (doc1, 0.032),
                (doc2, 0.041)
            ],                          然后x的结构是x = (doc, score)，x[1]=score,按照分数降序排序
            """
        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key = lambda x:x[1], reverse=True
        )
        return sorted_results[:_top_k] if _top_k else sorted_results



if __name__ == '__main__':
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟三路检索结果
    # chunk_1 命中 3 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4, chunk_5 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
        "kg_chunks": [
            {"id": None, "distance": 2.0, "entity": {"chunk_id": "chunk_5", "content": "知识图谱结果#1"}},
            {"id": None, "distance": 1.0, "entity": {"chunk_id": "chunk_1", "content": "知识图谱结果#2"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)

    rrf_node = RffNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")











