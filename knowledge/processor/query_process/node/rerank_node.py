from typing import Dict, Any, List

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.reranker_utils import get_reranker_model

class RerankNode(BaseNode):
    """
    职责：解决网络搜索这一路无法进入到rrf中进行排名的问题，因为并没有进入到milvus中进行查询，也就无法利用chunk_id进行投票。同时rrf输出可能也混杂不相关的文档，需进一步筛选
    而reranker是依靠语义来打分，网络结果也能参与。将问题跟每篇文档稳定拼在一起，也就是q跟d拼成一个序列[cls]q[sep]d来通过transformer,
    在每一层self_attention中，两者都互相捕捉交互信息。最后从CLS(放在序列开头汇总表示)处得到相关性得分。因此会对每篇文档进行独立打分，只要内容跟问题相关，就能得到高分
    具体：使用BGE_reranker模型来对RRF融合结果和网络搜索结果进行精排，并通过断崖检测算法来动态实现topk的截断
    """
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        #1. 获取
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 2. 合并多源文档
        merged_multi_docs = self._merge_mutil_source_docs(state)

        # 3.Rerank精排
        reranked_docs = self._rerank_merged_docs(user_query,merged_multi_docs)
        
        # 4.动态top_k截断
        cutoff_docs = self._cliff_cutoff(reranked_docs)
        self.logger.info(f"重排序完成:{len(reranked_docs)}->{len(cutoff_docs)}")

        state['reranked_docs'] = cutoff_docs

        return state
    def _merge_mutil_source_docs(self,state: QueryGraphState):
        """
        合并rrf和网络搜索结果
        Returns:
        """
        final_docs = []
        # 1. 获取本地RRF的文档
        for rrf_doc in (state.get('rrf_chunks') or []):

            # 1.1 判断当前文档对象类型
            if not isinstance(rrf_doc, dict):
                continue

            # 1.2 获取文档的内容
            content = rrf_doc.get('content', '').strip()

            # 1.3 判断文档内容
            if not content:
                continue
            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id', '').strip()
            # 1.4 格式化本地RRF的chunk结构
            format_rrf_doc = self._format_rrf_docs(content=content, title=title, chunk_id=chunk_id, source="local")
            final_docs.append(format_rrf_doc)

        # 2. 获取web远程的文档
        for web_doc in (state.get('web_search_docs') or []):

            # 2.1 判断当前文档对象类型
            if not isinstance(web_doc, dict):
                continue

            # 2.2 获取文档内容(snippet)
            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()

            # 2.3 判断内容
            if not content:
                continue
            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()

            # 2.4 格式化web的文档结构
            format_web_doc = self._format_rrf_docs(content=content, title=title, url=url, source="web")
            final_docs.append(format_web_doc)
        
        self.logger.info(f"收集到准备进行Rerank精排的文档 {len(final_docs)}")
        return final_docs
        
    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None, url: str = "", source: str = "") -> Dict[
        str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }

    def _rerank_merged_docs(self, user_query:str, merged_multi_docs:List[Dict[str,Any]]):
        """
        # 交叉编码器（精排阶段）— Reranker 用的就是这个
        # Query 和 Document 联合编码，精度更高
        pairs = [(query, doc1), (query, doc2), ...]
        scores = reranker.compute_score(pairs)  # 需要在线计算
        Args:
        Returns:

        """
        if not user_query or not merged_multi_docs:
            return []
        #2. 获取rerank模型
        reranker = get_reranker_model()

        # 3 .构建（问题，文档）元组列表

        pairs = [(user_query, docs.get('content')) for docs in merged_multi_docs]
        try:
            # 4. 计算得分
            reranker_scores = reranker.compute_score(sentence_pairs=pairs)
            # 5. 映射分数和文档
            score_doc = [ {**doc,"score":score} for doc, score in zip(merged_multi_docs,reranker_scores)]

            # 6.排序
            result = sorted(score_doc, key=lambda x:x['score'],reverse=True)
            return result
        except Exception as e:
            self.logger.error(f"重排序失败{str(e)}")
            return [{**merged_multi_docs,"score":None}]

    def _cliff_cutoff(self, reranked_docs: List[Dict[str,Any]]):
        """
        断崖检测截断，用绝对断崖捕捉高分的大幅下跌，相对断崖捕捉低分区的比例性下跌
        Args:
            reranked_docs:

        Returns:

        """
        if not reranked_docs:
            return []
        # 2. 计算截断范围
        upper_bound = min(self.config.rerank_max_top_k,len(reranked_docs))
        lower_bound = min(self.config.rerank_min_top_k,upper_bound)

        # 3.默认取最大值，遇到断崖提前截断
        cutoff_pos = upper_bound
        for i in range(lower_bound-1,upper_bound-1):
            current_score = reranked_docs[i].get('score')
            next_score = reranked_docs[i+1].get('score')

            # 3.1分数为空时跳过
            if current_score is None or next_score is None:
                continue
            # 3.2 计算绝对差距和相对差距
            abs_gap = current_score - next_score
            rel_gap = abs_gap / abs(current_score+1e-6)
            # 3.3 任一差距超过阈值即为断崖，立即截断
            if abs_gap >= self.config.rerank_gap_abs or rel_gap>= self.config.rerank_gap_ratio:
                cutoff_pos = i+1
                self.logger.debug(
                    f"断崖检测: 位置 {i + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
                )
                break
        # 4. 返回截断后的文档
        return reranked_docs[:cutoff_pos]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")