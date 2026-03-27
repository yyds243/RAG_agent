import json

from langgraph.graph import StateGraph, END

from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.processor.import_process.nodes.md_img_node import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.document_spilt_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding_chunks import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.base import setup_logging

# 路由函数，router就是一个流程分流器。根据当前state来判断下一步去哪
def import_router(state:ImportGraphState):
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return END

def create_import_graph() -> StateGraph:
    """
    定义导入业务的graph状态拓扑图（langraph构建流水线）
    Returns:

    """
    # 1.定义状态图
    graph_pipeline = StateGraph(ImportGraphState)
    # 2. 定义节点（入口、结束节点等）
    # 2.1  定义入口节点
    graph_pipeline.set_entry_point("entry_node")

    # 2.2 添加剩下的节点
    nodes = {
        "entry_node":EntryNode(),
        "pdf_to_md_node":PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
        "kg_graph_node": KnowLedgeGraphNode(),

    }
    # 然后把节点注册进图中去
    for key, value in nodes.items():
        graph_pipeline.add_node(key,value)


    # 3. 定义边（顺序边，条件边）
    # 动态分支
    # source:  路由开始节点
    # path:    路由函数，根据返回值来判断走哪一个节点
    # path_map 路由函数的映射，根据router返回值去匹配graph中的节点

    graph_pipeline.add_conditional_edges("entry_node",
                                         import_router,
                                {
                                        "md_img_node":"md_img_node",
                                        "pdf_to_md_node":"pdf_to_md_node",
                                         END:END
                                        }
        )

    # graph_pipeline.add_edge("entry_node","pdf_to_md_node")
    graph_pipeline.add_edge("pdf_to_md_node","md_img_node")
    graph_pipeline.add_edge("md_img_node","document_split_node")
    graph_pipeline.add_edge("document_split_node","item_name_rec_node")
    graph_pipeline.add_edge("item_name_rec_node","bge_embedding_node")
    graph_pipeline.add_edge("bge_embedding_node","import_milvus_node")
    graph_pipeline.add_edge("import_milvus_node","kg_graph_node")
    graph_pipeline.add_edge("kg_graph_node",END)
    # 4.编译，实现真正可执行的graph
    return graph_pipeline.compile()


#只需要构建一次图
graph_app = create_import_graph()

# 测试
def run_import_graph(import_file_path:str, file_dir:str):

    # 1. 构建state
    init_state = {
        'import_file_path': import_file_path,
        'file_dir': file_dir,
    }
    init_state = create_default_state(**init_state)

    # 2.调用stream
    final_state = None
    # 如果下述修改成 res = graph_app.invoke(init_state)就直接跑完图中所有节点，只会返回最终的state，没有中间过程
    for event in graph_app.stream(init_state):
        # 这里会先进入到核心创建graph图的流程，然后先执行entry节点中的process方法，返回方法中的state
        # event 就包含；event = {
        #     "entry_node": {...更新后的state...}
        # }
        for node_name, state in event.items():
            # print(f"运行的节点{node_name}\n state:{state}")
            final_state = state

    return final_state

if __name__ == '__main__':
    setup_logging()

    import_file_path =  r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"
    file_dir= r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir"
    final_state = run_import_graph(import_file_path=import_file_path, file_dir=file_dir)

    print(json.dumps(final_state, indent=3, ensure_ascii=False))

    # 打印图结构

    print("="*50)
    print("图结构：")
    graph_app.get_graph().print_ascii()










