import sys
import json
from pathlib import Path

from utils.task_utils import add_running_task, add_done_task

# 与课件一致，每次向量化 5 个切片，降低一次推理的内存压力。
EMBEDDING_BATCH_SIZE = 5

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState

@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    add_running_task(state["task_id"], "node_bge_embedding")
    chunks = step_1_validate_and_get_data(state)
    step_2_batch_generate_vector(chunks)
    # 仅在所有批次成功后发布完整结果，失败会抛异常，阻止下游入库。
    state["embeddings_content"] = chunks
    add_done_task(state["task_id"], "node_bge_embedding")
    logger.info(f"切片向量化完成：共 {len(chunks)} 条，稠密和稀疏向量均已生成")
    return state


def step_1_validate_and_get_data(state):
    """获取切片，必要时从节点 4 生成的 JSON 恢复，并补齐主体名。"""
    chunks = state.get("chunks")
    if not chunks:
        if not state.get("md_path"):
            raise ValueError("chunks 和 md_path 均为空，无法进行向量化")
        json_path = Path(state["md_path"]).with_suffix(".json")
        chunks = json.loads(json_path.read_text(encoding="utf-8"))
        logger.info(f"已从备份恢复切片：{json_path}")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks 必须是非空列表")

    prepared_chunks = []
    for number, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"第 {number} 条切片不是字典")
        # 用副本生成向量，避免某批失败时把半成品留在上游 chunks 中。
        prepared = dict(chunk)
        item_name = prepared.get("item_name") or state.get("item_name")
        content = prepared.get("content")
        if not isinstance(item_name, str) or not item_name.strip():
            raise ValueError(f"第 {number} 条切片缺少主体名，请先执行主体识别")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"第 {number} 条切片缺少非空正文 content")
        prepared["item_name"] = item_name.strip()
        prepared_chunks.append(prepared)
    logger.info(f"向量化输入校验完成：共 {len(prepared_chunks)} 个切片")
    return prepared_chunks


def step_2_batch_generate_vector(chunks):
    """将“主体名_正文”送入 BGE-M3，按顺序给切片添加两个向量字段。"""
    # 实际执行时才导入模型工具，单纯读取/校验数据无需加载模型依赖。
    from utils.lm.embedding_utils import generate_embeddings

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        texts = []
        for chunk in batch:
            # 增强主体信息，但不修改原始 content，便于检索后引用原文。
            texts.append(chunk["item_name"] + "_" + chunk["content"])
        batch_number = start // EMBEDDING_BATCH_SIZE + 1
        logger.info(f"开始向量化第 {batch_number} 批：{len(batch)} 个切片")
        result = generate_embeddings(texts)
        if len(result["dense"]) != len(batch) or len(result["sparse"]) != len(batch):
            raise ValueError(f"第 {batch_number} 批返回的向量数量与切片数量不一致")
        for index, chunk in enumerate(batch):
            if len(result["dense"][index]) != 1024:
                raise ValueError("BGE-M3 稠密向量维度不是预期的 1024")
            chunk["dense_vector"] = result["dense"][index]
            chunk["sparse_vector"] = result["sparse"][index]
        # 不打印完整向量，避免数千个浮点数淹没学习/调试日志。
        logger.info(f"第 {batch_number} 批向量化完成")
