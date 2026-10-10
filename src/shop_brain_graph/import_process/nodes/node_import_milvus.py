import sys
import json
import math

from pymilvus import DataType
from common.config.milvus_config import milvus_config
from utils.clients.milvus_utils import get_milvus_client
from utils.task_utils import add_running_task, add_done_task

INSERT_BATCH_SIZE = 100

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    add_running_task(state["task_id"], "node_import_milvus")
    records = step_1_validate_and_get_data(state)
    client = step_2_prepare_chunks_collection()
    step_3_insert_chunks(client, records)
    add_done_task(state["task_id"], "node_import_milvus")
    logger.info(f"文档 {records[0]['file_title']} 入库完成：共 {len(records)} 个切片")
    return state


def step_1_validate_and_get_data(state):
    """删除旧数据之前，先验证每条记录的字段、长度和向量是否合法。"""
    records = state.get("embeddings_content")
    if not isinstance(records, list) or not records:
        raise ValueError("embeddings_content 必须是非空列表，请先执行 BGE 嵌入节点")
    text_limits = {"content": 65535, "file_title": 512, "item_name": 512,
                   "title": 512, "parent_title": 512}
    for number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {number} 条入库数据不是字典")
        for field, limit in text_limits.items():
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"第 {number} 条缺少非空字符串字段 {field}")
            # Milvus VARCHAR 上限按 UTF-8 字节计算，中文通常一个字占三字节。
            if len(value.encode("utf-8")) > limit:
                raise ValueError(f"第 {number} 条的 {field} 超过 {limit} 字节")
        part = record.get("part")
        if type(part) is not int or not 1 <= part <= 2147483647:
            raise ValueError(f"第 {number} 条的 part 必须是正 INT32 整数")
        dense = record.get("dense_vector")
        sparse = record.get("sparse_vector")
        if not isinstance(dense, list) or len(dense) != 1024:
            raise ValueError(f"第 {number} 条稠密向量必须是 1024 维列表")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in dense):
            raise ValueError(f"第 {number} 条稠密向量含无效数值")
        if not isinstance(sparse, dict) or not sparse:
            raise ValueError(f"第 {number} 条稀疏向量必须是非空字典")
        for key, value in sparse.items():
            if type(key) is not int or not 0 <= key < 4294967295:
                raise ValueError(f"第 {number} 条稀疏向量索引非法")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"第 {number} 条稀疏向量权重非法")
        # 一个导入任务只处理一篇文档，防止错误混入其它文档再批量删除。
        if record["file_title"] != records[0]["file_title"]:
            raise ValueError("一次入库的切片必须属于同一个 file_title")
    logger.info(f"入库数据校验通过：{len(records)} 条")
    return records


def step_2_prepare_chunks_collection():
    """复用或创建切片集合；字段名与上游 chunks 和检索工具保持一致。"""
    name = milvus_config.chunks_collection
    if not name:
        raise ValueError("未配置 CHUNKS_COLLECTION")
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 连接失败")
    if not client.has_collection(collection_name=name):
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
        for field, limit in (("content", 65535), ("file_title", 512), ("item_name", 512),
                             ("title", 512), ("parent_title", 512)):
            schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=limit)
        # 课件用 INT8，只支持到 127；改用 INT32，避免长手册切片序号溢出。
        schema.add_field(field_name="part", datatype=DataType.INT32)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="dense_vector", index_name="dense_vector_index",
                          index_type="HNSW", metric_type="COSINE",
                          params={"M": 64, "efConstruction": 100})
        indexes.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                          index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                          params={"inverted_index_algo": "DAAT_MAXSCORE"})
        logger.info(f"开始创建切片集合及混合检索索引：{name}")
        client.create_collection(collection_name=name, schema=schema, index_params=indexes)
    else:
        logger.info(f"复用已有切片集合：{name}")
    # 已有集合也需要加载，使删除和后续查询可正常执行。
    client.load_collection(collection_name=name)
    return client


def step_3_insert_chunks(client, records):
    """按文档名替换旧切片，分批插入，并回填 Milvus 自动生成的 chunk_id。"""
    name = milvus_config.chunks_collection
    file_title = records[0]["file_title"]
    # 用 JSON 字符串字面量转义引号/反斜杠，避免文档名破坏过滤表达式。
    # 按课件实际代码使用 file_title；同产品的其它文档不会被删除。
    expression = "file_title == " + json.dumps(file_title, ensure_ascii=False)
    logger.info(f"开始替换文档的旧切片：{file_title}")
    client.delete(collection_name=name, filter=expression)
    for start in range(0, len(records), INSERT_BATCH_SIZE):
        batch = records[start:start + INSERT_BATCH_SIZE]
        data = []
        for record in batch:
            row = dict(record)
            # 主键 auto_id=True，重复运行时不能把上次生成的主键再次提交。
            row.pop("chunk_id", None)
            data.append(row)
        result = client.insert(collection_name=name, data=data)
        ids = result.get("ids", [])
        if len(ids) != len(batch):
            raise RuntimeError("Milvus 返回的主键数量与插入切片数量不一致")
        for record, chunk_id in zip(batch, ids):
            record["chunk_id"] = chunk_id
        logger.info(f"已写入 {start + len(batch)}/{len(records)} 个切片")
    # delete + insert 不是事务：中途失败可能留下部分新数据。
    # 异常会向上抛出；重新运行会再次按文档清理，然后重新插入全部切片。
