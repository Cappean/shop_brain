import sys
import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from common.config.milvus_config import milvus_config
from utils.clients.milvus_utils import get_milvus_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 识别名称只需要文档开头的部分内容，避免把整本手册发送给模型。
ITEM_NAME_CONTEXT_CHUNK_K = 5
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    add_running_task(state["task_id"], "node_item_name_recognition")
    chunks, file_title = step_1_validate_and_get_data(state)
    item_name = step_2_call_llm_return_item_name(chunks, file_title)

    # 主体名用于后续检索过滤；切片正文暂时不做向量化，那是节点 6 的工作。
    # 本节点生成的是“主体名称本身”的向量，存入独立的主体集合。
    client = step_4_prepared_item_name_collection()
    step_5_insert_item_name_data(client, item_name, file_title)
    step_3_padding_item_name_to_chunks(chunks, item_name)
    state["item_name"] = item_name
    state["chunks"] = chunks
    add_done_task(state["task_id"], "node_item_name_recognition")
    logger.info(f"主体识别完成：{file_title} → {item_name}，回填 {len(chunks)} 个切片")
    return state


def step_1_validate_and_get_data(state):
    """优先使用上游 chunks；缺失时读取节点 4 的同名 JSON 备份。"""
    chunks = state.get("chunks")
    md_path_value = state.get("md_path")
    if not chunks:
        if not md_path_value:
            raise ValueError("chunks 为空，且没有 md_path，无法定位切片备份")
        json_path = Path(md_path_value).with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"切片备份不存在，请先执行文档切分：{json_path}")
        chunks = json.loads(json_path.read_text(encoding="utf-8"))
        logger.info(f"已从 JSON 恢复切片：{json_path}")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks 必须是非空列表")
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("content"), str) or not chunk["content"].strip():
            raise ValueError(f"第 {index} 个切片必须是字典，并包含非空字符串 content")

    file_title = state.get("file_title")
    if not file_title:
        file_title = Path(md_path_value).stem if md_path_value else chunks[0].get("file_title")
        file_title = file_title or "default_title"
        logger.warning(f"file_title 为空，使用默认文档名：{file_title}")
    if not isinstance(file_title, str):
        raise ValueError("file_title 必须是字符串")
    state["file_title"] = file_title
    logger.info(f"主体识别输入校验完成：{file_title}，共 {len(chunks)} 个切片")
    return chunks, file_title


def step_2_call_llm_return_item_name(chunks, file_title):
    """把文件名和前几个切片交给现有 prompt，要求模型只返回商品名称。"""
    context = ""
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        title = chunk.get("parent_title") or chunk.get("title", "")
        context += f"标题：{title}\n内容：{chunk['content']}\n"
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]

    # load_prompt 会把 file_title 和 context 填入模板中的占位符。
    prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    chain = get_llm_client() | StrOutputParser()
    logger.info(f"开始调用 LLM 识别主体：上下文 {len(context)} 字符")
    item_name = chain.invoke([HumanMessage(content=prompt)]).strip()
    # 有些模型把空字符串输出成两个引号；这种情况也按“未识别到”处理。
    if item_name in ("", '""', "''"):
        item_name = file_title
        logger.warning(f"模型未返回主体名，使用文档名兜底：{item_name}")
    logger.info(f"主体识别结果：{item_name}")
    return item_name


def step_3_padding_item_name_to_chunks(chunks, item_name):
    """每个切片都带上同一个主体名，后续入库时可按商品进行检索。"""
    for chunk in chunks:
        chunk["item_name"] = item_name
    logger.info(f"主体名已回填到 {len(chunks)} 个切片")


def step_4_prepared_item_name_collection():
    """集合存在就复用；不存在则按课件创建字段和两个向量索引。"""
    collection_name = milvus_config.item_name_collection
    if not collection_name:
        raise ValueError("请配置 ITEM_NAME_COLLECTION")
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 连接失败，无法保存主体名")
    if client.has_collection(collection_name=collection_name):
        logger.info(f"复用已有主体集合：{collection_name}")
        return client

    # pk 由 Milvus 自动生成；VARCHAR 的 max_length 按 UTF-8 字节数限制。
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    # 老师提供的 BGE-M3 输出 1024 维稠密向量。
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    indexes = client.prepare_index_params()
    # HNSW 建图提高近邻检索效率；COSINE 用于计算稠密向量的方向相似度。
    indexes.add_index(field_name="dense_vector", index_type="HNSW",
                      index_name="dense_vector_index", metric_type="COSINE",
                      params={"M": 64, "efConstruction": 100})
    # 稀疏向量用倒排索引；IP（内积）对应课件的混合检索方案。
    indexes.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX",
                      index_name="sparse_vector_index", metric_type="IP",
                      params={"inverted_index_algo": "DAAT_MAXSCORE"})
    logger.info(f"开始创建主体集合：{collection_name}")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=indexes)
    logger.info(f"主体集合和向量索引创建完成：{collection_name}")
    return client


def step_5_insert_item_name_data(client, item_name, file_title):
    """生成名称向量，再按文档名替换旧主体记录，避免重复运行累加数据。"""
    # 先校验并生成向量，再删除旧记录，避免模型失败时误删已有数据。
    for field_name, value in (("file_title", file_title), ("item_name", item_name)):
        if len(value.encode("utf-8")) > 512:
            raise ValueError(f"{field_name} 超过主体集合允许的 512 UTF-8 字节")

    # 延迟导入：只有实际执行名称向量化时，才加载老师提供的 BGE 工具。
    from utils.lm.embedding_utils import generate_embeddings

    logger.info(f"开始生成主体名称向量：{item_name}")
    vectors = generate_embeddings([item_name])
    dense_vector = vectors["dense"][0]
    sparse_vector = vectors["sparse"][0]
    if len(dense_vector) != 1024:
        raise ValueError(f"主体稠密向量应为 1024 维，实际为 {len(dense_vector)} 维")

    # json.dumps 生成带双引号的字符串字面量，并转义文档名中的引号/反斜杠。
    # 删除条件只匹配当前 file_title，不删除其他文档的主体信息。
    collection_name = milvus_config.item_name_collection
    filter_expression = "file_title == " + json.dumps(file_title, ensure_ascii=False)
    logger.info(f"替换当前文档的主体记录：{file_title}")
    client.delete(collection_name=collection_name, filter=filter_expression)
    client.insert(collection_name=collection_name, data=[{
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }])
    # 注意：课件采用 delete + insert，两步不是事务；插入失败时旧记录已删除。
    # 此时异常向上传递，修复问题后重跑本节点即可重新写入。
    logger.info(f"主体名称及稠密/稀疏向量已写入：{collection_name}")
