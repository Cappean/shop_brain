import sys
import json
import re
from pathlib import Path

from utils.task_utils import add_running_task, add_done_task

# 与课件保持一致：一般切成约 600 字符；短块可合并，但不超过 1000 字符。
# 这里按字符计数，不是按 token 计数。标题也计入 content 的总长度。
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
CHUNK_MIN = 400
CHUNK_MAX_SIZE = 1000

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # 主函数只负责串联步骤，具体细节放在下面的小函数中，便于逐步阅读。
    add_running_task(state["task_id"], "node_document_split")
    md_content, file_title, md_path = step_1_validate_get_data(state)
    chunks = step_2_split_document_by_title(md_content, file_title)
    chunks = step_3_refine_split_and_merge_chunks(chunks)
    step_4_padding_chunks_metadata(chunks)
    step_5_backup_chunks_json(chunks, md_path)

    # 回填后，下一节点直接使用 state['chunks']，不用再次读取 Markdown。
    state["chunks"] = chunks
    add_done_task(state["task_id"], "node_document_split")
    logger.info(f"文档切分完成：{file_title}，共 {len(chunks)} 个切片")
    return state


def step_1_validate_get_data(state):
    """取出正文、文档名和路径；内存正文为空时，从 Markdown 文件恢复。"""
    if not state.get("md_path"):
        # JSON 备份要放在 Markdown 旁边，因此即使已有正文，也需要路径。
        raise ValueError("文档切分需要 md_path，用于定位 Markdown 和保存 JSON 备份")
    md_path = Path(state["md_path"])
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在：{md_path}")

    md_content = state.get("md_content")
    if not md_content:
        md_content = md_path.read_text(encoding="utf-8")
        logger.info(f"md_content 为空，已从文件读取：{md_path}")
    if not isinstance(md_content, str) or not md_content.strip():
        raise ValueError("Markdown 正文为空或不是字符串，无法切分")

    file_title = state.get("file_title") or md_path.stem
    # 保留空行：空行是递归切分的重要段落边界，也可能是代码块的一部分。
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    state["md_content"] = md_content
    state["file_title"] = file_title
    logger.info(f"参数校验完成：文档 {file_title}，正文 {len(md_content)} 字符")
    return md_content, file_title, md_path


def step_2_split_document_by_title(md_content, file_title):
    """按 # 到 ###### 建立标题层级；代码块内部的 # 不作为标题。"""
    chunks = []
    heading_stack = []
    current_title = None
    content_lines = []
    orphan_lines = []  # 第一个标题之前的正文，后面并入首个有正文的章节。
    code_fence = None

    for line in md_content.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if code_fence is None:
                code_fence = marker
            elif marker[0] == code_fence[0] and len(marker) >= len(code_fence) and stripped == marker:
                code_fence = None
            # 围栏本身也属于正文，不丢弃它。
            if current_title is None:
                orphan_lines.append(line)
            else:
                content_lines.append(line)
            continue

        heading_match = None
        if code_fence is None:
            heading_match = re.match(r"^ {0,3}(#{1,6})\s+(.+)$", line)
        if heading_match:
            if current_title is not None and "\n".join(content_lines).strip():
                body = "\n".join(orphan_lines + content_lines).strip()
                chunks.append({"title": current_title, "content": current_title + "\n" + body,
                               "file_title": file_title})
                orphan_lines = []
            content_lines = []

            # 遇到同级/上级标题时，清掉旧的下级标题；跳级处用 None 占位。
            level = len(heading_match.group(1))
            heading_stack = heading_stack[:level]
            while len(heading_stack) < level:
                heading_stack.append(None)
            heading_stack[level - 1] = stripped
            current_title = "_".join(title for title in heading_stack if title)
        elif current_title is None:
            orphan_lines.append(line)
        else:
            content_lines.append(line)

    # 循环结束还剩最后一个章节；无标题的文档使用文件名作为标题。
    body = "\n".join(orphan_lines + content_lines).strip()
    if body or current_title is not None:
        title = current_title or file_title
        chunks.append({"title": title, "content": title + "\n" + body, "file_title": file_title})
    logger.info(f"标题切分完成：得到 {len(chunks)} 个章节块")
    return chunks


def _split_chunk_content(chunk):
    """长章节先去掉标题前缀，再递归切正文，最后给每块补回标题。"""
    title = chunk["title"]
    prefix = title + "\n"
    body = chunk["content"][len(prefix):]
    # 标题特别长时，不能把 chunk_size 算成负数；允许该切片超过目标长度。
    body_size = max(CHUNK_SIZE - len(prefix), CHUNK_OVERLAP + 1)
    separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    # 把围栏代码块、连续表格行作为整体保留，普通正文才递归切分。
    # 它们若超长，允许单独形成大块，并记录警告，而不是破坏结构。
    protected_pattern = re.compile(
        r"(^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\2[ \t]*(?:\n|$)|"
        r"(?:^[ \t]*\|[^\n]*(?:\n|$))+)", re.MULTILINE | re.DOTALL
    )
    texts = []
    position = 0
    for match in protected_pattern.finditer(body):
        texts.extend(_recursive_split_text(body[position:match.start()], body_size,
                                           CHUNK_OVERLAP, separators))
        protected_text = match.group(0).strip("\n")
        texts.append(protected_text)
        if len(prefix) + len(protected_text) > CHUNK_MAX_SIZE:
            logger.warning(f"章节 {title} 的代码块/表格超过 {CHUNK_MAX_SIZE} 字符，整体保留")
        position = match.end()
    texts.extend(_recursive_split_text(body[position:], body_size,
                                       CHUNK_OVERLAP, separators))
    if not texts:
        texts = [body]

    sub_chunks = []
    for part, text in enumerate(texts, start=1):
        sub_chunks.append({"title": f"{title}_{part}", "parent_title": title,
                           "part": part, "content": prefix + text,
                           "file_title": chunk["file_title"]})
    return sub_chunks


def _recursive_split_text(text, chunk_size, chunk_overlap, separators):
    """
    一个基础、可读的递归字符切分器。

    优先按“段落 → 换行 → 句号等标点 → 空格”切分；只有完全没有语义
    分隔符的超长文本才按字符硬切。这样不依赖额外运行库，也便于观察算法。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]
    if separator == "":
        # 最终兜底：固定字符窗口，并保留相邻窗口的少量重叠。
        step = max(chunk_size - chunk_overlap, 1)
        return [text[start:start + chunk_size] for start in range(0, len(text), step)]

    # 当前分隔符不存在时，继续尝试更细一级的分隔符。
    if separator not in text:
        return _recursive_split_text(text, chunk_size, chunk_overlap, remaining_separators)

    # 分隔后把分隔符补回前一段，避免句号、换行等内容被丢失。
    raw_parts = text.split(separator)
    semantic_parts = []
    for index, part in enumerate(raw_parts):
        if index < len(raw_parts) - 1:
            part += separator
        if not part.strip():
            continue
        if len(part) > chunk_size:
            semantic_parts.extend(_recursive_split_text(
                part, chunk_size, chunk_overlap, remaining_separators
            ))
        else:
            semantic_parts.append(part)

    # 把小语义单元装入目标长度的 chunk；换块时保留上一块末尾作为重叠上下文。
    chunks = []
    current = ""
    for part in semantic_parts:
        if not current or len(current) + len(part) <= chunk_size:
            current += part
            continue
        chunks.append(current.strip())
        overlap_text = current[-chunk_overlap:] if chunk_overlap else ""
        # 如果重叠内容加新段超过上限，宁可缩短重叠，也不让普通文本块失控。
        allowed_overlap = max(chunk_size - len(part), 0)
        current = overlap_text[-allowed_overlap:] + part
    if current.strip():
        chunks.append(current.strip())
    return chunks


def step_3_refine_split_and_merge_chunks(chunks):
    """长块切小；只合并同一章节中相邻的短块，不跨章节混合知识点。"""
    refined_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > CHUNK_SIZE:
            refined_chunks.extend(_split_chunk_content(chunk))
        else:
            refined_chunks.append(chunk)
    logger.info(f"长块切分完成：得到 {len(refined_chunks)} 个切片")

    merged_chunks = []
    for chunk in refined_chunks:
        if merged_chunks:
            previous = merged_chunks[-1]
            parent = previous.get("parent_title")
            if parent and parent == chunk.get("parent_title") and len(previous["content"]) <= CHUNK_MIN:
                # 同章节切片的 content 都含相同标题，合并时去掉第二块的重复标题。
                next_body = chunk["content"][len(parent) + 1:]
                combined = previous["content"] + "\n" + next_body
                if len(combined) <= CHUNK_MAX_SIZE:
                    previous["content"] = combined
                    continue
        merged_chunks.append(dict(chunk))
    logger.info(f"短块合并完成：剩余 {len(merged_chunks)} 个切片")
    return merged_chunks


def step_4_padding_chunks_metadata(chunks):
    """补齐课件约定的 parent_title、part，并在合并后重新编号。"""
    part_counts = {}
    for chunk in chunks:
        was_split = "parent_title" in chunk
        parent = chunk.get("parent_title", chunk["title"])
        part_counts[parent] = part_counts.get(parent, 0) + 1
        chunk["parent_title"] = parent
        chunk["part"] = part_counts[parent]
        if was_split:
            chunk["title"] = f"{parent}_{chunk['part']}"
    logger.info("切片元数据已补齐：title、parent_title、part、content、file_title")


def step_5_backup_chunks_json(chunks, md_path):
    """备份到同名 JSON，例如 手册_processed.md → 手册_processed.json。"""
    json_path = Path(md_path).with_suffix(".json")
    # 重复运行会覆盖这份备份，不会追加旧切片。
    json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(f"切片 JSON 备份已保存：{json_path}")
