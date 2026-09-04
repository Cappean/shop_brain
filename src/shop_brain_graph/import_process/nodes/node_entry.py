import sys

from utils.task_utils import add_running_task,add_done_task
from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState,get_default_state,create_default_state
from pathlib import Path

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """
    # 节点一开始都要添加进运行中的节点列表，前端要用。
    add_running_task(state["task_id"],"node_entry")
    # TODO 校验输入状态是否合法。
    local_file_path = state["local_file_path"] if isinstance(state["local_file_path"], Path) else Path(state["local_file_path"])
    local_dir = state["local_dir"] if isinstance(state["local_dir"], Path) else Path(state["local_dir"])

    if str(local_file_path).strip() == '.' or str(local_dir).strip()== '.':
        logger.error("错误。没有输入文件路径或没有配置输出目录")
        return state # 这里直接返回state 是因为入口节点后面有路由函数。路由依赖的bool 没有true ，自动路由到end了。
     
    # TODO 校验路径是否有对应实体
    if not local_file_path.exists():
        logger.error(f"错误。文件路径不存在。{local_file_path}")
        return state
    if not local_dir.is_dir():
        logger.error(f"错误。输出目录不存在。{local_dir}")
        return state

    # TODO 判断文件类型 。三种，pdf,md 和其它。这里可以扩展。
    if local_file_path.suffix == ".pdf":
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.suffix == ".md":
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        logger.error(f"错误。暂时不支持的文件类型。{local_file_path}")
        return state
    # TODO 获取文件名
    state["file_title"] = local_file_path.stem

    # 节点运行完要添加进运行结束节点，也是前端
    add_done_task(state["task_id"],"node_entry")

    return state


if __name__ == "__main__":
    from utils.path_util import PROJECT_ROOT
    from rich import print as rprint
    state = create_default_state(local_file_path='',
                                 task_id="1234567890",
                                 local_dir=PROJECT_ROOT / "doc" / "output")
    result = node_entry(state)
    rprint(result)
    # logger.info(f"node_entry: {state["is_pdf_read_enabled"]}", )