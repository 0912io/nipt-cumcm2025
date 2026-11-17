"""读取官方附件时的可识别错误。缺少文件时给出说明，而不是半途抛出压缩包异常。"""

from __future__ import annotations


class WorkbookNotFoundError(FileNotFoundError):
    """本地没有 ``附件.xlsx``。该文件不进入 Git。"""


def missing_workbook_message(path) -> str:
    return (
        f"未找到官方附件工作簿：{path}\n"
        "请从全国大学生数学建模竞赛官网下载 C 题附件，并将「附件.xlsx」放在仓库根目录。"
        "该路径已被 .gitignore 忽略（*.xlsx），不要提交，也不要 force-add。"
        "修订分析在缺少该文件时不能计算数值结果。"
    )
