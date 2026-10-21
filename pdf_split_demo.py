import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   #递归字符文本分割器

DOC_DIR = Path(__file__).resolve().parent / "本地文档"


def resolve_pdf_path(pdf_path: str = None) -> Path:
   
    candidate = pdf_path or os.environ.get("RAG_DOC_PATH")
    if not candidate:
        raise SystemExit(
            "未指定 PDF 路径。请用 --pdf 显式传参，或设置环境变量 RAG_DOC_PATH。\n"
            "示例：python pdf_split_demo.py --pdf ./本地文档/政治.pdf"
        )
    p = Path(candidate)
    if not p.is_absolute():
        p = (DOC_DIR / p) if (DOC_DIR / p).exists() else (Path.cwd() / p)
    if not p.exists():
        raise SystemExit(f"PDF 不存在：{p}")
    return p


def load_and_split_pdf(pdf_load_path:str,chunk_size=500,chunk_overlap=80):
    """加载PDF文档,返回切块后的文档快"""
    loader=PyPDFLoader(pdf_load_path) #只记住读哪个pdf 不读取文件
    documents =loader.load()     ##执行真正的文件读取，打开 PDF，逐页解析文本内容，返回的是一个document对象列表
    text_splitter =RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
    split_docs =text_splitter.split_documents(documents)
    return split_docs



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="把一份 PDF 切块并打印分块信息")
    parser.add_argument("--pdf", help="待切块的 PDF 路径；也可用环境变量 RAG_DOC_PATH 指定")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    docs = load_and_split_pdf(str(pdf_path))
    print(f"文档：{pdf_path}")
    print(f"一共切成{len(docs)}个文本块")
    print(docs[0].page_content[:200])
