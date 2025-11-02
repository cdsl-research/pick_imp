#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pick_imp.py
data(=ファイル or ディレクトリ)から「命令・依頼」っぽい文を
ゼロショット分類（日本語対応XNLI）で抽出し、anser/ に日時付きで保存。

使い方:
  cd ~/hyouka/information
  python3 pick_imp.py
  # または
  python3 pick_imp.py --data ./data --threshold 0.5 --chunk paragraph
"""

from pathlib import Path
import re
import datetime
import argparse
from typing import List, Tuple, Iterable

# ====== 設定 ======
BASE = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE / "data"   # ファイルでもディレクトリでもOK
ANSER_DIR = BASE / "anser"

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CANDIDATE_LABELS = ["命令・依頼", "説明", "質問", "状況"]
HYPOTHESIS_TEMPLATE = "この文章は{}に当たる。"

DEFAULT_THRESHOLD = 0.50
BATCH_SIZE = 8

# ディレクトリ走査時の対象拡張子（テキスト系のみ）
DEFAULT_EXTS = {".txt", ".md", ".log", ".rst", ".json", ".yml", ".yaml", ""}

# フォールバック（表層ルール）
FALLBACK_PATTERNS = [
    r"(してください|して下さい|お願いいたします|お願い致します|お願いします)",
    r"(しなさい|せよ|すること|しましょう|しよう)",
    r"(確認してください|参照してください|実行してください|設定してください|入力してください|アクセスしてください)",
    r"(まず|次に|続いて|最後に).*(する|実行|確認|設定|アクセス)",
    r"^\s*\$+\s*.+",  # コマンド行
]

def jst_timestamp() -> str:
    jst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(jst).strftime("%Y%m%d_%H%M%S")

# -------- 文・段落分割 --------
def split_paragraphs(text: str) -> List[str]:
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", norm)
    return [b.strip() for b in blocks if b.strip()]

def split_sentences_in_paragraph(para: str) -> List[str]:
    # ```コードブロック```は塊として保護
    blocks = []
    code = re.compile(r"```.*?```", re.DOTALL)
    last = 0
    for m in code.finditer(para):
        if m.start() > last:
            blocks.append(("text", para[last:m.start()]))
        blocks.append(("code", m.group(0)))
        last = m.end()
    if last < len(para):
        blocks.append(("text", para[last:]))

    sents: List[str] = []
    for kind, chunk in blocks:
        if kind == "code":
            s = chunk.strip()
            if s:
                sents.append(s)
        else:
            tmp = re.split(r"[。．！？\n]+", chunk)
            for s in tmp:
                s = s.strip()
                if s:
                    sents.append(s)
    return sents

def split_sentences_all(text: str) -> Tuple[List[str], List[int]]:
    """
    文配列と、各文が属する段落インデックス配列を返す
    """
    paras = split_paragraphs(text) or [text.strip()]
    sentences = []
    para_indices = []
    for p_idx, para in enumerate(paras):
        ss = split_sentences_in_paragraph(para)
        for s in ss:
            sentences.append(s)
            para_indices.append(p_idx)
    return sentences, para_indices

# -------- 入力読み込み（テキストのみ） --------
def load_text_from_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in DEFAULT_EXTS:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""

def iter_files_in_dir(data_dir: Path, allow_exts: Iterable[str]) -> List[Path]:
    files = []
    exts = {e.strip().lower() for e in allow_exts}
    for p in sorted(data_dir.rglob("*")):
        if p.is_file():
            suf = p.suffix.lower()
            if suf in exts or (suf == "" and "" in exts):
                files.append(p)
    return files

# -------- 推論 --------
def zero_shot_setup():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        truncation=True,
        device=-1,  # CPU固定（accelerate不要）
    )

def zero_shot_predict(clf, sentences: List[str]):
    """
    返り値: 各文に対する {label->score} の辞書
    """
    outs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        res = clf(
            batch,
            CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True
        )
        if isinstance(res, dict):
            res = [res]
        for r in res:
            label2score = {lab: float(scr) for lab, scr in zip(r["labels"], r["scores"])}
            for lab in CANDIDATE_LABELS:
                label2score.setdefault(lab, 0.0)
            outs.append(label2score)
    return outs

def fallback_rule(s: str) -> bool:
    return any(re.search(p, s) for p in FALLBACK_PATTERNS)

# -------- メイン処理 --------
def process_text(text: str, source: str, threshold: float, chunk_mode: str):
    sentences, para_idx = split_sentences_all(text)
    if not sentences:
        return {"file": source, "hits": [], "chunks": [], "original": text}

    clf = zero_shot_setup()
    preds = zero_shot_predict(clf, sentences)

    hits = []
    for i, (s, pred) in enumerate(zip(sentences, preds), start=1):
        imp = pred.get("命令・依頼", 0.0)
        label = "命令・依頼"
        is_imp = (imp >= threshold)

        # フォールバック（モデルが低くても典型表現なら拾う）
        if not is_imp and fallback_rule(s) and imp < 0.55:
            is_imp = True
            label = "命令・依頼(fallback)"
            imp = max(imp, 0.60)

        if is_imp:
            hits.append({
                "line": i,
                "para_idx": para_idx[i-1],
                "text": s,
                "label": label,
                "score": round(imp, 3),
            })

    chunks = []
    if chunk_mode == "paragraph" and hits:
        paras = split_paragraphs(text)
        used = sorted(set(h["para_idx"] for h in hits))
        for p in used:
            chunks.append({
                "para_idx": p,
                "text": paras[p],
                "hit_lines": [h["line"] for h in hits if h["para_idx"] == p],
            })

    return {"file": source, "hits": hits, "chunks": chunks, "original": text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH),
                        help="ファイルまたはディレクトリ（default: ./data）")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="命令・依頼の採用閾値（0〜1）")
    parser.add_argument("--chunk", type=str, default="none", choices=["none", "paragraph"],
                        help="段落“塊”出力（none/paragraph）")
    args = parser.parse_args()

    target = Path(args.data).resolve()
    ANSER_DIR.mkdir(parents=True, exist_ok=True)
    ts = jst_timestamp()
    out = ANSER_DIR / f"answer_{ts}.txt"

    report = []
    report.append(f"# 命令/依頼 抽出レポート (最小版)  ({ts} JST)")
    report.append(f"data path: {target}")
    report.append(f"model: {MODEL_NAME}")
    report.append(f"labels: {CANDIDATE_LABELS}")
    report.append(f"threshold: {args.threshold}")
    report.append(f"chunk_mode: {args.chunk}")
    report.append("")

    results = []

    if target.is_file():
        text = load_text_from_file(target)
        if text:
            results.append(process_text(text, str(target), args.threshold, args.chunk))
        else:
            report.append("※ data がファイルですが、テキストを抽出できませんでした。")
    elif target.is_dir():
        files = iter_files_in_dir(target, DEFAULT_EXTS)
        report.append(f"files: {len(files)}")
        for p in files:
            text = load_text_from_file(p)
            if text:
                results.append(process_text(text, str(p), args.threshold, args.chunk))
            else:
                results.append({"file": str(p), "hits": [], "chunks": [], "original": "[[no text extracted]]"})
    else:
        report.append("※ 指定した --data のパスが存在しません。")
        out.write_text("\n".join(report), encoding="utf-8")
        print(f"Wrote: {out}")
        return

    total_hits = sum(len(r.get("hits", [])) for r in results)
    report.insert(3, f"total_extracted_sentences: {total_hits}")

    for r in results:
        report.append("=" * 80)
        report.append(f"FILE: {r['file']}")
        report.append(f"EXTRACTED: {len(r.get('hits', []))}")
        report.append("-" * 80)
        if r.get("hits"):
            for h in r["hits"]:
                report.append(f"[line {h['line']:>4}] {h['text']}")
                report.append(f"  -> label={h['label']}  score={h['score']}")
        else:
            report.append("(no imperative/request-like sentences found)")
        if r.get("chunks"):
            report.append("-" * 80)
            report.append(f"CHUNKS: {len(r['chunks'])}")
            for c in r["chunks"]:
                ls = ", ".join(map(str, c["hit_lines"]))
                report.append(f"[para {c['para_idx']}] hit_lines=[{ls}]")
                report.append(c["text"])
                report.append("-" * 40)
        report.append("-" * 80)
        report.append("【Original Text】")
        report.append(r.get("original", "").rstrip())
        report.append("")

    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()

