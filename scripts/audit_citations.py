import os, re, json
import duckdb
import pandas as pd

DB_PATH = "artifacts/mdc_challenge.db"
LABELS_CSV = "Data/train_labels.csv"
OUTDIR = "reports/citation_audit"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(LABELS_CSV)
df_nm = df[df["type"] != "Missing"][ ["article_id", "dataset_id", "type"] ].copy()
df_nm2 = df_nm.rename(columns={"article_id": "document_id", "dataset_id": "data_citation_label"})

con = duckdb.connect(DB_PATH)
df_cit = con.execute("SELECT document_id, data_citation, pages FROM citations").df()
con.close()
df_cit = df_cit.drop_duplicates()

merged = df_nm2.merge(df_cit, how="left", left_on=["document_id", "data_citation_label"], right_on=["document_id", "data_citation"])
found = merged[merged["data_citation"].notna()].copy()
missing = merged[merged["data_citation"].isna()].copy()

extras = df_cit.merge(df_nm2, how="left", left_on=["document_id", "data_citation"], right_on=["document_id", "data_citation_label"]) 
extras = extras[extras["data_citation_label"].isna()].copy()

def normalise_id(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"(?i)\.v\d+$", "", s)
    s = re.sub(r"(?i)/t\d+$", "", s)
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())

def egse_variant(s: str):
    s = str(s)
    return re.sub(r"(?i)^E-GEOD-", "GSE", s) if re.match(r"(?i)^E-GEOD-\d+$", s) else None

_df = df_cit.copy()
_df["norm"] = _df["data_citation"].apply(normalise_id)
doc2norms = {doc: set(sub["norm"].tolist()) for doc, sub in _df.groupby("document_id")}

if not missing.empty:
    def variant_hit(row) -> bool:
        doc = row["document_id"]
        raw = row["data_citation_label"]
        norms = doc2norms.get(doc, set())
        variants = [raw]
        cleaned = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", str(raw))
        variants.append(cleaned)
        nover = re.sub(r"(?i)\.v\d+$", "", cleaned)
        variants.append(nover)
        notbl = re.sub(r"(?i)/t\d+$", "", nover)
        variants.append(notbl)
        eg = egse_variant(raw)
        if eg:
            variants.append(eg)
        for v in variants:
            if normalise_id(v) in norms:
                return True
        return False
    missing["variant_hit"] = missing.apply(variant_hit, axis=1)
else:
    missing["variant_hit"] = pd.Series(dtype=bool)

found[["document_id", "data_citation_label", "type", "pages"]].to_csv(os.path.join(OUTDIR, "found.csv"), index=False)
missing[["document_id", "data_citation_label", "type", "variant_hit"]].to_csv(os.path.join(OUTDIR, "missing.csv"), index=False)
extras[["document_id", "data_citation"]].to_csv(os.path.join(OUTDIR, "extras.csv"), index=False)

miss_doc_counts = missing.groupby("document_id").size().reset_index(name="missing_count").sort_values("missing_count", ascending=False)
miss_doc_counts.to_csv(os.path.join(OUTDIR, "missing_by_document.csv"), index=False)
miss_type_counts = missing.groupby("type").size().reset_index(name="missing_count").sort_values("missing_count", ascending=False)
miss_type_counts.to_csv(os.path.join(OUTDIR, "missing_by_type.csv"), index=False)

summary = {
    "labels_non_missing": int(len(df_nm2)),
    "extracted_total": int(len(df_cit)),
    "found": int(len(found)),
    "missing": int(len(missing)),
    "missing_with_variant_hit": int(missing["variant_hit"].sum()) if not missing.empty else 0,
    "extras_not_in_labels": int(len(extras)),
}
print(json.dumps(summary, indent=2))
