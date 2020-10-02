import subprocess
import sys

# Needed to avoid error installing HDBSCAN
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-yq", "typing"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pandas==1.0.3",
                       "hdbscan==0.8.24",
                       "spacy==2.2.1",
                       "scispacy==0.2.4",
                       "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz",
                       "git+https://github.com/lmcinnes/umap.git@0.4.0rc2",
                       "scipy==1.4.1",
                       "tensorflow==2.0.1",
                       "tensorflow_hub==0.7.0",
                       "scikit-learn==0.20.3",
                       "numpy==1.18.1"])     

import json
import shutil
from pathlib import Path

import en_core_sci_md
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.umls_utils import UmlsEntity

DATA_DIR = Path("/kaggle/input/CORD-19-research-challenge")
# Avoid limits on number of generated files by writing to a temp directory
# and zipping + copying to the final directory
TMP_OUTPUT_DIR = Path("/tmp/output")
ANNOTATION_DIR = TMP_OUTPUT_DIR / "umls_annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("/kaggle/working")

def all_json_iter():
    """
    Iterate over all data files across all text subsets
    """
    all_files = DATA_DIR.glob(
        "document_parses/*/*.json"
    )

    for json_file in all_files:
        if json_file.name.startswith("."):
            # There are some garbage files in here for some reason
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            try:
                article_json = json.load(f)
            except Exception:
                raise RuntimeError(f"Failed to parse json from {json_file}")

        yield json_file.name, article_json


def get_article_id(article_json):
    return article_json["paper_id"]


def get_title(article_json):
    return article_json["metadata"]["title"]


def get_abstract(article_json):
    if "abstract" not in article_json:
        return ""
    return "\n\n".join(a["text"] for a in article_json["abstract"])


if __name__ == "__main__":
    print("ANNOTATING TEXTS")

    print("Loading models")
    nlp = en_core_sci_md.load()
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(linker)
    print("Models loaded")

    print("Annotating texts")

    def text_context_gen():
        for filename, article_json in all_json_iter():
            text = f"{get_title(article_json)}\n\n{get_abstract(article_json)}"
            yield text, (filename, article_json)

    for doc, (filename, article_json) in nlp.pipe(text_context_gen(), as_tuples=True):
        annotations = []
        for ent in doc.ents:
            for cui, score in ent._.umls_ents:
                umls_ent: UmlsEntity = linker.umls.cui_to_entity[cui]
                annotations.append(
                    {
                        "cui": cui,
                        "tuis": umls_ent.types,
                        "canonical_name": umls_ent.canonical_name,
                        "score": score,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

        with open(ANNOTATION_DIR / filename, "w") as f:
            json.dump(annotations, f)
            
    # Move tmp output to final output dir after zipping
    shutil.make_archive(OUTPUT_DIR / "umls_annotations", "zip", TMP_OUTPUT_DIR)

    print("DONE ANNOTATING TEXTS")
