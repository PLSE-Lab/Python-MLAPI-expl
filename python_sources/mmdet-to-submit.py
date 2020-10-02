import sys

sys.path.append("../../.")
sys.path.append("../.")

import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib

import argparse
import numpy as np
from multiprocessing import Pool
import os
from tqdm import tqdm
import mmcv
import os.path as osp
import pycocotools.mask as mutils
import pandas as pd

KLASSES = [
    "/m/01bms0",
    "/m/03jbxj",
    "/m/0jy4k",
    "/m/09gtd",
    "/m/01j5ks",
    "/m/01k6s3",
    "/m/05ctyq",
    "/m/015x5n",
    "/m/0jwn_",
    "/m/02zt3",
    "/m/05_5p_0",
    "/m/03qrc",
    "/m/01btn",
    "/m/02f9f_",
    "/m/01m4t",
    "/m/01x3jk",
    "/m/01pns0",
    "/m/01lcw4",
    "/m/084zz",
    "/m/0fx9l",
    "/m/0cjs7",
    "/m/096mb",
    "/m/02d1br",
    "/m/07dd4",
    "/m/02rgn06",
    "/m/012n7d",
    "/m/01_5g",
    "/m/0dq75",
    "/m/01f8m5",
    "/m/04g2r",
    "/m/029b3",
    "/m/047j0r",
    "/m/0hdln",
    "/m/03bbps",
    "/m/0jg57",
    "/m/0lt4_",
    "/m/01f91_",
    "/m/01b9xk",
    "/m/04ylt",
    "/m/043nyj",
    "/m/015wgc",
    "/m/03m3vtv",
    "/m/02tsc9",
    "/m/0h8n6f9",
    "/m/03q5t",
    "/m/04p0qw",
    "/m/0pcr",
    "/m/02l8p9",
    "/m/0388q",
    "/m/05bm6",
    "/m/02w3r3",
    "/m/046dlr",
    "/m/02g30s",
    "/m/01h8tj",
    "/m/0898b",
    "/m/076bq",
    "/m/0120dh",
    "/m/01lsmm",
    "/m/03d443",
    "/m/04c0y",
    "/m/0449p",
    "/m/0c29q",
    "/m/04h8sr",
    "/m/0frqm",
    "/m/02cvgx",
    "/m/01fh4r",
    "/m/01x_v",
    "/m/0d20w4",
    "/m/01dxs",
    "/m/09g1w",
    "/m/0kmg4",
    "/m/0584n8",
    "/m/02pv19",
    "/m/07dm6",
    "/m/0fbw6",
    "/m/03bk1",
    "/m/0633h",
    "/m/0by6g",
    "/m/06mf6",
    "/m/04tn4x",
    "/m/0h8ntjv",
    "/m/058qzx",
    "/m/06pcq",
    "/m/01kb5b",
    "/m/05z6w",
    "/m/078jl",
    "/m/027pcv",
    "/m/01h44",
    "/m/03y6mg",
    "/m/05n4y",
    "/m/0gd36",
    "/m/03fj2",
    "/m/0bwd_0j",
    "/m/09rvcxw",
    "/m/04rmv",
    "/m/0_cp5",
    "/m/06_72j",
    "/m/0cn6p",
    "/m/02hj4",
    "/m/0420v5",
    "/m/0h8lkj8",
    "/m/0h8my_4",
    "/m/01dwwc",
    "/m/0fldg",
    "/m/09f_2",
    "/m/01dwsz",
    "/m/020lf",
    "/m/03s_tn",
    "/m/02zvsm",
    "/m/029bxz",
    "/m/09qck",
    "/m/0cd4d",
    "/m/06j2d",
    "/m/04v6l4",
    "/m/061_f",
    "/m/0306r",
    "/m/06_fw",
    "/m/0wdt60w",
    "/m/0kpqd",
    "/m/0l14j_",
    "/m/0ccs93",
    "/m/03c7gz",
    "/m/06ncr",
    "/m/01j3zr",
    "/m/01s55n",
    "/m/02p3w7d",
    "/m/02gzp",
    "/m/0dkzw",
    "/m/0174k2",
    "/m/01xs3r",
    "/m/02jfl0",
    "/m/068zj",
    "/m/03v5tg",
    "/m/0dj6p",
    "/m/011k07",
    "/m/0162_1",
    "/m/0bh9flk",
    "/m/015x4r",
    "/m/0dbzx",
    "/m/05vtc",
    "/m/09ld4",
    "/m/04h7h",
    "/m/0176mf",
    "/m/03g8mr",
    "/m/06y5r",
    "/m/01fb_0",
    "/m/03fwl",
    "/m/04m9y",
    "/m/0gv1x",
    "/m/09d5_",
    "/m/0jly1",
    "/m/01xqw",
    "/m/04ctx",
    "/m/0gxl3",
    "/m/0fj52s",
    "/m/0cdn1",
    "/m/0hqkz",
    "/m/02jz0l",
    "/m/07clx",
    "/m/0cnyhnx",
    "/m/04yqq2",
    "/m/08pbxl",
    "/m/084rd",
    "/m/0hkxq",
    "/m/054fyh",
    "/m/02d9qx",
    "/m/071qp",
    "/m/08hvt4",
    "/m/01dy8n",
    "/m/0663v",
    "/m/019w40",
    "/m/03m3pdh",
    "/m/07bgp",
    "/m/0c06p",
    "/m/01tcjp",
    "/m/021mn",
    "/m/014j1m",
    "/m/05kyg_",
    "/m/016m2d",
    "/m/09b5t",
    "/m/0703r8",
    "/m/03grzl",
    "/m/05r5c",
    "/m/0bjyj5",
    "/m/02zn6n",
    "/m/0dftk",
    "/m/0pg52",
    "/m/09k_b",
    "/m/05zsy",
    "/m/0h23m",
    "/m/0cyhj_",
    "/m/07cmd",
    "/m/02vqfm",
    "/m/01z1kdw",
    "/m/0242l",
    "/m/0k1tl",
    "/m/0gjkl",
    "/m/09csl",
    "/m/0dbvp",
    "/m/0f6wt",
    "/m/025nd",
    "/m/0ftb8",
    "/m/02s195",
    "/m/01226z",
    "/m/0ph39",
    "/m/06k2mb",
    "/m/0cmx8",
    "/m/02jvh9",
    "/m/01gkx_",
    "/m/09ddx",
    "/m/01yrx",
    "/m/07j87",
    "/m/024g6",
    "/m/025rp__",
    "/m/01cmb2",
    "/m/01xq0k1",
    "/m/07fbm7",
    "/m/01yx86",
    "/m/034c16",
    "/m/015qff",
    "/m/03q5c7",
    "/m/0fszt",
    "/m/05gqfk",
    "/m/026qbn5",
    "/m/01599",
    "/m/02h19r",
    "/m/02p5f1q",
    "/m/081qc",
    "/m/052sf",
    "/m/0dv5r",
    "/m/06m11",
    "/m/080hkjn",
    "/m/02fq_6",
    "/m/01nq26",
    "/m/01m2v",
    "/m/050k8",
    "/m/01j51",
    "/m/03k3r",
    "/m/01b638",
    "/m/01940j",
    "/m/0h2r6",
    "/m/09728",
    "/m/0bt9lr",
    "/m/0cmf2",
    "/m/04_sv",
    "/m/0bt_c3",
    "/m/07jdr",
    "/m/039xj_",
    "/m/025dyy",
    "/m/07r04",
    "/m/083wq",
    "/m/01bjv",
    "/m/0283dt1",
    "/m/01n4qj",
    "/m/01jfm_",
    "/m/0342h",
    "/m/02wbtzl",
    "/m/04dr76w",
    "/m/01bqk0",
    "/m/01xyhv",
    "/m/04kkgm",
    "/m/04yx4",
    "/m/0fm3zh",
    "/m/01c648",
    "/m/01bl7v",
    "/m/06z37_",
    "/m/01bfm9",
    "/m/03bt1vf",
    "/m/099ssp",
    "/m/01rkbr",
    "/m/05r655",
    "/m/079cl",
    "/m/03120",
    "/m/0fly7",
    "/m/01d40f",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/train/mmdet_train_config.py", type=str)
    parser.add_argument("--annotation", default="test", type=str)
    parser.add_argument("--epoch", default=6, type=int)
    parser.add_argument("--pred_out", help="output result file", default="predict_test_t.pkl", type=str)
    parser.add_argument("--submission", type=str, default="subm1.csv")
    parser.add_argument("--output", default="predictions", type=str)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--conf_threshold", type=float, default=0.01)
    return parser.parse_args()


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError("encode_binary_mask expects a binary mask, received dtype == %s" % mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError("encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def convert_preds(data):
    prediction, ann, conf_threshold, classes = data
    attrs, masks = prediction

    sample_string = ""
    if masks is not None:
        for n, (klass_masks, klass_attr) in enumerate(zip(masks, attrs)):
            if len(klass_masks) > 0:
                for mask_dict, attr in zip(klass_masks, klass_attr):
                    conf = attr[-1]
                    if conf > conf_threshold:
                        klass = classes[n]
                        mask = mutils.decode(mask_dict).astype(np.bool)
                        text_mask = encode_binary_mask(mask)
                        sample_string += f"{klass} {conf:0.5f} {text_mask} "

        if len(sample_string) > 0:
            if sample_string[-1] == " ":
                sample_string = sample_string[:-1]

        return sample_string, ann["filename"], ann["width"], ann["height"]


def main():
    os.makedirs("subm", exist_ok=True)
    classes = KLASSES

    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    print("loading pickle")
    df_lst = []
    for index in tqdm(range(0, 12), total=12):
        ann_file = osp.join(PATHS["dataset"]["path"], PATHS["dataset"]["annotations"], "test", f"test_{index}.pkl")
        annotations = mmcv.load(ann_file)

        pred_file = osp.join(cfg.work_dir, f"epoch_{args.epoch}_{index}_{args.pred_out}")
        predictions = mmcv.load(pred_file)
        print("converting predictions", pred_file)
        with Pool() as p:
            lst = list(
                tqdm(
                    p.imap(
                        convert_preds,
                        zip(
                            predictions,
                            annotations,
                            [args.conf_threshold] * len(predictions),
                            [classes] * len(predictions),
                        ),
                    ),
                    total=len(predictions),
                )
            )

        string_preds, fns, widths, heights = zip(*lst)
        tdf = pd.DataFrame()
        tdf["ImageID"] = [el.split(".")[0] for el in fns]
        tdf["ImageWidth"] = widths
        tdf["ImageHeight"] = heights
        tdf["PredictionString"] = string_preds
        tdf.to_csv(f"subm/{index}_{args.submission}", index=False)
        df_lst.append(tdf)
        del annotations, predictions

    df = pd.concat(df_lst)
    df = fix_subm(df)

    df.to_csv(f"subm/{args.submission}", index=False)


def fix_string(lst):
    lst = [el.replace(" b'", " ") for el in lst]
    lst = [el.replace("' ", " ") for el in lst]
    lst = [el.replace("'", "") for el in lst]
    return lst


def fix_subm(df):
    empty = pd.read_csv("subm/sample_empty_submission.csv")
    print(empty.head())

    df_none = df.dropna()
    df_none["PredictionString"] = fix_string(df_none["PredictionString"])
    df_nan = df[~df["ImageID"].isin(df_none["ImageID"])]

    df = pd.concat([df_none, df_nan])

    miss = empty[~empty["ImageID"].isin(df["ImageID"])]
    print("miss", miss.shape)
    print(df.shape)
    return df


if __name__ == "__main__":
    main()
