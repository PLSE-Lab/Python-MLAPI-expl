#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Introduction](#introduction)
# 1. [Import libraries](#import_libraries)
# 1. [Configure hyper-parameters](#configure_hyper_parameters)
# 1. [Define helper-functions](#define_helper_functions)
# 1. [Define useful classes](#define_useful_classes)
# 1. [Get datasets](#get_datasets)
#   1. [Find all folders contain training data](#fine_all_folders_contain_training_data)
#   1. [Get and merge all metadata files](#get_and_merge_all_metadata_files)
#   1. [Clean data](#clean_data)
#     1. [Remove corrupt videos or ones in what cannot detect any faces](#remove_corrupt_videos_or_ones_in_what_cannot_detect_any_faces)
#     1. [Remove videos in which do not have enough faces](#remove_videos_in_which_do_not_have_enough_faces)
#   1. [Change label format](#change_label_format)
#   1. [Split dataset](#split_dataset)
# 1. [Start the training process](#start_the_training_process)
#   1. [Create dataloaders, classifier, etc.](#create_dataloaders_classifier_etc)
#   1. [Train the classifier](#train_the_classifier)
#     1. [Start the warm-up rounds](#start_the_warm_up_rounds)
#     1. [Visualize the results of the warm-up phase](#visualize_the_results_of_the_warm_up_phase)
#     1. [Start the fine-tune rounds](#start_the_fine_tune_rounds)
#     1. [Visualize the results of the fine-tune phase](#visualize_the_results_of_the_fine_tune_phase)
# 1. [Conclusion](#conclusion)

# <a id="introduction"></a>
# # Introduction
# 
# In this notebook, I will implement a simple pipeline which includes my suggested solution named `multiface classifier` to determine if an input video is FAKE or NOT.
# 
# Note:
# * This solution only used for learning and research purposes, and I do not guarantee that it will produce good results.
# * All the datasets I used to train the classifier are well prepared beforehand, and you can find the full list of them in the [*Other useful datasets*](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954) discussion. For simplicity, I only choose a few of these datasets to be added in this kernel; you can add more if you wish. I have tested by loading 32 datasets at the same time and it works just fine.
# 
# You are in the second step of the whole pipeline, to read about the data preparation process, let follow this [*link*](https://www.kaggle.com/phunghieu/deepfake-detection-face-extractor).
# 
# ---
# ## Idea
# Sometimes, it is hard to determine a video if it is `real` or `fake` by only using a single face appears in this video, or separately classify each face then combine the results in some ways, for example averaging, to predict the label of the input video. I think we can give the classifier more meaningful information by feeding it with multiple face images at once (the strategy to sample these images will be discussed later.) Then, based on a sequence of faces, the classifier can give a better judgment on the video it gets.
# 
# ---
# ## Multiface's general diagram
# ![diagram](data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%3F%3E%0A%3C%21DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%0A%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%20version%3D%221.1%22%20width%3D%221368%22%20height%3D%22389%22%20viewBox%3D%22-0.5%20-0.5%201368%20389%22%20content%3D%22%26lt%3Bmxfile%20host%3D%26quot%3BElectron%26quot%3B%20modified%3D%26quot%3B2020-02-16T02%3A32%3A31.746Z%26quot%3B%20agent%3D%26quot%3BMozilla%2F5.0%20%28X11%3B%20Linux%20x86_64%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20draw.io%2F12.2.2%20Chrome%2F78.0.3904.94%20Electron%2F7.1.0%20Safari%2F537.36%26quot%3B%20etag%3D%26quot%3B8-_NGCFXhlIKuX6CvSf-%26quot%3B%20version%3D%26quot%3B12.2.2%26quot%3B%20type%3D%26quot%3Bdevice%26quot%3B%20pages%3D%26quot%3B1%26quot%3B%26gt%3B%26lt%3Bdiagram%20id%3D%26quot%3BxoFWK3179xdSsBTeCwkt%26quot%3B%20name%3D%26quot%3BPage-1%26quot%3B%26gt%3B7Vpdc%2BMmFP01nmkfNqNvK4%2BJ7aQzm2Q79U67feoQCUvUSKgIxXZ%2FfUEC6wPFlmftKJPuU%2BAIELr3nHsvOBN7lmzvKcjiRxJCPLGMcDux5xPLMs2pzf8IZFchnu1WQERRKAfVwBL9CyVoSLRAIcxbAxkhmKGsDQYkTWHAWhiglGzaw1YEt9%2BagQhqwDIAWEf%2FQCGLK9R3jRr%2FBaIoVm82DfkkAWqwBPIYhGTTgOzFxJ5RQljVSrYziIXxlF2qeXevPN1vjMKUDZngzz%2FfP4Tr2eqfhXn%2FFYLZI%2Fjrk1wlZzv1wTDk3y%2B7hLKYRCQFeFGjt5QUaQjFqgbv1WMeCMk4aHLwb8jYTjoTFIxwKGYJlk%2FhFrFvjfafYqkrV%2FbmW7ly2dmpTsro7luz05gluvW0sqfmVd8nPupVsykbkIIG8ICtLEk%2FQCPIDtm09i6XBSQJ5BviEynEgKGX9kaA5Ge0H1e7kDekF0%2FwqNzlC8CFfNMd4J9lGXPIuEAInVge5tu%2FfRatSLR%2Bevw6e3r6WaNC29GbGDG4zEBpow1Xe9up8rWQMrg9bGrdMnKCo7QjY4U5lf1NrTxTySluqM4zLmRM94c8BsvDGygPf0x1eJo6fuf5hfRo4irJHF0S%2FBU888A3kYPfUYOvq8HvEYNzKTFMf4hhsBj8gWKwxhSDr6cKChJebY3J%2Bj1%2F3wvt1cJvzPs9h6%2BmbovGR0icc86xG1H5ciAlKVTYHRLfLWeEakSAQZ6joALlEPO8WlCl%2FHEx2GOqQW2zUzmNqwavWxKNrQbTH1UN5mkRPSjoS%2FneZipR%2BWJIKjmjDKzpUBm8QonBMpBTfyWI77EOrFabSpbntpeoNiZnNQ%2BS3YU6Edru7qUyhbZQybf993wHBa%2FHLEROYc9ZaDsKBb%2B3LhnKHGGLt%2BSO2nMjyP%2B2uHnQCMUjNmv7P2eUrOGMYH6C3ifWFU%2BYHQhgFKUisXLvQI7fiviPAoBv5IMEhSF%2BLXu0aXqGBNI5RFg9%2BcPpyR%2FWpfKH1XM%2FcfN58b9xgO2M7ABn1Cu%2FdvScHg6f76%2BUteyhAdQZs5RV22yITHP6kdoV5Fl1nb5CW%2BH4sxSzZicYXQ%2FTwsWK2cpLh%2B2UxyATzaB4HlDyP1dCeXjeAyBYR6V8vhQMI8HhEg8BXX%2FhyyBWsv3KcNugVaLnulj1jhve6zH8xe5VLfd9ErRTItvG2ATV7yvfhZ3sjp3Mke1kO2MmtVPuJls5rc5YzbRmttOaTHx1ThvtXDpuTptqUpiV5lshXuXp9%2FdzCDM%2B%2FAkWFOCywTaErsf%2FlUs7h%2FcEY7NPPReLxkrNDdMuQZJxA33YqtxzO05w3EER7GJlua2XbEvGa4eP6wJXmfzEZHu5k5F%2BNNWsX1eDFO9uKXeQCJnHokZtu6rHeLwkwuKfxH3ueQh9fTyqvOlP545O6JR3V71X6h%2BG1FPzeGXknofUvFv%2F81B1P1b%2FC5a9%2BA8%3D%26lt%3B%2Fdiagram%26gt%3B%26lt%3B%2Fmxfile%26gt%3B%22%20style%3D%22background-color%3A%20rgb%28255%2C%20255%2C%20255%29%3B%22%3E%3Cdefs%3E%3Cfilter%20id%3D%22dropShadow%22%3E%3CfeGaussianBlur%20in%3D%22SourceAlpha%22%20stdDeviation%3D%221.7%22%20result%3D%22blur%22%2F%3E%3CfeOffset%20in%3D%22blur%22%20dx%3D%223%22%20dy%3D%223%22%20result%3D%22offsetBlur%22%2F%3E%3CfeFlood%20flood-color%3D%22%233D4574%22%20flood-opacity%3D%220.4%22%20result%3D%22offsetColor%22%2F%3E%3CfeComposite%20in%3D%22offsetColor%22%20in2%3D%22offsetBlur%22%20operator%3D%22in%22%20result%3D%22offsetBlur%22%2F%3E%3CfeBlend%20in%3D%22SourceGraphic%22%20in2%3D%22offsetBlur%22%2F%3E%3C%2Ffilter%3E%3C%2Fdefs%3E%3Cg%20filter%3D%22url%28%23dropShadow%29%22%3E%3Cpath%20d%3D%22M%20880.67%2060%20L%201027.93%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%201038.43%2060%20L%201024.43%2067%20L%201027.93%2060%20L%201024.43%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%22640%22%20y%3D%220%22%20width%3D%22240%22%20height%3D%22120%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28683.5%2C33.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2276%22%20height%3D%2226%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2076px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFace%20Detector%3Cbr%20%2F%3E%28MTCNN%29%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2238%22%20y%3D%2219%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFace%20Detector%26lt%3Bbr%26gt%3B%28MTCNN%29%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20160.67%2060%20L%20307.93%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20318.43%2060%20L%20304.43%2067%20L%20307.93%2060%20L%20304.43%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%2280%22%20cy%3D%2260%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%2845.5%2C33.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2234%22%20height%3D%2226%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2036px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EVideo%3Cbr%20%2F%3E%28.mp4%29%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2217%22%20y%3D%2219%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3E%5BNot%20supported%20by%20viewer%5D%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20480.67%2060%20L%20627.93%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20638.43%2060%20L%20624.43%2067%20L%20627.93%2060%20L%20624.43%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%22400%22%20cy%3D%2260%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28359.5%2C47.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2240%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2042px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFrames%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2220%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFrames%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%201120%20100%20L%201120.67%20170%20L%201120.67%20227.26%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%201120.67%20237.76%20L%201113.67%20223.76%20L%201120.67%20227.26%20L%201127.67%20223.76%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%221120%22%20cy%3D%2260%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%281087.5%2C47.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2232%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2034px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFaces%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2216%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFaces%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20280.67%20300%20Q%20180.67%20300%20180.67%20270%20Q%20180.67%20240%2093.4%20240%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%2082.9%20240%20L%2096.9%20233%20L%2093.4%20240%20L%2096.9%20247%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20280.67%20300%20Q%20180.67%20300%20180.67%20330%20Q%20180.67%20360%2093.4%20360%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%2082.9%20360%20L%2096.9%20353%20L%2093.4%20360%20L%2096.9%20367%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%220%22%20y%3D%22220%22%20width%3D%2280%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%287.5%2C227.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2232%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2032px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EREAL%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2216%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EREAL%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Crect%20x%3D%220%22%20y%3D%22340%22%20width%3D%2280%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%289.5%2C347.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2230%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2032px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFAKE%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2215%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFAKE%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%201060.67%20300%20L%20892.74%20300%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20882.24%20300%20L%20896.24%20293%20L%20892.74%20300%20L%20896.24%20307%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%221060%22%20y%3D%22240%22%20width%3D%2280%22%20height%3D%2280%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20760%20240%20L%20840%20240%20L%20880%20280%20L%20880%20360%20L%20800%20360%20L%20760%20320%20L%20760%20240%20Z%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20760%20240%20L%20840%20240%20L%20880%20280%20L%20800%20280%20Z%22%20fill-opacity%3D%220.05%22%20fill%3D%22%23000000%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20760%20240%20L%20800%20280%20L%20800%20360%20L%20760%20320%20Z%22%20fill-opacity%3D%220.1%22%20fill%3D%22%23000000%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20800%20360%20L%20800%20280%20L%20760%20240%20M%20800%20280%20L%20880%20280%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%221080%22%20y%3D%22260%22%20width%3D%2280%22%20height%3D%2280%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%221100%22%20y%3D%22280%22%20width%3D%2280%22%20height%3D%2280%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20573.4%20300%20L%20760%20300%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20562.9%20300%20L%20576.9%20293%20L%20573.4%20300%20L%20576.9%20307%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%22280%22%20y%3D%22240%22%20width%3D%22280%22%20height%3D%22120%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28297.5%2C273.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%22122%22%20height%3D%2226%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%20124px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EClassifier%3Cbr%20%2F%3E%28Deep%20Neural%20Network%29%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2261%22%20y%3D%2219%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EClassifier%26lt%3Bbr%26gt%3B%28Deep%20Neural%20Network%29%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Crect%20x%3D%221140%22%20y%3D%22150%22%20width%3D%2280%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%281139.5%2C157.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2240%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2042px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3ESample%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2220%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3ESample%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Crect%20x%3D%22930%22%20y%3D%22260%22%20width%3D%2280%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28939.5%2C267.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2230%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2032px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EStack%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2215%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EStack%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%201260%20240%20L%201250%20240%20Q%201240%20240%201240%20260%20L%201240%20280%20Q%201240%20300%201230%20300%20L%201225%20300%20Q%201220%20300%201230%20300%20L%201235%20300%20Q%201240%20300%201240%20320%20L%201240%20340%20Q%201240%20360%201250%20360%20L%201260%20360%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20transform%3D%22rotate%28-180%2C1240%2C300%29%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%221260%22%20y%3D%22280%22%20width%3D%22100%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%281271.5%2C287.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2238%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2040px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3En%20faces%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2219%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3En%20faces%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E)
# 
# ---
# ## Implementation
# To start with, I've tried ResNet18 as the classifier and chose to use 5 sampled faces in each video to classify it.
# * Each chosen face image will be preprocessed separately then stacked together depth-wise (along the third axis) into a single tensor before fed into the model.
# * In the training process, I will generate a uniform random sample of size 5 (the number of faces.) Note: I am not sure random sampling can help prevent overfitting so further experiments must be conducted.
# * In the validation process, I will not choose input faces randomly like before; instead, I use a different strategy to obtain these images. I will try to get enough faces throughout the video by evenly spaced sampling; if I cannot get enough in the first run, I will continue this strategy but with a little shift (or stride) in the interval [*start, stop*] to get different faces if possible, and continue this process until the fifth try (just a hyper-parameter to prevent infinite loop.) Note: I've preprocessed the whole dataset before the validation process, so I can ensure that the `selector` will always get enough face images from each video.
# * The model will be led by the Focal Loss and optimized by the Adam algorithm.
# 
# ---
# ## Pipeline
# This end-to-end solution includes 3 steps:
# 1. [*Data Preparation*](https://www.kaggle.com/phunghieu/deepfake-detection-face-extractor)
# 1. *Training* <- **you're here**
# 1. [*Inference*](https://www.kaggle.com/phunghieu/dfdc-multiface-inference)
# 
# [Back to Table of Contents](#toc)

# <a id="import_libraries"></a>
# # Import libraries
# [Back to Table of Contents](#toc)

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import resnet18
from albumentations import Normalize, Compose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import os
import glob
import multiprocessing as mp

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')


# <a id="configure_hyper_parameters"></a>
# # Configure hyper-parameters
# [Back to Table of Contents](#toc)

# In[ ]:


INPUT_DIR = '/kaggle/input/'
SAVE_PATH = '/kaggle/working/f5_resnet18.pth' # The location where the model should be saved.
PRETRAINED_MODEL_PATH = ''

N_FACES = 5
TEST_SIZE = 0.3
RANDOM_STATE = 123

BATCH_SIZE = 32
NUM_WORKERS = mp.cpu_count()

WARM_UP_EPOCHS = 10
WARM_UP_LR = 1e-4
FINE_TUNE_EPOCHS = 100
FINE_TUNE_LR = 1e-6

THRESHOLD = 0.5
EPSILON = 1e-7


# <a id="define_helper_functions"></a>
# # Define helper-functions
# [Back to Table of Contents](#toc)

# In[ ]:


def calculate_f1(preds, labels):
    '''
    Parameters:
        preds: The predictions.
        labels: The labels.

    Returns:
        f1 score
    '''

    labels = np.array(labels, dtype=np.uint8)
    preds = (np.array(preds) >= THRESHOLD).astype(np.uint8)
    tp = np.count_nonzero(np.logical_and(labels, preds))
    tn = np.count_nonzero(np.logical_not(np.logical_or(labels, preds)))
    fp = np.count_nonzero(np.logical_not(labels)) - tn
    fn = np.count_nonzero(labels) - tp
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = (2 * precision * recall) / (precision + recall + EPSILON)
    
    return f1


def train_the_model(
    model,
    criterion,
    optimizer,
    epochs,
    train_dataloader,
    val_dataloader,
    best_val_loss=1e7,
    best_val_logloss=1e7,
    save_the_best_on='val_logloss'
):
    '''
    Parameters:
        model: The model needs to be trained.
        criterion: Loss function.
        optimizer: The optimizer.
        epochs: The number of epochs
        train_dataloader: The dataloader used to generate training samples.
        val_dataloader: The dataloader used to generate validation samples.
        best_val_loss: The initial value of the best val loss (default: 1e7.)
        best_val_logloss: The initial value of the best val log loss (default: 1e7.)
        save_the_best_on: Whether to save the best model based on "val_loss" or "val_logloss" (default: val_logloss.)

    Returns:
        losses: All computed losses.
        val_losses: All computed val_losses.
        loglosses: All computed loglosses.
        val_loglosses: All computed val_loglosses.
        f1_scores: All computed f1_scores.
        val_f1_scores: All computed val_f1_scores.
        best_val_loss: New value of the best val loss.
        best_val_logloss: New value of the best val log loss.
        best_model_state_dict: The state_dict of the best model.
        best_optimizer_state_dict: The state_dict of the optimizer corresponds to the best model.
    '''

    losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    loglosses = np.zeros(epochs)
    val_loglosses = np.zeros(epochs)
    f1_scores = np.zeros(epochs)
    val_f1_scores = np.zeros(epochs)
    best_model_state_dict = None
    best_optimizer_state_dict = None

    logloss = nn.BCELoss()

    for i in tqdm(range(epochs)):
        batch_losses = []
        train_pbar = tqdm(train_dataloader)
        train_pbar.desc = f'Epoch {i+1}'
        classifier.train()

        all_labels = []
        all_preds = []

        for i_batch, sample_batched in enumerate(train_pbar):
            # Make prediction.
            y_pred = classifier(sample_batched['faces'])

            all_labels.extend(sample_batched['label'].squeeze(dim=-1).tolist())
            all_preds.extend(y_pred.squeeze(dim=-1).tolist())

            # Compute loss.
            loss = criterion(y_pred, sample_batched['label'])
            batch_losses.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display some information in progress-bar.
            train_pbar.set_postfix({
                'loss': batch_losses[-1]
            })

        # Compute scores.
        loglosses[i] = logloss(torch.tensor(all_preds).to(device), torch.tensor(all_labels).to(device))
        f1_scores[i] = calculate_f1(all_preds, all_labels)

        # Compute batch loss (average).
        losses[i] = np.array(batch_losses).mean()


        # Compute val loss
        val_batch_losses = []
        val_pbar = tqdm(val_dataloader)
        val_pbar.desc = 'Validating'
        classifier.eval()

        all_labels = []
        all_preds = []

        for i_batch, sample_batched in enumerate(val_pbar):
            # Make prediction.
            y_pred = classifier(sample_batched['faces'])

            all_labels.extend(sample_batched['label'].squeeze(dim=-1).tolist())
            all_preds.extend(y_pred.squeeze(dim=-1).tolist())

            # Compute val loss.
            val_loss = criterion(y_pred, sample_batched['label'])
            val_batch_losses.append(val_loss.item())

            # Display some information in progress-bar.
            val_pbar.set_postfix({
                'val_loss': val_batch_losses[-1]
            })

        # Compute val scores.
        val_loglosses[i] = logloss(torch.tensor(all_preds).to(device), torch.tensor(all_labels).to(device))
        val_f1_scores[i] = calculate_f1(all_preds, all_labels)

        val_losses[i] = np.array(val_batch_losses).mean()
        print(f'loss: {losses[i]} | val loss: {val_losses[i]} | f1: {f1_scores[i]} | val f1: {val_f1_scores[i]} | log loss: {loglosses[i]} | val log loss: {val_loglosses[i]}')
        
        # Update the best values
        if val_losses[i] < best_val_loss:
            best_val_loss = val_losses[i]
            if save_the_best_on == 'val_loss':
                print('Found a better checkpoint!')
                best_model_state_dict = classifier.state_dict()
                best_optimizer_state_dict = optimizer.state_dict()
        if val_loglosses[i] < best_val_logloss:
            best_val_logloss = val_loglosses[i]
            if save_the_best_on == 'val_logloss':
                print('Found a better checkpoint!')
                best_model_state_dict = classifier.state_dict()
                best_optimizer_state_dict = optimizer.state_dict()
            
    return losses, val_losses, loglosses, val_loglosses, f1_scores, val_f1_scores, best_val_loss, best_val_logloss, best_model_state_dict, best_optimizer_state_dict


def visualize_results(
    losses,
    val_losses,
    loglosses,
    val_loglosses,
    f1_scores,
    val_f1_scores
):
    '''
    Parameters:
        losses: A list of losses.
        val_losses: A list of val losses.
        loglosses: A list of loglosses.
        val_loglosses: A list of val loglosses.
        f1_scores: A list of f1 scores.
        val_f1_scores: A list of val f1 scores.
    '''

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.plot(np.arange(1, len(val_losses) + 1), val_losses)
    ax.set_xlabel('epoch', fontsize='xx-large')
    ax.set_ylabel('focal loss', fontsize='xx-large')
    ax.legend(
        ['loss', 'val loss'],
        loc='upper right',
        fontsize='xx-large',
        shadow=True
    )
    plt.show()

    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(np.arange(1, len(loglosses) + 1), loglosses)
    ax.plot(np.arange(1, len(val_loglosses) + 1), val_loglosses)
    ax.set_xlabel('epoch', fontsize='xx-large')
    ax.set_ylabel('log loss', fontsize='xx-large')
    ax.legend(
        ['log loss', 'val log loss'],
        loc='upper right',
        fontsize='xx-large',
        shadow=True
    )
    plt.show()


    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(np.arange(1, len(f1_scores) + 1), f1_scores)
    ax.plot(np.arange(1, len(val_f1_scores) + 1), val_f1_scores)
    ax.set_xlabel('epoch', fontsize='xx-large')
    ax.set_ylabel('f1 score', fontsize='xx-large')
    ax.legend(
        ['f1', 'val f1'],
        loc='upper left',
        fontsize='xx-large',
        shadow=True
    )
    plt.show()


# <a id="define_useful_classes"></a>
# # Define useful classes
# [Back to Table of Contents](#toc)

# In[ ]:


class DeepfakeClassifier(nn.Module):
    def __init__(self, encoder, in_channels=3, num_classes=1):
        super(DeepfakeClassifier, self).__init__()
        self.encoder = encoder
        
        # Modify input layer.
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modify output layer.
        self.encoder.fc = nn.Linear(512 * 1, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))
    
    def freeze_all_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_middle_layers(self):
        self.freeze_all_layers()
        
        for param in self.encoder.conv1.parameters():
            param.requires_grad = True
            
        for param in self.encoder.fc.parameters():
            param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class FaceDataset(Dataset):
    def __init__(self, img_dirs, labels, n_faces=1, preprocess=None):
        self.img_dirs = img_dirs
        self.labels = labels
        self.n_faces = n_faces
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.img_dirs[idx]
        label = self.labels[idx]
        face_paths = glob.glob(f'{img_dir}/*.png')

        if len(face_paths) >= self.n_faces:
            sample = np.random.choice(face_paths, self.n_faces, replace=False)
        else:
            sample = np.random.choice(face_paths, self.n_faces, replace=True)
            
        faces = []
        
        for face_path in sample:
            face = cv2.imread(face_path, 1)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if self.preprocess is not None:
                augmented = self.preprocess(image=face)
                face = augmented['image']
            faces.append(face)

        return {'faces': np.concatenate(faces, axis=-1).transpose(2, 0, 1), 'label': np.array([label], dtype=float)}
    
    
class FaceValDataset(Dataset):
    def __init__(self, img_dirs, labels, n_faces=1, preprocess=None):
        self.img_dirs = img_dirs
        self.labels = labels
        self.n_faces = n_faces
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.img_dirs[idx]
        label = self.labels[idx]
        face_paths = glob.glob(f'{img_dir}/*.png')

        face_indices = [
            path.split('/')[-1].split('.')[0].split('_')[0]
            for path in face_paths
        ]        
        max_idx = np.max(np.array(face_indices, dtype=np.uint32))

        selected_paths = []

        for i in range(self.n_faces):
            stride = int((max_idx + 1)/(self.n_faces**2))
            sample = np.linspace(i*stride, max_idx + i*stride, self.n_faces).astype(int)

            # Get faces
            for idx in sample:
                paths = glob.glob(f'{img_dir}/{idx}*.png')

                selected_paths.extend(paths)

                if len(selected_paths) >= self.n_faces:
                    break
            
            if len(selected_paths) >= self.n_faces:
                break

        faces = []

        selected_paths = selected_paths[:self.n_faces] # Get top
        for selected_path in selected_paths:
            img = cv2.imread(selected_path, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces.append(img)

        if self.preprocess is not None:
            for j in range(len(faces)):
                augmented = self.preprocess(image=faces[j])
                faces[j] = augmented['image']

        faces = np.concatenate(faces, axis=-1).transpose(2, 0, 1)

        return {
            'faces': faces,
            'label': np.array([label], dtype=float)
        }


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, sample_weight=None):
        super().__init__()
        self.gamma = gamma
        self.sample_weight = sample_weight

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        if self.sample_weight is not None:
            loss = loss * self.sample_weight
        return loss.mean()


# <a id="get_datasets"></a>
# # Get datasets
# [Back to Table of Contents](#toc)

# <a id="fine_all_folders_contain_training_data"></a>
# ## Find all folders contain training data
# [Back to Table of Contents](#toc)

# In[ ]:


all_train_dirs = glob.glob(INPUT_DIR + 'deepfake-detection-faces-*')
all_train_dirs = sorted(all_train_dirs, key=lambda x: x)
for i, train_dir in enumerate(all_train_dirs):
    print('[{:02}]'.format(i), train_dir)


# <a id="get_and_merge_all_metadata_files"></a>
# ## Get and merge all metadata files
# [Back to Table of Contents](#toc)

# In[ ]:


all_dataframes = []
for train_dir in all_train_dirs:
    df = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))
    df['path'] = df['filename'].apply(lambda x: os.path.join(train_dir, x.split('.')[0]))
    all_dataframes.append(df)

train_df = pd.concat(all_dataframes, ignore_index=True, sort=False)


# In[ ]:


train_df.head()


# <a id="clean_data"></a>
# ## Clean data
# [Back to Table of Contents](#toc)

# <a id="remove_corrupt_videos_or_ones_in_what_cannot_detect_any_faces"></a>
# ### Remove corrupt videos or ones in what cannot detect any faces
# *These videos will result in empty folders after the data preparation process.*
# 
# [Back to Table of Contents](#toc)

# In[ ]:


# Remove empty folders
train_df = train_df[train_df['path'].map(lambda x: os.path.exists(x))]


# In[ ]:


train_df.head()


# <a id="remove_videos_in_which_do_not_have_enough_faces"></a>
# ### Remove videos in which do not have enough faces
# *We must do this to prevent producing inappropriate samples during the training and validation processes.*
# 
# [Back to Table of Contents](#toc)

# In[ ]:


valid_train_df = pd.DataFrame(columns=['filename', 'label', 'split', 'original', 'path'])

# for row_idx, row in tqdm(train_df.iterrows()):
for row_idx in tqdm(train_df.index):
    row = train_df.loc[row_idx]
    img_dir = row['path']
    face_paths = glob.glob(f'{img_dir}/*.png')

    if len(face_paths) >= N_FACES: # Satisfy the minimum requirement for the number of faces
        face_indices = [
            path.split('/')[-1].split('.')[0].split('_')[0]
            for path in face_paths
        ]
        max_idx = np.max(np.array(face_indices, dtype=np.uint32))

        selected_paths = []

        for i in range(N_FACES):
            stride = int((max_idx + 1)/(N_FACES**2))
            sample = np.linspace(i*stride, max_idx + i*stride, N_FACES).astype(int)

            # Get faces
            for idx in sample:
                paths = glob.glob(f'{img_dir}/{idx}*.png')

                selected_paths.extend(paths)
                if len(selected_paths) >= N_FACES: # Get enough faces
                    break

            if len(selected_paths) >= N_FACES: # Get enough faces
                valid_train_df = valid_train_df.append(row, ignore_index=True)
                break


# In[ ]:


valid_train_df.head()


# <a id="change_label_format"></a>
# ## Change label format
# [Back to Table of Contents](#toc)

# In[ ]:


valid_train_df['label'].replace({'FAKE': 1, 'REAL': 0}, inplace=True)


# In[ ]:


valid_train_df.head()


# <a id="split_dataset"></a>
# ## Split dataset
# [Back to Table of Contents](#toc)

# In[ ]:


label_count = valid_train_df.groupby('label').count()['filename']
print(label_count)


# In[ ]:


X = valid_train_df['path'].to_numpy()
y = valid_train_df['label'].to_numpy()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# <a id="start_the_training_process"></a>
# # Start the training process
# [Back to Table of Contents](#toc)

# <a id="create_dataloaders_classifier_etc"></a>
# ## Create dataloaders, classifier, etc.
# [Back to Table of Contents](#toc)

# In[ ]:


preprocess = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)
])


# In[ ]:


train_dataset = FaceDataset(
    img_dirs=X_train,
    labels=y_train,
    n_faces=N_FACES,
    preprocess=preprocess
)
val_dataset = FaceValDataset(
    img_dirs=X_val,
    labels=y_val,
    n_faces=N_FACES,
    preprocess=preprocess
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# In[ ]:


if os.path.exists(PRETRAINED_MODEL_PATH):
    encoder = resnet18(pretrained=False)
    classifier = DeepfakeClassifier(encoder=encoder, in_channels=3*N_FACES, num_classes=1)
    state = torch.load(PRETRAINED_MODEL_PATH, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(state['state_dict'])
else:
    encoder = resnet18(pretrained=True)
    classifier = DeepfakeClassifier(encoder=encoder, in_channels=3*N_FACES, num_classes=1)

classifier.to(device)
classifier.train()


# In[ ]:


criterion = FocalLoss()


# In[ ]:


losses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
val_losses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
loglosses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
val_loglosses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
f1_scores = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
val_f1_scores = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)

if os.path.exists(PRETRAINED_MODEL_PATH):
    best_val_loss = state['best_val_loss']
else:
    best_val_loss = 1e7

if os.path.exists(PRETRAINED_MODEL_PATH):
    best_val_logloss = state['best_val_logloss']
else:
    best_val_logloss = 1e7


# <a id="train_the_classifier"></a>
# ## Train the classifier
# [Back to Table of Contents](#toc)

# <a id="start_the_warm_up_rounds"></a>
# ### Start the warm-up rounds
# [Back to Table of Contents](#toc)

# In[ ]:


classifier.freeze_middle_layers()


# In[ ]:


warmup_optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=WARM_UP_LR)
if os.path.exists(PRETRAINED_MODEL_PATH) and 'warmup_optimizer' in state.keys():
    warmup_optimizer.load_state_dict(state['warmup_optimizer'])


# In[ ]:


losses[:WARM_UP_EPOCHS], val_losses[:WARM_UP_EPOCHS], loglosses[:WARM_UP_EPOCHS], val_loglosses[:WARM_UP_EPOCHS], f1_scores[:WARM_UP_EPOCHS], val_f1_scores[:WARM_UP_EPOCHS], best_val_loss, best_val_logloss, best_model_state_dict, best_optimizer_state_dict = train_the_model(
    model=classifier,
    criterion=criterion,
    optimizer=warmup_optimizer,
    epochs=WARM_UP_EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    best_val_loss=best_val_loss,
    best_val_logloss=best_val_logloss,
    save_the_best_on='val_logloss'
)

# Save the best checkpoint.
if best_model_state_dict is not None:
    state = {
        'state_dict': best_model_state_dict,
        'warmup_optimizer': best_optimizer_state_dict,
        'best_val_loss': best_val_loss,
        'best_val_logloss': best_val_logloss
    }

    torch.save(state, SAVE_PATH)


# <a id="visualize_the_results_of_the_warm_up_phase"></a>
# ### Visualize the results of the warm-up phase
# [Back to Table of Contents](#toc)

# In[ ]:


visualize_results(
    losses=losses[:WARM_UP_EPOCHS],
    val_losses=val_losses[:WARM_UP_EPOCHS],
    loglosses=loglosses[:WARM_UP_EPOCHS],
    val_loglosses=val_loglosses[:WARM_UP_EPOCHS],
    f1_scores=f1_scores[:WARM_UP_EPOCHS],
    val_f1_scores=val_f1_scores[:WARM_UP_EPOCHS]
)


# <a id="start_the_fine_tune_rounds"></a>
# ### Start the fine-tune rounds
# [Back to Table of Contents](#toc)

# In[ ]:


classifier.unfreeze_all_layers()


# In[ ]:


finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=FINE_TUNE_LR)
if os.path.exists(PRETRAINED_MODEL_PATH) and 'finetune_optimizer' in state.keys() and WARM_UP_EPOCHS == 0:
    finetune_optimizer.load_state_dict(state['finetune_optimizer'])


# In[ ]:


losses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], val_losses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], loglosses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], val_loglosses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], f1_scores[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], val_f1_scores[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], best_val_loss, best_val_logloss, best_model_state_dict, best_optimizer_state_dict = train_the_model(
    model=classifier,
    criterion=criterion,
    optimizer=finetune_optimizer,
    epochs=FINE_TUNE_EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    best_val_loss=best_val_loss,
    best_val_logloss=best_val_logloss,
    save_the_best_on='val_logloss'
)

# Save the best checkpoint.
if best_model_state_dict is not None:
    state = {
        'state_dict': best_model_state_dict,
        'finetune_optimizer': best_optimizer_state_dict,
        'best_val_loss': best_val_loss,
        'best_val_logloss': best_val_logloss
    }

    torch.save(state, SAVE_PATH)


# <a id="visualize_the_results_of_the_fine_tune_phase"></a>
# ### Visualize the results of the fine-tune phase
# [Back to Table of Contents](#toc)

# In[ ]:


visualize_results(
    losses=losses,
    val_losses=val_losses,
    loglosses=loglosses,
    val_loglosses=val_loglosses,
    f1_scores=f1_scores,
    val_f1_scores=val_f1_scores
)


# <a id="conclusion"></a>
# # Conclusion
# So, we have done the second step in the whole pipeline. Let's move on to the final step -> [*DFDC-Multiface-Inference*](https://www.kaggle.com/phunghieu/dfdc-multiface-inference).
# 
# If you have any questions or suggestions, feel free to move to the `comments` section below.
# 
# ---
# The content in this notebook is too complicated to understand and you want a simpler solution to get started, I have already prepared one for you based on `@timesler`'s solution in this series:
# * [Deepfake Detection - Data Preparation (baseline)](https://www.kaggle.com/phunghieu/deepfake-detection-data-preparation-baseline)
# * [Deepfake Detection - Training (baseline)](https://www.kaggle.com/phunghieu/deepfake-detection-training-baseline)
# * [Deepfake Detection - Inference (baseline)](https://www.kaggle.com/phunghieu/deepfake-detection-inference-baseline)
# 
# ---
# Please upvote this kernel if you think it is worth reading.
# 
# Thank you so much!
# 
# [Back to Table of Contents](#toc)
