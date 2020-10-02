#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Introduction](#introduction)
# 1. [Configure hyper-parameters](#configure_hyper_parameters)
# 1. [Install dependencies](#install_dependencies)
# 1. [Import libraries](#import_libraries)
# 1. [Define useful classes](#define_useful_classes)
# 1. [Define helper-functions](#define_helper_functions)
# 1. [Start inference process](#start_inference_process)
# 1. [Create submission.csv](#create_submission_csv)
# 1. [Conclusion](#conclusion)

# <a id="introduction"></a>
# # Introduction
# Let's use our trained Logistic Regression model in the previous step to infer all test videos, and submit the result to see how our simple model perform :)
# 
# ---
# ## Baseline's diagram
# ![diagram](data:image/svg+xml,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%3F%3E%0A%3C%21DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%0A%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%20version%3D%221.1%22%20width%3D%221288%22%20height%3D%22478%22%20viewBox%3D%22-0.5%20-0.5%201288%20478%22%20content%3D%22%26lt%3Bmxfile%20host%3D%26quot%3BElectron%26quot%3B%20modified%3D%26quot%3B2020-02-16T02%3A35%3A34.546Z%26quot%3B%20agent%3D%26quot%3BMozilla%2F5.0%20%28X11%3B%20Linux%20x86_64%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20draw.io%2F12.2.2%20Chrome%2F78.0.3904.94%20Electron%2F7.1.0%20Safari%2F537.36%26quot%3B%20etag%3D%26quot%3By65MKnhNS-8JGmsLDxx9%26quot%3B%20version%3D%26quot%3B12.2.2%26quot%3B%20type%3D%26quot%3Bdevice%26quot%3B%20pages%3D%26quot%3B1%26quot%3B%26gt%3B%26lt%3Bdiagram%20id%3D%26quot%3By-GUQvWtg-YzmS7vu5I_%26quot%3B%20name%3D%26quot%3BPage-1%26quot%3B%26gt%3B7ZrZctowFIafhpn2ohnvkMsESJop6XRIpw2Xqn1iK5EtRhZbn74yyBhbLG6Ll2m5wjqSLOmc79dmOmY%2FXN4zNA0eqQekY2jesmMOOoahW1pP%2FCSW1cbSdYyNwWfYk4UywxP%2BCdKoSesMexDnCnJKCcfTvNGlUQQuz9kQY3SRL%2FZCSb7VKfJBMTy5iKjW79jjwcbas7XM%2FhGwH6Qt65rMCVFaWBriAHl0sWMyhx2zzyjlm6dw2QeSOC%2F1y6be3YHcbccYRLxMBfT2%2FOrgB23%2Bspg4o4c5ouPJh2vZN75KBwyeGL9MUsYD6tMIkWFmvWV0FnmQvFUTqazMiNKpMOrC%2BAqcr2Qw0YxTYQp4SGQuLDF%2F3nmeJK%2B6smVqsJRvXidWaSLibPW8m9iplSSzautUWk%2F1knRcTGfMhSOuSWlDzAd%2BpJwEPPHbTgMyBvdAQxD9EQUYEMTxPM8Vknj623JZBMWDDOJvBFS%2Bd47ITLZ0h8QwDW0AXOiDso7hEDGc2x%2FJk588vXv82v%2F8%2Bb1CQj7OiwBzeJqitc8WQuz5mB708xwYh%2BVRz8hcM1WKnCr0rkwvMuHpaZlgR3SOVpUzG5VHJonJTs5%2Befw55kZJzHW9VZwbKueA%2BIwlqA%2BXnKEDrD9ELkw5ptEY4gh489TbvRLUW3VS71zWhEOusUqKxW6VVixFK9%2FEporuUcdVOLVUSYgmxHYLapFDUQ09VQ29PWKwqhJD9yKGE5CfXjlaJQZbXTgYCsURo0nqDa1l2KfHrwv3Bzf8J7k3WsV9T%2BVeAN0s9la3ZdgbKvaVQH18M58hnVE8yUF89rNu2cOu3i6otwOsIF5a6XjV4XerVX5P%2B733%2BDVf3zM0O7PY1%2FmZxXC6V7Yyt%2By9TahucjHqgbX0iqnXdpVW%2BpKhZZirtwz9xCdU1GqSbsfK021aKtv1rptmk9vFlsNf9tJAd9oFv3ptMMAxR1Hju0bHPkn%2Fda30W%2F86%2Fe6Mzdd9PRrC01Jwykqhe24pyKpfKBadzs7dhU8Opl5gZNNTWauAybYbf0GOfSHnzOS07HOco0yiI%2BqLaRS7wjoGn0EcJx8qjm4Ma7iAKu4ozJK75eq%2BvXUVz42HNyPFU2KAPO%2BOWGzQ3qBPifCrOYholIjmBRNSMCGC%2FUgkXeE2EPbbxF3YReRGZoTY88ihNS0foTOEoHDzbRpqBPZ9BjIqC8CeG6ObT8P%2FJwC9ygIgktn%2FPjZrSfbvGXP4Cw%3D%3D%26lt%3B%2Fdiagram%26gt%3B%26lt%3B%2Fmxfile%26gt%3B%22%20style%3D%22background-color%3A%20rgb%28255%2C%20255%2C%20255%29%3B%22%3E%3Cdefs%3E%3Cfilter%20id%3D%22dropShadow%22%3E%3CfeGaussianBlur%20in%3D%22SourceAlpha%22%20stdDeviation%3D%221.7%22%20result%3D%22blur%22%2F%3E%3CfeOffset%20in%3D%22blur%22%20dx%3D%223%22%20dy%3D%223%22%20result%3D%22offsetBlur%22%2F%3E%3CfeFlood%20flood-color%3D%22%233D4574%22%20flood-opacity%3D%220.4%22%20result%3D%22offsetColor%22%2F%3E%3CfeComposite%20in%3D%22offsetColor%22%20in2%3D%22offsetBlur%22%20operator%3D%22in%22%20result%3D%22offsetBlur%22%2F%3E%3CfeBlend%20in%3D%22SourceGraphic%22%20in2%3D%22offsetBlur%22%2F%3E%3C%2Ffilter%3E%3C%2Fdefs%3E%3Cg%20filter%3D%22url%28%23dropShadow%29%22%3E%3Cpath%20d%3D%22M%20720%2060%20L%20767.26%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20777.76%2060%20L%20763.76%2067%20L%20767.26%2060%20L%20763.76%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%22480%22%20y%3D%220%22%20width%3D%22240%22%20height%3D%22120%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28523.5%2C33.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2276%22%20height%3D%2226%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2076px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFace%20Detector%3Cbr%20%2F%3E%28MTCNN%29%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2238%22%20y%3D%2219%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFace%20Detector%26lt%3Bbr%26gt%3B%28MTCNN%29%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%201140%20120%20L%201140%20160%20L%201140%20156%20L%201140%20182.26%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%201140%20192.76%20L%201133%20178.76%20L%201140%20182.26%20L%201147%20178.76%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%221000%22%20y%3D%220%22%20width%3D%22280%22%20height%3D%22120%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%281045.5%2C33.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2294%22%20height%3D%2226%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2096px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFeature%20Extractor%3Cbr%20%2F%3E%28InceptionResnet%29%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2247%22%20y%3D%2219%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFeature%20Extractor%26lt%3Bbr%26gt%3B%28InceptionResnet%29%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20160%2060%20L%20227.26%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20237.76%2060%20L%20223.76%2067%20L%20227.26%2060%20L%20223.76%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%2280%22%20cy%3D%2260%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%2845.5%2C33.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2234%22%20height%3D%2226%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2036px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EVideo%3Cbr%20%2F%3E%28.mp4%29%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2217%22%20y%3D%2219%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3E%5BNot%20supported%20by%20viewer%5D%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20400%2060%20L%20467.26%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20477.76%2060%20L%20463.76%2067%20L%20467.26%2060%20L%20463.76%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%22320%22%20cy%3D%2260%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28279.5%2C47.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2240%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2042px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFrames%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2220%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFrames%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20940%2060%20L%20987.26%2060%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20997.76%2060%20L%20983.76%2067%20L%20987.26%2060%20L%20983.76%2053%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%22860%22%20cy%3D%2260%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28827.5%2C47.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2232%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2034px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFaces%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2216%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFaces%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%201224.85%20263.28%20L%201203.51%20337.76%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%201200.62%20347.85%20L%201197.74%20332.46%20L%201203.51%20337.76%20L%201211.2%20336.32%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%201055.15%20263.28%20L%20921.51%20350.05%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20912.7%20355.77%20L%20920.63%20342.28%20L%20921.51%20350.05%20L%20928.26%20354.02%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%221140%22%20cy%3D%22235%22%20rx%3D%22120%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%281055.5%2C222.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2284%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2084px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFeature%20vectors%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2242%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFeature%20vectors%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%201120%20390%20L%20962.74%20390%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20952.24%20390%20L%20966.24%20383%20L%20962.74%20390%20L%20966.24%20397%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%221200%22%20cy%3D%22390%22%20rx%3D%2280%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%281153.5%2C377.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2246%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2046px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3ECentroid%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2223%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3ECentroid%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20770%20390%20L%20572.74%20390%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%20562.24%20390%20L%20576.24%20383%20L%20572.74%20390%20L%20576.24%20397%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cellipse%20cx%3D%22860%22%20cy%3D%22390%22%20rx%3D%2290%22%20ry%3D%2240%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28807.5%2C377.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2252%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2054px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EDistances%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2226%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EDistances%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Cpath%20d%3D%22M%20320%20390%20Q%20200%20390%20200%20360%20Q%20200%20330%2092.74%20330%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%2082.24%20330%20L%2096.24%20323%20L%2092.74%20330%20L%2096.24%20337%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Cpath%20d%3D%22M%20320%20390%20Q%20200%20390%20200%20420%20Q%20200%20450%2092.74%20450%22%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22stroke%22%2F%3E%3Cpath%20d%3D%22M%2082.24%20450%20L%2096.24%20443%20L%2092.74%20450%20L%2096.24%20457%20Z%22%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20stroke-miterlimit%3D%2210%22%20pointer-events%3D%22all%22%2F%3E%3Crect%20x%3D%22320%22%20y%3D%22330%22%20width%3D%22240%22%20height%3D%22120%22%20fill%3D%22%23ffffff%22%20stroke%3D%22%23000000%22%20stroke-width%3D%222%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%28339.5%2C377.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%22100%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%20100px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3ELogistic%20Regressor%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2250%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3ELogistic%20Regressor%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Crect%20x%3D%220%22%20y%3D%22310%22%20width%3D%2280%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%287.5%2C317.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2232%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2032px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EREAL%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2216%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EREAL%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3Crect%20x%3D%220%22%20y%3D%22430%22%20width%3D%2280%22%20height%3D%2240%22%20fill%3D%22none%22%20stroke%3D%22none%22%20pointer-events%3D%22all%22%2F%3E%3Cg%20transform%3D%22translate%289.5%2C437.5%29scale%282%29%22%3E%3Cswitch%3E%3CforeignObject%20style%3D%22overflow%3Avisible%3B%22%20pointer-events%3D%22all%22%20width%3D%2230%22%20height%3D%2212%22%20requiredFeatures%3D%22http%3A%2F%2Fwww.w3.org%2FTR%2FSVG11%2Ffeature%23Extensibility%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20font-size%3A%2012px%3B%20font-family%3A%20Helvetica%3B%20color%3A%20rgb%280%2C%200%2C%200%29%3B%20line-height%3A%201.2%3B%20vertical-align%3A%20top%3B%20width%3A%2032px%3B%20white-space%3A%20nowrap%3B%20overflow-wrap%3A%20normal%3B%20text-align%3A%20center%3B%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3Ainline-block%3Btext-align%3Ainherit%3Btext-decoration%3Ainherit%3Bwhite-space%3Anormal%3B%22%3EFAKE%3C%2Fdiv%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3Ctext%20x%3D%2215%22%20y%3D%2212%22%20fill%3D%22%23000000%22%20text-anchor%3D%22middle%22%20font-size%3D%2212px%22%20font-family%3D%22Helvetica%22%3EFAKE%3C%2Ftext%3E%3C%2Fswitch%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E)
# 
# ---
# ## Pipeline
# This end-to-end solution includes 3 steps:
# 1. [*Data Preparation* ](https://www.kaggle.com/phunghieu/deepfake-detection-data-preparation-baseline)
# 1. [*Training*](https://www.kaggle.com/phunghieu/deepfake-detection-training-baseline)
# 1. *Inference* <- **you're here**
# 
# ---
# [Back to Table of Contents](#toc)

# <a id="configure_hyper_parameters"></a>
# # Configure hyper-parameters
# [Back to Table of Contents](#toc)

# In[ ]:


TEST_DIR = '/kaggle/input/deepfake-detection-challenge/test_videos/'
MODEL_PATH = '/kaggle/input/deepfake-detection-logistic-regression/model.pth'

BATCH_SIZE = 60
SCALE = 0.25
N_FRAMES = None # 'None' means using all available frames
DEFAULT_PROB = 0.5


# <a id="install_dependencies"></a>
# # Install dependencies
# [Back to Table of Contents](#toc)

# In[ ]:


# Install facenet-pytorch
get_ipython().system('pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl')

from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
torch_home = get_torch_home()

# Copy model checkpoints to torch cache so they are loaded automatically by the package
get_ipython().system('mkdir -p $torch_home/checkpoints/')
get_ipython().system('cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt')
get_ipython().system('cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt')


# <a id="import_libraries"></a>
# # Import libraries
# [Back to Table of Contents](#toc)

# In[ ]:


import os
import glob
import json
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')


# <a id="define_useful_classes"></a>
# # Define useful classes
# [Back to Table of Contents](#toc)

# In[ ]:


# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()

        return faces
    

class LogisticRegression(nn.Module):
    def __init__(self, D_in=1, D_out=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(D_in, D_out)
        
    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = torch.sigmoid(y_pred)
        
        return y_pred


# <a id="define_helper_functions"></a>
# # Define helper-functions
# [Back to Table of Contents](#toc)

# In[ ]:


# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
def process_faces(faces, feature_extractor):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    if len(faces) == 0:
        return None
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = feature_extractor(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    centroid = embeddings.mean(dim=0)
    x = (embeddings - centroid).norm(dim=1).cpu().numpy()
    
    return x


# <a id="start_inference_process"></a>
# # Start inference process
# [Back to Table of Contents](#toc)

# In[ ]:


# Load model.
classifier = LogisticRegression()
classifier.load_state_dict(torch.load(MODEL_PATH))
classifier.eval()


# In[ ]:


# Get all test videos.
all_test_videos = glob.glob(os.path.join(TEST_DIR, '*.mp4'))


# In[ ]:


# Load face detector.
face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

# Load facial recognition model.
feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# Define face detection pipeline.
detection_pipeline = DetectionPipeline(detector=face_detector, n_frames=N_FRAMES, batch_size=BATCH_SIZE, resize=SCALE)


# In[ ]:


X_test = []

with torch.no_grad():
    for path in tqdm(all_test_videos):
        try:
            # Detect all faces occur in the video.
            faces = detection_pipeline(path)

            # Calculate the distances of all faces' feature vectors to the centroid.
            distances = process_faces(faces, feature_extractor)
            X_test.append(distances)
        except:
            X_test.append(None)


# In[ ]:


submission = []

with torch.no_grad():
    for path, distances in zip(all_test_videos, X_test):
        file_name = os.path.basename(path)

        if distances is not None:
            distances = torch.tensor(distances).unsqueeze(dim=1).float().to(device)
            y_pred = classifier(distances)
            y_pred = float(y_pred.mean().cpu().numpy())
        else:
            y_pred = DEFAULT_PROB

        submission.append([file_name, y_pred])


# <a id="create_submission_csv"></a>
# # Create submission.csv
# [Back to Table of Contents](#toc)

# In[ ]:


submission = pd.DataFrame(submission, columns=['filename', 'label'])
submission.sort_values('filename').to_csv('submission.csv', index=False)

plt.hist(submission.label, 20)
plt.show()


# <a id="conclusion"></a>
# # Conclusion
# Finally, we made it! Let's submit the result to see whether we can get a better position in the Public Leaderboard =]]
# 
# If you have any questions or suggestions, feel free to move to the `comments` section below.
# 
# Please upvote this kernel if you think it is worth reading; and remember to upvote `@timesler`'s [*kernel*](https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch), too. Thank you so much!
# 
# ---
# [Back to Table of Contents](#toc)
