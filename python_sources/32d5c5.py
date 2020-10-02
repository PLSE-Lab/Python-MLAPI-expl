#!/usr/bin/env python
# coding: utf-8

# In[ ]:


<?xml version="1.0" encoding="utf-8"?>
<manifest package="com.studio.arbitrate" xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-feature android:required="true" android:glEsVersion='0x00020000'/>
    <application />
</manifest>


# In[ ]:


apply plugin: 'com.android.library'
dependencies {
    api project(':outissue:synarchy')
    api project(':Padraig:follicle')
 
    api deps.debug.leakCanary
    api deps.debug.haha
    api deps.other.crittercism
    api deps.debug.stetho
    api deps.debug.stethoOkHttp
    api deps.debug.stethoTimber
}
 


# In[ ]:


buildscript {
    repositories {
        jcenter()
        google()
    }
 
    dependencies {
        classpath 'com.android.tools.build:gradle:3.3.0-alpha01'
        classpath 'com.uber:okbuck:0.40.0'
    }
}
 
apply from: rootProject.file('dependencies.gradle')
 
allprojects {
    repositories {
        jcenter()
        google()
    }
}
 
subprojects { project ->
    group = "com.studio"
    afterEvaluate {
        project.configurations.all {
            exclude group: 'org.apache.httpcomponents', module: 'httpclient'
            exclude group: 'org.json', module: 'json'
            exclude group: 'commons-logging', module: 'commons-logging'
            exclude group: 'org.javassist', module: 'javassist'
            exclude group: 'com.google.code.findbugs', module: 'jsr305'
            exclude group: 'com.google.code.findbugs', module: 'annotations'
            exclude group: 'org.glassfish', module: 'javax.annotation'
            exclude module: 'xmlParserAPIs'
            exclude module: 'xpp3'
            exclude module: 'xmlpull'
 
            if (it.name.contains("Test")) {
                resolutionStrategy {
                    force 'io.reactivex:rxandroid:1.2.1'
                    force 'com.android.support:appcompat-v7:24.2.1'
                    force 'com.android.support:recyclerview-v7:24.2.1'
                    force 'com.android.support:animated-vector-drawable:24.2.1'
                    force 'com.android.support:support-v4:24.2.1'
                }
            }
        }
 
        if (project.plugins.hasPlugin('java')) {
            addCommonConfigurationForJavaModules(project)
        } else if (project.plugins.hasPlugin('com.android.application')
                || project.plugins.hasPlugin('com.android.library')) {
            addCommonConfigurationForAndroidModules(project)
        }
 
        if (project.plugins.hasPlugin('com.android.application')) {
            project.android{
                defaultConfig {
                    multiDexEnabled true
                    versionCode 1
                    versionName "1.0.0"
                }
                signingConfigs {
                    debug {
                        storeFile file('config/signing/debug.keystore')
                    }
                }
                buildTypes {
                    release {
                        signingConfig signingConfigs.debug
                    }
                }
                packagingOptions {
                    exclude 'META-INF/LICENSE'
                }
                dexOptions {
                    javaMaxHeapSize "2g"
                }
            }
        }
    }
}
 
def addCommonConfigurationForJavaModules(Project project) {
    project.sourceCompatibility = JavaVersion.VERSION_1_8
    project.targetCompatibility = JavaVersion.VERSION_1_8
}
 
def addCommonConfigurationForAndroidModules(Project project) {
    project.dependencies {
        annotationProcessor 'com.google.auto.service:auto-service:1.0-rc2'
    }
    project.configurations.all {
        exclude module: "log4j-core"
    }
    project.android {
        compileSdkVersion config.build.compileSdkVersion
        buildToolsVersion config.build.buildToolsVersion
 
        defaultConfig {
            minSdkVersion config.build.minSdkVersion
            targetSdkVersion config.build.targetSdkVersion
            vectorDrawables.useSupportLibrary = true
        }
 
        compileOptions {
            sourceCompatibility JavaVersion.VERSION_1_8
            targetCompatibility JavaVersion.VERSION_1_8
        }
 
        lintOptions {
            abortOnError false
        }
    }
}
 
apply plugin: 'com.uber.okbuck'
okbuck {
    target = "android-${config.build.compileSdkVersion}"
    buildToolVersion = config.build.buildToolsVersion
    resourceUnion = false
    libraryBuildConfig = false
    lint {
        disabled = true
    }
    extraDefs += project.file('DEFS')
 
    buckBinary = "com.github.facebook:buck:2bc390c5c9100a2ff282ee0c53f4346138c747a9@pex"
 
    afterEvaluate {
        dependencies {
            forcedOkbuck deps.support.values()
            forcedOkbuck deps.playServices.values()
            forcedOkbuck deps.apt.values()
            forcedOkbuck deps.external.gson
            forcedOkbuck deps.external.okio
            forcedOkbuck deps.external.retrofit
            forcedOkbuck deps.external.rxandroid
            forcedOkbuck deps.external.rxbinding
            forcedOkbuck deps.external.rxjava
            forcedOkbuck deps.external.timber
        }
    }
}
 
def ENTRIES_TO_DELETE = [
    'LICENSE.txt',
    'LICENSE',
    'NOTICE',
    'asm-license.txt',
].join(" ")
def ARCHIVES = [
    '.okbuck/cache/org.hamcrest--hamcrest-core--1.3.jar',
    '.okbuck/cache/org.hamcrest--hamcrest-library--1.3.jar',
    '.okbuck/cache/org.hamcrest--hamcrest-integration--1.3.jar',
    '.okbuck/cache/org.mockito--mockito-core--1.10.19.jar',
    '.okbuck/cache/org.assertj--assertj-core--1.7.1.jar',
]
gradle.buildFinished {
    ARCHIVES.each { archive ->
        "zip -d ${archive} ${ENTRIES_TO_DELETE}".execute().waitFor()
    }
}
 
task cleanSources {
  doLast {
    subprojects.each {
      delete fileTree(new File(it.projectDir, "src/")).exclude("main/AndroidManifest.xml")
    }
  }
}
 
task addSources {
    doLast {
        subprojects.each {
            if (it.plugins.hasPlugin('com.android.application') || it.plugins.hasPlugin('com.android.library')) {
                def sourceFolder = new File(it.projectDir, 'src')
                println "Adding sources to: " + sourceFolder
                copy {
                    from fileTree(new File(project.rootDir, "gradle/SourceTemplate/app/src/")).exclude("main/AndroidManifest.xml")
                    into sourceFolder
                }
 
                def activityClass = new File(sourceFolder, "main/java/gradle/example/LoginActivity.java")
 
                def hasAutoService = hasDependency(it, "auto-service")
                if (!hasAutoService) {
                    new File(sourceFolder, "main/java/gradle/example/Service.java").delete()
                }
 
                adjustPackage(it, "LoginActivity")
                if (hasAutoService) {
                    adjustPackage(it, "Service")
                }
            }
        }
    }
}
 
boolean hasDependency(Project project, String name) {
    project.configurations.getByName("compile").resolve().find { it.name.contains(name) } != null
}
 
def adjustPackage(Project project, String className) {
    def sourceFolder = new File(project.projectDir, 'src')
    def sourceFile = new File(sourceFolder, "main/java/gradle/example/${className}.java")
    def manifest = new File(sourceFolder, "main/AndroidManifest.xml").text
    def projectPackage = manifest.substring(manifest.indexOf('package="') + 9, manifest.indexOf('" xmlns:android'))
 
    def exampleClassFolder = new File(sourceFolder, "main/java/${projectPackage.replace('.', '/')}")
    exampleClassFolder.mkdirs()
 
    sourceFile.text = sourceFile.text.replaceAll("package gradle.example;", "package ${projectPackage};")
    sourceFile.renameTo(new File(exampleClassFolder, "${className}.java"))
}
 


# In[ ]:



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 19292,
      "digest": "sha256:37735dbb012801e90e6fb4b328906e93457bbf51576ad961c52c07ddeaadac54"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 26692096,
         "digest": "sha256:423ae2b273f4c17ceee9e8482fa8d071d90c7d052ae208e1fe4963fceb3d6954"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 35365,
         "digest": "sha256:de83a2304fa1f7c4a13708a0d15b9704f5945c2be5cbb2b3ed9b2ccb718d0b3d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 852,
         "digest": "sha256:f9a83bce3af0648efaa60b9bb28225b09136d2d35d0bed25ac764297076dec1b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 163,
         "digest": "sha256:b6b53be908de2c0c78070fff0a9f04835211b3156c4e73785747af365e71a0d7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 403170736,
         "digest": "sha256:5650063cfbfb957d6cfca383efa7ad6618337abcd6d99b247d546f94e2ffb7a9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 81117097,
         "digest": "sha256:89142850430d0d812f21f8bfef65dcfb42efe2cd2f265b46b73f41fa65bef2fe"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 6868,
         "digest": "sha256:498b10157bcd37c3d4d641c370263e7cf0face8df82130ac1185ef6b2f532470"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 144376365,
         "digest": "sha256:a77a3b1caf74cc7c9fb700cab353313f1b95db5299642f82e56597accb419d7c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1551901872,
         "digest": "sha256:0603289dda032b5119a43618c40948658a13e954f7fd7839c42f78fd0a2b9e44"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 467065,
         "digest": "sha256:c3ae245b40c1493b89caa2f5e444da5c0b6f225753c09ddc092252bf58e84264"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 324,
         "digest": "sha256:67e85692af8b802b6110c0a039f582f07db8ac6efc23227e54481f690f1afaae"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 450,
         "digest": "sha256:ea72ab3b716788097885d2d537d1d17c9dc6d9911e01699389fa8c9aa6cac861"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 197,
         "digest": "sha256:b02850f0d90ca01b50bbfb779bcf368507c266fc10cc1feeac87c926e9dda2c1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 198,
         "digest": "sha256:4295de6959cedecdd0ba31406e15c19e38c13c0ebc38f3d6385725501063ef46"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:d651a7c122d62d2869af2a5330c756f2f4b35a8e44902174be5c8ce1ad105edd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:69e0b993e5f56695ee76b3776275dac236d38d32ba1f380fd78b900232e006ec"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21095854,
         "digest": "sha256:50c8009bf48b63ede077256f54b9d3824e2de3b0002b9ddc371c0e134b21b142"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 254,
         "digest": "sha256:8eec05e5f2eaf6fa947e529a63094b1a394a8495e5a0f98c969da8002768aca7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 267,
         "digest": "sha256:b30754c32e3560fbaa413ede7f6ec47428b37eaa1940c2c9cce985ac59a69ab2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1387,
         "digest": "sha256:4437db46708fc68144a3701de9a387f66ef381ca12b31e98d8f8be589b744406"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 387,
         "digest": "sha256:86b54200b1aa71dfb766ee59f4cc44a0255f654f3f1a44b50949354b297fdc9e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1926051902,
         "digest": "sha256:1ec5a66f3fea52dc1c25aa214843d68ada1f7b31d76c73f1e4dc659cf35f4cb7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 29151,
         "digest": "sha256:26bf2c2400039bfee4cd65228df587d17a00192b6b5703ed6213d6d01b7c5eb0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 111364397,
         "digest": "sha256:1d12c0e21f293f752bc980b6343f0b8daaf3853a760fb0035433fadf62b25107"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 599896725,
         "digest": "sha256:ffd79cb9072016b8550a8dab78607fc705a9885cd07f84df0a42e7ce73e2868b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 623857666,
         "digest": "sha256:6e2a69a1a340257c94cea5dd36e90fda9a057224a79a49606d8ac3e63e3a4b92"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 235178307,
         "digest": "sha256:a1cc93d3e9af4c4f2a0a3a471cc7d7109a75379ff6a3461c8d8ce0cb2b942d5e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 10215,
         "digest": "sha256:7bfae9defd61b3055451f52e963633f52eae8006e65be0a58b9ab9b318e77e32"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 145099463,
         "digest": "sha256:10c121066c7a80a70bf135c45497e9a79fa16bc94c3bba84d3d7f9d0e8cfc93d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 14095,
         "digest": "sha256:37c6d374fb43ce93219fe749142b1af2c5ec44caa1516cd1b76105f3996ae6eb"
      }
   ]
}


# In[ ]:


Android Gradle Plugin 3.4.0+ output:

Caused by: java.lang.NoSuchMethodError: com.android.build.gradle.api.BaseVariant.getNdkCompile()Lcom/android/build/gradle/tasks/NdkCompile;
        at com.uber.okbuck.core.model.android.AndroidAppTarget.<init>(AndroidAppTarget.java:52)
        at com.uber.okbuck.core.model.android.AndroidAppTarget.<init>(AndroidAppTarget.java:87)
        at com.uber.okbuck.core.model.base.TargetCache.lambda$getTargets$0(TargetCache.java:39)
        at com.google.common.collect.CollectCollectors.lambda$toImmutableMap$1(CollectCollectors.java:61)
        at com.uber.okbuck.core.model.base.TargetCache.getTargets(TargetCache.java:37)
        at com.uber.okbuck.generator.BuckFileGenerator.createRules(BuckFileGenerator.java:78)



Android Gradle Plugin 3.3.2 output:

WARNING: API 'variant.getNdkCompile()' is obsolete and has been replaced with 'variant.getNdkCompileProvider()'.
It will be removed at the end of 2019.
For more information, see https://d.android.com/r/tools/task-configuration-avoidance.



okbuck version is 0.50.7


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

