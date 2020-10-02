#!/usr/bin/env python
# coding: utf-8

# In[ ]:


---
title: List role assignments using Azure RBAC and the Azure portal
description: Learn how to determine what resources users, groups, service principals, or managed identities have access to using Azure role-based access control (RBAC) and the Azure portal.
services: active-directory
documentationcenter: ''
author: rolyon
manager: mtillman

ms.assetid: 8078f366-a2c4-4fbb-a44b-fc39fd89df81
ms.service: role-based-access-control
ms.devlang: na
ms.topic: conceptual
ms.tgt_pltfrm: na
ms.workload: identity
ms.date: 03/18/2020
ms.author: rolyon
ms.reviewer: bagovind
---

# List role assignments using Azure RBAC and the Azure portal

[!INCLUDE [Azure RBAC definition list access](../../includes/role-based-access-control-definition-list.md)] This article describes how to list role assignments using the Azure portal.

> [!NOTE]
> If your organization has outsourced management functions to a service provider who uses [Azure delegated resource management](../lighthouse/concepts/azure-delegated-resource-management.md), role assignments authorized by that service provider won't be shown here.

## List role assignments for a user or group

The easiest way to see the roles assigned to a user or group in a subscription is to use the **Azure resources** pane.

1. In the Azure portal, click **All services** and then select **Users** or **Groups**.

1. Click the user or group you want list the role assignments for.

1. Click **Azure resources**.

    You see a list of roles assigned to the selected user or group at various scopes such as management group, subscription, resource group, or resource. This list includes all role assignments you have permission to read.

    get_ipython().system('[Role assignments for a user](./media/role-assignments-list-portal/azure-resources-user.png)    ')

1. To change the subscription, click the **Subscriptions** list.

## List owners of a subscription

Users that have been assigned the [Owner](built-in-roles.md#owner) role for a subscription can manage everything in the subscription. Follow these steps to list the owners of a subscription.

1. In the Azure portal, click **All services** and then **Subscriptions**.

1. Click the subscription you want to list the owners of.

1. Click **Access control (IAM)**.

1. Click the **Role assignments** tab to view all the role assignments for this subscription.

1. Scroll to the **Owners** section to see all the users that have been assigned the Owner role for this subscription.

   ![Subscription Access control - Role assignments tab](./media/role-assignments-list-portal/access-control-role-assignments-subscription.png)

## List role assignments at a scope

1. In the Azure portal, click **All services** and then select the scope. For example, you can select **Management groups**, **Subscriptions**, **Resource groups**, or a resource.

1. Click the specific resource.

1. Click **Access control (IAM)**.

1. Click the **Role assignments** tab to view all the role assignments at this scope.

   get_ipython().system('[Access control - Role assignments tab](./media/role-assignments-list-portal/access-control-role-assignments.png)')

   On the Role assignments tab, you can see who has access at this scope. Notice that some roles are scoped to **This resource** while others are **(Inherited)** from another scope. Access is either assigned specifically to this resource or inherited from an assignment to the parent scope.

## List role assignments for a user at a scope

To list access for a user, group, service principal, or managed identity, you list their role assignments. Follow these steps to list the role assignments for a single user, group, service principal, or managed identity at a particular scope.

1. In the Azure portal, click **All services** and then select the scope. For example, you can select **Management groups**, **Subscriptions**, **Resource groups**, or a resource.

1. Click the specific resource.

1. Click **Access control (IAM)**.

1. Click the **Check access** tab.

    get_ipython().system('[Access control - Check access tab](./media/role-assignments-list-portal/access-control-check-access.png)')

1. In the **Find** list, select the type of security principal you want to check access for.

1. In the search box, enter a string to search the directory for display names, email addresses, or object identifiers.

    get_ipython().system('[Check access select list](./media/role-assignments-list-portal/check-access-select.png)')

1. Click the security principal to open the **assignments** pane.

    get_ipython().system('[assignments pane](./media/role-assignments-list-portal/check-access-assignments.png)')

    On this pane, you can see the roles assigned to the selected security principal and the scope. If there are any deny assignments at this scope or inherited to this scope, they will be listed.

## List role assignments for a system-assigned managed identity

1. In the Azure portal, open a system-assigned managed identity.

1. In the left menu, click **Identity**.

    get_ipython().system('[System-assigned managed identity](./media/role-assignments-list-portal/identity-system-assigned.png)')

1. Under **Role assignments**, click **Show the Azure RBAC roles assigned to this managed identity**.

    You see a list of roles assigned to the selected system-assigned managed identity at various scopes such as management group, subscription, resource group, or resource. This list includes all role assignments you have permission to read.

    get_ipython().system('[Role assignments for a system-assigned managed identity](./media/role-assignments-list-portal/azure-resources-system-assigned.png)')

## List role assignments for a user-assigned managed identity

1. In the Azure portal, open a user-assigned managed identity.

1. Click **Azure resources**.

    You see a list of roles assigned to the selected user-assigned managed identity at various scopes such as management group, subscription, resource group, or resource. This list includes all role assignments you have permission to read.

    get_ipython().system('[Role assignments for a system-assigned managed identity](./media/role-assignments-list-portal/azure-resources-user-assigned.png)')

1. To change the subscription, click the **Subscriptions** list.

## List number of role assignments

You can have up to **2000** role assignments in each subscription. To help you keep track of this limit, the **Role assignments** tab includes a chart that lists the number of role assignments for the current subscription.

get_ipython().system('[Access control - Number of role assignments chart](./media/role-assignments-list-portal/access-control-role-assignments-chart.png)')

If you are getting close to the maximum number and you try to add more role assignments, you'll see a warning in the **Add role assignment** pane. For ways that you can reduce the number of role assignments, see [Troubleshoot Azure RBAC](troubleshooting.md#azure-role-assignments-limit).

get_ipython().system('[Access control - Add role assignment warning](./media/role-assignments-list-portal/add-role-assignment-warning.png)')

## Next steps

- [Add or remove role assignments using Azure RBAC and the Azure portal](role-assignments-portal.md)
- [Troubleshoot RBAC for Azure resources](troubleshooting.md)


# In[ ]:



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 33632,
      "digest": "sha256:ade05cfcd8379749f8847106fdf27a59c7a93b8511520302f29b3fe639eac695"
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
         "size": 526,
         "digest": "sha256:79bc66276f843221e0c4d13f1b34d960c341d67c188af6dc50092d90348037bc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 458,
         "digest": "sha256:bb4c961b816781997e0a992abecd58ee164f41994d6c516f980b2c1e5247c0e2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21102081,
         "digest": "sha256:3d193cdfd896eaf5953eac05d6d28ffe6a430650d0dc99379a9542bd485dbbc9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 126344599,
         "digest": "sha256:e51a5d6cfd7abd612392a137fc3413cc8423f754586c736633c113f074ccffed"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247356144,
         "digest": "sha256:46ce42750308ac914d734873588cb8ad5364abc18a2fd3d06ca0058287bdbcea"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 207981925,
         "digest": "sha256:b9909ab76858784a60f71b8e1e31d5a01e040631bdfae2e23c276d40de196647"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 96474622,
         "digest": "sha256:79555596337d60ce332f3d56568b81e9dadb49fdb27b075cc3aafda9001d6a68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 108017684,
         "digest": "sha256:38b2eaceb3ac381162bc1fcc22729e28dd19249eeb7659f826100174201b2005"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1026794330,
         "digest": "sha256:9ced624a00fa9b215cfd8971c1bb7b53457b6b85413a67c11a81aa97865872f0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 75053766,
         "digest": "sha256:d1c39afce8ac9cefd98cc227428d8fc6421ac3da23dfd093c2c0bd9677b86077"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46050943,
         "digest": "sha256:6827f3030ffac0ea8f61636017b3f7ee8000412a9ae5e7fbdc361ee682fd5e40"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 59100672,
         "digest": "sha256:a62a2bad207986be16c085ce3d46afdbe9b5d601e52f653c84b63cdb804830a0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 390896502,
         "digest": "sha256:16f4f5e72f1a7708d3da70e0b517a2e75d6d92c7bac2324594cefb7a90e0d195"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 46394299,
         "digest": "sha256:5fabe9e3725817d1671a2f9db086b79821e0c0b6bdf93567b2a738c2016721d3"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 106373156,
         "digest": "sha256:42510f7bd45151031bcc95b33252be111e364a5de05ebb69f6be528abffe8082"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 889994109,
         "digest": "sha256:7c397fdd935b8f4bf0fba8b5c9fea15d50ec4184c0d4e72c40b4797eb779458a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 177210260,
         "digest": "sha256:e461f336cb6e17abdae5cfd78493f7ef16daf3ca0ddc33e7281ad6d643bc77e0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 10625290,
         "digest": "sha256:cce987500e44b3121472624066612c689603dd5b7f56c6b0cf486cbdc935b731"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1845213,
         "digest": "sha256:fda620ea72e08d08d5a1804860250d78ae9258a9fbc7548270a1353751674f16"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 79538116,
         "digest": "sha256:b209bc243c56dafbd661d7c79e5806c0edec7f4ebf4bd4dcf736eab5fee1ca53"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3265,
         "digest": "sha256:0a80b13b4a0f63647e3d385fba4cddf217ce5f3b17bc3459bf7b7dc493729c82"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2165,
         "digest": "sha256:cc06eb49d933ccde0904cc93eaf009d7fe5835bd4c7431bed47ba124ec09a865"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1269,
         "digest": "sha256:64bc3380f72a43fb335bac9c60b7cf71dd877584c4610f39674e7a485c0bbb93"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 643,
         "digest": "sha256:da0e18d51de0c2d4405627aff6e25249d3398bc379b420f14cff0c2a00dae26a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2053,
         "digest": "sha256:66ece140bcf967692365a75ec887d096c5d12f93bda74745443debc98783a0e4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 877,
         "digest": "sha256:213e3a244ae9b7fe90dd3d04146a523e536b105669a4c78b5464bfdfd42f0f15"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 381,
         "digest": "sha256:67938cb4eadcef9abb78e31472776402892181bba010e7e045ace73bbf1663e1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 213,
         "digest": "sha256:cf296a506ff5aa864ae9115bd6a9b32491dc63facd186a6c1b10ccded35f033b"
      }
   ]
}


# In[ ]:


workspace(name = "tensorflow_gcs_config")

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

tf_configure(
    name = "local_config_tf",
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "1bf082fb3016154d3f806da8eb5876caf05743da4b2e8130fadd000df74b5bb6",
    strip_prefix = "grpc-1.21.1",
    urls = [
        "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.21.1.tar.gz",
        "https://github.com/grpc/grpc/archive/v1.21.1.tar.gz",
    ],
)

# 3.7.1 with a fix to BUILD file
http_archive(
    name = "com_google_protobuf",
    sha256 = "1c020fafc84acd235ec81c6aac22d73f23e85a700871466052ff231d69c1b17a",
    strip_prefix = "protobuf-5902e759108d14ee8e6b0b07653dac2f4e70ac73",
    urls = [
        "http://mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

http_archive(
    name = "jsoncpp_git",
    build_file = "//third_party:jsoncpp.BUILD",
    sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
    strip_prefix = "jsoncpp-1.8.4",
    urls = [
        "http://mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        "https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
    ],
)


# In[ ]:


licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "_gcs_config_ops.so",
    srcs = [
        "gcs_config_op_kernels.cc",
        "gcs_config_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-DNDEBUG",
    ],
    linkshared = 1,
    deps = [
        "@jsoncpp_git//:jsoncpp",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)


# In[ ]:


"""Setup TensorFlow as external dependency"""

_TF_HEADER_DIR = "TF_HEADER_DIR"
_TF_SHARED_LIBRARY_DIR = "TF_SHARED_LIBRARY_DIR"
_TF_SHARED_LIBRARY_NAME = "TF_SHARED_LIBRARY_NAME"

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl
    repository_ctx.template(
        out,
        Label("//third_party/tensorflow:%s.tpl" % tpl),
        substitutions,
    )

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    os_name = repository_ctx.os.name.lower()
    if os_name.find("windows") != -1:
        return True
    return False

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Helper for executes an arbitrary shell command.

    Args:
      repository_ctx: the repository_ctx object.
      cmdline: list of strings, the command to execute.
      error_msg: string, a summary of the error if the command fails.
      error_details: string, details about the error or steps to fix it.
      empty_stdout_fine: bool, if True, an empty stdout result is fine, otherwise
        it's an error.

    Returns:
      The result of repository_ctx.execute(cmdline).
    """
    result = repository_ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        _fail("\n".join([
            error_msg.strip() if error_msg else "Repository command failed",
            result.stderr.strip(),
            error_details if error_details else "",
        ]))
    return result

def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks. The returned string contains the full path of all files
    separated by line breaks.

    Args:
        repository_ctx: the repository_ctx object.
        src_dir: directory to find files from.

    Returns:
        A string of all files inside the given dir.
    """
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            repository_ctx,
            ["cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return result

def _genrule(genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.

    Args:
        genrule_name: A unique name for genrule target.
        command: The command to run.
        outs: A list of files generated by this rule.

    Returns:
        A genrule target.
    """
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _symlink_genrule_for_dir(
        repository_ctx,
        src_dir,
        dest_dir,
        genrule_name,
        src_files = [],
        dest_files = []):
    """Returns a genrule to symlink(or copy if on Windows) a set of files.

    If src_dir is passed, files will be read from the given directory; otherwise
    we assume files are in src_files and dest_files.

    Args:
        repository_ctx: the repository_ctx object.
        src_dir: source directory.
        dest_dir: directory to create symlink in.
        genrule_name: genrule name.
        src_files: list of source files instead of src_dir.
        dest_files: list of corresonding destination files.

    Returns:
        genrule target that creates the symlinks.
    """
    if src_dir != None:
        src_dir = _norm_path(src_dir)
        dest_dir = _norm_path(dest_dir)
        files = "\n".join(sorted(_read_dir(repository_ctx, src_dir).splitlines()))

        # Create a list with the src_dir stripped to use for outputs.
        dest_files = files.replace(src_dir, "").splitlines()
        src_files = files.splitlines()
    command = []
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]

            # Copy the headers to create a sandboxable setup.
            cmd = "cp -f"
            command.append(cmd + ' "%s" "%s"' % (src_files[i], dest))
            outs.append('        "' + dest_dir + dest_files[i] + '",')
    genrule = _genrule(
        genrule_name,
        " && ".join(command),
        "\n".join(outs),
    )
    return genrule

def _tf_pip_impl(repository_ctx):
    tf_header_dir = repository_ctx.os.environ[_TF_HEADER_DIR]
    tf_header_rule = _symlink_genrule_for_dir(
        repository_ctx,
        tf_header_dir,
        "include",
        "tf_header_include",
    )

    tf_shared_library_dir = repository_ctx.os.environ[_TF_SHARED_LIBRARY_DIR]
    tf_shared_library_name = repository_ctx.os.environ[_TF_SHARED_LIBRARY_NAME]
    tf_shared_library_path = "%s/%s" % (tf_shared_library_dir, tf_shared_library_name)

    tf_shared_library_rule = _symlink_genrule_for_dir(
        repository_ctx,
        None,
        "",
        "libtensorflow_framework.so",
        [tf_shared_library_path],
        ["libtensorflow_framework.so"],
    )

    _tpl(repository_ctx, "BUILD", {
        "%{TF_HEADER_GENRULE}": tf_header_rule,
        "%{TF_SHARED_LIBRARY_GENRULE}": tf_shared_library_rule,
    })

tf_configure = repository_rule(
    implementation = _tf_pip_impl,
    environ = [
        _TF_HEADER_DIR,
        _TF_SHARED_LIBRARY_DIR,
        _TF_SHARED_LIBRARY_NAME,
    ],
)


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

