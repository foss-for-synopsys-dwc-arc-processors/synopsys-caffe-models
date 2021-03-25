# ==============================================================================
# Copyright 2021  Synopsys, Inc.
#
# This file and the associated documentation are proprietary to Synopsys,
# Inc., and may only be used in accordance with the terms and conditions of
# a written license agreement with Synopsys, Inc.
# Notwithstanding contrary terms in the DFPUC, Licensee may provide the
# binaries of the EV CNN SDK to its end-customer that purchase Licensee ICs
# that incorporate the Synopsys EV processor core with the CNN option,
# subject to confidentiality terms no less restrictive than those contained in
# the DFPUC.  All other use, reproduction, or distribution of this file
# is strictly prohibited.
# ==============================================================================

"""
.. module:: download_unpack_cnn_modeles
   :platform: Unix, Windows
   :synopsis: Download zipped models from GitHub and unpack them

.. moduleauthors:: Dmitry Golovkin <golovkin@synopys.com

"""

import os
from pathlib import Path
import argparse
import re
import shutil
import zipfile
import time
import multiprocessing
import requests


def download_unpack_zip(zip_name, output_path, zip_path,
                        git_hub_repo_url, force_download):
    """
    Download and unpack one zip file
    """

    digit_match = re.search(r'\d.zip$', zip_name)  # if we have multi archive
    if digit_match:
        name = zip_name[:-5]  # cut last character with number
    else:
        name = zip_name[:-4]

    model_out_folder = output_path/name

    if model_out_folder.is_dir():
        if force_download:
            print("[INFO]  --force option is set. {} will be rewritten".
                  format(model_out_folder))
            shutil.rmtree(model_out_folder)
        else:
            print("[INFO] Skip downloading {}. Model already exists here {}".
                  format(zip_name, model_out_folder))
            return False

    file_name_url = git_hub_repo_url + '/' + zip_name
    file_name_zip = zip_path / zip_name

    print("[INFO] Start downloading {}".format(zip_name))
    request_check = requests.get(file_name_url)
    if request_check.status_code != 200:
        print("   [ERROR]. Can not access to Zip file by address {}".
              format(file_name_url))
        return False

    with requests.get(file_name_url, stream=True) as r_zip_file:
        with open(file_name_zip, 'wb') as f_download:
            shutil.copyfileobj(r_zip_file.raw, f_download)

    with zipfile.ZipFile(file_name_zip, 'r') as zip_ref:
        zip_ref.extractall(output_path)

    print("[INFO] {} is downloaded and unpacked"
          .format(zip_name))
    return True


def arg2bool(bool_arg):
    """
    argparse does not allow bools by default, very annoying
    """
    # In case options come from files they may have weird characters such as \r
    bool_arg = bool_arg.replace('\n', "").replace('\r', "")
    if bool_arg.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True

    if bool_arg.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    git_hub_repo_url = 'https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/raw/master/caffe_models_zipped'

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, required=False,
        help="Path to NN Models home")

    parser.add_argument(
        "--model_list", type=str, required=True,
        help="File with list of models")

    parser.add_argument(
        "--force", type=arg2bool, nargs='?', const=True, required=False,
        help="Force to download/unpack models. Overwrite existed")

    args = parser.parse_args()

    list_file = Path(args.model_list)

    force_download = False  # Rewrite existed models
    if args.force is not None:
        if args.force:
            force_download = True

    if not list_file.is_file():
        print("[ERROR]. List file {} is not found".format(list_file))
        exit(1)

    # Get a file with models
    with open(list_file, "r") as f:
        zips_files_lines = f.readlines()

    # Create zip lists from list_file
    model_list = []
    for line in zips_files_lines:
        model_name = line.strip()
        # Avoid comments and empty strings
        if model_name.isspace() or model_name == "" or model_name[0] == '#':
            # print("[DEBUG] string is empty or comment {}".format(model_name))
            continue
        model_list.extend([model_name])

    model_list_num = len(model_list)

    if model_list_num == 0:
        print("[ERROR]. There is no model names in model list file {}".
              format(list_file))
        exit(1)

    print("[INFO] Number of models in the model list is {}"
          .format(model_list_num))

    # Get and check output path
    if args.model_path:
        output_path = Path(args.model_path)/"caffe_models"
    elif os.environ['EV_CNNMODELS_HOME']:
        output_path = Path(os.environ['EV_CNNMODELS_HOME']) / "caffe_models"
    else:
        print("[ERROR]. Output path is not defined. Use --model_path or "
              "setup EV_CNNMODELS_HOME environment variable")
        exit(1)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except:
        print("[ERROR]. Problem of creating CNN Models folder {}".
              format(output_path))
        exit(1)

    zip_path = output_path/"download"  # folder for downloaded zip files
    try:
        zip_path.mkdir(parents=True, exist_ok=True)
    except:
        print(
            "[ERROR]. Problem of creating folder to download zipped CNN Models:"
            " {}".format(zip_path))
        exit(1)

    print("[INFO] NN Models path is {}".format(output_path))
    print("[INFO] Zipped NN Models path is {}".format(zip_path))

    # check GitHub site access
    request = requests.get(git_hub_repo_url)
    if request.status_code != 200:
        print("[ERROR]. Can not access to GitHub by address {}"
              .format(git_hub_repo_url))
        exit(1)

    # Download and unpack zip files in parallel mode
    list_of_params = []

    # Use multi-processing to speed-up zipping
    for zip_file_name in model_list:
        list_of_params += [[zip_file_name, output_path, zip_path,
                            git_hub_repo_url, force_download]]
    
    MAX_NUM_CORE = 8
    num_cores = multiprocessing.cpu_count()
    parallel_task = num_cores if num_cores < MAX_NUM_CORE else MAX_NUM_CORE

    start_time = time.time()
    with multiprocessing.Pool(parallel_task) as p:
        results_values = p.starmap(download_unpack_zip, list_of_params)

    # Count success and failed downloads
    download_cnt = 0
    issue_cnt = 0

    for value in results_values:
        if value:
            download_cnt += 1
        else:
            issue_cnt += 1

    end_time = int(time.time() - start_time)
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("[INFO] Finish. \n"
          "   Total models: {}\n"
          "   Successful:   {}\n"
          "   Failed :      {}\n"
          "   Time:         {:0>2}:{:0>2}:{:05.2f}".
          format(int(model_list_num), int(download_cnt),
                 int(issue_cnt), int(hours), int(minutes), int(seconds)))


if __name__ == "__main__":
    main()
