# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


FROM nvcr.io/nvidia/pytorch:21.08-py3
LABEL description="warpdrive-env"
WORKDIR /home/
RUN chmod a+rwx /home/
# Install other packages
RUN pip3 install pycuda==2021.1
