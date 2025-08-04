# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file constitutes end  part of runmodel.py
# this is appended to the model.py in test dir

import sys, argparse
import torch_mlir
import numpy
import io, pickle

# Fx importer related
from typing import Optional
import torch.export
from torch_mlir.extras.fx_importer import FxImporter # No module named 'torch_mlir.extras'
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir import fx # ImportError: cannot import name 'fx' from 'torch_mlir' 

# old torch_mlir.compile path
# from torch_mlir import torchscript # ImportError: cannot import name 'torchscript' from 'torch_mlir'
from commonutils import getOutputTensorList, E2ESHARK_CHECK_DEF, postProcess

msg = "The script to run a model test"
parser = argparse.ArgumentParser(description=msg, epilog="")

parser.add_argument(
    "-d",
    "--todtype",
    choices=["default", "fp32", "fp16", "bf16"],
    default="default",
    help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
)
parser.add_argument(
    "-m",
    "--mode",
    choices=["direct", "onnx", "ort"],
    default="direct",
    help="Generate torch MLIR, ONNX or ONNX plus ONNX RT stub",
)
parser.add_argument(
    "-p",
    "--torchmlirimport",
    choices=["compile", "fximport"],
    default="fximport",
    help="Use torch_mlir.torchscript.compile, or Fx importer",
)
parser.add_argument(
    "-o",
    "--outfileprefix",
    default="model",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
runmode = args.mode
outfileprefix = args.outfileprefix + "." + args.todtype


def getTorchDType(dtypestr):
    if dtypestr == "fp32":
        return torch.float32
    elif dtypestr == "fp16":
        return torch.float16
    elif dtypestr == "bf16":
        return torch.bfloat16
    else:
        print("Unknown dtype {dtypestr} returning torch.float32")
        return torch.float32


if args.todtype != "default":
    # convert the model to given dtype
    dtype = getTorchDType(args.todtype)
    model = model.to(dtype)
    # not all model need the input re-casted
    
    if E2ESHARK_CHECK["inputtodtype"]:
        if isinstance(E2ESHARK_CHECK["input"], list):
            E2ESHARK_CHECK["input"] = list(element.to(dtype) for element in E2ESHARK_CHECK["input"])
        else:
           E2ESHARK_CHECK["input"] = E2ESHARK_CHECK["input"].to(dtype)
    if isinstance(E2ESHARK_CHECK["input"], list):
        new_ele = [element for element in E2ESHARK_CHECK["input"]]
        print("lyl-tuple", new_ele)
        E2ESHARK_CHECK["output"] = model(*new_ele)
    else:
        E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])

if runmode == "onnx" or runmode == "ort": # PyTorch ---> ONNX
    onnx_name = outfileprefix + ".onnx"
    if args.outfileprefix == "llama2-7b-GPTQ" or args.outfileprefix == "opt-125m-gptq":
        if not isinstance(E2ESHARK_CHECK["input"], list):
            onnx_program = torch.onnx.export(model, E2ESHARK_CHECK["input"], onnx_name, opset_version=18)
        else:
            onnx_program = torch.onnx.export(
                model, tuple(E2ESHARK_CHECK["input"]), onnx_name, opset_version=18
        )
    else:
        if not isinstance(E2ESHARK_CHECK["input"], list):
            onnx_program = torch.onnx.export(model, E2ESHARK_CHECK["input"], onnx_name, opset_version=17)
        else:
            onnx_program = torch.onnx.export(
                model, tuple(E2ESHARK_CHECK["input"]), onnx_name, opset_version=17
        )
elif runmode == "direct": ## PyTorch ---> MLIR
    torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
    torch_mlir_model = None
    # override mechanism to get torch MLIR as per model
    if (
        args.torchmlirimport == "compile"
        or E2ESHARK_CHECK["torchmlirimport"] == "compile"
    ):
        print("lyl-check-2583:",args.torchmlirimport)
        print("lyl-check-2583:",E2ESHARK_CHECK["torchmlirimport"])
        torch_mlir_model = torchscript.compile(
            model,
            (E2ESHARK_CHECK["input"]),
            # output_type="torch", 
            output_type="LINALG_ON_TENSORS",
            # output_type="tosa",
            use_tracing=True,
            verbose=True,
        )
    else:
        # check for seq2seq model
        if not isinstance(E2ESHARK_CHECK["input"], list):
            torch_mlir_model = fx.export_and_import(model, E2ESHARK_CHECK["input"])
        else:
            torch_mlir_model = fx.export_and_import(model, *E2ESHARK_CHECK["input"])
    with open(torch_mlir_name, "w+") as f:
        f.write(torch_mlir_model.operation.get_asm())

inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"

test_input_list = E2ESHARK_CHECK["input"]
test_output_list = E2ESHARK_CHECK["output"]

if not isinstance(E2ESHARK_CHECK["input"], list):
    test_input_list = [E2ESHARK_CHECK["input"]]

if isinstance(test_output_list, tuple):
    # handles only nested tuples for now
    print(f"Found tuple {len(test_output_list)} {test_output_list}")
    test_output_list = getOutputTensorList(E2ESHARK_CHECK["output"])

# model result expected to be List[Tensors]
if not isinstance(test_output_list, list):
    test_output_list = [E2ESHARK_CHECK["output"]]

E2ESHARK_CHECK["input"] = [t.detach() for t in test_input_list]
E2ESHARK_CHECK["output"] = [t.detach() for t in test_output_list]

E2ESHARK_CHECK["postprocessed_output"] = postProcess(E2ESHARK_CHECK)
# TBD, move to using E2ESHARK_CHECK pickle save
torch.save(E2ESHARK_CHECK["input"], inputsavefilename)
torch.save(E2ESHARK_CHECK["output"], outputsavefilename)
# out_sizes = [i.size() for i in E2ESHARK_CHECK["output"]]
# print(f"output sizes: {out_sizes}")
# Save the E2ESHARK_CHECK
with open("E2ESHARK_CHECK.pkl", "wb") as tchkf:
    pickle.dump(E2ESHARK_CHECK, tchkf)
