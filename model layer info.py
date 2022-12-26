import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import onnx
import onnx.helper
import pandas as pd
from onnx import ModelProto

from qonnx.core.modelwrapper import ModelWrapper

def load_onnx_model(model_path: Path, update_model: bool = False) -> ModelProto:
    """Loads an onnx model given a path and returns the model.
    Args:
        model_path: onnx model path
        update_model: whether to update the model with the shape inference results.
    Returns:
        onnx model as a protobuf object.
    """
    onnx_model = onnx.load(model_path)
    # annotate the operator shapes
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    if update_model:
        # save onnx model again, with updated shapes
        onnx.save(onnx_model, model_path)
    return onnx_model

# Calculating the product of shapes
def calc(x, sum, loc, file) -> int:
    count = len(x)
    if count == 0:
        file.write("Couldn't retrieve shape at {}\n".format(loc))
        return 0    
    elif count == 1:
        sum += x[0]
    #elif x[0] == 0:
    else:
        temp = 1
        for y in range(count-1):
            #padding for width to be multiple of 16
            if y==1 and x[y+1]%16 != 0:
                pad = x[y+1] + (16 - x[y+1]%16)  #taking y+1 to skip the 0
                temp = temp * pad
            else:
                temp = temp*x[y+1]
        sum += temp
    '''''
    else:
        temp = 1
        for y in range(count):
            #padding for width to be multiple of 16
            if y==2 and x[y]%16 != 0:
                pad = x[y] + (16 - x[y]%16)
                temp = temp * pad
            else:
                temp = temp * x[y]
        sum += temp
    '''''
    return sum
# For checking 1.5Mb Constraints
def checkSize(limit, ins, ops, name, loc, file):
    sum = 0
    if name == "Conv" or "Gemm":
        x = ins[0]
        sum = calc(x, sum, loc, file)
    else:
        for x in ins:
            sum = calc(x, sum, loc, file)
    for x in ops:
        sum = calc(x, sum, loc, file)
    #assert sum < 1.5e+6, "Exceeds Size limit at *{}* node number {}".format(name, cnt)
    if sum > limit: 
        file.write("Exceeds Size limit for *{}* at node {}\n".format(name, loc))
    
    return sum
@dataclass
class Operator:
    """Represents an ONNX operator
    Args:
        count: frequency of the operator
        type: operator type: e.g. Add, Conv...
    """
    op_count: int
    type: str    

def extract_model_info(onnx_model: ModelWrapper, size, file, check1, check3) -> pd.DataFrame:
    """Create a dataframe containing operator information
    This may include attributes and input/output shapes.
    Args:
        onnx_model: model we are investigating
        size: The size constraint Ex:1500000kb (1.5mb)
        check1: wether we want to do a size constraint check
        check2: wether we want to do a symmetric padding check
    Returns:
        dataframe containing the list of operators in the model.
    """
    ops: Dict[str, Operator] = {}
    max_size = 0
    for node in onnx_model.graph.node:
        op = ops.get(
            node.op_type,
            Operator(
                0, node.op_type
            ),
        )
        op.op_count = op.op_count + 1
        input_shapes_list = [
            onnx_model.get_tensor_shape(x)
            for x in node.input
        ]
        output_shapes_list = [
            onnx_model.get_tensor_shape(x)
            for x in node.output
        ]
        # Skip empty nodes
        if not input_shapes_list:
            continue
        if not output_shapes_list:
            continue
        node_name = node.name
        
        # Skip QuantizeLinear and DequantizeLinear
        if node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear":
            continue

        # Detecting asymmetric padding
        if check3 == "True":
            try:
                lst = node.attribute[1].ints
                result = all(element == lst[0] for element in lst)
                if (not result):
                    file.write("Asymmetric padding at {}\n".format(node_name))
            except:
                pass

        # Checking the 1.5Mb Size Constraints
        if check1 == "True":
            size_tst = checkSize(size, input_shapes_list, output_shapes_list, node.op_type, node_name, file)
            if size_tst > max_size:
                max_size = size_tst
                max_loc = node_name

        ops[node.op_type] = op

    if check1 == 'True':
        file.write("The maximum (input+output) size is at {}: {} bytes\n".format(max_loc, max_size))
    return pd.DataFrame(list(ops.values()))     


def main(args: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Utility script to generate "
        "information about operators that appear either in torchvision or "
        "given onnx models."
    )
    parser.add_argument(
        "--model", 
        type=str, required=True, 
        help="name of the model to be analysed"
    )
    parser.add_argument(
        "--size",
        type=int, default=1.5e+6,
        help="What is the size constraint ex: 1.5Mb",
    )
    parser.add_argument(
        "--output",
        type=str, default="analysis.txt",
        help="Name of the output file",
    )
    parser.add_argument(
        "--mem_check",
        type=str, default="True",
        help="Size constraints check, check1",
    )
    parser.add_argument(
        "--nop_check",
        type=str, default="True",
        help="New operator check, check2",
    )
    parser.add_argument(
        "--smp_check",
        type=str, default="True",
        help="Symmetric padding check, check3",
    )

    args = parser.parse_args(args)
    model_name = args.model
    onnx_model = "onnx_models/{}.onnx".format(model_name)
    op_table = pd.DataFrame(columns=[x.name for x in dataclasses.fields(Operator)])

    opfile = args.output
    with open(opfile, 'w+') as file:
        file.write(f"Processing Model: {model_name}\n")
        model = load_onnx_model(onnx_model)

        size = args.size
        check1 = args.mem_check
        check3 = args.smp_check
        df = extract_model_info(ModelWrapper(model), size, file, check1, check3)
        file.write("Memory Constraints check Done!!\n")

        op_table = pd.concat([op_table, df])

        # Checking for operators not supported by FlexML
        check2 = args.nop_check
        if check2 == "True":
            # supported operator list
            list1 = ['Conv', 'Relu', 'MaxPool', 'LRN', 'Dropout', 'Softmax', 'BatchNormalization']
            # operators present in our model
            list2 = op_table["type"]
            # list of supported operators
            sup_op = list(set(list1).intersection(list2))
            # yields the elements in `list2` that are NOT in `sup_op`
            new_op = np.setdiff1d(list2,sup_op)

            file.write('Supported operators in {}: \n'.format(model_name))
            for p in sup_op: file.write(str(p) + "\n")

            if len(new_op) == 0:
                file.write('There are no unsupported operators in {} \n'.format(model_name))
            else:    
                file.write('Unsupported operators in {}: \n'.format(model_name))
                for p in new_op: file.write(str(p) + "\n")


        file.write('The parameter check for {} model is done.'.format(model_name))
        print('Refer file ' +str(opfile) + ' for results.')
        file.write("\n\n")

    return 0
