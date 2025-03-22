import os
import pickle
import warnings
from typing import List, OrderedDict, Tuple, Union, Optional,NamedTuple
import torch
import numpy as np
import tensorrt as trt
from collections import namedtuple
from pathlib import Path
import time
import sys
import cv2
import torch.nn.functional as F
from rich import print
from typing import Tuple

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path], device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')

        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        self.context = model.create_execution_context()
        self.num_bindings = model.num_io_tensors
        self.num_inputs = sum(model.get_tensor_mode(model.get_tensor_name(i)) == trt.TensorIOMode.INPUT for i in range(self.num_bindings))
        self.num_outputs = self.num_bindings - self.num_inputs
        self.bindings = [0] * self.num_bindings
        self.input_names = [model.get_tensor_name(i) for i in range(self.num_inputs)]
        self.output_names = [model.get_tensor_name(i) for i in range(self.num_inputs, self.num_bindings)]
        self.idx = list(range(self.num_outputs))
        self.model = model

    def __init_bindings(self) -> None:
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        self.inp_info = []
        self.out_info = []
        self.idynamic = False
        self.odynamic = False

        for i, name in enumerate(self.input_names + self.output_names):
            is_input = i < self.num_inputs
            dtype = self.dtypeMapping[self.model.get_tensor_dtype(self.model.get_tensor_name(i))]
            shape = tuple(self.model.get_tensor_shape(self.model.get_tensor_name(i)))
            tensor_info = Tensor(name, dtype, shape)
            if is_input:
                self.inp_info.append(tensor_info)
                if -1 in shape:
                    self.idynamic = True
            else:
                self.out_info.append(tensor_info)
                if -1 in shape:
                    self.odynamic = True

        if not self.odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in self.out_info
            ]

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired, (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        assert len(inputs) == self.num_inputs, "Number of inputs does not match model's requirements"
        contiguous_inputs = [i.contiguous() for i in inputs]

        outputs = [
            torch.empty(self.context.get_binding_shape(i + self.num_inputs),
                        dtype=self.out_info[i].dtype, 
                        device=self.device)
            if self.odynamic else self.output_tensor[i]
            for i in range(self.num_outputs)
        ]

        # Link each input tensor to its corresponding memory address
        for i, input_tensor in enumerate(contiguous_inputs):
            input_tensor_address = int(input_tensor.data_ptr())
            self.context.set_tensor_address(self.input_names[i], input_tensor_address)

        # Prepare and link output tensors similarly
        for i, output_tensor in enumerate(outputs):
            output_tensor_address = int(output_tensor.data_ptr())
            self.context.set_tensor_address(self.output_names[i], output_tensor_address)

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)


        self.stream.synchronize()  # Synchronize to ensure completion


        return tuple(outputs[i] for i in self.idx) if len(outputs) > 1 else outputs[0]






class Utility:
    def __init__(self):
        self.fps_counter = 0
        self.start_time = time.time()

    def check_cuda_device(self):
        # Verifica la disponibilità di CUDA
        if torch.cuda.is_available():
            cuda_status = "[yellow]CUDA device found:[/yellow] [orange]{}[/orange]".format(torch.cuda.get_device_name(0))
            print(cuda_status)
        else:
            cuda_status = "[red]No CUDA device found.[/red]"
            print(cuda_status)

    def draw_visuals(self, img, bboxes):
        for bbox in bboxes:
            cx, cy, w, h = bbox
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
        cv2.imshow("Object Detection", img)
        cv2.waitKey(1)

    def count_fps(self):
        elapsed_time = time.time() - self.start_time
        self.fps_counter += 1

        if elapsed_time >= 1.0:
            fps = self.fps_counter / elapsed_time
            sys.stdout.write(f"\rFPS: {fps:.2f}")
            sys.stdout.flush()
            self.fps_counter = 0
            self.start_time = time.time()

    import torch

    def det_postprocess6(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                         confidence_threshold: float = 0.48, 
                         class_id: int = 0,  
                         max_results: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_dets, bboxes, scores, labels = (i[0] for i in data)

        # Applica i filtri di soglia di confidenza e ID della classe
        selected = (scores >= confidence_threshold)&(labels == class_id)


        # Applica la selezione
        scores_selected = scores[selected]
        labels_selected = labels[selected]
        bboxes_selected = bboxes[selected]

        # Avoid moving to CPU prematurely; keep computations on GPU
        x1, y1, x2, y2 = bboxes_selected[:, 0], bboxes_selected[:, 1], bboxes_selected[:, 2], bboxes_selected[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        new_boxes = torch.stack([cx, cy, w, h], dim=1)

        return new_boxes.cpu().numpy()



