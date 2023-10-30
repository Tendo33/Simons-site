---
tags:
- 深度学习环境配置
---
# 深度学习环境配置

##### Transfomer 中 trainer 的 CUDA 报错

```python title="CUDA error"

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
For effortless bug reporting copy-paste your error into this form: https://docs.google.com/forms/d/e/1FAIpQLScPB8emS3Thkp66nvqwmjTEgxp8Y9ufuWTzFyr9kJ5AoI47dQ/viewform?usp=sf_link
================================================================================
CUDA SETUP: Loading binary d:\anaconda\envs\roboflow\lib\site-packages\bitsandbytes\libbitsandbytes_cuda116.dll...

{
	"name": "AttributeError",
	"message": "function 'cadam32bit_grad_fp32' not found",
	"stack": "---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
d:\\github_projet\\transformers-code\\01-Getting Started\\02-pipeline\\pipeline.ipynb 单元格 2 line 1
----> <a href='vscode-notebook-cell:/d%3A/github_projet/transformers-code/01-Getting%20Started/02-pipeline/pipeline.ipynb#W1sZmlsZQ%3D%3D?line=0'>1</a> from transformers.pipelines import SUPPORTED_TASKS

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\transformers\\pipelines\\__init__.py:44
     34 from ..tokenization_utils import PreTrainedTokenizer
     35 from ..utils import (
     36     HUGGINGFACE_CO_RESOLVE_ENDPOINT,
     37     is_kenlm_available,
   (...)
     42     logging,
     43 )
---> 44 from .audio_classification import AudioClassificationPipeline
     45 from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
     46 from .base import (
     47     ArgumentHandler,
     48     CsvPipelineDataFormat,
   (...)
     56     infer_framework_load_model,
     57 )

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\transformers\\pipelines\\audio_classification.py:21
     18 import requests
     20 from ..utils import add_end_docstrings, is_torch_available, logging
---> 21 from .base import PIPELINE_INIT_ARGS, Pipeline
     24 if is_torch_available():
     25     from ..models.auto.modeling_auto import MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\transformers\\pipelines\\base.py:35
     33 from ..feature_extraction_utils import PreTrainedFeatureExtractor
     34 from ..image_processing_utils import BaseImageProcessor
---> 35 from ..modelcard import ModelCard
     36 from ..models.auto.configuration_auto import AutoConfig
     37 from ..tokenization_utils import PreTrainedTokenizer

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\transformers\\modelcard.py:48
     31 from . import __version__
     32 from .models.auto.modeling_auto import (
     33     MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
     34     MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
   (...)
     46     MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
     47 )
---> 48 from .training_args import ParallelMode
     49 from .utils import (
     50     MODEL_CARD_NAME,
     51     cached_file,
   (...)
     57     logging,
     58 )
     61 TASK_MAPPING = {
     62     \"text-generation\": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
     63     \"image-classification\": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
   (...)
     74     \"zero-shot-image-classification\": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
     75 }

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\transformers\\training_args.py:67
     64     import torch.distributed as dist
     66 if is_accelerate_available():
---> 67     from accelerate.state import AcceleratorState, PartialState
     68     from accelerate.utils import DistributedType
     70 if is_torch_tpu_available(check_device=False):

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\accelerate\\__init__.py:3
      1 __version__ = \"0.21.0\"
----> 3 from .accelerator import Accelerator
      4 from .big_modeling import (
      5     cpu_offload,
      6     cpu_offload_with_hook,
   (...)
     11     load_checkpoint_and_dispatch,
     12 )
     13 from .data_loader import skip_first_batches

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\accelerate\\accelerator.py:35
     32 import torch
     33 import torch.utils.hooks as hooks
---> 35 from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
     36 from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
     37 from .logging import get_logger

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\accelerate\\checkpointing.py:24
     21 import torch
     22 from torch.cuda.amp import GradScaler
---> 24 from .utils import (
     25     MODEL_NAME,
     26     OPTIMIZER_NAME,
     27     RNG_STATE_NAME,
     28     SCALER_NAME,
     29     SCHEDULER_NAME,
     30     get_pretty_name,
     31     is_tpu_available,
     32     is_xpu_available,
     33     save,
     34 )
     37 if is_tpu_available(check_device=False):
     38     import torch_xla.core.xla_model as xm

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\accelerate\\utils\\__init__.py:131
    121 if is_deepspeed_available():
    122     from .deepspeed import (
    123         DeepSpeedEngineWrapper,
    124         DeepSpeedOptimizerWrapper,
   (...)
    128         HfDeepSpeedConfig,
    129     )
--> 131 from .bnb import has_4bit_bnb_layers, load_and_quantize_model
    132 from .fsdp_utils import load_fsdp_model, load_fsdp_optimizer, save_fsdp_model, save_fsdp_optimizer
    133 from .launch import (
    134     PrepareForLaunch,
    135     _filter_args,
   (...)
    140     prepare_tpu,
    141 )

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\accelerate\\utils\\bnb.py:42
     31 from .modeling import (
     32     find_tied_parameters,
     33     get_balanced_memory,
   (...)
     37     set_module_tensor_to_device,
     38 )
     41 if is_bnb_available():
---> 42     import bitsandbytes as bnb
     44 from copy import deepcopy
     47 logger = logging.getLogger(__name__)

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\__init__.py:6
      1 # Copyright (c) Facebook, Inc. and its affiliates.
      2 #
      3 # This source code is licensed under the MIT license found in the
      4 # LICENSE file in the root directory of this source tree.
----> 6 from . import cuda_setup, utils, research
      7 from .autograd._functions import (
      8     MatmulLtState,
      9     bmm_cublas,
   (...)
     13     matmul_4bit
     14 )
     15 from .cextension import COMPILED_WITH_CUDA

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\research\\__init__.py:1
----> 1 from . import nn
      2 from .autograd._functions import (
      3     switchback_bnb,
      4     matmul_fp8_global,
      5     matmul_fp8_mixed,
      6 )

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\research\
n\\__init__.py:1
----> 1 from .modules import LinearFP8Mixed, LinearFP8Global

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\research\
n\\modules.py:8
      5 from torch import Tensor, device, dtype, nn
      7 import bitsandbytes as bnb
----> 8 from bitsandbytes.optim import GlobalOptimManager
      9 from bitsandbytes.utils import OutlierTracer, find_outlier_dims
     11 T = TypeVar(\"T\", bound=\"torch.nn.Module\")

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\optim\\__init__.py:8
      1 # Copyright (c) Facebook, Inc. and its affiliates.
      2 #
      3 # This source code is licensed under the MIT license found in the
      4 # LICENSE file in the root directory of this source tree.
      6 from bitsandbytes.cextension import COMPILED_WITH_CUDA
----> 8 from .adagrad import Adagrad, Adagrad8bit, Adagrad32bit
      9 from .adam import Adam, Adam8bit, Adam32bit, PagedAdam, PagedAdam8bit, PagedAdam32bit
     10 from .adamw import AdamW, AdamW8bit, AdamW32bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\optim\\adagrad.py:5
      1 # Copyright (c) Facebook, Inc. and its affiliates.
      2 #
      3 # This source code is licensed under the MIT license found in the
      4 # LICENSE file in the root directory of this source tree.
----> 5 from bitsandbytes.optim.optimizer import Optimizer1State
      8 class Adagrad(Optimizer1State):
      9     def __init__(
     10         self,
     11         params,
   (...)
     21         block_wise=True,
     22     ):

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\optim\\optimizer.py:12
      8 from itertools import chain
     10 import torch
---> 12 import bitsandbytes.functional as F
     15 class MockArgs:
     16     def __init__(self, initial_data):

File d:\\anaconda\\envs\\roboflow\\lib\\site-packages\\bitsandbytes\\functional.py:31
     29 \"\"\"C FUNCTIONS FOR OPTIMIZERS\"\"\"
     30 str2optimizer32bit = {}
---> 31 str2optimizer32bit[\"adam\"] = (lib.cadam32bit_grad_fp32, lib.cadam32bit_grad_fp16, lib.cadam32bit_grad_bf16)
     32 str2optimizer32bit[\"momentum\"] = (
     33     lib.cmomentum32bit_grad_32,
     34     lib.cmomentum32bit_grad_16,
     35 )
     36 str2optimizer32bit[\"rmsprop\"] = (
     37     lib.crmsprop32bit_grad_32,
     38     lib.crmsprop32bit_grad_16,
     39 )

File d:\\anaconda\\envs\\roboflow\\lib\\ctypes\\__init__.py:386, in CDLL.__getattr__(self, name)
    384 if name.startswith('__') and name.endswith('__'):
    385     raise AttributeError(name)
--> 386 func = self.__getitem__(name)
    387 setattr(self, name, func)
    388 return func

File d:\\anaconda\\envs\\roboflow\\lib\\ctypes\\__init__.py:391, in CDLL.__getitem__(self, name_or_ordinal)
    390 def __getitem__(self, name_or_ordinal):
--> 391     func = self._FuncPtr((name_or_ordinal, self))
    392     if not isinstance(name_or_ordinal, int):
    393         func.__name__ = name_or_ordinal

AttributeError: function 'cadam32bit_grad_fp32' not found"
}
```

暂定解决方法：使用bitsandbytes_windows，应该去配一个自己的服务器了
```python

bitsandbytes 本来在Windows上兼容就差，只能使用bitsandbytes_windows
首先需要安装0.35.0版本的bitsandbytes，然后按照如下操作

git clone https://github.com/bmaltais/kohya_ss.git
一共五个文件
cd kohya_ss
cp .\bitsandbytes_windows*.dll .\venv\Lib\site-packages\bitsandbytes
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```
