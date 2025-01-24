from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import accelerate



torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



torch.set_num_threads(1)
if torch.cuda.is_available():
        torch.cuda.set_device(0)



accelerator = accelerate.Accelerator()


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


model.enable_input_require_grads()

model.config.max_length = 1024


