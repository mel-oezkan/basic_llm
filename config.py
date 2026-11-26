MAX_TOKENS = 256
TRAIN_SIZE = 2000
EVAL_SIZE = 200
TEST_SIZE = 100

SYSTEM = "You are a helpful assistant."

MODEL = "HuggingFaceTB/SmolLM2-360M"
TEMPLATE_FILE = "templates/qwen2-5_template.jinja"

LR = 2e-4

LORA_R = 8
LORA_ALPHA = 16.0
EPOCHS = 10