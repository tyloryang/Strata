import importlib.util
import sys

path = "/data/strata-npu-vector-v4-20261106/source-grouped-cann/tests/serve_engine.py"
spec = importlib.util.spec_from_file_location("strata_serve_engine", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.BIN = "/data/strata-npu-vector-v4-20261106/model_seq-grouped"
module.DEFAULT_MODEL = "/data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
module.MODEL_NAME = "qwen3.8-flash-next-iq3-xxs-strata-cann"
module.LLAMA = "http://127.0.0.1:18789"
sys.argv = [path, "--host", "127.0.0.1", "--port", "18790", "--model", module.DEFAULT_MODEL]
module.main()
