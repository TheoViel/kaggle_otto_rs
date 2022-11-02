import shutil
from pathlib import Path


def update_tokenizers(package_path="", input_dir=""):
    transformers_path = Path(package_path)
    input_dir = Path(input_dir)

    convert_file = input_dir / "convert_slow_tokenizer.py"
    conversion_path = transformers_path/convert_file.name

    if conversion_path.exists():
        conversion_path.unlink()

    shutil.copy(convert_file, transformers_path)
    deberta_v2_path = transformers_path / "models" / "deberta_v2"

    for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
        filepath = deberta_v2_path/filename
        if filepath.exists():
            filepath.unlink()

        shutil.copy(input_dir/filename, filepath)
