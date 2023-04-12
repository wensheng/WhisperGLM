# WhisperGLM

This is a GUI interface for [OpenAI Whisper](https://github.com/openai/whisper) and [GhatGLM](https://github.com/THUDM/ChatGLM-6B) running locally.

This program can only run on Windows for now.

With Whisper, it can listen to system audio or microphone and transribe speech to text.

It can also download youtube video and transcribe it to text.

Combined with ChatGLM, it can summerize the text, as well as can be chatted with directly.

## Installation

A NVIDIA GPU with >=8GB vram is required to run this program.

~~**Option 1**~~: (not ready yet) 

Download from Releases, unzip it, and run main.pyw.

**Option 2**:

Clone this repo:

    git clone --recurse-submodules https://github.com/wensheng/WhisperGLM.git
    cd WhisperGLM

Install [PyTorch with cuda](https://pytorch.org/get-started/locally/) (Tested with PyTorch v1.13.1 and v2.0, cuda 11.7)

Install rest of dependencies:

    pip install -r requirements.txt

**(Optional)** Download ChatGLM-6B model from Huggingface (you need to have [git-lfs](https://github.com/git-lfs/git-lfs#installing) already installed):

    cd data
    git clone https://huggingface.co/THUDM/chatglm-6b-int4

If you don't have git-lfs, you can check out without large file:

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b-int4

Then go to: [here](https://huggingface.co/THUDM/chatglm-6b-int4/tree/main) and download `pytorch_model.bin` directly.

### After everything is setup, run main.py

    python main.py
    
If you skipped installing ChatGLM models, start the program without the model:

    python main.py -n
    
    

