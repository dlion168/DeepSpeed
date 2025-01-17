import os
import sys
import torch
import typing
import logging
import argparse
import importlib
import s3prl.hub as hub
import torchaudio
from time import time
from pathlib import Path

sys.path.append(os.path.realpath(os.path.join(__file__, "../../")))

from flops_profiler import get_model_profile, FlopsProfiler, number_to_string, macs_to_string, params_to_string

# logger setting
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

def synchronize(device):
    if device != "cpu":
        torch.cuda.synchronize()

# args
def get_profiling_args():
    parser=argparse.ArgumentParser()
    # upstream
    upstreams=[attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream', default="hubert", help="This is also the filename of logfile")
    parser.add_argument('--upstream_ckpt', default="", help="The ckpt path for upstream")
    # huggingface
    parser.add_argument('--from_hf_hub', action="store_true")
    parser.add_argument('--hf_org_name', type=str)
    parser.add_argument('--hf_repo_name', type=str)
    parser.add_argument('--hf_revision', type=str)
    # profiling setup
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="only for pseudo input")
    parser.add_argument('-l', '--seq_len', type=int, default=160000, help="only for pseudo input")
    parser.add_argument("--sample_rate", type=float, default=16000., help="The input sample rate")
    parser.add_argument('-d', "--device", default="cuda")
    # information present
    parser.add_argument('-s', "--show_untracked", help="Show all untracked functions in torch and torchaudio.", action="store_true")
    parser.add_argument("--show_time", help="Show time related info.", action="store_true")
    parser.add_argument("--execution_time_only", help="Show execution time only (currently only support cuda)", action="store_true")
    parser.add_argument("-p", "--precision", type=int, default=2)
    parser.add_argument("--as_string", action="store_true", help="print result as formated string")
    parser.add_argument("--without_bucket", action="store_true", help="not divide inputs into four buckets by sequence length and report the result seperately")
    # path
    parser.add_argument("--libri_root", type=str, default="/mnt/diskb/corpora/LibriSpeech/", help="The root dir of LibriSpeech")
    parser.add_argument("--log_path", type=str, help="The path for log file storing. (not include file name)", default=os.path.join(os.path.dirname(__file__), "log/"))
    return parser.parse_args()


def s3prl_input_constructor(batch_size, seq_len, device, dtype):
    return [torch.randn(seq_len, dtype=dtype, device=device) for _ in range(batch_size)]

def pseudo_input_profiling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    model_args: list=[],
    model_kwargs: dict={},
    ignore_modules: typing.List[torch.nn.Module]=[],
):
    global s3prl_input_constructor

    with torch.no_grad():
        # setup model
        model = model.eval().to(args.device)
        # get dtype of model parameters
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float
        # construct inputs
        inputs = s3prl_input_constructor(args.batch_size, args.seq_len, args.device, dtype)
        # profiling
        flops, macs, params = get_model_profile(
            model=model,
            args=[inputs, *model_args],
            kwargs=model_kwargs,
            top_modules=3,
            warm_up=10,
            as_string=args.as_string,
            output_file=os.path.join(args.log_path, "{}_pseudo.txt".format(args.upstream)),
            ignore_modules=ignore_modules,
            show_untracked=args.show_untracked,
            show_time=args.show_time,
            precision=args.precision,
        )
        del model
    
    M = macs/args.seq_len*args.sample_rate/args.batch_size if not args.as_string else "Not support --as_string"
    # summary
    logger.info("summary, sequence length = {}, batch size = {}, sample rate = {}\nsum of flops: {}\nsum of macs: {}\nparams: {} \nmacs/(l/sr)/bs: {}\n".format(args.seq_len, args.batch_size, args.sample_rate ,flops, macs, params, M))


if __name__ == "__main__":
    args = get_profiling_args()
    # initialize your model here
    if args.from_hf_hub == True:
        from huggingface_hub import snapshot_download

        print(f'[Runner] - Downloading upstream model {args.upstream} from the Hugging Face Hub')
        filepath = snapshot_download(
            repo_id = f"{args.hf_org_name}/{args.hf_repo_name}",
            revision = args.hf_revision,
            use_auth_token = True
        )
        sys.path.append(filepath)

        dependencies = (Path(filepath) / 'requirements.txt').resolve()
        print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
        for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
            print(f"{idx}. {line.strip()}")
        print(f"You can install them by:\n")
        print(f"pip install -r {dependencies}\n")

        from expert import UpstreamExpert
        Upstream = UpstreamExpert
        args.upstream_ckpt = os.path.join(filepath, "model.pt")
    else:
        try:
            Upstream = getattr(hub, args.upstream)
        except AttributeError:
            print("[UpstreamExpert] - Try to import upstream locally")
            module_path = f's3prl.upstream.{args.upstream}.expert'
            Upstream = getattr(importlib.import_module(module_path), 'UpstreamExpert')
    
    if args.upstream_ckpt:
        # if initialization need ckpt
        model = Upstream(args.upstream_ckpt)
    else:
        model = Upstream()
    
    model_args = []  # forward args
    model_kwargs = {}# forward kwargs
    pseudo_input_profiling(model, args, model_args, model_kwargs)
