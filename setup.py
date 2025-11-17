import os
import shutil
import subprocess
from subprocess import CalledProcessError

import sys

from setuptools import find_packages, setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib as _install_lib

try:
    import git  # from GitPython
    import cryptography
except:
    try:
        print("[INFO] GitPython not found. Installing...")
        subprocess.check_call(["pip", "install", "gitpython", "cryptography"])
        import git
    except:
        print(
            "[ERROR] Failed to install flagscale. Please use 'pip install . --no-build-isolation' to reinstall when the pip version > 23.1."
        )
        sys.exit(1)

SUPPORTED_DEVICES = ["cpu", "gpu", "ascend", "cambricon", "bi", "metax", "kunlunxin", "musa"]
SUPPORTED_BACKENDS = ["llama.cpp", "Megatron-LM", "sglang", "vllm", "Megatron-Energon"]
VLLM_UNPATCH_DEVICES = ["ascend", "cambricon", "bi", "metax", "kunlunxin"]


def _check_backend(backend):
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Invalid backend {backend}. Supported backends are {SUPPORTED_BACKENDS}.")


def check_backends(backends):
    for backend in backends:
        _check_backend(backend)


def check_vllm_unpatch_device(device):
    is_supported = False
    for supported_device in VLLM_UNPATCH_DEVICES:
        if supported_device in device.lower():
            is_supported = True
            return is_supported
    return is_supported


def check_device(device):
    is_supported = False
    for supported_device in SUPPORTED_DEVICES:
        if supported_device in device.lower():
            is_supported = True
            return
    if not is_supported:
        raise ValueError(f"Unsupported device {device}. Supported devices are {SUPPORTED_DEVICES}.")


def _run_stream(cmd, cwd=None, env=None, prefix="[install]"):
    """
    以流式方式将子进程 stdout/stderr 同步到父进程，确保实时输出。
    """
    cmd_str = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    print(f"{prefix} $ {cmd_str}")
    try:
        # 不设置 stdout/stderr，直接继承父进程终端，保证同步输出
        result = subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return result.returncode
    except CalledProcessError as e:
        print(f"{prefix} command failed (code {e.returncode}): {cmd_str}")
        raise


# Call for the extensions
def _build_vllm(device):
    assert device != "cpu"
    vllm_path = os.path.join(os.path.dirname(__file__), "third_party", "vllm")
    if device != "gpu":
        vllm_path = os.path.join(
            os.path.dirname(__file__), "build", device, "FlagScale", "third_party", "vllm"
        )
    # Set env
    env = os.environ.copy()
    if "metax" in device.lower():
        if "MACA_PATH" not in env:
            env["MACA_PATH"] = "/opt/maca"
        if "CUDA_PATH" not in env:
            env["CUDA_PATH"] = "/usr/local/cuda"
        env["CUCC_PATH"] = f'{env["MACA_PATH"]}/tools/cu-bridge'
        env["PATH"] = (
            f'{env["CUDA_PATH"]}/bin:'
            f'{env["MACA_PATH"]}/mxgpu_llvm/bin:'
            f'{env["MACA_PATH"]}/bin:'
            f'{env["CUCC_PATH"]}/tools:'
            f'{env["CUCC_PATH"]}/bin:'
            f'{env.get("PATH", "")}'
        )
        env["LD_LIBRARY_PATH"] = (
            f'{env["MACA_PATH"]}/lib:'
            f'{env["MACA_PATH"]}/ompi/lib:'
            f'{env["MACA_PATH"]}/mxgpu_llvm/lib:'
            f'{env.get("LD_LIBRARY_PATH", "")}'
        )
        env["VLLM_INSTALL_PUNICA_KERNELS"] = "1"
    _run_stream(["uv", "pip", "install", ".", "--no-build-isolation", "--verbose"], cwd=vllm_path, env=env, prefix="[build_ext]")


def _build_sglang(device):
    assert device != "cpu"
    sglang_path = os.path.join(os.path.dirname(__file__), "third_party", "sglang")
    if device != "gpu":
        sglang_path = os.path.join(
            os.path.dirname(__file__), "build", device, "FlagScale", "third_party", "sglang"
        )
    _run_stream(
        ["uv", "pip", "install", "-e", "python[all]", "--no-build-isolation", "--verbose"],
        cwd=sglang_path,
        prefix="[build_ext]",
    )


def _build_llama_cpp(device):
    llama_cpp_path = os.path.join(os.path.dirname(__file__), "third_party", "llama.cpp")
    print(f"[build_ext] Build llama.cpp for {device}")
    if device == "gpu":
        subprocess.check_call(["cmake", "-B", "build", "-DGGML_CUDA=ON"], cwd=llama_cpp_path)
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", "-j64"], cwd=llama_cpp_path
        )
    elif device == "musa":
        subprocess.check_call(["cmake", "-B", "build", "-DGGML_MUSA=ON"], cwd=llama_cpp_path)
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", "-j8"], cwd=llama_cpp_path
        )
    elif device == "cpu":
        subprocess.check_call(["cmake", "-B", "build"], cwd=llama_cpp_path)
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", "-j8"], cwd=llama_cpp_path
        )
    else:
        raise ValueError(f"Unsupported device {device} for llama.cpp backend.")


def _build_megatron_energon(device):
    try:
        import editables
        import hatch_vcs
        import hatchling
    except:
        try:
            print("[INFO] hatchling not found. Installing...")
            subprocess.check_call(
                ["uv", "pip", "install", "hatchling", "--no-build-isolation"]
            )
            subprocess.check_call(
                ["uv", "pip", "install", "hatch-vcs", "--no-build-isolation"]
            )
            subprocess.check_call(
                ["uv", "pip", "install", "editables", "--no-build-isolation"]
            )
            import editables
            import hatch_vcs
            import hatchling
        except:
            print("[ERROR] Failed to install hatchling, hatch-vcs and editables.")
            sys.exit(1)
    energon_path = os.path.join(os.path.dirname(__file__), "third_party", "Megatron-Energon")
    _run_stream(["uv", "pip", "install", "-e", ".", "--no-build-isolation", "--verbose"], cwd=energon_path, prefix="[build_ext]")


class FlagScaleBuild(_build):
    """
    Build the FlagScale backends.
    """

    user_options = _build.user_options + [
        ('backend=', None, 'Build backends'),
        ('device=', None, 'Device type for build'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.backend = None
        self.device = None

    def finalize_options(self):
        super().finalize_options()
        if self.backend is None:
            self.backend = os.environ.get("FLAGSCALE_BACKEND")
        if self.device is None:
            self.device = os.environ.get("FLAGSCALE_DEVICE", "gpu")
        if self.backend is not None:
            # Set the environment variables for backends and device to use in the install command
            # os.environ["FLAGSCALE_BACKEND"] = self.backend
            # os.environ["FLAGSCALE_DEVICE"] = self.device
            check_device(self.device)

            from tools.patch.patch import normalize_backend

            backends = self.backend.split(",")
            self.backend = [normalize_backend(backend.strip()) for backend in backends]
            print(f"[build] Received backend = {self.backend}")
            print(f"[build] Received device = {self.device}")
        else:
            print(f"[build] No backend specified, just build FlagScale python codes.")

    def run(self):
        print("111111", self.backend)
        # self.backend = ["Megatron-LM"]
        # self.device = "gpu"
        if self.backend is not None:
            build_install_cmd = FlagScaleInstall(backend=self.backend, device=self.device)

            build_py_cmd = self.get_finalized_command('build_py')
            build_py_cmd.backend = self.backend
            build_py_cmd.device = self.device
            build_py_cmd.ensure_finalized()

            build_ext_cmd = self.get_finalized_command('build_ext')
            build_ext_cmd.backend = self.backend
            build_ext_cmd.device = self.device
            build_ext_cmd.ensure_finalized()
           
            build_install_cmd.run()
            # self.run_command('install_requirements')
            self.run_command('build_py')
            self.run_command('build_ext')
        super().run()


class FlagScaleBuildPy(_build_py):
    """
    Unpatch the FlagScale backends.
    """

    user_options = _build_py.user_options + [
        ('backend=', None, 'Build backends'),
        ('device=', None, 'Device type for build'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.backend = None
        self.device = None

    def unpatch_backend(self):
        from tools.patch.unpatch import unpatch

        main_path = os.path.dirname(__file__)
        for backend in self.backend:
            if backend == "FlagScale":
                continue
            backend_commit = None
            if backend == "Megatron-LM":
                backend_commit = os.getenv(f"FLAGSCALE_MEGATRON_COMMIT", None)
            elif backend == "Megatron-Energon":
                backend_commit = os.getenv(f"FLAGSCALE_ENERGON_COMMIT", None)
            elif backend == "sglang":
                backend_commit = os.getenv(f"FLAGSCALE_SGLANG_COMMIT", None)
            elif backend == "vllm":
                backend_commit = os.getenv(f"FLAGSCALE_VLLM_COMMIT", None)
            elif backend == "llama.cpp":
                backend_commit = os.getenv(f"FLAGSCALE_LLAMA_CPP_COMMIT", None)
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            print(f"[build_py] Device {self.device} initializing the {backend} backend.")
            force = os.getenv("FLAGSCALE_FORCE_INIT", False)
            unpatch(
                main_path,
                src,
                dst,
                backend,
                force=force,
                backend_commit=backend_commit,
                fs_extension=True,
            )
            # ===== Copy for packaging =====
            if backend == "Megatron-LM":
                rel_src = os.path.join("third_party", backend, "megatron")
                abs_src = os.path.join(main_path, rel_src)
                abs_dst = os.path.join(self.build_lib, "flag_scale", rel_src)
                print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                if os.path.exists(abs_dst):
                    shutil.rmtree(abs_dst)
                shutil.copytree(abs_src, abs_dst)

        # ===== Copy for packaging for Megatron-Energon =====
        if "Megatron-Energon" in self.backend:
            assert "Megatron-LM" in self.backend, "Megatron-Energon requires Megatron-LM"
            abs_src = os.path.join(
                main_path, "third_party", "Megatron-Energon", "src", "megatron", "energon"
            )
            abs_dst = os.path.join(
                self.build_lib, "flag_scale", "third_party", "Megatron-LM", "megatron", "energon"
            )
            print(f"[build_py] Copying {abs_src} -> {abs_dst}")
            if os.path.exists(abs_dst):
                shutil.rmtree(abs_dst)
            shutil.copytree(abs_src, abs_dst)

            # Source code for Megatron-Energon is copied to the megatron directory
            abs_dst = os.path.join(main_path, "third_party", "Megatron-LM", "megatron", "energon")
            print(f"[build_py] Copying {abs_src} -> {abs_dst}")
            if os.path.exists(abs_dst):
                shutil.rmtree(abs_dst)
            shutil.copytree(abs_src, abs_dst)

    def run(self):
        super().run()
        if self.backend:
            print(f"[build_py] Running with backend = {self.backend}")
            assert self.device is not None
            from tools.patch.unpatch import apply_hardware_patch

            # At present, only vLLM supports domestic chips, and the remaining backends have not been supported yet.
            # FlagScale just modified the vLLM and Megatron-LM
            main_path = os.path.dirname(__file__)
            if "vllm" in self.backend or "Megatron-LM" in self.backend:
                if check_vllm_unpatch_device(self.device):
                    print(f"[build_py] Device {self.device} unpatching the vllm backend.")
                    # Unpatch the backed in specified device
                    from git import Repo

                    main_repo = Repo(main_path)
                    commit = os.getenv("FLAGSCALE_UNPATCH_COMMIT", None)
                    if commit is None:
                        commit = main_repo.head.commit.hexsha
                    # Checkout to the commit and apply the patch to build FlagScale
                    key_path = os.getenv("FLAGSCALE_KEY_PATH", None)
                    apply_hardware_patch(
                        self.device, self.backend, commit, main_path, True, key_path=key_path
                    )
                    build_lib_flagscale = os.path.join(self.build_lib, "flag_scale")
                    src_flagscale = os.path.join(main_path, "build", self.device, "FlagScale")

                    for f in os.listdir(build_lib_flagscale):
                        if f.endswith(".py"):
                            file_path = os.path.join(build_lib_flagscale, f)
                            print(f"[build_py] Removing file {file_path}")
                            os.remove(file_path)

                    for f in os.listdir(src_flagscale):
                        if f.endswith(".py"):
                            src_file = os.path.join(src_flagscale, f)
                            dst_file = os.path.join(build_lib_flagscale, f)
                            print(f"[build_py] Copying file {src_file} -> {dst_file}")
                            shutil.copy2(src_file, dst_file)

                    dirs_to_copy = ["flagscale", "examples", "tools", "tests"]
                    for d in dirs_to_copy:
                        src_dir = os.path.join(src_flagscale, d)
                        dst_dir = os.path.join(build_lib_flagscale, d)
                        if os.path.exists(dst_dir):
                            print(f"[build_py] Removing directory {dst_dir}")
                            shutil.rmtree(dst_dir)
                        if os.path.exists(src_dir):
                            print(f"[build_py] Copying directory {src_dir} -> {dst_dir}")
                            shutil.copytree(src_dir, dst_dir)

                    # ===== Copy for packaging =====
                    if "Megatron-LM" in self.backend:
                        rel_src = os.path.join(
                            main_path,
                            "build",
                            self.device,
                            "FlagScale",
                            "third_party",
                            "Megatron-LM",
                            "megatron",
                        )
                        abs_src = os.path.join(main_path, rel_src)
                        abs_dst = os.path.join(
                            self.build_lib, "flag_scale", "third_party", "Megatron-LM", "megatron"
                        )
                        print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                        if os.path.exists(abs_dst):
                            shutil.rmtree(abs_dst)
                        shutil.copytree(abs_src, abs_dst)

                    if "Megatron-Energon" in self.backend:
                        assert (
                            "Megatron-LM" in self.backend
                        ), "Megatron-Energon requires Megatron-LM"
                        abs_src = os.path.join(
                            main_path,
                            "build",
                            self.device,
                            "FlagScale",
                            "third_party",
                            "Megatron-Energon",
                            "src",
                            "megatron",
                            "energon",
                        )
                        abs_dst = os.path.join(
                            self.build_lib,
                            "flag_scale",
                            "third_party",
                            "Megatron-LM",
                            "megatron",
                            "energon",
                        )
                        print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                        if os.path.exists(abs_dst):
                            shutil.rmtree(abs_dst)
                        shutil.copytree(abs_src, abs_dst)

                        abs_dst = os.path.join(
                            main_path,
                            "build",
                            self.device,
                            "FlagScale",
                            "third_party",
                            "Megatron-LM",
                            "megatron",
                            "energon",
                        )
                        print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                        if os.path.exists(abs_dst):
                            shutil.rmtree(abs_dst)
                        shutil.copytree(abs_src, abs_dst)
                else:
                    self.unpatch_backend()
            else:
                self.unpatch_backend()


class FlagScaleBuildExt(_build_ext):
    """
    Build or pip install the FlagScale backends.
    """

    user_options = _build_py.user_options + [
        ('backend=', None, 'Build backends'),
        ('device=', None, 'Device type for build'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.backend = None
        self.device = None

    def finalize_options(self):
        super().finalize_options()
        if self.backend:
            print(f"[build_ext] Backend received: {self.backend}")

    def run(self):
        if self.backend:
            print(f"[build_ext] Building extensions for backend = {self.backend}")
            for backend in self.backend:
                if backend == "FlagScale":
                    continue
                elif backend == "vllm":
                    # 在构建 vllm 之前安装其依赖
                    if self.device == "gpu":
                        main_path = os.path.dirname(__file__)
                        vllm_req_dir = os.path.join(main_path, "third_party", "vllm", "requirements")
                        if os.path.exists(vllm_req_dir):
                            print("[build_ext] Installing vllm build dependencies...")
                            req_files = ["build.txt", "cuda.txt", "common.txt", "dev.txt"]
                            for req_file in req_files:
                                req_path = os.path.join(vllm_req_dir, req_file)
                                if os.path.exists(req_path):
                                    _run_stream(["uv", "pip", "install", "-r", req_path], prefix="[build_ext]")
                            # 安装 mamba
                            _run_stream(["uv", "pip", "install", "git+https://github.com/state-spaces/mamba.git@v2.2.4"], prefix="[build_ext]")
                    _build_vllm(self.device)
                elif backend == "sglang":
                    _build_sglang(self.device)
                elif backend == "llama.cpp":
                    _build_llama_cpp(self.device)
                elif backend == "Megatron-LM":
                    print(
                        f"[build_ext] Megatron-LM does not need to be built, just copy the source code."
                    )
                elif backend == "Megatron-Energon":
                    _build_megatron_energon(self.device)
                    print(
                        f"[build_ext] Megatron-Energon will be copied to megatron after installed."
                    )
                else:
                    raise ValueError(f"Unknown backend: {backend}")
        super().run()


def _determine_env_type(backends):
    """根据 backend 判断是 train 还是 inference 环境"""
    train_backends = ["Megatron-LM", "Megatron-Energon"]
    inference_backends = ["vllm", "sglang", "llama.cpp"]
    
    has_train = any(b in train_backends for b in backends)
    has_inference = any(b in inference_backends for b in backends)
    
    if has_train and has_inference:
        return "both"  # 需要安装 train 和 inference 的依赖
    elif has_train:
        return "train"
    elif has_inference:
        return "inference"
    else:
        return None  # 没有指定 backend，不安装额外依赖


def _read_requirements(file_path):
    """读取 requirements 文件为列表（忽略注释与空行）"""
    reqs = []
    if not os.path.exists(file_path):
        return reqs
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


def _install_heavy_dependencies(env_type, device, main_path):
    """安装复杂依赖（按需执行）：torch/cuDNN/TE/flash-attn/apex 等"""
    import tempfile

    if device == "gpu" and env_type in ["train", "both", "inference"]:
        # PyTorch + deepspeed
        try:
            _run_stream([
                "uv", "pip", "install",
                "torch==2.7.1+cu128", "torchaudio==2.7.1+cu128", "torchvision==0.22.1+cu128",
                "--extra-index-url", "https://download.pytorch.org/whl/cu128"
            ])
            _run_stream(["uv", "pip", "install", "deepspeed"])
        except Exception as e:
            print(f"[install] Warning: Torch installation failed/skipped: {e}")

    if device == "gpu" and env_type in ["train", "both"]:
        # TransformerEngine
        try:
            print("[install] Installing TransformerEngine...")
            with tempfile.TemporaryDirectory() as tmpdir:
                te_path = os.path.join(tmpdir, "TransformerEngine")
                subprocess.check_call([
                    "git", "clone", "--recursive", "https://github.com/NVIDIA/TransformerEngine.git", te_path
                ])
                subprocess.check_call(["git", "checkout", "e9a5fa4e"], cwd=te_path)
                _run_stream(["uv", "pip", "install", "."], cwd=te_path)
        except Exception as e:
            print(f"[install] Warning: TransformerEngine install skipped: {e}")

        # cuDNN frontend
        try:
            print("[install] Installing cuDNN frontend...")
            _run_stream(["uv", "pip", "install", "nvidia-cudnn-cu12==9.7.1.26"])
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            _run_stream(["pip", "install", "--no-build-isolation ", "nvidia-cudnn-frontend"], env=env)
        except Exception as e:
            print(f"[install] Warning: cuDNN frontend install skipped: {e}")

        # flash-attn（尽力安装）
        try:
            print("[install] Installing flash-attention (best-effort)...")
            nvcc_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
            cu_match = None
            for line in nvcc_output.split('\n'):
                if "Cuda compilation tools" in line:
                    cu_match = line.split()[4]
                    break
            if cu_match:
                cu = cu_match.split('.')[0]
                torch_output = subprocess.check_output(["uv", "pip", "show", "torch"]).decode()
                torch_version = None
                for line in torch_output.split('\n'):
                    if line.startswith("Version:"):
                        torch_version = line.split()[1].split('+')[0]
                        break
                if torch_version:
                    torch_v = '.'.join(torch_version.split('.')[:2])
                    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
                    try:
                        gxx_output = subprocess.check_output(["g++", "--version"]).decode()
                        gxx_version = gxx_output.split('\n')[0].split()[-1].split('.')[0]
                    except:
                        gxx_version = "11"
                    flash_attn_version = "2.8.0.post2"
                    wheel_name = f"flash_attn-{flash_attn_version}+cu{cu}torch{torch_v}cxx{gxx_version}abiFALSE-cp{python_version}-cp{python_version}-linux_x86_64.whl"
                    wheel_url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{flash_attn_version}/{wheel_name}"
                    with tempfile.TemporaryDirectory() as tmpdir:
                        wheel_path = os.path.join(tmpdir, wheel_name)
                        subprocess.check_call([
                            "wget", "--continue", "--timeout=60", "--no-check-certificate",
                            "--tries=5", "--waitretry=10", wheel_url, "-O", wheel_path
                        ])
                        _run_stream(["uv", "pip", "install", "--no-cache-dir", wheel_path])
                    _run_stream([
                        "uv", "pip", "install", "--no-build-isolation",
                        "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
                    ])
                    python_path = subprocess.check_output([
                        sys.executable, "-c", "import site; print(site.getsitepackages()[0])"
                    ]).decode().strip()
                    hopper_path = os.path.join(python_path, "flashattn_hopper")
                    os.makedirs(hopper_path, exist_ok=True)
                    subprocess.check_call([
                        "wget", "-P", hopper_path,
                        "https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py"
                    ])
        except Exception as e:
            print(f"[install] Warning: flash-attention install skipped: {e}")

        # apex
        try:
            print("[install] Installing apex...")
            with tempfile.TemporaryDirectory() as tmpdir:
                apex_path = os.path.join(tmpdir, "apex")
                subprocess.check_call(["git", "clone", "https://github.com/NVIDIA/apex", apex_path])
                _run_stream([
                    "pip", "install", "-v", "--disable-pip-version-check",
                    "--no-cache-dir", "--no-build-isolation",
                    "--config-settings=--build-option=--cpp_ext",
                    "--config-settings=--build-option=--cuda_ext",
                    "."
                ], cwd=apex_path)
        except Exception as e:
            print(f"[install] Warning: apex install skipped: {e}")


        _fix_torch_elastic()


def _fix_torch_elastic():
    """修复 torch distributed elastic 的自动容错功能"""
    try:
        import torch
        torch_version = torch.__version__
        site_packages = subprocess.check_output([
            sys.executable, "-c", "import site; print(site.getsitepackages()[0])"
        ]).decode().strip()
        file_path = os.path.join(site_packages, "torch", "distributed", "elastic", "agent", "server", "api.py")
        
        if not os.path.exists(file_path):
            return
        
        if "2.5.1" in torch_version:
            # 修复 2.5.1 版本
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 893:
                if 'if num_nodes_waiting > 0:' in lines[892]:
                    lines[892] = '                if num_nodes_waiting > 0 and self._remaining_restarts > 0:\n'
                if len(lines) >= 902 and 'self._restart_workers(self._worker_group)' in lines[901]:
                    lines[901] = '                    self._remaining_restarts -= 1; self._restart_workers(self._worker_group)\n'
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
        
        elif "2.6.0" in torch_version or "2.7.0" in torch_version or "2.7.1" in torch_version:
            # 修复 2.6.0/2.7.0/2.7.1 版本
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 908:
                if 'if num_nodes_waiting > 0:' in lines[907]:
                    lines[907] = '                if num_nodes_waiting > 0 and self._remaining_restarts > 0:\n'
                if len(lines) >= 917 and 'self._restart_workers(self._worker_group)' in lines[916]:
                    lines[916] = '                    self._remaining_restarts -= 1; self._restart_workers(self._worker_group)\n'
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
    except Exception as e:
        print(f"[install] Warning: Failed to fix torch elastic: {e}")


class FlagScaleInstall:
    """
    安装 FlagScale 及其依赖
    """
    
    def __init__(self, backend=None, device=None, skip_deps=False):
        self.backend = backend
        self.device = device
        self.skip_deps = skip_deps
    
    def run(self):
        # 在 build 之前安装依赖（轻依赖通过 extras，重依赖通过脚本）
        print(1111111111111)
        if not self.skip_deps and self.backend:
            env_type = _determine_env_type(self.backend)

            main_path = os.path.dirname(__file__)
            # 轻依赖：读取 requirements 作为 extras 列表并安装
            base_reqs = os.path.join(main_path, "requirements", "requirements-base.txt")
            common_reqs = os.path.join(main_path, "requirements", "requirements-common.txt")
            train_reqs = os.path.join(main_path, "requirements", "train", "requirements.txt")
            inference_reqs = os.path.join(main_path, "requirements", "inference", "requirements.txt")
            serving_reqs = os.path.join(main_path, "requirements", "serving", "requirements.txt")
            megatron_cuda_reqs = os.path.join(main_path, "requirements", "train", "megatron", "requirements-cuda.txt")
            print(f"[install] Detected environment type: {env_type}, device: {self.device}, backends: {self.backend}")
            # 始终先升级 pip
            _run_stream(["uv", "pip", "install", "--upgrade", "pip"])

            # 安装通用轻依赖
            if env_type in ["train", "inference", "both"]:
                print(f"[install] Installing base pip dependencies ({base_reqs}) before build...")
                _run_stream(["uv", "pip", "install", "-r", base_reqs, "--verbose"])
                _run_stream(["pip", "install", "-r", common_reqs, "--verbose"])

            if env_type in ["train", "both"]:
                print(f"[install] Installing training pip dependencies ({train_reqs}) before build...")
                _run_stream(["uv", "pip", "install", "-r", train_reqs, "--verbose"])
                if self.device == "gpu":
                    _run_stream(["uv", "pip", "install", "--no-build-isolation", "-r", megatron_cuda_reqs, "--verbose"])

            if env_type in ["inference", "both"]:
                print(f"[install] Installing inference pip dependencies ({inference_reqs}) before build...")
                _run_stream(["uv", "pip", "install", "-r", inference_reqs, "--verbose"])
                if self.device == "gpu":
                    _run_stream(["uv", "pip", "install", "-r", serving_reqs, "--verbose"])

    

            # 安装复杂依赖
            print(f"[install] Installing heavy pip dependencies before build... ,env_type={env_type}")
            import time
            time.sleep(10)
            _install_heavy_dependencies(env_type, self.device, main_path)
            # _fix_torch_elastic()
            # 公共：安装 FlagGems
            try:
                _run_stream(["uv", "pip", "install", "--no-build-isolation", "git+https://github.com/FlagOpen/FlagGems.git@release_v1.0.0"])
            except Exception as e:
                print(f"[install] Warning: FlagGems install skipped: {e}")


class FlagScaleInstallLib(_install_lib):
    def run(self):
        build_py_cmd = self.get_finalized_command("build_py")
        backend = getattr(build_py_cmd, "backend", None)
        print(f"[install_lib] Got backends from build_py: {backend}")
        super().run()


from version import FLAGSCALE_VERSION

setup(
    name="flag_scale",
    version=FLAGSCALE_VERSION,
    description="FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). ",
    url="https://github.com/FlagOpen/FlagScale",
    packages=[
        "flag_scale",
        "flag_scale.flagscale",
        "flag_scale.examples",
        "flag_scale.tools",
        "flag_scale.tests",
    ],
    package_dir={
        "flag_scale": "",
        "flag_scale.flagscale": "flagscale",
        "flag_scale.examples": "examples",
        "flag_scale.tools": "tools",
        "flag_scale.tests": "tests",
    },
    package_data={
        "flag_scale.flagscale": ["**/*"],
        "flag_scale.examples": ["**/*"],
        "flag_scale.tools": ["**/*"],
        "flag_scale.tests": ["**/*"],
    },
    install_requires=[
        "click",
        "gitpython",
        "cryptography",
        "setuptools>=77.0.0",
        "packaging>=24.2",
        "importlib_metadata>=8.5.0",
    ],
    entry_points={"console_scripts": ["flagscale=flag_scale.flagscale.cli:flagscale"]},
    cmdclass={
        "build": FlagScaleBuild,
        "build_py": FlagScaleBuildPy,
        "build_ext": FlagScaleBuildExt,
    },
)
