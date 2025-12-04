import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from setuptools._distutils._log import log
from setuptools.command.install_lib import install_lib as _install_lib

TORCH_CUDA_VERSION_MAP = {
    "cu128": "torch==2.7.1",
    "cu124": "torch==2.5.1"
}

def _is_in_build_isolation():
    """Check if in the pip build isolation environment"""
    for path in sys.path:
        if '/pip-build-env-' in path:
            return True
    # Check the path of the current Python executable
    if '/pip-build-env-' in sys.executable:
        return True
    # Check if the site-packages path contains the isolation environment
    import site
    try:
        site_packages = site.getsitepackages()
        for sp in site_packages:
            if '/pip-build-env-' in sp:
                return True
    except:
        pass
    return False

def _get_cuda_tag():
    """获取 CUDA 标签，如 cu128"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # 从 "Cuda compilation tools, release 12.8, V12.8.93" 提取版本号
            import re
            match = re.search(r'release (\d+)\.(\d+)', result.stdout)
            if match:
                major, minor = match.groups()
                return f"cu{major}{minor}"
    except FileNotFoundError:
        pass
    return None


# If not in an isolated environment, it means that --no-build-isolation is used
_using_no_build_isolation = not _is_in_build_isolation()
if _using_no_build_isolation:
    print(f"[build] Using no build isolation, installing build system dependencies...")
    build_sys_requires = ["setuptools>=77.0", "wheel", "gitpython", "pyyaml", "cryptography", "pip", "hatchling", "hatch-vcs", "editables", "pybind11==2.13.6"]
    install_cmd = [sys.executable, "-m", "pip", "install"] + build_sys_requires
    subprocess.check_call(install_cmd)
else:
    raise ValueError("Not in an isolated environment, please use --no-build-isolation flag.")

from tools.builder import (
    check_device,
    check_vllm_unpatch_device,
    unpatch_backend,
    unpatch_hardware_backend,
    build_backend,
)


def _read_requirements_files(requirements_paths):
    """Read the requirements file and return the dependency list"""
    requirements = []
    for requirements_path in requirements_paths:
        requirements.extend(_read_requirements_file(requirements_path))
    requirements = deduplicate_dependencies(requirements)
    return requirements


def _read_requirements_file(requirements_path):
    """Read the requirements file and return the dependency list"""
    requirements = []
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                requirements.append(line)
    except FileNotFoundError:
        print(f"[WARNING] Requirements file not found: {requirements_path}")
        return []
    return requirements


def deduplicate_dependencies(dependencies):
    """Deduplicate the dependencies"""
    seen = set()
    result = []
    for dep in dependencies:
        pkg_name = dep.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].split("!=")[0].strip()
        pkg_name_lower = pkg_name.lower()
        if pkg_name_lower not in seen:
            seen.add(pkg_name_lower)
            result.append(dep)
    return result


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
        self.domain = None
        self.extras_to_install = []  # List of extra names to install

    def finalize_options(self):
        super().finalize_options()
        if self.backend is None:
            self.backend = os.environ.get("FLAGSCALE_BACKEND")
        if self.device is None:
            self.device = os.environ.get("FLAGSCALE_DEVICE", "gpu")
        if self.domain is None:
            self.domain = os.environ.get("FLAGSCALE_DOMAIN")
        
        # Check if we need to install extra dependencies based on backend-device combination
        self.extras_to_install = []
        if self.backend and self.device:
            if hasattr(self.distribution, 'extras_require') and self.distribution.extras_require:
                available_extras = self.distribution.extras_require.keys()
                original_backends = [b.strip() for b in self.backend.split(",")]
                for backend in original_backends:
                    extra_name = f"{backend.lower()}-{self.device.lower()}"
                    if extra_name in available_extras:
                        self.extras_to_install.append(extra_name)
                        print(f"[build] Detected backend={backend} and device={self.device}, will install {extra_name} extra dependencies")
                    else:
                        print(f"[build] No extra '{extra_name}' found in extras_require, skipping")

        if self.domain is not None:
            from tools.patch.patch import domain_to_backends
            self.backend = domain_to_backends(self.domain)
            print(f"[build] Received domain = {self.domain}, will build backends = {self.backend}")

        if self.backend is not None:
            # Set the environment variables for backends and device to use in the install command
            # os.environ["FLAGSCALE_BACKEND"] = self.backend
            # os.environ["FLAGSCALE_DEVICE"] = self.device
            check_device(self.device)

            from tools.patch.patch import normalize_backend
            backends = self.backend.split(",")
            normalized = [normalize_backend(backend.strip()) for backend in backends]
            self.backend = normalized
            print(f"[build] Received backend = {self.backend}")
            print(f"[build] Received device = {self.device}")
        else:
            print(f"[build] No backend specified, just build FlagScale python codes.")

        
    def install_extras(self):
        """Install extra requirements from extras_require"""
        if not self.extras_to_install:
            return

        if hasattr(self.distribution, 'extras_require') and self.distribution.extras_require:
            all_deps_to_install = []
            
            for extra_name in self.extras_to_install:
                if extra_name in self.distribution.extras_require:
                    deps = self.distribution.extras_require[extra_name]
                    if deps:
                        print(f"[build] Found {extra_name} extra with {len(deps)} dependencies")
                        all_deps_to_install.extend(deps)
                    else:
                        print(f"[build] Warning: {extra_name} extra has no dependencies defined")
                else:
                    print(f"[build] Warning: {extra_name} extra not found in extras_require")
            
            if all_deps_to_install:
                # Remove duplicates while preserving order
                seen = set()
                unique_deps = []
                for dep in all_deps_to_install:
                    if dep not in seen:
                        seen.add(dep)
                        unique_deps.append(dep)
                
                print(f"[build] Installing {len(unique_deps)} unique dependencies from extras: {self.extras_to_install}")
                install_cmd = [sys.executable, "-m", "pip", "install"] + unique_deps
                try:
                    subprocess.check_call(install_cmd)
                    print(f"[build] Successfully installed dependencies from extras: {self.extras_to_install}")
                except subprocess.CalledProcessError as e:
                    print(f"[build] Warning: Failed to install some dependencies from extras {self.extras_to_install}: {e}")
                    # Continue build even if some dependencies fail to install
            else:
                print(f"[build] No dependencies to install from extras: {self.extras_to_install}")
        else:
            print(f"[build] Warning: distribution has no extras_require defined")


    def install_torch(self, device: str):
        if device == "gpu":
            cuda_tag = _get_cuda_tag()
            
            if cuda_tag:
                print(f"[build] Detected CUDA tag: {cuda_tag}")
                install_torch_cmd = [
                    sys.executable, "-m", "pip", "install",
                    TORCH_CUDA_VERSION_MAP[cuda_tag],
                    f"torchvision",
                    f"torchaudio",
                    "--extra-index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
                ]
            else:
                print(f"[build] CUDA not detected, installing CPU version of torch")
                install_torch_cmd = [
                    sys.executable, "-m", "pip", "install",
                    "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"
                ]
            
            subprocess.check_call(install_torch_cmd)

    def run(self):
        self.install_extras()
        self.install_torch(self.device)
        if self.backend is not None:
            build_py_cmd = self.get_finalized_command('build_py')
            build_py_cmd.backend = self.backend
            build_py_cmd.device = self.device
            build_py_cmd.domain = self.domain
            build_py_cmd.ensure_finalized()

            build_ext_cmd = self.get_finalized_command('build_ext')
            build_ext_cmd.backend = self.backend
            build_ext_cmd.device = self.device
            build_ext_cmd.domain = self.domain
            build_ext_cmd.ensure_finalized()

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
        self.domain = None

    def _unpatch_backend(self):
        main_path = os.path.dirname(os.path.abspath(__file__))
        for backend in self.backend:
            unpatch_backend(backend, self.device, main_path)

    def run(self):
        super().run()
        if self.backend:
            print(f"[build_py] Running with backend = {self.backend}")
            assert self.device is not None
            
            # At present, only vLLM supports domestic chips, and the remaining backends have not been supported yet.
            # FlagScale just modified the vLLM and Megatron-LM
            main_path = os.path.dirname(os.path.abspath(__file__))
            if "vllm" in self.backend or "Megatron-LM" in self.backend:
                if check_vllm_unpatch_device(self.device):
                    unpatch_hardware_backend(self.backend, self.device, self.build_lib, main_path)
                else:
                    self._unpatch_backend()
            else:
                self._unpatch_backend()


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
        self.domain = None

    def finalize_options(self):
        super().finalize_options()
        log.info(f"[build_ext] finalize_options called")
        log.info(f"[build_ext] self.backend = {self.backend}")
        log.info(f"[build_ext] self.device = {self.device}")
        log.info(f"[build_ext] FLAGSCALE_BACKEND env = {os.environ.get('FLAGSCALE_BACKEND')}")
        log.info(f"[build_ext] FLAGSCALE_DEVICE env = {os.environ.get('FLAGSCALE_DEVICE')}")
        if self.backend:
            log.info(f"[build_ext] Backend received: {self.backend}")
        else:
            log.info(f"[build_ext] No backend specified, skipping extension build")

    def run(self):
        if self.backend:

            main_path = os.path.dirname(os.path.abspath(__file__))
            for backend in self.backend:
                if backend == "FlagScale":
                    continue
                else:
                    # Use build_backend (unpatch already done in build_py)
                    log.info(f"[build_ext] Building backend: {backend}")
                    try:
                        build_backend(backend, self.device, main_path)
                    except Exception as e:
                        log.info(f"[build_ext] Error building {backend}: {e}")
                        raise
        else:
            log.info(f"[build_ext] No backend to build, skipping")
        super().run()


class FlagScaleInstallLib(_install_lib):
    def run(self):
        build_py_cmd = self.get_finalized_command("build_py")
        backend = getattr(build_py_cmd, "backend", None)
        print(f"[install_lib] Got backends from build_py: {backend}")
        super().run()

        raise ValueError(self.install_dir)


def _get_install_requires():
    """获取 install_requires 列表"""
    install_requires = []
    
    install_requires.extend(_read_requirements_file('requirements/requirements-base.txt'))
    install_requires.extend(_read_requirements_file('requirements/requirements-common.txt'))
    core_deps = [
        "setuptools>=77.0.0",
        "packaging>=24.2",
        "importlib_metadata>=8.5.0",
        "torch==2.7.1", 
        "torchaudio==2.7.1",
        "torchvision==0.22.1",
    ]
    
    all_deps = install_requires + core_deps
    result = deduplicate_dependencies(all_deps)
    return result


# TODO: replace with megatron-lm-fl/vllm-fl when they are published
def _get_extras_require():
    """Build the extras_require dictionary"""
    serving_common_deps = [
        'requirements/serving/requirements.txt',
    ]
    train_common_deps = [
        'requirements/train/requirements.txt',
    ]
    inference_common_deps = [
        'requirements/inference/requirements.txt',
    ]
    return {
        # domains
        'robotics-gpu': _read_requirements_files([
            'requirements/serving/robotics/requirements.txt', 
            'requirements/train/robotics/requirements.txt',
            *serving_common_deps, 
            *train_common_deps,
            *inference_common_deps,
        ]),

        # vllm
        'vllm-gpu': _read_requirements_files([
            *inference_common_deps,
            *serving_common_deps,
        ]),
        'vllm-metax': _read_requirements_files([
            *inference_common_deps,
            *serving_common_deps,
        ]),

        # megatron
        'megatron-gpu': _read_requirements_files([
            'requirements/train/megatron/requirements-cuda.txt',
            *train_common_deps,
        ]),
    }


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
    install_requires=_get_install_requires(),
    extras_require=_get_extras_require(),
    entry_points={"console_scripts": ["flagscale=flag_scale.flagscale.cli:flagscale"]},
    cmdclass={
        "build": FlagScaleBuild,
        "build_py": FlagScaleBuildPy,
        "build_ext": FlagScaleBuildExt,
    },
)
