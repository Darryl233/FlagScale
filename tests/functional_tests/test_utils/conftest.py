import os
import shutil

from datetime import datetime

import pytest

REPORT_DIR = "/workspace/report/error_log"


def _cleanup_old_logs_if_needed(report_dir, size_limit_gb=1.0):
    """
    Check the size of report_dir and delete the oldest half of logs if size exceeds limit

    Args:
        report_dir: Directory path to check and clean
        size_limit_gb: Size limit in GB (default: 1.0 GB)
    """
    if not os.path.exists(report_dir):
        return

    try:
        # Calculate total size of all files in the directory
        total_size = 0
        file_info_list = []

        for root, dirs, files in os.walk(report_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    file_mtime = os.path.getmtime(file_path)
                    total_size += file_size
                    file_info_list.append((file_path, file_mtime, file_size))
                except (OSError, IOError) as e:
                    # Skip files that can't be accessed
                    print(f"[WARNING] Cannot access file {file_path}: {e}")
                    continue

        # Convert size limit to bytes
        size_limit_bytes = size_limit_gb * 1024 * 1024 * 1024

        # If total size exceeds limit, delete oldest half of files
        if total_size > size_limit_bytes:
            # Sort files by modification time (oldest first)
            file_info_list.sort(key=lambda x: x[1])

            # Calculate how many files to delete (half of total)
            num_files_to_delete = len(file_info_list) // 2

            deleted_size = 0
            deleted_count = 0

            for i in range(num_files_to_delete):
                file_path, _, file_size = file_info_list[i]
                try:
                    os.remove(file_path)
                    deleted_size += file_size
                    deleted_count += 1
                except (OSError, IOError) as e:
                    print(f"[WARNING] Cannot delete file {file_path}: {e}")

            if deleted_count > 0:
                print(
                    f"\n[LOG CLEANUP] REPORT_DIR size ({total_size / (1024**3):.2f} GB) exceeded limit ({size_limit_gb} GB)"
                )
                print(
                    f"  Deleted {deleted_count} oldest log files, freed {deleted_size / (1024**3):.2f} GB\n"
                )

    except Exception as e:
        print(f"[WARNING] Error occurred while checking/cleaning REPORT_DIR: {e}")


def pytest_addoption(parser):
    parser.addoption(
        "--test_path", action="store", default="none", help="Base path for the test cases"
    )
    parser.addoption(
        "--test_type",
        action="store",
        default="none",
        help="Different Types of Testing (train/inference/....)",
    )
    parser.addoption(
        "--test_task", action="store", default="none", help="Model name for the test cases"
    )
    parser.addoption(
        "--test_case", action="store", default="none", help="Specific test case to run"
    )


def save_failed_log_to_report(result_path, test_name, test_type, test_task, test_case):
    """
    Save the log file from result_path to /workspace/report directory when test fails

    Args:
        result_path: Original log file path (corresponds to result_path at four locations in test_result.py)
        test_name: Test function name
        test_type: Test type
        test_task: Test task
        test_case: Test case
    """
    if not result_path:
        print(f"\n[WARNING] result_path is empty, cannot save log file\n")
        return
    # Generate saved filename with test information for easy identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.basename(result_path)
    safe_test_name = test_name.replace("test_", "")
    safe_test_case = test_case.replace("/", "_").replace("\\", "_")
    saved_filename = (
        f"{safe_test_name}_{test_type}_{test_task}_{safe_test_case}_{timestamp}_{log_filename}"
    )
    saved_path = os.path.join(REPORT_DIR, saved_filename)

    try:
        if not os.path.exists(REPORT_DIR):
            os.makedirs(REPORT_DIR)

        # Check REPORT_DIR size and clean up if exceeds 1GB
        _cleanup_old_logs_if_needed(REPORT_DIR, size_limit_gb=1.0)

        # Copy log file to report directory
        shutil.copy2(result_path, saved_path)

        # Print marker information to easily locate corresponding log
        print("\n" + "=" * 80)
        print(f"[FAILED LOG SAVED] Test failed, log file saved to {REPORT_DIR}")
        print(f"  Original log path: {result_path}")
        print(f"  Saved path: {saved_path}")
    except Exception as e:
        print(f"[ERROR] Error occurred while saving failed log to {REPORT_DIR}: {e}")


@pytest.fixture
def test_path(request):
    return request.config.getoption("--test_path")


@pytest.fixture
def test_type(request):
    return request.config.getoption("--test_type")


@pytest.fixture
def test_task(request):
    return request.config.getoption("--test_task")


@pytest.fixture
def test_case(request):
    return request.config.getoption("--test_case")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook executed after test completion, checks if test failed and saves log file if so
    """
    # Execute test and get result
    outcome = yield
    rep = outcome.get_result()

    # Only execute when test fails
    if rep.when == "call" and rep.failed:
        # Get stored result_path and test information from item.node
        result_path = getattr(item, "result_path", None)
        test_type = getattr(item, "test_type", None)
        test_task = getattr(item, "test_task", None)
        test_case = getattr(item, "test_case", None)

        if result_path:
            try:
                test_name = item.name
                save_failed_log_to_report(
                    result_path=result_path,
                    test_name=test_name,
                    test_type=test_type or "unknown",
                    test_task=test_task or "unknown",
                    test_case=test_case or "unknown",
                )
            except Exception as e:
                # If import or save fails, at least print error message
                print(f"\n[ERROR] Error occurred while saving failed log: {e}\n")
                import traceback

                traceback.print_exc()
        else:
            print(f"\n[WARNING] result_path is None, skipping log save for test: {item.name}\n")
