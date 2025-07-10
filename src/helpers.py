import logging
import os, sys, time
from functools import wraps
import inspect
from typing import List, Any
from pydantic import BaseModel
from typing_extensions import get_type_hints
from concurrent.futures import ThreadPoolExecutor, as_completed


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Ensure the logs directory exists
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Ensure the log file exists
helpers_log_file = os.path.join(logs_dir, 'helpers.log')
if not os.path.exists(helpers_log_file):
    with open(helpers_log_file, 'w') as f:
        pass # Create an empty file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(helpers_log_file, 'a')
    ]
)
logger = logging.getLogger(__name__)

def initialize_logging(log_file: str, project_root: str = project_root):
    # Configure logging
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

    # Set up logging configuration
    log_filename = log_file.split(".")[0]
    log_file = os.path.join(project_root, 'logs', f"{log_filename}.log")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {log_file}")
    return logger

def timer_wrap(func):
    if inspect.iscoroutinefunction(func):
        # Handle async functions
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__}...")
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
            return result
        return async_wrapper
    elif inspect.isasyncgenfunction(func):
        # Handle async generator functions
        @wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__} (async generator)...")
            start_time = time.time()
            async for item in func(*args, **kwargs):
                yield item
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Async generator {func.__name__} took {elapsed_time:.4f} seconds to complete.")
        return async_gen_wrapper
    else:
        # Handle sync functions
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
            return result
        return sync_wrapper

def batch_list(lst, batch_size):
    """Yield successive batches from the list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def _all_iterables(items: List[Any]) -> bool:
    """Return *True* if every element is a list/tuple (but **not** a str)."""
    return all(isinstance(x, (list, tuple)) for x in items)

def flatten_results(func, results: list):
    """
    Flattens a list of results from parallel batches into a single output.
    
    For fields that are lists, the results are concatenated.
    For non-list fields, the first non-None and non-empty string value is taken.

    Rules applied in order:

    1. **Nested iterables** – If *every* element in ``results`` is a list/tuple,
       concatenate them into a single flat list.
    2. **Dict / Pydantic BaseModel** – If elements are dicts or BaseModels,
       merge them field‑wise: list‑typed fields are concatenated, scalar fields
       take the first non‑null/non‑empty value encountered.
    3. **Fallback** – For anything else, return ``results`` unchanged.
    
    Parameters:
        func: The original function (to extract its return type annotation)
        results: A list of results from the decorated function.
        
    Returns:
        A single merged result matching the expected output type.
    """
    # ------------------------------------------------------------------
    # Guard clauses & trivial cases
    # ------------------------------------------------------------------
    if not results:
        return None

    # ------------------------------------------------------------------
    # 1) Flatten one‑level nested lists produced by batched calls
    # ------------------------------------------------------------------
    if _all_iterables(results):
        flattened: List[Any] = []
        for batch in results:
            flattened.extend(batch)  # type: ignore[arg-type]
        logger.debug("Flattened %d nested batches into a single list.", len(results))
        return flattened

    # ------------------------------------------------------------------
    # 2) Merge dictionaries or Pydantic models
    # ------------------------------------------------------------------
    first = results[0]

    if isinstance(first, dict) or isinstance(first, BaseModel):
        # Convert everything to dict for uniform processing
        if isinstance(first, dict):
            dicts = results  # type: ignore[assignment]
            build_obj = lambda d: d  # noqa: E731
        else:
            dicts = [r.model_dump() for r in results]
            ret_type = get_type_hints(func).get("return", type(first))
            build_obj = (
                lambda d: ret_type(**d) if isinstance(ret_type, type) else first.__class__(**d)  # noqa: E731
            )

        merged: dict = {}
        for key in dicts[0].keys():
            sample_val = dicts[0][key]
            if isinstance(sample_val, list):
                # Concatenate list fields across batches
                merged[key] = [item for d in dicts for item in d.get(key, [])]
            else:
                # Take the first truthy scalar value
                merged[key] = next((d.get(key) for d in dicts if d.get(key) not in (None, "")), None)

        logger.debug("Merged %d dict/BaseModel batches into one object.", len(results))
        return build_obj(merged)

    # ------------------------------------------------------------------
    # 3) Fallback – leave result list untouched
    # ------------------------------------------------------------------
    logger.debug("Results contain heterogeneous or unsupported types; returning unchanged.")
    return results


# create generic parallel processing wrapper function
def parallel_processing_decorator(
        batch_size=10, 
        max_workers=5,  # Reduced concurrency from 5 to 3 by default.
        batch_param_name=None, 
        timeout=120,
        flatten=False,
    ):
    """
    Decorator to process function inputs in parallel using ThreadPoolExecutor.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the argument to be batched.
            try:
                if batch_param_name:
                    if batch_param_name not in kwargs:
                        raise ValueError(f"Parameter '{batch_param_name}' not found in function arguments")
                    obj_list = kwargs[batch_param_name]
                else:
                    if len(args) == 0:
                        raise ValueError("No positional arguments provided to the function")
                    obj_list = args[0]
                if not obj_list:
                    raise ValueError("No objects available to process")
                logger.info(f"Running {func.__name__} on {len(obj_list)} objects in batches of {batch_size}")

                results = []
                batches = list(batch_list(obj_list, batch_size))

                # Process batches in parallel using a ThreadPoolExecutor.
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {}
                    for batch in batches:
                        if batch_param_name:
                            # Instead of directly submitting func, we submit run_with_retries.
                            future = executor.submit(func, *args, **{batch_param_name: batch, **kwargs})
                        else:
                            future = executor.submit(func, batch, *args[1:], **kwargs)
                        future_to_batch[future] = batch

                    for future in as_completed(future_to_batch, timeout=timeout):
                        try:
                            result = future.result()
                            results.extend(result if isinstance(result, list) else [result])
                        except TimeoutError:
                            logger.warning("Processing batch timed out")
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                    if flatten:
                        try:
                            flat = flatten_results(func, results)
                            logger.info(f"Flattened results: {flat}")
                            return flat
                        except Exception as e:
                            logger.error(f"Error flattening results: {e}")
                            return results
                logger.info(f"Successfully executed {func.__name__} on {len(results)} objects")
                return results
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator