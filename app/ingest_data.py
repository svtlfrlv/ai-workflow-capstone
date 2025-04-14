import json
import logging
from pathlib import Path
from typing import Optional, Union
import glob
import pandas as pd

logger = logging.getLogger("data_ingestion")


def load_json_file(file_path: Path) -> list:
    """Loads a single JSON file and returns its contents as a list of records.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of records from the JSON file.

    Raises:
        ValueError: If JSON is invalid or not a list.
    """
    try:
        with file_path.open("r") as f:
            content = json.load(f)
            if not isinstance(content, list):
                logger.warning(f"JSON in {file_path} is not a list, skipping")
                return []
            return content
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {file_path}")
        return []
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []


def validate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensures DataFrame has required columns, adding defaults if needed.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with required columns.

    Raises:
        ValueError: If critical columns are missing.
    """
    needed = {"country", "price", "year", "month", "day"}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"Data missing columns: {missing}")

    result = data[["country", "price", "year", "month", "day"]].copy()
    if "times_viewed" not in data.columns:
        logger.info("No times_viewed column, setting to 0")
        result["times_viewed"] = 0
    else:
        result["times_viewed"] = data["times_viewed"]

    return result


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans numeric columns and removes invalid rows.

    Args:
        data: Input DataFrame.

    Returns:
        Cleaned DataFrame.

    Raises:
        ValueError: If no valid data remains.
    """
    numeric_cols = ["year", "month", "day", "price", "times_viewed"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["times_viewed"] = data["times_viewed"].fillna(0)

    cleaned = data.dropna(subset=["country", "price", "year", "month"])
    if cleaned.empty:
        raise ValueError("No valid data after cleaning")

    return cleaned


def filter_top_countries(data: pd.DataFrame, count: int) -> pd.DataFrame:
    """Filters DataFrame to top N countries by total revenue.

    Args:
        data: Input DataFrame.
        count: Number of top countries to keep.

    Returns:
        Filtered DataFrame.
    """
    revenue_by_country = data.groupby("country")["price"].sum()
    top_countries = revenue_by_country.nlargest(count).index
    logger.info(f"Keeping top {count} countries: {list(top_countries)}")
    return data[data["country"].isin(top_countries)]


def process(
    input_folder: str,
    output_file: Optional[str] = None,
    top_revenue_countries_count: Optional[int] = None
) -> pd.DataFrame:
    """Processes JSON invoice files from a folder into monthly revenue data.

    Args:
        input_folder: Path to folder with JSON files.
        output_file: Optional path to save CSV output.
        top_revenue_countries_count: Optional number of top countries to filter.

    Returns:
        DataFrame with monthly revenue per country.

    Raises:
        FileNotFoundError: If input folder or JSON files are missing.
        ValueError: If data is invalid or empty.
    """
    folder = Path(input_folder)
    if not folder.exists():
        logger.error(f"Input folder not found: {input_folder}")
        raise FileNotFoundError(f"Folder does not exist: {input_folder}")

    # Find JSON files
    json_files = list(folder.glob("invoices-*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {input_folder}")
        raise FileNotFoundError(f"No invoices-*.json files in {input_folder}")

    logger.info(f"Processing {len(json_files)} JSON files from {input_folder}")

    # Load all data
    records = []
    for file in json_files:
        records.extend(load_json_file(file))

    if not records:
        logger.error("No valid data loaded")
        raise ValueError("No data could be loaded from JSON files")

    # Create DataFrame
    data = pd.DataFrame(records)

    # Validate and clean
    data = validate_columns(data)
    data = clean_data(data)

    # Filter top countries if specified
    if top_revenue_countries_count is not None:
        data = filter_top_countries(data, top_revenue_countries_count)

    # Aggregate to monthly revenue
    result = (
        data.groupby(["year", "month", "country"])
        .agg({"price": "sum", "times_viewed": "sum"})
        .reset_index()
        .rename(columns={"price": "revenue"})
    )

    # Add date column
    result["date"] = pd.to_datetime(
        result[["year", "month"]].assign(day=1)
    )

    # Sort results
    result = result[["year", "month", "country",
                     "revenue", "times_viewed", "date"]]
    result = result.sort_values(["year", "month", "country"])

    # Save output if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info(f"Saved output to {output_path}")

    return result
