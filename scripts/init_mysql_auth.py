from __future__ import annotations

from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.db import ensure_auth_schema_updates, execute_sql_script


def main() -> None:
    schema_path = Path("sql") / "mysql_auth_schema.sql"
    # Connect at the server level because the schema file is responsible for creating the database.
    execute_sql_script(schema_path, include_database=False)
    ensure_auth_schema_updates()
    print(f"MySQL auth schema applied from {schema_path}")


if __name__ == "__main__":
    main()
