"""QuestDB HTTP-client. Gebruikt het GET-gebaseerde /exec endpoint."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


class QuestDB:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, sql: str) -> pd.DataFrame:
        """Voer een SQL-query uit en geef het resultaat als DataFrame.

        Gebruikt /exp (CSV-export); sneller en veiliger dan /exec bij grote resultaten.
        """
        resp = requests.get(
            f"{self.base_url}/exp",
            params={"query": sql},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))
